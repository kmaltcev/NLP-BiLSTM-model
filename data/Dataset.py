import re
import ssl
import nltk
import numpy as np
import pandas as pd

from tqdm import tqdm
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from utils.utils import read_books
from joblib import Parallel, delayed

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')


def lemmatize(text):
    m = Mystem()
    lemma = m.lemmatize(' '.join(text))
    return ''.join(lemma)


def embedding(text, elmo):
    return elmo().get_elmo_vectors(text)


class Dataset:
    prep_data = None

    def __init__(self, data):
        if data is None:
            raise ValueError("No dataset provided")
        self.data = pd.DataFrame({"label": range(len(data)),
                                  "author": [author for author in data],
                                  "text": [read_books(author) for author in data]})

    def __str__(self):
        return str(self.data)

    def preprocess(self):
        final_dataset = pd.DataFrame(columns=self.data.columns.values)
        for index, data_row in self.data.iterrows():
            text = re.sub(r'[^ЁёА-я\s]', ' ', data_row['text'])
            text = ' '.join([w for w in text.split() if len(w) > 1])
            text = re.sub(r' {2,}', ' ', text)
            text = text.lower().split(' ')
            text_batch = [text[i: i + 1000] for i in range(0, len(text), 1000)]
            text = Parallel(n_jobs=-1)(
                delayed(lemmatize)(t) for t in tqdm(text_batch, desc=f"Lemmatizing {data_row['author']}"))
            text = ' '.join(text).split(' ')
            tokens = [text[i] for i in tqdm(range(len(text)), desc=f"Preprocessing {data_row['author']}")
                      if text[i] not in stopwords.words('russian')
                      and text[i] != " " and text[i].strip() not in punctuation]
            final_dataset = final_dataset \
                .append({"label": index, "author": data_row['author'], "text": " ".join(tokens)}, ignore_index=True)
        self.data = final_dataset

    def chunking(self, chunk_size=40):
        chunked = pd.DataFrame(columns=self.data.columns.values)
        for idx, book in self.data.iterrows():
            words = book['text'].split()
            chunks = [words[i - chunk_size:i] for i in range(chunk_size, len(words), chunk_size)]
            temp_df = pd.DataFrame({'label': book['label'], 'author': book['author'], 'text': [chunks]})
            chunked = chunked.append(temp_df)
        self.prep_data = chunked

    def embedding_depr(self, embeddder):
        embeddings = []

        for index, data_row in tqdm(self.prep_data.iterrows(), total=self.prep_data.shape[0],
                                    desc="ELMo embedding process:"):
            simple_elmo = embeddder()
            res = simple_elmo.get_elmo_vectors(data_row['text'])
            embeddings.append(res)

        self.prep_data['embeddings'] = list(np.concatenate(embeddings))
        return self.prep_data['embeddings'].shape

    def embedding(self, elmo):
        embeddings = Parallel(n_jobs=2)(delayed(embedding)(data_row['text'], elmo) for index, data_row in
                                        tqdm(self.prep_data.iterrows(), total=self.prep_data.shape[0],
                                             desc="ELMo embedding process:"))
        embeddings = [list(emb) for emb in embeddings]
        embeddings = [pd.DataFrame({'label': self.prep_data['label'].values[idx],
                                    'author': self.prep_data['author'].values[idx],
                                    'text': self.prep_data['text'].values[idx],
                                    'embeddings': work})
                      for idx, work in enumerate(embeddings)]

        self.prep_data = pd.concat(embeddings)
        return self.prep_data['embeddings'].shape
