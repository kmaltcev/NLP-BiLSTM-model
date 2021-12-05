import re
import pandas as pd

from tqdm import tqdm
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from utils.utils import read_books
from joblib import Parallel, delayed


def lemmatize(text):
    m = Mystem()
    lemma = m.lemmatize(' '.join(text))
    return ''.join(lemma)


def embedding(text, elmo):
    return elmo().get_elmo_vectors(text)


class Dataset:
    data, embeddings = None, None

    def __init__(self, names):
        if names is None:
            raise ValueError("No data provided")
        books = read_books(names)
        self.data = pd.DataFrame(columns=["label", "author", "text"])
        for idx, name in enumerate(names):
            self.data = self.data.append({"label": idx,
                                          "author": name,
                                          "text": " ".join(books[name])}, ignore_index=True)

    def __str__(self):
        return str(self.data)

    def preprocess(self):
        final_dataset = pd.DataFrame(columns=self.data.columns)
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
            final_dataset = final_dataset.append({"label": index,
                                                  "author": data_row['author'],
                                                  "text": " ".join(tokens)}, ignore_index=True)
        self.data = final_dataset

    def chunking(self, chunk_size=40):
        chunked = pd.DataFrame(columns=self.data.columns.values)
        for idx, book in self.data.iterrows():
            words = book['text'].split()
            chunks = [words[i - chunk_size:i] for i in range(chunk_size, len(words), chunk_size)]
            temp_df = pd.DataFrame({'label': book['label'], 'author': book['author'], 'text': [chunks]})
            chunked = chunked.append(temp_df)
        self.data = chunked

    def embedding(self, elmo):
        self.embeddings = [elmo().get_elmo_vectors(data_row['text']) for index, data_row in
                           tqdm(self.data.iterrows(), total=self.data.shape[0], desc="ELMo embedding process:")]
        list_embeddings = [list(emb) for emb in self.embeddings]
        list_embeddings = [pd.DataFrame({'label': self.data['label'].values[idx],
                                         'author': self.data['author'].values[idx],
                                         'text': self.data['text'].values[idx],
                                         'embeddings': work})
                           for idx, work in enumerate(list_embeddings)]
        self.data = pd.concat(list_embeddings)
        return self.data['embeddings'].shape
