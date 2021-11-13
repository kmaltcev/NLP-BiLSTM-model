import re
import nltk
import numpy as np
import pandas as pd

from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from tqdm import tqdm

from utils.utils import read_books

nltk.download('stopwords')


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
        mystem = Mystem()
        for i in tqdm(range(2), desc='Preprocessing:'):
            text = re.sub(r'[^ЁёА-я\s]', ' ', self.data.values[i][2])
            text = ' '.join([w for w in text.split() if len(w) > 1])
            text = re.sub(r' {2,}', ' ', text)
            lemmas = mystem.lemmatize(text.lower())
            tokens = [token for token in lemmas if token not in stopwords.words('russian')
                      and token != " "
                      and token.strip() not in punctuation]
            final_dataset = final_dataset \
                .append({"label": i, "author": self.data.values[i][1], "text": " ".join(tokens)}, ignore_index=True)
        self.data = final_dataset

    def chunking(self, chunk_size=40):
        chunked = pd.DataFrame(columns=self.data.columns.values)
        for idx, book in self.data.iterrows():
            words = book['text'].split()
            chunks = [words[i - chunk_size:i] for i in range(chunk_size, len(words), chunk_size)]
            temp_df = pd.DataFrame({'label': book['label'], 'author': book['author'], 'text': chunks})
            chunked = chunked.append(temp_df)
        self.prep_data = chunked

    def embedding(self, embeddder):
        embeddings = []
        for i in range(2):
            data = self.prep_data.loc[self.prep_data['label'] == i]['text']
            embeddings.append(embeddder.get_elmo_vectors(data))
        self.prep_data['embeddings'] = list(np.concatenate(embeddings))
        return self.prep_data['embeddings'].shape
        # data_as_list = list(self.prep_data['text'])
        # self.prep_data['embeddings'] = list(embeddder.get_elmo_vectors(data_as_list))
        # return self.prep_data['embeddings'].shape
