import re

import numpy as np
import pandas as pd

from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords

from utils.utils import read_books


class Dataset:
    def __init__(self, data):
        if data is None:
            raise ValueError("No dataset provided")
        self.data = pd.DataFrame({"label": [0, 1],
                                  "author": [data[0], data[1]],
                                  "text": [read_books(data[0]), read_books(data[1])]})
        self.russian_stopwords = stopwords.words('russian')
        self.mystem = Mystem()

    def __str__(self):
        return str(self.data)

    def preprocess(self):
        texts = []
        for i, raw_text in enumerate(self.data['text']):
            text = re.sub(r'[^ЁёА-я\s]', ' ', raw_text)
            text = ' '.join([w for w in text.split() if len(w) > 1])
            text = re.sub(r' {2,}', ' ', text)
            lemmas = self.mystem.lemmatize(text.lower())
            tokens = [token for token in lemmas if token not in self.russian_stopwords
                      and token != " "
                      and token.strip() not in punctuation]
            texts.append(" ".join(tokens))
        self.data = pd.DataFrame({"label": [0, 1], "author": self.data['author'].tolist(), "text": texts})

    def chunking(self, chunk_size=40):
        chunked = pd.DataFrame(columns=self.data.columns.values)
        for idx, book in self.data.iterrows():
            words = book['text'].split()
            chunks = [words[i - chunk_size:i] for i in range(chunk_size, len(words), chunk_size)]
            temp_df = pd.DataFrame({'label': book['label'], 'author': book['author'], 'text': chunks, 'embedding': None})
            chunked = chunked.append(temp_df)
        self.data = chunked

    def set_embeddings(self, embeddings):
        self.data['embeddings'] = embeddings.tolist()
        pass
