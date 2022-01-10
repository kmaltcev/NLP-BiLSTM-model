import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from stqdm import stqdm

from data.absDataset import AbsDataset, pre_clean, not_stopword
from utils.utils import load_embeddings, lemmatize
from joblib import Parallel, delayed


class TestSet(AbsDataset):
    data, embeddings = None, None

    def __init__(self, names: str):
        super().__init__(names, columns=["author", "book", "text"])
        for k in self.books:
            for book in self.books[k]:
                self.data = self.data.append({"author": names,
                                              "book": list(book.keys())[0],
                                              "text": list(book.values())[0]}, ignore_index=True)

    def __getitem__(self, item):
        return np.concatenate(self.data[item].values).reshape(-1)

    def preprocess(self):
        preprocessed_data = pd.DataFrame(columns=self.data.columns)
        for index, data_row in self.data.iterrows():
            cache_fp = f"./prep_data_cached/{data_row['author']}/{data_row['book']}"
            if f"{data_row['author']}" not in os.listdir("./prep_data_cached/"):
                os.mkdir(f"./prep_data_cached/{data_row['author']}/")
            if data_row['book'] not in os.listdir(f"./prep_data_cached/{data_row['author']}/"):
                text = pre_clean(data_row)
                tokens = []
                for i in stqdm(range(len(text)),
                               desc=f"Cleaning and stemming {data_row['author']}'s {data_row['book']}"):
                    if not_stopword(text[i]):
                        tokens.append(text[i])
                text = Parallel(n_jobs=-1)(delayed(lemmatize)(tokens[i: i + 100000])
                                           for i in tqdm(range(0, len(tokens), 100000),
                                                         desc=f"Stemming {data_row['author']}'s {data_row['book']}"))
                text = " ".join(text)
                with open(cache_fp, "w") as fp:
                    fp.write(text)
            else:
                with open(cache_fp, "r") as fp:
                    text = fp.read()
            preprocessed_data = preprocessed_data.append({"author": data_row['author'],
                                                          "book": data_row['book'],
                                                          "text": text}, ignore_index=True)
        self.data = preprocessed_data

    def chunking(self, chunk_size=40):
        chunked = pd.DataFrame(columns=self.data.columns.values)
        for idx, row in self.data.iterrows():
            words = row['text'].split()
            chunks = [words[i - chunk_size:i] for i in range(chunk_size, len(words), chunk_size)]
            temp_df = pd.DataFrame({'author': row['author'], 'book': row['book'], 'text': [chunks]})
            chunked = chunked.append(temp_df)
        self.data = chunked

    def create_embeddings(self, elmo):
        self.embeddings = [load_embeddings(data_row, elmo, book=data_row['book']) for index, data_row in
                           tqdm(self.data.iterrows(), total=self.data.shape[0],
                                desc=f"ELMo embedding process for {self.data['author'].values[0]}'s books")]
        list_embeddings = [list(emb) for emb in self.embeddings]
        self.data = pd.concat([pd.DataFrame({'author': self.data['author'].values[idx],
                                             'book': self.data['book'].values[idx],
                                             'text': [self.data['text'].values[idx]],
                                             'embeddings': [work]})
                               for idx, work in enumerate(tqdm(list_embeddings, desc="Building embeddings df"))])
        return self.data['embeddings'].shape
