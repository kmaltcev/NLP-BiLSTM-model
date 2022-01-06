import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from stqdm import stqdm
from string import punctuation
from nltk.corpus import stopwords
from joblib import Parallel, delayed

from data.absDataset import AbsDataset, pre_clean, not_stopword
from utils.utils import load_embeddings, lemmatize


class RawDataset(AbsDataset):
    data, embeddings = None, None

    def __init__(self, names):
        super().__init__(names, columns=["label", "author", "text", "length"])
        for idx, name in enumerate(names):
            text = " ".join([" ".join([book[k] for k in book]) for book in self.books[name]])
            self.data = self.data.append({"label": idx,
                                          "author": name,
                                          "text": text,
                                          "length": len(text)}, ignore_index=True)

    def preprocess(self):
        preprocessed_data = pd.DataFrame(columns=self.data.columns)
        for index, data_row in self.data.iterrows():
            cache_fp = f"./prep_data_cached/{data_row['author']}/data.txt"
            if f"{data_row['author']}" not in os.listdir("./prep_data_cached/"):
                os.mkdir(f"./prep_data_cached/{data_row['author']}/")
            if "data.txt" not in os.listdir(f"./prep_data_cached/{data_row['author']}/"):
                text = pre_clean(data_row)
                tokens = []
                for i in stqdm(range(len(text)), desc=f"Cleaning and stemming {data_row['author']}"):
                    if not_stopword(text[i]):
                        tokens.append(text[i])
                text = Parallel(n_jobs=-1)(delayed(lemmatize)(tokens[i: i + 1000])
                                           for i in tqdm(range(0, len(tokens), 1000),
                                                         desc=f"Stemming {data_row['author']}"))
                text = " ".join(text)
                with open(cache_fp, "w") as fp:
                    fp.write(text)
            else:
                with open(cache_fp, "r") as fp:
                    text = fp.read()
            preprocessed_data = preprocessed_data.append({"label": index,
                                                          "author": data_row['author'],
                                                          "text": text}, ignore_index=True)
        self.data = preprocessed_data

    def chunking(self, chunk_size=40):
        chunked = pd.DataFrame(columns=self.data.columns.values)
        for idx, row in self.data.iterrows():
            words = row['text'].split()
            chunks = [words[i - chunk_size:i] for i in range(chunk_size, len(words), chunk_size)]
            temp_df = pd.DataFrame({'label': row['label'], 'author': row['author'], 'text': [chunks]})
            chunked = chunked.append(temp_df)
        self.data = chunked

    def create_embedding(self, elmo):
        self.embeddings = [load_embeddings(data_row, elmo) for index, data_row in
                           tqdm(self.data.iterrows(), total=self.data.shape[0],
                                desc=f"ELMo embedding process for {self.data['author'].values[0]}")]

        min_len = min([x.shape[0] for x in self.embeddings])
        list_embeddings = [list(emb[np.random.randint(low=0, high=emb.shape[0], size=min_len), :])
                           for emb in self.embeddings]
        list_embeddings = [pd.DataFrame({'label': self.data['label'].values[idx],
                                         'author': self.data['author'].values[idx],
                                         'text': self.data['text'].values[idx][:min_len],
                                         'embeddings': work})
                           for idx, work in enumerate(list_embeddings)]
        self.data = pd.concat(list_embeddings)
        return self.data['embeddings'].shape
