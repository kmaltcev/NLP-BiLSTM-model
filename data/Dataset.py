import os
import re

import pandas as pd
from tqdm import tqdm
from stqdm import stqdm
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from utils.utils import read_books, load_embeddings
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
        self.data = pd.DataFrame(columns=["label", "author", "text", "length"])
        for idx, name in enumerate(names):
            text = " ".join(books[name])
            self.data = self.data.append({"label": idx,
                                          "author": name,
                                          "text": text,
                                          "length": len(text)}, ignore_index=True)

    def __str__(self):
        return str(self.data)

    def preprocess(self):
        preprocessed_data = pd.DataFrame(columns=self.data.columns)
        for index, data_row in self.data.iterrows():
            cache_fp = f"./prep_data_cached/{data_row['author']}/data.txt"
            if f"{data_row['author']}" not in os.listdir("./prep_data_cached/"):
                os.mkdir(f"./prep_data_cached/{data_row['author']}/")

            if "data.txt" not in os.listdir(f"./prep_data_cached/{data_row['author']}/"):
                text = re.sub(r'[^ЁёА-я\s]', ' ', data_row['text'])
                text = ' '.join([w for w in text.split() if len(w) > 1])
                text = re.sub(r' {2,}', ' ', text)
                text = text.lower().split(' ')
                text_batch = [text[i: i + 1000] for i in range(0, len(text), 1000)]
                text = Parallel(n_jobs=-1)(delayed(lemmatize)(t)
                                           for t in tqdm(text_batch, desc=f"Lemmatizing {data_row['author']}"))
                text = ' '.join(text).split(' ')

                tokens = []
                for i in stqdm(range(len(text)), desc=f"Cleaning and lemmatizing {data_row['author']}"):
                    if text[i] not in stopwords.words('russian') \
                            and text[i] != " " \
                            and text[i].strip() not in punctuation:
                        tokens.append(text[i])

                text = " ".join(tokens)
                with open(cache_fp, "w") as fp:
                    fp.write(text)
            else:
                with open(cache_fp, "r") as fp:
                    text = fp.read()
            preprocessed_data = preprocessed_data.append({"label": index,
                                                          "author": data_row['author'],
                                                          "text": text,
                                                          "length": len(text)}, ignore_index=True)
        self.data = preprocessed_data

    def chunking(self, chunk_size=40):
        chunked = pd.DataFrame(columns=self.data.columns.values)
        for idx, book in self.data.iterrows():
            words = book['text'].split()
            chunks = [words[i - chunk_size:i] for i in range(chunk_size, len(words), chunk_size)]
            temp_df = pd.DataFrame({'label': book['label'], 'author': book['author'], 'text': [chunks]})
            chunked = chunked.append(temp_df)
        self.data = chunked

    def create_embedding(self, elmo):
        self.embeddings = [load_embeddings(data_row, elmo) for index, data_row in
                           tqdm(self.data.iterrows(), total=self.data.shape[0], desc="ELMo embedding process")]

        min_len = min([x.shape[0] for x in self.embeddings])

        list_embeddings = [list(emb[:min_len]) for emb in self.embeddings]
        list_embeddings = [pd.DataFrame({'label': self.data['label'].values[idx],
                                         'author': self.data['author'].values[idx],
                                         'text': self.data['text'].values[idx][:min_len],
                                         'embeddings': work})
                           for idx, work in enumerate(list_embeddings)]
        self.data = pd.concat(list_embeddings)
        return self.data['embeddings'].shape
