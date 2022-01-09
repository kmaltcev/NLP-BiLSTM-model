import os
import numpy as np
import pandas as pd
from dtaidistance import dtw
from pymystem3 import Mystem
from scipy import stats

from utils.constants import BOOKS_DIR, EMBEDDINGS_DIR
from numpy import savez_compressed, load


def read_books(names):
    books = {}
    for name in names:
        books[name] = []
        for book_name in os.listdir(f"./{BOOKS_DIR}/{name}"):
            with open(f"./{BOOKS_DIR}/{name}/{book_name}", "r", encoding='utf8', errors='ignore') as book:
                book = book.read()
                books[name].append({book_name: book})
    return books


def build_graph_data(X, y):
    graph_data = pd.DataFrame(columns=["book", "label", "count"])
    for k, v in X.items():
        for i in range(2):
            counts = len(np.where(v == i)[0])
            graph_data = graph_data.append({"book": k, "label": y[i], "count": counts}, ignore_index=True)
    return graph_data


def compute_distance(texts):
    ascii_texts = [stats.zscore([np.average([ord(ch) for ch in word]) for word in text]) for text in texts]
    distance = dtw.distance(ascii_texts[0], ascii_texts[1], window=25, max_dist=500, use_c=True)
    return distance


def load_embeddings(creation, elmo, book=None):
    if creation['author'] not in os.listdir(f"{EMBEDDINGS_DIR}/"):
        os.mkdir(f"{EMBEDDINGS_DIR}/{creation['author']}")
    search_q = creation['book'].split('.')[0] if book else creation['author']

    if f"{search_q}_embeddings.npz" in os.listdir(f"{EMBEDDINGS_DIR}/{creation['author']}"):
        data = load(f"{EMBEDDINGS_DIR}/{creation['author']}/{search_q}_embeddings.npz")
        data = data['arr_0']
    else:
        data = elmo().get_elmo_vectors(creation['text'])
        savez_compressed(f"{EMBEDDINGS_DIR}/{creation['author']}/{search_q}_embeddings", data)
    return data


def convert_embeddings_to_tensor(array):
    return np.array([np.array(row) for row in array])


def lemmatize(text):
    m = Mystem()
    lemma = m.lemmatize(' '.join(text))
    return ''.join(lemma)


def embedding(text, elmo):
    return elmo().get_elmo_vectors(text)


def circular_generator(array):
    while True:
        for connection in array:
            yield connection
