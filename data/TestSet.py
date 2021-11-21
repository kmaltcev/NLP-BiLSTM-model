import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from data.Dataset import Dataset
from utils.constants import BOOKS_DIR, EMBEDDINGS_DIR


def embedding_process(author, elmo):
    dataset = Dataset([author])
    dataset.preprocess()
    dataset.chunking()
    dataset.embedding(elmo)

    return dataset.prep_data


class TestSet:
    def __init__(self, path, labels, elmo):
        self.preds = None
        directory, self.author, self.work = path.split("/")
        self.labels = labels
        self.work, _ = self.work.split('.')
        if f"{self.work}_embeddings.csv" in os.listdir(f"./{EMBEDDINGS_DIR}/{self.author}"):
            print("Embeddings loaded successfully")
            self.dataframe = pd.read_pickle(f'./{EMBEDDINGS_DIR}/{self.author}/{self.work}_embeddings.csv')
        elif f"{self.work}.{_}" in os.listdir(f"./{BOOKS_DIR}/{self.author}"):
            print("Embeddings not found, processing...")
            self.dataframe = embedding_process(self.author, elmo)
            self.dataframe.to_pickle(f'./{EMBEDDINGS_DIR}/{self.author}/{self.work}_embeddings.csv')
        else:
            raise FileNotFoundError(path)

    def plot_prediction(self):
        cnts = [len(np.where(self.preds == i)[0]) for i in range(2)]
        fig = plt.figure(figsize=(12, 4))
        fig.set_facecolor('white')

        sns.barplot(self.labels, cnts, alpha=0.8)
        plt.ylabel('Chunks Attribution', fontsize=12)
        plt.xlabel('Author', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()
        plt.savefig(f"./plots/{self.author}_{self.work}.png")

