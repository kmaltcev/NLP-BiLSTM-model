import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from models.merge import ELMo
from utils.embedding_utils import load_embeddings


class TestSet:
    def __init__(self, path, labels):
        self.preds = None
        directory, self.author, self.work = path.split("/")
        self.labels = labels
        self.work, ext = self.work.split('.')

        self.data = load_embeddings(self.author, ELMo)

        if self.data is None:
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
