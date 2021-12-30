import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
'''
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
'''