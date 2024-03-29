import re
import pandas as pd
from string import punctuation
from typing import Union
from nltk.corpus import stopwords
from utils.utils import read_books


# pre-Clean text from spaces and leave only Russian letters
def pre_clean(data_row):
    text = re.sub(r'[^ЁёА-я\s]', ' ', data_row['text'])
    text = ' '.join([w for w in text.split() if len(w) > 1])
    text = re.sub(r' {2,}', ' ', text)
    return text.lower().split(' ')


# Clean text from stopwords and spaces
def not_stopword(string):
    return string not in stopwords.words('russian') \
           and string != " " \
           and string.strip() not in punctuation


# Abstract dataset class, our datasets implements it
class AbsDataset:
    def __init__(self, names: Union[list, str], columns):
        if names is None:
            raise ValueError("No data provided")
        self.books = read_books(names if type(names) is list else [names])
        self.labels_size = len(names) if type(names) is list else 1
        self.data = pd.DataFrame(columns=columns)

    def __str__(self):
        return str(self.data)
