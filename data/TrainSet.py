import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


class TrainSet:
    def __init__(self, dataset, test_size=0.15, random_state=42):
        y = np.asarray(dataset['label'].tolist())
        x = np.asarray(dataset['embeddings'].tolist())

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(x, y, test_size=test_size, random_state=random_state)
        self.X_train, self.X_val, self.Y_train, self.Y_val = \
            train_test_split(self.X_train, self.Y_train, test_size=test_size, random_state=random_state)

        self.Y_train_cat = to_categorical(self.Y_train)
        self.Y_val_cat = to_categorical(self.Y_val)

        self.shape = pd.DataFrame({'Train': [self.X_train.shape, self.Y_train.shape],
                                   'Test': [self.X_test.shape, self.Y_test.shape]}, index=['X', 'Y'])

    def X_shape(self):
        return self.X_train[0].shape
