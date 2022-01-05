import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


class TrainSet:
    def __init__(self, dataset, num_classes=2):
        self.Y_train = np.asarray(dataset['label'].tolist())
        self.X_train = np.asarray(dataset['embeddings'].tolist())
        self.Y_train = to_categorical(self.Y_train, num_classes=num_classes)
        self.shape = pd.DataFrame({'Train': [self.X_train.shape, self.Y_train.shape]}, index=['X', 'Y'])

    def X_shape(self):
        return self.X_train[0].shape
