import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


# from sklearn.model_selection import train_test_split


class TrainSet:
    def __init__(self, dataset, num_classes=2, test_size=0.15, random_state=42):
        self.Y_train = np.asarray(dataset['label'].tolist())
        self.X_train = np.asarray(dataset['embeddings'].tolist())

        self.Y_train = to_categorical(self.Y_train, num_classes=num_classes)

        self.shape = pd.DataFrame({'Train': [self.X_train.shape, self.Y_train.shape]}, index=['X', 'Y'])

    def X_shape(self):
        return self.X_train[0].shape


'''
#self.Y_test = to_categorical(self.Y_test, num_classes=num_classes)
#'Test': [self.X_test.shape, self.Y_test.shape]}, index=['X', 'Y'])

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(x, y, test_size=test_size, random_state=random_state)
'''
