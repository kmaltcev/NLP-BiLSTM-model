import numpy as np
from keras.optimizer_v2.adam import Adam
from scikeras.wrappers import KerasClassifier
from mlxtend.classifier import EnsembleVoteClassifier

from utils.utils import plot_eval


class Ensemble:
    model = None

    def __init__(self, train_set):
        self.name = "CNN-BiLSTM"
        self.train_set = train_set
        self.X = train_set.X_train
        self.Y = train_set.Y_train[:, 1]
        self.clfs = []

    def add(self, model, path):
        optimizer = Adam(learning_rate=model.learning_rate)
        clf = KerasClassifier(model=model.build(), epochs=model.epochs, batch_size=model.batch_size, warm_start=True,
                              validation_split=.2, optimizer=optimizer, validation_batch_size=model.batch_size)
        clf._estimator_type = "classifier"
        history = clf.fit(self.X, self.train_set.Y_train)
        self.clfs.append(clf)
        return plot_eval(history.history_, history.epochs, model.name, path)

    def build(self):
        self.model = EnsembleVoteClassifier(clfs=self.clfs, voting='soft', fit_base_estimators=False)

    def fit(self):
        history = self.model.fit(self.X, self.Y)
        return history

    def predict_proba(self, X=None):
        if X is None:
            print("Using Testing Set for Probabilities Prediction")
            X = self.X
        return self.model.predict_proba(X)

    def predict(self, X=None):
        if X is None:
            print("Using Testing Set for Classes Prediction")
            X = self.train_set.X_train
        return np.array(self.model.predict(X), dtype='uint8')
