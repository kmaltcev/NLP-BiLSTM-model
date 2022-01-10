import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from mlxtend.classifier import EnsembleVoteClassifier

from utils.plots import plot_eval


class Ensemble:
    model = None

    def __init__(self, train_set):
        self.name = "CNN-BiLSTM"
        self.train_set = train_set
        self.X = train_set.X_train
        self.Y = train_set.Y_train[:, 1]
        self.clfs = []

    def add(self, model, path):
        clf = KerasClassifier(build_fn=model.build, batch_size=model.batch_size,
                              validation_split=.2, epochs=model.epochs)
        clf._estimator_type = "classifier"

        history = clf.fit(self.X, self.train_set.Y_train)
        self.clfs.append(clf)
        return plot_eval(history.history, history.epoch, model.name, path)

    def build(self):
        self.model = EnsembleVoteClassifier(clfs=self.clfs, voting='soft', verbose=1,
                                            fit_base_estimators=False, use_clones=False)

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
        return np.array(np.concatenate([self.model.predict(X[i: i + 100]) for i in range(0, len(X), 100)]),
                        dtype='uint8')
