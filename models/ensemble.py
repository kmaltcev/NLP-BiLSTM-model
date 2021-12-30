import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from mlxtend.classifier import EnsembleVoteClassifier


class Ensemble:
    model = None

    def __init__(self, train_set):
        self.name = "CNN-BiLSTM"
        self.train_set = train_set
        self.X = train_set.X_train
        self.Y = train_set.Y_train[:, 1]
        self.clfs = []

    def add(self, model):
        clf = KerasClassifier(build_fn=model.build, epochs=model.epochs, batch_size=model.batch_size)
        clf._estimator_type = "classifier"
        clf.fit(self.X, self.train_set.Y_train)
        self.clfs.append(clf)

    def build(self):
        self.model = EnsembleVoteClassifier(clfs=self.clfs, voting='soft',
                                            fit_base_estimators=False)

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
