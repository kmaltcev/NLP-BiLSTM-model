import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from mlxtend.classifier import EnsembleVoteClassifier

from utils.utils import evaluate, plot_eval


class Ensemble:
    def __init__(self, clf1, clf2, train_set, epochs=10, batch_size=50):
        self.batch_size = batch_size
        self.epochs = epochs
        self.name = "CNN-BiLSTM"
        self.train_set = train_set
        self.X = train_set.X_train
        self.Y = train_set.Y_train[:, 1]

        cnn_clf = KerasClassifier(build_fn=clf1.build, epochs=epochs, batch_size=batch_size)
        cnn_clf._estimator_type = "classifier"
        cnn_clf.fit(self.X, train_set.Y_train)

        bilstm_clf = KerasClassifier(build_fn=clf2.build, epochs=epochs, batch_size=batch_size)
        bilstm_clf._estimator_type = "classifier"
        bilstm_clf.fit(self.X, train_set.Y_train)
        self.model = EnsembleVoteClassifier(clfs=[cnn_clf, bilstm_clf], voting='soft', fit_base_estimators=False)

    def fit(self):
        history = self.model.fit(self.X, self.Y)
        #evaluate(self.model, self.train_set)
        #plot_eval(history, self.epochs, self.name)
        return history

    def predict_proba(self, X=None):
        if X is None:
            print("Using Testing Set for Probabilities Prediction")
        return self.model.predict_proba(X if X else self.X)

    def predict(self, X=None):
        if X is None:
            print("Using Testing Set for Classes Prediction")
        return np.array(self.model.predict(X if X else self.X), dtype='uint8')

