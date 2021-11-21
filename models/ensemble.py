import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from mlxtend.classifier import EnsembleVoteClassifier


class Ensemble:
    def __init__(self, clf1, clf2, train_set):
        self.X = train_set.X_train
        self.Y = train_set.Y_train[:, 1]

        cnn_clf = KerasClassifier(build_fn=clf1.build, epochs=10, batch_size=50)
        cnn_clf._estimator_type = "classifier"
        cnn_clf.fit(self.X, train_set.Y_train)

        bilstm_clf = KerasClassifier(build_fn=clf2.build, epochs=10, batch_size=50)
        bilstm_clf._estimator_type = "classifier"
        bilstm_clf.fit(self.X, train_set.Y_train)
        #tf.compat.v1.experimental.output_all_intermediates(True)
        self.voting = EnsembleVoteClassifier(clfs=[cnn_clf, bilstm_clf], voting='soft', fit_base_estimators=False)

    def fit(self):
        self.voting.fit(self.X, self.Y)

    def predict_proba(self, X=None):
        if not X:
            print("Using Testing Set for Probabilities Prediction")
        self.voting.predict_proba(X if X else self.X)

    def predict(self, X=None):
        if not X:
            print("Using Testing Set for Classes Prediction")
        np.array(self.voting.predict(X if X else self.X), dtype='uint8')

