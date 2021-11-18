from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier


class Ensemble:
    def __init__(self, model1, model2):
        cnn_clf = KerasClassifier(build_fn=model1.build_fn, epochs=10, batch_size=50)
        cnn_clf._estimator_type = "classifier"

        bilstm_clf = KerasClassifier(build_fn=model2.build_fn, epochs=10, batch_size=50)
        bilstm_clf._estimator_type = "classifier"

        self.voting = VotingClassifier(
            estimators=[('cnn', cnn_clf),
                        ('bilstm', bilstm_clf)],
            voting='soft')

    def fit(self, train_set):
        self.voting.fit(train_set.X_train, train_set.Y_train.T)
