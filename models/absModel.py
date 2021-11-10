import numpy as np
import pandas as pd
from keras.optimizer_v2.adam import Adam
from sklearn.metrics import confusion_matrix, classification_report

from utils.utils import evaluate, plot_eval


class AbsModel:
    history = None
    model = None

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def fit(self, dataset, epochs=10, batch_size=50):
        # compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['acc'])
        # start training
        self.history = self.model.fit(dataset.X_train, dataset.Y_train,
                                      epochs=epochs, batch_size=batch_size, verbose=2)
        evaluate(self.model, dataset)
        plot_eval(self.history.history)

    def predict(self, train_set):
        predict_x = self.model.predict(train_set.X_test)
        X_pred = np.argmax(predict_x, axis=1)

        df_test = pd.DataFrame({'true': train_set.Y_test, 'pred': X_pred})
        tn, fp, fn, tp = confusion_matrix(df_test.true, df_test.pred).ravel()
        c_matrix = pd.DataFrame({1: [tp, fp], 0: [fn, tn]}, index=[1, 0])
        # print(f"confusion matrix:\n {}")
        print("Confusion Matrix:")
        print(c_matrix)
        print(classification_report(df_test.true, df_test.pred))

    def build(self):
        pass
