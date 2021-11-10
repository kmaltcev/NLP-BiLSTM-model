import numpy as np
import pandas as pd
import seaborn as sns

from keras.optimizer_v2.adam import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from utils.utils import evaluate, plot_eval


class AbsModel:
    history = None
    model = None
    X_pred = None
    plot_title = None

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
        return self.history

    def predict(self, train_set):
        predict_x = self.model.predict(train_set.X_test)
        self.X_pred = np.argmax(predict_x, axis=1)
        return self.X_pred

    def confusion_matrix(self, train_set):
        df_test = pd.DataFrame({'true': train_set.Y_test, 'pred': self.X_pred})
        c_matrix = confusion_matrix(df_test.true, df_test.pred)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(c_matrix, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")

        plt.title(self.plot_title)
        plt.xlabel('Y predict')
        plt.ylabel('Y test')
        plt.show()
        return c_matrix

    def validation(self, train_set):
        validation_size = int(train_set.X_test.shape[0]/2)

        X_validate = train_set.X_test[-validation_size:]
        Y_validate = train_set.Y_test[-validation_size:]
        X_test = train_set.X_test[:-validation_size]
        Y_test = train_set.Y_test[:-validation_size]

        score, acc = self.model.evaluate(X_test, Y_test, verbose=2, batch_size=50)

        print("score: %.2f" % score)
        print("acc: %.2f" % acc)
