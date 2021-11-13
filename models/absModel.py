import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from matplotlib import pyplot as plt
from keras.optimizer_v2.adam import Adam
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
        self.history = self.model.fit(dataset.X_train, dataset.Y_train, epochs=epochs, batch_size=batch_size, verbose=2,
                                      validation_data=(dataset.X_val, dataset.Y_val))
        evaluate(self.model, dataset)
        plot_eval(self.history.history, epochs, self.plot_title)
        return self.history

    def predict(self, train_set):
        y_pred = self.model.predict(train_set.X_test)
        self.X_pred = np.argmax(y_pred, axis=1)
        return self.X_pred

    def confusion_matrix(self, train_set):
        df_test = pd.DataFrame({'true': train_set.Y_test, 'pred': self.X_pred})
        c_matrix = confusion_matrix(df_test.true, df_test.pred)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(c_matrix, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")

        plt.title(f"{self.plot_title} Classification Confusion Matrix")
        plt.xlabel('Y predict')
        plt.ylabel('Y test')
        plt.show()
        print(classification_report(df_test.true, df_test.pred))
        return c_matrix

    def validation(self, train_set):
        score_acc = self.model.evaluate(train_set.X_test, train_set.Y_test, verbose=2, batch_size=50)
        return score_acc

    def make_prediction(self, test_set):
        y_pred = self.model.predict(np.array(list(test_set.dataframe['embeddings'].array)))
        X_pred = np.argmax(y_pred, axis=1)
        test_set.dataframe['label'] = X_pred
        return test_set.dataframe
