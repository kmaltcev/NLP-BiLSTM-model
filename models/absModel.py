import numpy as np
import pandas as pd
import seaborn as sns
from keras import Model
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
from keras.optimizer_v2.adam import Adam
from sklearn.metrics import confusion_matrix, classification_report
from utils.utils import plot_eval


class AbsModel:
    name, learning_rate, input_shape, fc_layer, epochs, batch_size = None, None, None, None, None, None
    model, history, X_pred = None, None, None

    def build(self):
        self.model = Model(inputs=self.input_shape, outputs=self.fc_layer, name=self.name)
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
        self.plot_model()
        return self.model

    def fit(self, train_set):
        self.history = self.model.fit(train_set.X_train, train_set.Y_train,
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      verbose=2, validation_split=0.15)
        return plot_eval(self.history.history, self.epochs, self.name)

    def predict(self, train_set):
        y_pred = self.model.predict(train_set.X_test)
        self.X_pred = np.argmax(y_pred, axis=1)
        return self.X_pred

    def make_prediction(self, test_set):
        y_pred = self.model.predict(np.array(list(test_set.dataframe['embeddings'].array)))
        X_pred = np.argmax(y_pred, axis=1)
        test_set.dataframe['label'] = X_pred
        return test_set.dataframe

    def plot_model(self):
        return plot_model(self.model, f"./plots/{self.name}.png", show_shapes=True)

'''
    def validation(self, train_set):
        score_acc = self.model.evaluate(train_set.X_test, train_set.Y_test, verbose=2, batch_size=50)
        return score_acc

    def confusion_matrix(self, train_set):
        df_test = pd.DataFrame({'true': train_set.Y_test[:, 1], 'pred': self.X_pred})
        c_matrix = confusion_matrix(df_test.true, df_test.pred)
        f, ax = plt.subplots()
        sns.heatmap(c_matrix, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")
        plt.title(f"{self.name} Classification Confusion Matrix")
        plt.xlabel('Y predict')
        plt.ylabel('Y test')
        plt.show()
        print(classification_report(df_test.true, df_test.pred))
        return c_matrix
'''