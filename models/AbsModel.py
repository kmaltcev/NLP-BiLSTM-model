import numpy as np
from keras import Model
from keras.utils.vis_utils import plot_model
from keras.optimizer_v2.adam import Adam
from utils.plots import plot_eval


class AbsModel:
    input_shape, fc_layer = None, None
    model, history, X_pred = None, None, None

    def __init__(self, name, learning_rate, epochs, batch_size):
        self.name = name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def build(self):
        self.model = Model(inputs=self.input_shape, outputs=self.fc_layer, name=self.name)
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
        return self.model

    def fit(self, train_set, path_to_plot):
        self.history = self.model.fit(train_set.X_train, train_set.Y_train,
                                      epochs=self.epochs, batch_size=self.batch_size,
                                      verbose=2, validation_split=.2)
        return plot_eval(self.history.history, self.epochs, self.name, path_to_plot)

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
