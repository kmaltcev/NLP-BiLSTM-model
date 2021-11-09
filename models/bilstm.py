from keras import Model
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from utils.utils import plot_eval, evaluate
from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Embedding


class BiLSTM:
    model = None
    history = None

    def __init__(self, shape, hidden_state_dim=200,
                 dropout_rate=.2, fc_layer_size=30,
                 learning_rate=0.001, output_units=3,
                 epochs=10, batch_size=50):
        if shape is None:
            raise AttributeError('Input shape not set')
        self.input = Input(shape=shape)
        self.hidden_state_dim = hidden_state_dim
        self.dropout_rate = dropout_rate
        self.fc_layer_size = fc_layer_size
        self.learning_rate = learning_rate
        self.output_units = output_units
        self.epochs = epochs
        self.batch_size = batch_size

    def build(self):
        forward_lstm = LSTM(self.hidden_state_dim)  # (input_shape)
        backward_lstm = LSTM(self.hidden_state_dim, go_backwards=True)  # (input_shape)
        self.model = Bidirectional(forward_lstm, backward_layer=backward_lstm)(self.input)
        self.model = Dropout(rate=self.dropout_rate)(self.model)
        self.model = Dense(self.fc_layer_size, activation='relu')(self.model)
        self.model = Dropout(rate=self.dropout_rate)(self.model)
        self.model = Dense(3, activation='softmax')(self.model)
        self.model = Model(inputs=self.input, outputs=self.model, name="BiLSTM")
        # output model skeleton
        print(self.model.summary())
        plot_model(self.model, "bilstm.png", show_shapes=True)

    def compile(self):
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['acc'])

    def fit(self, dataset):
        self.history = self.model.fit(dataset.X_train, dataset.Y_train,
                                      epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        evaluate(self.model, dataset)
        plot_eval(self.history.history)
