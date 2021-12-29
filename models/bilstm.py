from models.absModel import AbsModel
from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout


class BiLSTM(AbsModel):
    def __init__(self, shape, hidden_state_dim=200, dropout_rate=.2,
                 fc_layer_size=30, learning_rate=0.001, output_units=3,
                 epochs = 10, batch_size = 50):
        self.name = "BiLSTM"
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if shape is None:
            raise AttributeError('Input shape not set')

        self.input_shape = Input(shape=shape)
        forward_lstm = LSTM(hidden_state_dim)
        backward_lstm = LSTM(hidden_state_dim, go_backwards=True)
        bi = Bidirectional(forward_lstm, backward_layer=backward_lstm)(self.input_shape)
        dropout = Dropout(rate=dropout_rate)(bi)
        self.fc_layer = Dense(fc_layer_size, activation='relu')(dropout)
        self.fc_layer = Dropout(rate=dropout_rate)(self.fc_layer)
        self.fc_layer = Dense(output_units, activation='softmax')(self.fc_layer)
