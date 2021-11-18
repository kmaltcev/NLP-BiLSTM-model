from models.absModel import AbsModel
from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout


class BiLSTM(AbsModel):
    def __init__(self, shape, hidden_state_dim=200, dropout_rate=.2,
                 fc_layer_size=30, learning_rate=0.001, output_units=3):
        if shape is None:
            raise AttributeError('Input shape not set')

        input_shape = Input(shape=shape)
        forward_lstm = LSTM(hidden_state_dim)
        backward_lstm = LSTM(hidden_state_dim, go_backwards=True)
        bi = Bidirectional(forward_lstm, backward_layer=backward_lstm)(input_shape)
        dropout = Dropout(rate=dropout_rate)(bi)
        fc_layer = Dense(fc_layer_size, activation='relu')(dropout)
        fc_layer = Dropout(rate=dropout_rate)(fc_layer)
        fc_layer = Dense(output_units, activation='softmax')(fc_layer)
        super().__init__(input_shape, fc_layer, "BiLSTM", learning_rate)
