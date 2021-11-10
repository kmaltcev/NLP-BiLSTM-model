from keras import Model
from keras.utils.vis_utils import plot_model

from models.absModel import AbsModel
from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Embedding


class BiLSTM(AbsModel):

    def __init__(self, shape, hidden_state_dim=40, dropout_rate=.2,
                 fc_layer_size=30, learning_rate=0.001, output_units=3):
        super().__init__(learning_rate)
        if shape is None:
            raise AttributeError('Input shape not set')
        self.input = Input(shape=shape)
        self.hidden_state_dim = hidden_state_dim
        self.dropout_rate = dropout_rate
        self.fc_layer_size = fc_layer_size
        self.learning_rate = learning_rate
        self.output_units = output_units

    def build(self):
        forward_lstm = LSTM(self.hidden_state_dim)  # (input_shape)
        backward_lstm = LSTM(self.hidden_state_dim, go_backwards=True)  # (input_shape)
        bi = Bidirectional(forward_lstm, backward_layer=backward_lstm)(self.input)
        dropout = Dropout(rate=self.dropout_rate)(bi)
        fc_layer = Dense(self.fc_layer_size, activation='relu')(dropout)
        fc_layer = Dropout(rate=self.dropout_rate)(fc_layer)
        fc_layer = Dense(3, activation='softmax')(fc_layer)
        self.model = Model(inputs=self.input, outputs=fc_layer, name="BiLSTM")
        # output skeleton
        print(self.model.summary(line_length=80))
        plot_model(self.model, "./plots/bilstm.png", show_shapes=True)
