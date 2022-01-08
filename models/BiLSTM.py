from models.AbsModel import AbsModel
from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout


class BiLSTM(AbsModel):
    def __init__(self, shape, parameters):
        self.name = "BiLSTM"
        self.epochs = parameters[self.name]["epochs"]
        self.batch_size = parameters[self.name]["batch_size"]
        self.learning_rate = parameters[self.name]['lr']
        if shape is None:
            raise AttributeError('Input shape not set')

        self.input_shape = Input(shape=shape)
        forward_lstm = LSTM(parameters[self.name]["hidden_state_dim"])
        backward_lstm = LSTM(parameters[self.name]["hidden_state_dim"], go_backwards=True)
        bi = Bidirectional(forward_lstm, backward_layer=backward_lstm)(self.input_shape)
        dropout = Dropout(rate=parameters[self.name]["dropout_rate"])(bi)
        self.fc_layer = Dense(parameters[self.name]["fc_layer_size"], activation='relu')(dropout)
        self.fc_layer = Dropout(rate=parameters[self.name]["dropout_rate"])(self.fc_layer)
        self.fc_layer = Dense(2, activation='softmax')(self.fc_layer)