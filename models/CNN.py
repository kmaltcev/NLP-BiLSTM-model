from models.AbsModel import AbsModel
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Concatenate, Dense, Dropout


class CNN(AbsModel):
    def __init__(self, shape, parameters):
        super().__init__("CNN", parameters["CNN"]['lr'],
                         parameters["CNN"]["epochs"], parameters["CNN"]["batch_size"])
        if shape is None:
            raise AttributeError('Input shape not set')

        self.input_shape = Input(shape=shape)
        cnn_1 = Conv1D(parameters[self.name]["num_filters"], kernel_size=parameters[self.name]["kernel_size_1"],
                       use_bias=True)(self.input_shape)
        cnn_1 = GlobalMaxPooling1D()(cnn_1)
        cnn_2 = Conv1D(parameters[self.name]["num_filters"], kernel_size=parameters[self.name]["kernel_size_2"],
                       use_bias=True)(self.input_shape)
        cnn_2 = GlobalMaxPooling1D()(cnn_2)
        cnn_3 = Conv1D(parameters[self.name]["num_filters"], kernel_size=parameters[self.name]["kernel_size_2"],
                       use_bias=True)(self.input_shape)
        cnn_3 = GlobalMaxPooling1D()(cnn_3)
        merged = Concatenate(axis=1)([cnn_1, cnn_2, cnn_3])
        self.fc_layer = Dense(parameters["CNN"]["fc_layer_size"], activation='relu')(merged)
        self.fc_layer = Dropout(rate=parameters["CNN"]["dropout_rate"])(self.fc_layer)
        self.fc_layer = Dense(2, activation='softmax')(self.fc_layer)
