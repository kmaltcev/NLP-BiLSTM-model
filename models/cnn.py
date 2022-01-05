from models.absModel import AbsModel
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Concatenate, Dense, Dropout


class CNN(AbsModel):
    def __init__(self, shape, num_filters=200, kernel_size: list = None,
                 dropout_rate=.5, fc_layer_size=100,
                 learning_rate=0.0001, epochs=10, batch_size=50):
        self.epochs = epochs
        self.batch_size = batch_size
        self.name = "CNN"
        self.learning_rate = learning_rate
        if shape is None:
            raise AttributeError('Input shape not set')

        kernel_size = [3, 4, 5] if kernel_size is None else kernel_size
        self.input_shape = Input(shape=shape)
        cnn_1 = Conv1D(num_filters, kernel_size=kernel_size[0], use_bias=True)(self.input_shape)
        cnn_1 = GlobalMaxPooling1D()(cnn_1)
        cnn_2 = Conv1D(num_filters, kernel_size=kernel_size[1], use_bias=True)(self.input_shape)
        cnn_2 = GlobalMaxPooling1D()(cnn_2)
        cnn_3 = Conv1D(num_filters, kernel_size=kernel_size[2], use_bias=True)(self.input_shape)
        cnn_3 = GlobalMaxPooling1D()(cnn_3)
        merged = Concatenate(axis=1)([cnn_1, cnn_2, cnn_3])
        self.fc_layer = Dense(fc_layer_size, activation='relu')(merged)
        self.fc_layer = Dropout(rate=dropout_rate)(self.fc_layer)
        self.fc_layer = Dense(2, activation='softmax')(self.fc_layer)
