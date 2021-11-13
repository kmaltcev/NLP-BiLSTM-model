from keras import Model

from models.absModel import AbsModel
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Concatenate, Dense, Dropout


class CNN(AbsModel):
    def __init__(self, shape, num_filters=200, kernel_size: list = None,
                 use_bias=True, dropout_rate=.5, fc_layer_size=100,
                 learning_rate=0.001, output_units=3):
        super().__init__(learning_rate)
        if shape is None:
            raise AttributeError('Input shape not set')
        if kernel_size is None:
            kernel_size = [3, 4, 5]
        input_shape = Input(shape=shape)
        cnn_1 = Conv1D(num_filters, kernel_size=kernel_size[0], use_bias=use_bias)(input_shape)
        cnn_1 = GlobalMaxPooling1D()(cnn_1)
        cnn_2 = Conv1D(num_filters, kernel_size=kernel_size[1], use_bias=use_bias)(input_shape)
        cnn_2 = GlobalMaxPooling1D()(cnn_2)
        cnn_3 = Conv1D(num_filters, kernel_size=kernel_size[2], use_bias=use_bias)(input_shape)
        cnn_3 = GlobalMaxPooling1D()(cnn_3)
        merged = Concatenate(axis=1)([cnn_1, cnn_2, cnn_3])
        fc_layer = Dense(fc_layer_size, activation='relu')(merged)
        fc_layer = Dropout(rate=dropout_rate)(fc_layer)
        fc_layer = Dense(output_units, activation='softmax')(fc_layer)
        self.model = Model(inputs=input_shape, outputs=fc_layer, name="CNN")
        # output skeleton
        print(self.model.summary(line_length=140))
        plot_model(self.model, "./plots/cnn.png", show_shapes=True)
        self.plot_title = 'CNN'
