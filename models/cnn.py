from keras import Model
from utils.utils import plot_eval
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Concatenate, Dense, Dropout


class CNN:
    model = None
    history = None

    def __init__(self, shape, num_filters=200,
                 kernel_size: list = None, use_bias=True,
                 dropout_rate=.5, fc_layer_size=100,
                 learning_rate=0.001, output_units=3,
                 epochs=10, batch_size=50):
        if shape is None:
            raise AttributeError('Input shape not set')
        if kernel_size is None:
            kernel_size = [3, 4, 5]
        self.input_shape = shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.fc_layer_size = fc_layer_size
        self.learning_rate = learning_rate
        self.output_units = output_units
        self.epochs = epochs
        self.batch_size = batch_size

    def build(self):
        input_shape = Input(shape=self.input_shape)
        cnn_1 = Conv1D(self.num_filters, kernel_size=self.kernel_size[0], use_bias=self.use_bias)(input_shape)
        cnn_1 = GlobalMaxPooling1D()(cnn_1)
        cnn_2 = Conv1D(self.num_filters, kernel_size=self.kernel_size[1], use_bias=self.use_bias)(input_shape)
        cnn_2 = GlobalMaxPooling1D()(cnn_2)
        cnn_3 = Conv1D(self.num_filters, kernel_size=self.kernel_size[2], use_bias=self.use_bias)(input_shape)
        cnn_3 = GlobalMaxPooling1D()(cnn_3)
        merged = Concatenate(axis=1)([cnn_1, cnn_2, cnn_3])
        fc_layer = Dense(self.fc_layer_size, activation='relu')(merged)
        fc_layer = Dropout(rate=self.dropout_rate)(fc_layer)
        fc_layer = Dense(self.output_units, activation='softmax')(fc_layer)
        self.model = Model(inputs=input_shape, outputs=fc_layer, name="CNN")

        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['acc'])
        print(self.model.summary(line_length=140))
        plot_model(self.model, "cnn.png", show_shapes=True)

    def fit(self, x_train, y_train):
        self.history = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        plot_eval(self.history.history)
