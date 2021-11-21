from matplotlib import pyplot as plt

from data.Dataset import Dataset
from data.TrainSet import TrainSet
from models.merge import ELMo, CNN, BiLSTM, Ensemble
from utils.utils import plot_words_cloud, plot_words_count, plot_compare_bars

names = ["Furman", "Garshin"]
name_C = "Sholokhov"

if __name__ == '__main__':
    dataset = Dataset(names)
    dataset.preprocess()
    plot_words_cloud(dataset.data)
    plot_words_count(dataset.data)
    plot_compare_bars(dataset.data)
    dataset.chunking()
    dataset.embedding(ELMo)
    train_set = TrainSet(dataset.prep_data)
    cnn = CNN(train_set.X_shape(), output_units=3)
    cnn.build()
    plt.figure()
    cnn.plot_model()
    plt.show()
    cnn.fit(train_set)
    bilstm = BiLSTM(train_set.X_shape(), hidden_state_dim=500)
    bilstm.build()
    plt.figure()
    bilstm.plot_model()
    plt.show()
    bilstm.fit(train_set)
    cnn_bilstm = Ensemble(cnn, bilstm, train_set)
    cnn_bilstm.fit()
    X = train_set.X_train,
    Y = train_set.Y_train[:, 1]
    print(cnn_bilstm.voting.predict_proba(X))
    print(cnn_bilstm.voting.predict(X))
