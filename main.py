from simple_elmo import ElmoModel

from data.Dataset import Dataset
from models.cnn import CNN
from models.elmo import Elmo
from utils.utils import evaluate, plot_eval, plot_bars, words_count

name_A = "Nekrasov"
name_B = "Pushkin"
name_C = "Sholokhov"
elmo = Elmo()

if __name__ == '__main__':
    # 1. load dataset
    dataset = Dataset([name_A, name_B])
    # 2. preprocess text
    dataset.preprocess()
    # 3. plot some beauty
    words_count(dataset.data)
    plot_bars(dataset.data)
    # 4. chunking
    dataset.chunking()
    # 5. embedding
    aslist = list(dataset.data['text'])
    authors_embeddings = elmo.get_elmo_vectors(aslist)
    dataset.set_embeddings(authors_embeddings)
    print(dataset)
    # cnn = CNN()
    # bilstm = BiLSTM()
