import os
from timeit import default_timer as timer
from data.Dataset import Dataset
from numpy import savez_compressed, load
from utils.constants import EMBEDDINGS_DIR


def load_embeddings(author, elmo):
    root, folder = EMBEDDINGS_DIR.split('/')
    if folder not in os.listdir(root + '/'):
        os.mkdir(f"{EMBEDDINGS_DIR}")
    if author not in os.listdir(f"{EMBEDDINGS_DIR}/"):
        os.mkdir(f"{EMBEDDINGS_DIR}/{author}")
    start = timer()
    if f"{author}_embeddings.npz" in os.listdir(f"{EMBEDDINGS_DIR}/{author}"):
        print(f"{author} embeddings loading...")
        data = load(f'{EMBEDDINGS_DIR}/{author}/{author}_embeddings.npz')
        data = data['arr_0']
        print(f"{author} embeddings loaded successfully.")
    else:
        print(f"{author} embeddings not found, processing...")
        data = embedding_process(author, elmo)
        savez_compressed(f'{EMBEDDINGS_DIR}/{author}/{author}_embeddings', data)
        print(f"{author} embedding done.")
    end = timer()
    print(f"Elapsed time: {end - start:.2f} sec.")
    return data


def embedding_process(author, elmo):
    dataset = Dataset([author])
    dataset.preprocess()
    dataset.chunking()
    embeddings = elmo().get_elmo_vectors(dataset.data['text'].tolist()[0])
    del dataset
    return embeddings
