import os
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns

from utils.constants import BOOKS_DIR, EMBEDDINGS_DIR
from wordcloud import WordCloud
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from numpy import savez_compressed, load


def read_books(names):
    books = {}
    for name in names:
        books[name] = []
        for book_name in os.listdir(f"./{BOOKS_DIR}/{name}"):
            with open(f"./{BOOKS_DIR}/{name}/{book_name}", "r", encoding='utf8', errors='ignore') as book:
                book = book.read()
                books[name].append(book)  # [:int(len(book) / 100)])
    return books


'''
def evaluate(model, dataset):
    _, train_acc = model.evaluate(dataset.X_train, dataset.Y_train, verbose=2)
    # _, test_acc = model.evaluate(dataset.X_test, dataset.Y_test, verbose=2)
    # print('Train: %.3f, Test: %.4f' % (train_acc, test_acc))
'''


def plot_eval(history, n_epochs, title):
    fig_1 = plt.figure()
    loss_train = history['loss']
    loss_val = history['val_loss']
    epochs = range(1, n_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title(f'{title} Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./plots/{title}_train_vs_val_loss.png')

    fig_2 = plt.figure()
    loss_train = history['accuracy']
    loss_val = history['val_accuracy']
    epochs = range(1, n_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'./plots/{title}_train_vs_val_acc.png')
    return fig_1, fig_2


def plot_words_bar(dataset, path):
    sns_palette = ["rocket", "mako", "magma", "rocket_r"]
    sns.set_style("whitegrid")
    figs = []
    for i, book in dataset.iterrows():
        words = book['text'].split()
        length = len(words)
        counters = Counter(words)
        cnt_pro = np.asarray(counters.most_common(20))
        probs = [int(num) / length for num in cnt_pro[:, 1]]

        occ_df = pd.DataFrame({'word': cnt_pro[:, 0],
                               'count': probs})
        fig = plt.figure()
        sns.barplot(x='word', y='count', alpha=0.8, data=occ_df, palette=sns_palette[i])
        plt.title(f"{book['author']}'s frequent words")
        plt.ylabel('Frequency', fontsize=12)
        plt.xlabel(f'Word', fontsize=12)
        plt.xticks(rotation=90)
        figs.append(fig)
        plt.savefig(f'{path}/words_bar_{book["author"]}.png')
    return figs


def plot_compare_bars(dataset, path):
    words = []
    names = []
    fig = plt.figure()
    fig.set_facecolor('white')
    for i, book in dataset.iterrows():
        words.append(len(book['text'].split()))
        names.append(book['author'])

    sns.barplot(names, words, alpha=0.8)
    plt.title("Length of works")
    plt.ylabel('Length of works', fontsize=12)
    plt.xlabel('Author', fontsize=12)
    plt.xticks(rotation=90)
    plt.savefig(f'{path}/words_count.png')
    return fig


def plot_words_cloud(dataset, path):
    for i, book in dataset.iterrows():
        wordcloud = WordCloud(background_color="white",
                              stopwords=stopwords.words('russian'),
                              mode="RGBA",
                              width=400,
                              height=330,
                              colormap='inferno').generate(book['text'])
        fig = plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        plt.savefig(f'{path}/wordcloud_{book["author"]}.png')
        return fig


def plot_scores(score_a, score_b, path):
    data = {'abbrev': ['CNN', 'BiLSTM'],
            'score': [score_a[0], score_b[0]],
            'accuracy': [score_a[1], score_b[1]]}

    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots()

    sns.set_color_codes("pastel")
    sns.barplot(x="accuracy", y="abbrev", data=data,
                label="Accuracy", color="b")

    sns.set_color_codes("muted")
    sns.barplot(x="score", y="abbrev", data=data,
                label="Score", color="b")

    ax.legend(bbox_to_anchor=(1, 1), ncol=1, loc='lower right', frameon=True)
    ax.set(ylabel="NN Type",
           xlabel="Score/Accuracy Value")

    for i in range(len(data['abbrev'])):
        plt.annotate(f"{data['score'][i]:.2f}", xy=(data['score'][i], i))
        plt.annotate(f"{data['accuracy'][i]:.2f}", xy=(data['accuracy'][i], i))

    plt.title(f"Score and Accuracy")
    plt.show()
    plt.savefig(f'{path}/Score_acc_plot.png')


def plot_prediction(preds, labels, path):
    cnts = [len(np.where(preds == i)[0]) for i in range(2)]
    fig = plt.figure()
    fig.set_facecolor('white')
    sns.barplot(labels, cnts, alpha=0.8)
    plt.title("Chunks Distribution")
    plt.ylabel('Chunk', fontsize=12)
    plt.xlabel('Author', fontsize=12)
    plt.xticks(rotation=90)
    plt.savefig(f"{path}/preds_distribution.png")
    return fig


def load_embeddings(creation, elmo):
    if creation['author'] not in os.listdir(f"{EMBEDDINGS_DIR}/"):
        os.mkdir(f"{EMBEDDINGS_DIR}/{creation['author']}")
    if f"{creation['author']}_embeddings.npz" in os.listdir(f"{EMBEDDINGS_DIR}/{creation['author']}"):
        print(f"{creation['author']} embeddings loading...")
        data = load(f"{EMBEDDINGS_DIR}/{creation['author']}/{creation['author']}_embeddings.npz")
        data = data['arr_0']
        print(f"{creation['author']} embeddings loaded successfully.")
    else:
        print(f"{creation['author']} embeddings not found, processing...")
        data = elmo().get_elmo_vectors(creation['text'])
        savez_compressed(f"{EMBEDDINGS_DIR}/{creation['author']}/{creation['author']}_embeddings", data)
    return data
