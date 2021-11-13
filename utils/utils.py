import os
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns

from wordcloud import WordCloud
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from utils.constants import BOOKS_DIR


def read_books(author_name):
    books_text = ""
    for book_name in os.listdir(f"./{BOOKS_DIR}/{author_name}"):
        with open(f"./{BOOKS_DIR}/{author_name}/{book_name}", "r", encoding='utf8', errors='ignore') as book:
            books_text += " " + book.read()
    return books_text


def evaluate(model, dataset):
    _, train_acc = model.evaluate(dataset.X_train, dataset.Y_train, verbose=2)
    _, test_acc = model.evaluate(dataset.X_test, dataset.Y_test, verbose=2)
    print('Train: %.3f, Test: %.4f' % (train_acc, test_acc))


def plot_eval(history, n_epochs, title):
    loss_train = history['loss']
    loss_val = history['val_loss']
    epochs = range(1, n_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title(f'{title} Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(f'./plots/{title}_train_vs_val_loss.png')

    loss_train = history['acc']
    loss_val = history['val_acc']
    epochs = range(1, n_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig(f'./plots/{title}_train_vs_val_acc.png')


def plot_words_count(dataset):
    sns_palette = ["rocket", "mako", "magma", "rocket_r"]
    sns.set_style("whitegrid")
    for i, book in dataset.iterrows():
        cnt_pro = np.asarray(Counter(book['text'].split()).most_common(20))
        occ_df = pd.DataFrame({'word': cnt_pro[:, 0], 'count': [int(num) for num in cnt_pro[:, 1]]})
        plt.figure(figsize=(12, 4))
        sns.barplot(x='word', y='count', alpha=0.8, data=occ_df, palette=sns_palette[i])
        plt.title(book['author'])
        plt.ylabel('Frequency', fontsize=12)
        plt.xlabel('Word', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()
        plt.savefig(f'./plots/words_bar_{book["author"]}.png')


def plot_compare_bars(dataset):
    words = []
    names = []
    fig = plt.figure(figsize=(12, 4))
    fig.set_facecolor('white')
    for i, book in dataset.iterrows():
        words.append(len(book['text'].split()))
        names.append(book['author'])

    sns.barplot(names, words, alpha=0.8)
    plt.ylabel('Length of works', fontsize=12)
    plt.xlabel('Author', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()
    plt.savefig('./plots/words_count.png')


def plot_words_cloud(dataset):
    for i, book in dataset.iterrows():
        wordcloud = WordCloud(background_color="white",
                              stopwords=stopwords.words('russian'),
                              mode="RGBA",
                              width=400,
                              height=330,
                              colormap='inferno').generate(book['text'])
        plt.figure(figsize=[7, 7])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        plt.savefig(f'./plots/wordcloud_{book["author"]}.png')


def plot_scores(score_a, score_b):
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
    plt.savefig(f'./plots/Score_acc_plot.png')
