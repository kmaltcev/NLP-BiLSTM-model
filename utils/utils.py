import os
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns

from wordcloud import WordCloud
from nltk.corpus import stopwords
from matplotlib import pyplot as plt


def read_books(author_name):
    books_text = ""
    for book_name in os.listdir(f"./books_ru/poems/{author_name}"):
        book_file = open(f"./books_ru/poems/{author_name}/{book_name}", "r", encoding='utf8', errors='replace')
        books_text += " " + book_file.read()
        book_file.close()
    return books_text


def evaluate(model, dataset):
    _, train_acc = model.evaluate(dataset.X_train, dataset.Y_train, verbose=2)
    _, test_acc = model.evaluate(dataset.X_test, dataset.Y_test, verbose=2)
    print('Train: %.3f, Test: %.4f' % (train_acc, test_acc))


def plot_eval(history):
    plt.plot(history['acc'])
    plt.title('Model Accuracy')
    plt.ylabel('acc')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()
    plt.savefig('./plots/model_accuracy.png')

    # summarize history for loss
    plt.plot(history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()
    plt.savefig('./plots/model_loss.png')


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
