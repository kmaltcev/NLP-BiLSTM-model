import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from utils.utils import circular_generator


# Compute ratios and expected value
def build_graph_data_z_test(dataframe, impostors_pair, creation_under_test):
    value = dataframe[dataframe['book'] == creation_under_test]['count'].values
    value = value[0] / value[1] if value[1] != 0 else 0

    dataframe = dataframe[dataframe['book'] != creation_under_test]
    arr = [dataframe[dataframe['label'] == impostors_pair[0]]['count'].values,
           dataframe[dataframe['label'] == impostors_pair[1]]['count'].values]

    ratios = []
    for v1, v2 in zip(arr[0], arr[1]):
        if v2 == 0:
            ratios.append(v1)
        elif v1 == 0:
            ratios.append(v2)
        else:
            ratios.append(v1 / v2)

    return ratios, value


# Compute sum's of predictions except Book under test, and scale book under test
def build_graph_data_summary(dataframe, authors_pair, test_creation):
    graph_data = []
    for impostor in authors_pair:
        A = dataframe[dataframe['book'] != test_creation][dataframe['label'] == impostor]
        graph_data.append(pd.DataFrame({
            "book": "Summary except test creation",
            "label": A['label'].values[0],
            "count": A['count'].sum()
        }, index=[0]))
    df = pd.concat(graph_data)
    max_value = df[df['count'] == max(df['count'])]
    graph_data.append(dataframe[dataframe['book'] == test_creation])
    ratio = max_value['count'][0] / graph_data[2][graph_data[2]['label'] == max_value['label'][0]]['count'].values[0]
    for i in range(2):
        graph_data[2].at[graph_data[2].index[i], 'count'] = graph_data[2]['count'].values[i] * ratio
    return pd.concat(graph_data)


def plot_eval(history, epochs, title, path_to_plot):
    cmap = circular_generator(['g', 'b'])

    fig_1 = plt.figure()
    labels = (label for label in ['Training loss', 'Validation loss'])
    plt.title(f'{title} Training and Validation loss')
    for k in {'loss', 'val_loss'}.intersection(history.keys()):
        v = history[k]
        plt.plot(epochs, v, next(cmap), label=next(labels))
        plt.xlabel(k)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./{path_to_plot}/{title}_train_vs_val_loss.png')

    fig_2 = plt.figure()
    labels = (label for label in ['Training accuracy', 'Validation accuracy'])
    plt.title(f'{title} Training and Validation accuracy')
    for k in {'accuracy', 'val_accuracy'}.intersection(history.keys()):
        v = history[k]
        plt.plot(epochs, v, next(cmap), label=next(labels))
        plt.xlabel(k)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'./{path_to_plot}/{title}_train_vs_val_acc.png')
    return fig_1, fig_2


def plot_prediction(graph_data, path):
    fig = plt.figure()
    fig.set_facecolor('white')
    sns.barplot(x="book", y="count", alpha=0.8, hue="label", data=graph_data)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title("Chunks Distribution")
    plt.ylabel('Chunk', fontsize=12)
    plt.xlabel('Author', fontsize=12)
    plt.xticks(rotation=90)
    plt.savefig(path)
    return fig


def plot_train_prediction(graph_data, path):
    fig = plt.figure()
    fig.set_facecolor('white')
    sns.barplot(x=list(graph_data.keys()), y=list(graph_data.values()), alpha=0.8)
    plt.title("Chunks Distribution")
    plt.ylabel('Chunk', fontsize=12)
    plt.xlabel('Author', fontsize=12)
    plt.xticks(rotation=90)
    plt.savefig(path)
    return fig
