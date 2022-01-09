import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from utils.utils import circular_generator


def build_graph_data_test(dataframe, authors_pair, test_creation):
    graph_data = []
    for impostor in authors_pair:
        A = dataframe[dataframe['book'] != test_creation][dataframe['label'] == impostor]
        graph_data.append(pd.DataFrame({
            "book": "Summary except test creation",
            "label": A['label'].values[0],
            "count": A['count'].sum()
        }, index=[0]))
    graph_data.append(dataframe[dataframe['book'] == test_creation])
    df = pd.concat(graph_data)
    s1 = df[df['label'] == authors_pair[0]]['count'].values
    s2 = df[df['label'] == authors_pair[1]]['count'].values
    return s1, s2


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
