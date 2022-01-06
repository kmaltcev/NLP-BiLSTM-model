import json
import os

import streamlit as st
import tensorflow as tf
from data.RawDataset import RawDataset
from data.TestSet import TestSet
from data.TrainSet import TrainSet
from models.Models import ELMo, CNN, BiLSTM, Ensemble
from utils.utils import plot_prediction, convert_embeddings_to_tensor, circular, build_graph_data, \
    build_graph_data_summary, plot_train_prediction, count_distance
from utils.experiment_params import ExperimentsParams


def preprocess(dataset):
    with st.spinner(text="Preprocessing in progress..."):
        dataset.preprocess()
    dataset.chunking()
    with st.spinner(text="Embedding in progress..."):
        dataset.create_embedding(ELMo)


def plot_by_cols(desc, *args):
    for arg in args:
        with next(col):
            st.pyplot(arg)
            if len(desc) > 0:
                st.write(desc)


# Set streamlit configuration
st.set_page_config(page_title="Plagiarism detection", initial_sidebar_state="expanded", layout="wide")
st.markdown('# Plagiarism detection using Impostors Method')

# Initiate default configs for tensorflow outputs
tf.compat.v1.experimental.output_all_intermediates(True)

# Load default model settings from settings.json
with open('settings.json') as f:
    settings = json.load(f)

# Base data directory path
base_dir = st.sidebar.text_input("Path to data", value="./books")

if base_dir:
    all_data = os.listdir(base_dir)
    first_impostor = st.sidebar.multiselect("First Impostors", all_data)
    second_impostor = st.sidebar.multiselect("Second Impostors", all_data)
    author_under_test = st.sidebar.selectbox("Author under test", all_data)
    creation_under_test = st.sidebar.selectbox("Creation under test", os.listdir(f"{base_dir}/{author_under_test}"))
    btn_cols = st.sidebar.columns([1, 3])
    start_training = st.sidebar.button("Analyse Authorship!")
    # Hyper-Parameters controls
    sb_cols = st.sidebar.columns([1, 1])
    # Initialize variables
    parameters = dict()
    for i, (category_key, category_param_list) in enumerate(settings['params'].items()):
        with sb_cols[i]:
            parameters[category_key] = dict()
            f'{category_key}'
            parameters[category_key]['lr'] = float(st.text_input(label='Learning rate', value=0.0001,
                                                                 key=i, max_chars=6))
            for j, (label, config) in enumerate(category_param_list.items()):
                parameters[category_key][label] = st.number_input(config['label'], config['min_value'],
                                                                  config['max_value'], config['value'],
                                                                  config['step'], key=(i + j + 3),
                                                                  format=config['format'])
    if start_training:
        if len(first_impostor) == 0 or len(second_impostor) == 0:
            st.error("Impostors are empty")
        elif len(first_impostor) != len(second_impostor):
            st.error("Impostors are not the same length. Please provide the same number of authors for each impostor")
        else:
            st.markdown(f"## Test set: {author_under_test}, {creation_under_test}")
            test_set = TestSet(author_under_test)
            if len(test_set.data['book'].values) < 2:
                st.error(
                    f"Author under test has only {len(test_set.data['book'].values)} creations, please provide more.")
            preprocess(test_set)

            for idx, (impostor1, impostor2) in enumerate(zip(first_impostor, second_impostor)):
                if impostor1 == impostor2:
                    st.warning(f"{impostor1} paired with himself, predictions quality will not be valid. "
                               "Please provide different authors.")
                st.markdown(f"### {idx + 1}. {impostor1} vs. {impostor2}")
                authors_pair = [impostor1, impostor2]
                params = ExperimentsParams(author_under_test, creation_under_test, impostor1, impostor2)
                raw_train_set = RawDataset(authors_pair)
                preprocess(raw_train_set)
                with st.spinner(text="Counting distance..."):
                    distance, path = count_distance(raw_train_set.embeddings, distance_measure_cut=100)

                if distance == 0:
                    st.error(f"These books looks are the same. The distance is: {distance}")
                elif 0 < distance < 10:
                    st.warning(f"These books looks way too similar. The distance is: {distance}")
                elif 10 <= distance < 11:
                    st.success(f"Good choice for impostors! The distance is: {distance}")
                else:
                    st.warning(f"These books looks far too different. The distance is: {distance}")

                cols = st.columns([1, 1, 1, 1])
                col = circular(cols)
                train_set = TrainSet(raw_train_set.data)
                cnn = CNN(train_set.X_shape(), parameters)
                bilstm = BiLSTM(train_set.X_shape(), parameters)
                cnn_bilstm = Ensemble(train_set)

                with st.spinner(text="CNN Training in progress..."):
                    loss_fig, acc_fig = cnn_bilstm.add(cnn, params.path_to_plot)
                    plot_by_cols("", loss_fig, acc_fig)
                with st.spinner(text="BiLSTM Training in progress..."):
                    loss_fig, acc_fig = cnn_bilstm.add(bilstm, params.path_to_plot)
                    plot_by_cols("", loss_fig, acc_fig)
                with st.spinner(text="Ensemble Training in progress..."):
                    cnn_bilstm.build()
                    cnn_bilstm.fit()

                Y = cnn_bilstm.predict()
                preds = dict()
                for i, impostor in enumerate(authors_pair):
                    preds[impostor] = Y[Y == i].shape[0]

                predictions = plot_train_prediction(preds, f"{params.path_to_plot}/preds_train_set_distribution.png")
                plot_by_cols("Validation barplot, training set is used for predictions. "
                             "Therefore, distribution must be close to equal. "
                             "If it's not, something is wrong", predictions)

                preds = dict()
                for i, row in test_set.data.iterrows():
                    X = convert_embeddings_to_tensor(row['embeddings'])
                    Y = cnn_bilstm.predict(X)
                    preds[row['book']] = Y

                graph_data = build_graph_data(preds, authors_pair)
                predictions = plot_prediction(graph_data, f"{params.path_to_plot}/preds_by_book_distribution.png")
                plot_by_cols("Predictions distribution by every creation", predictions)

                graph_data_summary = build_graph_data_summary(graph_data, authors_pair, creation_under_test)
                predictions = plot_prediction(graph_data_summary,
                                              f"{params.path_to_plot}/preds_summary_distribution.png")
                plot_by_cols("Predictions are summarized except creation under test."
                             "Creation under test is scaled up for easier analysis", predictions)
                del cols
