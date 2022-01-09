import json
import os
import streamlit as st
import tensorflow as tf
from scipy.stats import ttest_ind
from data.DataTypes import TestSet, TrainSet, RawDataset
from models.Models import ELMo, CNN, BiLSTM, Ensemble
from utils import strings as R
from utils.constants import Constants, BOOKS_DIR
from utils.plots import plot_prediction, plot_train_prediction, build_graph_data_summary, build_graph_data_test
from utils.utils import convert_embeddings_to_tensor, circular_generator, build_graph_data, compute_distance

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


def show_hide_help_button_onclick():
    st.session_state['show_help'] = not st.session_state['show_help']


def preprocess(dataset):
    with st.spinner(text=R.preprocess_progress_label):
        dataset.preprocess()
    dataset.chunking(chunk_size=200)
    with st.spinner(text=R.embedding_progress_label):
        dataset.create_embedding(ELMo)


def plot_by_cols(desc, *args):
    for arg in args:
        with next(col):
            st.pyplot(arg)
            if len(desc) > 0:
                st.caption(desc)


def distances_check(dist):
    for k in range(len(dist)):
        prev = dist[k - 1]
        curr = dist[k]
        if curr['distance'] == 0:
            st.metric(label=f"Distance between {curr['a']} and {curr['b']}",
                      value=f"{curr['distance']:.2f}")
        else:
            st.metric(label=f"Distance between {curr['a']} and {curr['b']}",
                      value=f"{curr['distance']:.2f}",
                      delta=f"{curr['distance'] - prev['distance']}")


# Set streamlit configuration
st.set_page_config(page_title=R.page_title, initial_sidebar_state="expanded", layout="wide")
st.markdown(R.main_page_title)

show_hide_help_button = st.sidebar.button("Show/Hide instructions", on_click=show_hide_help_button_onclick)
if 'show_help' not in st.session_state:
    st.session_state['show_help'] = False
if st.session_state['show_help']:
    with open("HELPME.md", "r") as fp:
        st.markdown(fp.read(), unsafe_allow_html=True)

# Initiate default configs for tensorflow outputs
tf.compat.v1.experimental.output_all_intermediates(True)

# Load default model settings from settings.json
with open('settings.json') as f:
    settings = json.load(f)

# Base data directory path
base_dir = st.sidebar.text_input(R.path_to_data_label, value=f"./{BOOKS_DIR}")

if base_dir:
    all_data = os.listdir(base_dir)
    first_impostor = st.sidebar.multiselect(R.first_impostor_label, all_data)
    second_impostor = st.sidebar.multiselect(R.second_impostor_label, all_data)
    author_under_test = st.sidebar.selectbox(R.author_under_test, all_data)
    creation_under_test = st.sidebar.selectbox(R.select_box_label, os.listdir(f"{base_dir}/{author_under_test}"))
    btn_cols = st.sidebar.columns([1, 3])
    start_training = st.sidebar.button(R.button_label)
    # Hyper-Parameters controls
    sb_cols = st.sidebar.columns([1, 1])
    # Initialize variables
    parameters = dict()
    for i, (category, category_param_list) in enumerate(settings['params'].items()):
        with sb_cols[i]:
            parameters[category] = dict()
            f'{category}'
            parameters[category]['lr'] = float(
                st.text_input(label='Learning rate', value=0.0001, key=i, max_chars=6, help=R.lr_desc))
            for j, (label, config) in enumerate(category_param_list.items()):
                parameters[category][label] = st.number_input(config['label'], config['min_value'],
                                                              config['max_value'], config['value'],
                                                              config['step'], key=(i + j + 3),
                                                              format=config['format'], help=config['help'])
    # begin training
    if start_training:
        if len(first_impostor) == 0 or len(second_impostor) == 0:
            st.error(R.empty_impostors_err)
        elif len(first_impostor) != len(second_impostor):
            st.error(R.length_err)
        else:
            st.markdown(R.test_set_title(author_under_test, creation_under_test))
            test_set = TestSet(author_under_test)
            if len(test_set.data['book'].values) < 2:
                st.error(R.test_set_len_err(test_set))
            else:
                # Preprocess test_set
                preprocess(test_set)
                # Loops over impostors pairs
                for idx, (impostor1, impostor2) in enumerate(zip(first_impostor, second_impostor)):

                    # Initialize constants
                    constants = Constants(author_under_test, creation_under_test, impostor1, impostor2)
                    # Check if pairs is the same
                    if impostor1 == impostor2:
                        st.warning(R.same_impostors_err(impostor1))

                    st.markdown(R.experiment_title(idx, impostor1, impostor2))
                    impostors_pair = [impostor1, impostor2]
                    raw_train_set = RawDataset(impostors_pair)
                    preprocess(raw_train_set)

                    cols = st.columns([1, 1, 1, 1])
                    col = circular_generator(cols)
                    train_set = TrainSet(raw_train_set.data)
                    cnn = CNN(train_set.X_shape(), parameters)
                    bilstm = BiLSTM(train_set.X_shape(), parameters)
                    cnn_bilstm = Ensemble(train_set)

                    with st.spinner(text=R.training_title(cnn.name)):
                        loss_fig, acc_fig = cnn_bilstm.add(cnn, constants.path_to_plot)
                        plot_by_cols("", loss_fig, acc_fig)
                    with st.spinner(text=R.training_title(bilstm.name)):
                        loss_fig, acc_fig = cnn_bilstm.add(bilstm, constants.path_to_plot)
                        plot_by_cols("", loss_fig, acc_fig)
                    with st.spinner(text=R.training_title(cnn_bilstm.name)):
                        cnn_bilstm.build()
                        cnn_bilstm.fit()
                    # Validate predictions of train set
                    Y = cnn_bilstm.predict()
                    preds = dict()
                    for i, impostor in enumerate(impostors_pair):
                        preds[impostor] = Y[Y == i].shape[0]
                    # Plot validation bar plot
                    predictions = plot_train_prediction(preds, R.plot_train_path(constants.path_to_plot))
                    plot_by_cols(R.validation_barplot_desc, predictions)
                    # Get predictions by every book
                    preds = dict()
                    for i, row in test_set.data.iterrows():
                        X = convert_embeddings_to_tensor(row['embeddings'])
                        preds[row['book']] = cnn_bilstm.predict(X)
                    # Plot predictions by creation
                    graph_data = build_graph_data(preds, impostors_pair)
                    predictions = plot_prediction(graph_data, R.plot_by_book_preds_path(constants.path_to_plot))
                    plot_by_cols(R.prediction_by_creation_desc, predictions)
                    # Plot sum of predictions vs. scaled prediction of creation under test
                    graph_data_summary = build_graph_data_summary(graph_data, impostors_pair, creation_under_test)
                    predictions = plot_prediction(graph_data_summary, R.plot_summary_path(constants.path_to_plot))
                    plot_by_cols(R.summarized_pred_desc, predictions)
                    # Count P-value
                    s1, s2 = build_graph_data_test(graph_data, impostors_pair, creation_under_test)
                    statistic, pvalue = ttest_ind(s1, s2)
                    with next(col):
                        delta, delta_color = R.metric_result(pvalue * 100)
                        st.metric(label="P-value",
                                  value=f"{pvalue * 100:.4f}%",
                                  delta=delta, delta_color=delta_color)
                        texts = raw_train_set['text']
                        distances = []
                        with st.spinner(text=R.computing_distance_label(impostors_pair)):
                            distances.append({"a": impostor1,
                                              "b": impostor2,
                                              "distance": compute_distance(texts)})
                        distances_check(distances)
