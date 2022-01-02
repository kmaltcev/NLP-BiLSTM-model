import json
import os

import streamlit as st
import tensorflow as tf
from data.Dataset import Dataset
from data.TestSet import TestSet
from data.TrainSet import TrainSet
from models.merge import ELMo, CNN, BiLSTM, Ensemble
from utils.utils import plot_compare_bars, plot_prediction, convert_embeddings_to_tensor, circular
from utils.experiment_params import ExperimentsParams


def preprocess(dataset):
    with st.spinner(text="Preprocessing in progress..."):
        dataset.preprocess()
    # plots = [st.pyplot(plot) for plot in plot_words_bar(dataset.data, params.path_to_plot)]
    # st.pyplot(plot_compare_bars(dataset.data, params.path_to_plot))
    dataset.chunking()
    with st.spinner(text="Embedding in progress..."):
        dataset.create_embedding(ELMo)


# Set streamlit configuration
st.set_page_config(page_title="Plagiarism detection", initial_sidebar_state="expanded", layout="wide")
st.markdown('# Plagiarism detection using Impostors Method')
# Initiate default configs for tensorflow outputs
tf.compat.v1.experimental.output_all_intermediates(True)

# Initialize variables
parameters = dict()

# Load default model settings from settings.json
with open('settings.json') as f:
    settings = json.load(f)

# Base data directory path
base_dir = st.sidebar.text_input("Path to data", value="books")
if base_dir:
    all_data = os.listdir(base_dir)

    first_impostor = st.sidebar.multiselect("Impostor1", all_data)
    second_impostor = st.sidebar.multiselect("impostor2", [data for data in all_data if data not in first_impostor])
    author_under_test = st.sidebar.selectbox("Author under test", all_data)
    creation_under_test = st.sidebar.selectbox("Creation under test", os.listdir(f"./{base_dir}/{author_under_test}"))
    training = st.sidebar.button("Start training!")
    # Hyper-Parameters controls
    sb_cols = st.sidebar.columns([1, 1])
    for i, (category_key, category_param_list) in enumerate(settings['params'].items()):
        with sb_cols[i]:
            parameters[category_key] = dict()
            f'{category_key}'
            for j, (label, config) in enumerate(category_param_list.items()):
                parameters[category_key][label] = st.number_input(config['label'], config['min_value'],
                                                                  config['max_value'], config['value'],
                                                                  config['step'], key=(i + j + 3),
                                                                  format=config['format'])
    if training:
        for impostor1, impostor2 in zip(first_impostor, second_impostor):
            st.markdown(f"### Impostor 1: {impostor1}, Impostor 2: {impostor2}, Test author: {author_under_test}")
            authors_pair = [impostor1, impostor2]
            cols = st.columns([1, 1, 1, 1])
            col = circular(cols)
            params = ExperimentsParams(author_under_test, creation_under_test, impostor1, impostor2)
            raw_train_set = Dataset(authors_pair)
            preprocess(raw_train_set)
            train_set = TrainSet(raw_train_set.data)
            cnn = CNN(train_set.X_shape(),
                      num_filters=parameters["CNN"]["num_filters"],
                      kernel_size=[parameters["CNN"]["kernel_size_1"],
                                   parameters["CNN"]["kernel_size_2"],
                                   parameters["CNN"]["kernel_size_3"]],
                      dropout_rate=parameters["CNN"]["dropout_rate"],
                      fc_layer_size=parameters["CNN"]["fc_layer_size"],
                      learning_rate=0.0001,
                      epochs=parameters["CNN"]["epochs"],
                      batch_size=parameters["CNN"]["batch_size"])
            cnn.build()
            loss_fig, acc_fig = cnn.fit(train_set)
            with next(col):
                st.pyplot(loss_fig)
            with next(col):
                st.pyplot(acc_fig)
            bilstm = BiLSTM(train_set.X_shape(),
                            hidden_state_dim=parameters["BiLSTM"]["hidden_state_dim"],
                            dropout_rate=parameters["BiLSTM"]["dropout_rate"],
                            fc_layer_size=parameters["BiLSTM"]["fc_layer_size"],
                            learning_rate=0.0001,
                            epochs=parameters["BiLSTM"]["epochs"],
                            batch_size=parameters["BiLSTM"]["batch_size"])
            bilstm.build()
            loss_fig, acc_fig = bilstm.fit(train_set)
            with next(col):
                st.pyplot(loss_fig)
            with next(col):
                st.pyplot(acc_fig)
            cnn_bilstm = Ensemble(train_set)

            with st.spinner(text="CNN Training in progress..."):
                loss_fig, acc_fig = cnn_bilstm.add(cnn)
            with next(col):
                st.pyplot(loss_fig)
            with next(col):
                st.pyplot(acc_fig)

            with st.spinner(text="BiLSTM Training in progress..."):
                loss_fig, acc_fig = cnn_bilstm.add(bilstm)
                with next(col):
                    st.pyplot(loss_fig)
                with next(col):
                    st.pyplot(acc_fig)

            with st.spinner(text="Ensemble Training in progress..."):
                cnn_bilstm.build()
                cnn_bilstm.fit()

            test_set = TestSet(author_under_test)
            preprocess(test_set)

            preds = dict()
            for idx, row in test_set.data.iterrows():
                X = convert_embeddings_to_tensor(row['embeddings'])
                Y = cnn_bilstm.predict(X)
                preds[row['book']] = Y

            fig2 = plot_prediction(preds, authors_pair, params.path_to_plot)
            with next(col):
                st.pyplot(fig2)
            del cols
