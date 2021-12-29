import json
import os
import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from data.Dataset import Dataset
from data.TrainSet import TrainSet
from models.merge import ELMo, CNN, BiLSTM, Ensemble
from utils.utils import plot_words_cloud, plot_words_count, plot_compare_bars
from data_objects.experiment_params import ExperimentsParams
# Set streamlit configuration
st.set_page_config(page_title="Plagiarism detection", initial_sidebar_state="expanded",layout = "wide")
# Load default model settings from settings.json
with open('settings.json') as f:
    settings = json.load(f)
# Initiating default configs for tensorflow outputs
tf.compat.v1.experimental.output_all_intermediates(True)

base_dir = st.sidebar.text_input("Path to data", value="books")

if base_dir:

    all_data = os.listdir(base_dir)

    first_impostor = st.sidebar.multiselect("Impostor1", all_data)

    second_impostor = st.sidebar.multiselect("impostor2", [data for data in all_data if data not in first_impostor])

    author_under_test = st.sidebar.selectbox("Author under test", all_data)

    if author_under_test:

        creation_under_test = st.sidebar.selectbox("Creation under test", os.listdir(base_dir+"/"+author_under_test))

    pressed = st.sidebar.button("Start training!")


cols = st.columns([6, 1, 1])

parameters = dict()

for i,(category_key,category_param_list) in enumerate(settings['params'].items()):

    with cols[i] if category_key != "ELMo" else cols[1]:

        parameters[category_key] = dict()

        f'{category_key}'

        for j,(label,config) in enumerate(category_param_list.items()):

            parameters[category_key][label] = st.number_input(config['label'],

                                                    config['min_value'],

                                                    config['max_value'],

                                                    config['value'],

                                                    config['step'],

                                                    key=(i + j + 3),

                                                   format=config['format'])


with cols[0]:

    st.markdown('# Plagiarism detection using Impostors Method')


    if pressed:

        all_authors = []

        all_authors.extend(first_impostor)

        all_authors.extend(second_impostor)

        for impostor1, impostor2 in zip(first_impostor, second_impostor):
            params = ExperimentsParams(author_under_test,creation_under_test,impostor1,impostor2)
            dataset = Dataset([impostor1, impostor2])
            dataset.preprocess()
            plots = [st.pyplot(plot) for plot in plot_words_count(dataset.data,params.path_to_plot)]
            st.pyplot(plot_compare_bars(dataset.data,params.path_to_plot))
            dataset.chunking()
            dataset.embedding(ELMo)
            train_set = TrainSet(dataset.data)
            kernels_sizes = [parameters["CNN"]["kernel_size_1"],
                            parameters["CNN"]["kernel_size_2"],
                            parameters["CNN"]["kernel_size_3"]]
            cnn = CNN(train_set.X_shape(),
                       num_filters = parameters["CNN"]["num_filters"],
                       kernel_size=kernels_sizes,
                       dropout_rate = parameters["CNN"]["dropout_rate"],
                       fc_layer_size= parameters["CNN"]["fc_layer_size"],
                       learning_rate=parameters["CNN"]["learning_rate"],
                       epochs=parameters["CNN"]["epochs"],
                      batch_size=parameters["CNN"]["batch_size"])
            cnn.build()
            cnn.fit(train_set)
            bilstm = BiLSTM(train_set.X_shape(),
                            hidden_state_dim=parameters["BiLSTM"]["hidden_state_dim"],
                            dropout_rate= parameters["BiLSTM"]["dropout_rate"],
                            fc_layer_size= parameters["BiLSTM"]["fc_layer_size"],
                            learning_rate= parameters["BiLSTM"]["learning_rate"],
                            epochs= parameters["BiLSTM"]["epochs"],
                           batch_size=parameters["BiLSTM"]["batch_size"])
            bilstm.build()
            cnn_bilstm = Ensemble(train_set)
            cnn_bilstm.add(cnn)
            cnn_bilstm.add(bilstm)
            cnn_bilstm.build()
            cnn_bilstm.fit()
            cnn_bilstm.predict()

