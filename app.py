import json
import os

import streamlit as st
import numpy as np
from matplotlib import pyplot as plt

from data.Dataset import Dataset
from data.TrainSet import TrainSet
from models.merge import ELMo, CNN, BiLSTM, Ensemble
from utils.utils import plot_words_cloud, plot_words_count, plot_compare_bars

# Set streamlit configuration
st.set_page_config(page_title="Plagiarism detection", initial_sidebar_state="expanded")

# Load default model settings from settings.json
with open('settings.json') as f:
    settings = json.load(f)

base_dir = st.sidebar.text_input("Path to data", value="D:\\Study\\Semester 9\\Project\\NLP-BiLSTM-model\\books")
if base_dir:
    all_data = os.listdir(base_dir)
    first_impostor = st.sidebar.multiselect("Impostor1", all_data)
    second_impostor = st.sidebar.multiselect("impostor2", [data for data in all_data if data not in first_impostor])
    author_under_test = st.sidebar.selectbox("Author under test", all_data)
    pressed = st.sidebar.button("Start training!")

cols = st.columns([6, 1, 1])

for i, label in enumerate(settings['params_labels']):
    idx = -i if label != "Preprocessing" else 1
    with cols[idx]:
        f"{label}"
        for j, parameter in enumerate(settings['params'][i]):
            parameter['value'] = st.number_input(parameter['label'], parameter['min_value'],
                                                 parameter['max_value'], parameter['value'],
                                                 parameter['step'], key=(i + j + 3), format=parameter['format'])

with cols[0]:
    st.markdown('# Plagiarism detection using Impostors Method')

    if pressed:
        all_authors = []
        all_authors.extend(first_impostor)
        all_authors.extend(second_impostor)

        for impostor1, impostor2 in zip(first_impostor, second_impostor):
            dataset = Dataset([impostor1, impostor2])

            dataset.data

            dataset.preprocess()

            dataset.data

            [st.pyplot(plot) for plot in plot_words_count(dataset.data)]
            st.pyplot(plot_compare_bars(dataset.data))
            dataset.chunking()

            shape = dataset.embedding(ELMo)

            dataset.embeddings

