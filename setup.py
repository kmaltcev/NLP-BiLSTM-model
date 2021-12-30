import os
import io
import nltk
import wget
import zipfile
import requests
import conda.cli
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='final_project',
    version='1',
    packages=find_packages(),
    url='',
    license='None',
    author='kmalt',
    author_email='k.maltcev@outlook.com',
    description='Setup CNN-BiLSTM Authorship Analysis',
    python_requires='==3.9',
    install_requires=required
)
conda.cli.main('conda', 'install',  '-y', 'cudatoolkit')
conda.cli.main('conda', 'install',  '-y', 'cudnn')
nltk.download('stopwords')

if not os.path.exists("plots"):
    os.mkdir("plots")
if not os.path.exists("embeddings"):
    os.mkdir("embeddings")
if not os.path.exists("prep_data_cached"):
    os.mkdir("prep_data_cached")
if not os.path.exists("elmo"):
    os.mkdir("elmo")
    if "model.hdf5" not in os.listdir("./elmo"):
        print("Downloading ELMo (araneum_lemmas_elmo_2048_2020) 1.5 GB...")
        r = requests.get("http://vectors.nlpl.eu/repository/20/212.zip")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("./elmo/")
        print("ELMo is ready!")
