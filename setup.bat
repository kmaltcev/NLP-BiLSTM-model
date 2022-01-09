conda create -n tf-py39 -p ./venv python=3.9
conda activate tf-py39
pip3 install -r requirements.txt
conda install -n tf-py39 conda cudatoolkit cudnn -y
python setup.py install