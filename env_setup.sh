#!/bin/bash
# create Python 3 virtual environment
python3 -m pip install --user virtualenv
virtualenv -p python3 mask_designer_env
source mask_designer_env/bin/activate

# install package
pip install -e .
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 # Can't be done inside setup.py
pip install -e "git+https://github.com/ebezzam/slm-controller.git@master#egg=slm_controller"
pip install -e "git+https://github.com/ebezzam/waveprop.git@master#egg=waveprop" # TODO fails on my linux partition, numpy build fails
pip install numpy==1.22                                                          # TODO waveprop uses old numpy (1.19.5)
