#!/bin/bash
# create Python 3 virtual environment
python3 -m pip install --user virtualenv
virtualenv -p python3 .slm_designer_env
source .slm_designer_env/bin/activate

# install package
pip install -e .
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 # TODO can't be done inside setup.py
pip install -e "git+https://github.com/ebezzam/slm-controller.git@holoeye#egg=slm_controller"     # TODO change branch once merged
pip install -e "git+https://github.com/ebezzam/waveprop.git@master#egg=waveprop"                  # TODO fails on my linux partition -> numpy build fails
pip install numpy==1.22                                                                           # TODO waveprop uses old numpy (1.19.5)
