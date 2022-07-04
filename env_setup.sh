#!/bin/sh

# Clone from GitHub
git@github.com:nbaehler/slm-designer.git

# create Python 3 virtual environment
python3 -m pip install --user virtualenv
virtualenv -p python3 slm_designer_env
source slm_designer_env/bin/activate

# install package
pip install -e .
pip install -e "git+https://github.com/ebezzam/slm-controller.git@holoeye#egg=slm_controller" # TODO change branch once merged
pip install -e "git+https://github.com/ebezzam/waveprop.git@master#egg=waveprop"
pip install numpy==1.22 # TODO waveprop uses old numpy (1.19.5)
