#!/bin/sh

# create Python 3 virtual environment
python3 -m pip install --user virtualenv
virtualenv -p python3 slm_designer_env # TODO rename
source slm_designer_env/bin/activate

# install package
pip install -e .[dev]
