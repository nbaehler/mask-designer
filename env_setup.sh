#!/bin/bash

# https://stackoverflow.com/a/18434831
case "$OSTYPE" in
solaris*) OS="SOLARIS" ;;
darwin*) OS="OSX" ;;
linux*) OS="LINUX" ;;
bsd*) OS="BSD" ;;
msys*) OS="WINDOWS" ;;
cygwin*) OS="WINDOWS" ;;
*) OS= "unknown" ;;
esac

if [[ "$OS" == "LINUX" ]]; then
    # create Python 3 virtual environment
    python3 -m pip install --user virtualenv
    virtualenv -p python3 mask_designer_env
    source mask_designer_env/bin/activate
elif [[ "$OS" == "WINDOWS" ]]; then
    # create Python 3 virtual environment
    python -m pip install virtualenv
    python -m virtualenv mask_designer_env
    source ./mask_designer_env/Scripts/activate
else
    echo "Unknown OS"
    exit 1
fi

# install package
python -m pip install -e .
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 # Can't be done inside setup.py
python -m pip install -e "git+https://github.com/ebezzam/slm-controller.git@master#egg=slm_controller"
python -m pip install waveprop
