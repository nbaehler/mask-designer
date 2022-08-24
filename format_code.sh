#!/bin/bash

black -l 100 *.py
black -l 100 examples/*.py
black -l 100 examples/**/*.py
black -l 100 mask_designer/*.py
black -l 100 mask_designer/**/*.py
black -l 100 mask_designer/neural_holography/**/*.py
black -l 100 tests/*.py

# TODO black formats differently in Linux and Windows: https://github.com/psf/black/issues/3037#issuecomment-1110607036
# TODO better version for matching all files and all files in subdirectories: black -l 100 --include $(find -name "*.py")
