#!/bin/bash

black *.py -l 100
black examples/*.py -l 100
black examples/**/*.py -l 100
black mask_designer/*.py -l 100
black mask_designer/**/*.py -l 100
black mask_designer/neural_holography/**/*.py -l 100
black tests/*.py -l 100

# TODO black formats differently in Linux and Windows...
# TODO better version for matching all files and all files in subdirectories
