#!/bin/bash

black *.py -l 100
black examples/*.py -l 100
black mask_designer/**/*.py -l 100
black tests/*.py -l 100

# TODO black formats differently in Linux and Windows...
