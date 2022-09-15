#!/bin/bash
sphinx-apidoc -f -o docs/source/ mask_designer/
sphinx-build docs/source docs/build
