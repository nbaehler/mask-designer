#!/bin/bash
sphinx-apidoc -o docs/source/ mask_designer/
sphinx-build docs/source docs/build
