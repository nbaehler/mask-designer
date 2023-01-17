#!/bin/bash
sphinx-apidoc -f -o source/ ../mask_designer/
sphinx-build source build
