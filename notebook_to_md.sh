#!/bin/bash

cd "${BASH_SOURCE%/*}"

# Execute all notebooks
#jupyter nbconvert --to notebook --inplace --execute notebooks/*.ipynb

# ipynb to markdown, exluding code cells.
jupyter nbconvert --output-dir='.' --to markdown notebooks/*.ipynb
