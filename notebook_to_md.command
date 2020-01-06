#!/bin/bash

cd "${BASH_SOURCE%/*}"

rm ./notebooks/markdown*.md
jupyter nbconvert --output-dir='./notebooks/markdown' --to markdown notebooks/*.ipynb

git add --all
git commit -m 'auto-commit'
git push
