#!/bin/bash

cd "${BASH_SOURCE%/*}"

rm ./notebooks/*.md
jupyter nbconvert --output-dir='./notebooks' --to markdown notebooks/original/*.ipynb

git add --all
git commit -m 'auto-commit'
git push
