#!/bin/bash

cd "${BASH_SOURCE%/*}"

jupyter nbconvert --output-dir='./notebooks' --to markdown notebooks/original/*.ipynb

git add --all
git commit -m 'auto-commit'
git push
