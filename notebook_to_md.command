#!/bin/bash

cd "${BASH_SOURCE%/*}"


rsync -rPz --include="*.ipynb" --include="*/img/*.png" --exclude="/Notes/" --exclude="*.*" --delete-excluded rosles@gales.cs.ox.ac.uk:projects/crisis-data/notebooks/ notebooks/


# Create Markdown copies of notebooks for Github viewing
rm -rf ./notebooks/markdown/*
jupyter nbconvert --output-dir='./notebooks/markdown' --to markdown notebooks/*.ipynb

git add --all
git commit -m 'auto-commit'
git push
