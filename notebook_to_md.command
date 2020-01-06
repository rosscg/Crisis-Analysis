#!/bin/bash

cd "${BASH_SOURCE%/*}"

rsync -rPz --include="*.ipynb" --exclude="*" --delete-excluded rosles@gales.cs.ox.ac.uk:projects/crisis-data/notebooks/ notebooks/

rm ./notebooks/markdown*.md
jupyter nbconvert --output-dir='./notebooks/markdown' --to markdown notebooks/*.ipynb

git add --all
git commit -m 'auto-commit'
git push
