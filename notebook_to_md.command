#!/bin/bash

cd "${BASH_SOURCE%/*}"
#
# # Copy all notebooks, and png files in an img folders.
rsync -rPz --include="*.ipynb" --include="*/img/*.png" --exclude="/notes/" --exclude="*.*" --delete-excluded rosles@gales.cs.ox.ac.uk:projects/crisis-data/notebooks/ notebooks/
#
#
# # Create Markdown copies of notebooks for Github viewing
# rm -rf ./notebooks/markdown/*
# jupyter nbconvert --output-dir='./notebooks/markdown' --to markdown notebooks/*.ipynb
# cp -r ./notebooks/data ./notebooks/markdown/data
#

# Remove style tag contents from .md files as they do not render on Github:
# python3 <<EOF
# import os, re
# dir = './notebooks/markdown/'
# for fname in os.listdir(dir):
# 	if fname.split('.')[-1] != 'md':
# 		continue
# 	os.rename(dir + fname, dir + fname + '.orig')
# 	with open(dir + fname + '.orig', 'r') as fin, open(dir + fname, 'w') as fout:
# 		data = fin.read()
# 		data = re.sub(r'(\n\<style scoped\>).*?(style\>)',
# 			'',
# 			data, flags=re.DOTALL)
# 		fout.write(data)
# 	os.remove(dir + fname + '.orig')
# EOF


git add --all
git commit -m 'auto-commit'
git push
