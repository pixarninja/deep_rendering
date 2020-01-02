#!/bin/bash

if [ $# == 1 ]; then
	mkdir "output"
	name=$(echo "$1" | cut -f 1 -d '.')
	ext=".pdf"
	path="./output/"$name$ext

	pdflatex --shell-escape --output-directory "./output" $1
	xdg-open "$path"
elif [ $# == 2 ] && [ $2 = "-b" ]; then
	mkdir "output"
	name=$(echo "$1" | cut -f 1 -d '.')
	ext=".pdf"
	path="./output/"$name$ext
	subpath="./output/"$name

	pdflatex --shell-escape --output-directory "./output" $1
	bibtex "$subpath"
	pdflatex --shell-escape --output-directory "./output" $1
	xdg-open "$path"
else
	echo "Usage: ./compile [LaTeX file] [optional: -b (BibTeX compile)]"
fi
