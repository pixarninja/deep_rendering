#!/bin/bash

if (($# < 1)); then
	echo "Usage: ./compile [LaTeX file]"
else
	mkdir "output"
	pdflatex --shell-escape --output-directory "./output" $1
	name=$(echo "$1" | cut -f 1 -d '.')
	ext=".pdf"
	path="./output/"$name$ext
	xdg-open "$path"
fi
