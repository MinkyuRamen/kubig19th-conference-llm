main.pdf: $(wildcard *.tex) roberta.bib 
	@pdflatex main
	@bibtex main
	@pdflatex main
	@pdflatex main

clean:
	rm -f *.aux *.log *.bbl *.blg present.pdf *.bak *.ps *.dvi *.lot *.bcf main.pdf

dist: main.pdf
	@pdflatex --file-line-errors main

default: main.pdf

