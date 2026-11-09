MAIN=00_main_aaai_2027
PDFLATEX?=pdflatex
BIBTEX?=bibtex

# The bundled AAAI-27 style requires pdfTeX; Tectonic uses XeTeX.
all: pdftex

pdftex:
	mkdir -p _build
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(MAIN).tex
	$(BIBTEX) _build/$(MAIN)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(MAIN).tex
	cp _build/$(MAIN).pdf .

latexmk:
	mkdir -p _build
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=_build $(MAIN).tex && cp _build/$(MAIN).pdf .
