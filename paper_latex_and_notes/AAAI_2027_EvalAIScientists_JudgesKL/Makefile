MAIN=00_main_aaai_2027
ARXIV=00_main_arxiv
BODY_ONLY=00_main_aaai_2027_body_only
PDFLATEX?=pdflatex
BIBTEX?=bibtex

# The bundled AAAI-27 style requires pdfTeX; Tectonic uses XeTeX.
all: pdftex arxiv

pdftex:
	mkdir -p _build
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(MAIN).tex
	$(BIBTEX) _build/$(MAIN)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(MAIN).tex
	cp _build/$(MAIN).pdf .

arxiv:
	mkdir -p _build
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(ARXIV).tex
	$(BIBTEX) _build/$(ARXIV)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(ARXIV).tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(ARXIV).tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(ARXIV).tex
	cp _build/$(ARXIV).pdf .

body-only:
	mkdir -p _build
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(BODY_ONLY).tex
	$(BIBTEX) _build/$(BODY_ONLY)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(BODY_ONLY).tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=_build $(BODY_ONLY).tex

latexmk:
	mkdir -p _build
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=_build $(MAIN).tex && cp _build/$(MAIN).pdf .
