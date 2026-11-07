MAIN=00_main_aaai_2027
all:
	tectonic -X compile --outdir _build $(MAIN).tex && cp _build/$(MAIN).pdf .

latexmk:
	latexmk -pdf -interaction=nonstopmode -f -outdir=_build $(MAIN).tex && cp _build/$(MAIN).pdf .
