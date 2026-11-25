# Technical memo, 11-25-2026 (ICML 2026 format, preprint option).
# `make` rebuilds figures/tables from committed results, then the PDF. `make pdf` skips the asset step.
MEMO = memo_11-25-2026

.PHONY: all assets pdf clean
all: assets pdf

assets:
	python3 -I make_memo_assets.py

pdf:
	latexmk -pdf -interaction=nonstopmode $(MEMO).tex

clean:
	latexmk -C $(MEMO).tex
