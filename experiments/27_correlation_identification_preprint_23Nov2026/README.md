# Correlation identification preprint — 11-23-2026

**What Correlation Can and Cannot Identify in Human-Rating Judges**  
Brando Miranda, Stanford University. 11-23-2026. CC BY 4.0.

This is a self-contained theory preprint developed from the detailed derivation in experiment 18.
It characterizes correlation of conditional means, model restrictions, proper scoring rules, and finite-data limitations.
The exact counterexample is arithmetic, with no fitted model, sampled ratings, or new paid inference.
It does not claim a universal finite-data advantage for CE-trained judges.

## Read and reproduce

- `PAPER_11-23-2026.pdf`: compiled preprint.
- `PAPER_11-23-2026.tex` and `sections/`: canonical sectioned LaTeX source.
- `PAPER_standalone_11-23-2026.tex`: self-contained export, equivalent to the sectioned source.
- `PAPER_11-23-2026.md`: matching Markdown text.
- `CONTENT_MAP.md`: trace of the derivation and corrections to its scope.

Build twice from this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error PAPER_11-23-2026.tex
pdflatex -interaction=nonstopmode -halt-on-error PAPER_11-23-2026.tex
python3 experiments/exact_counterexample/reproduce.py
```

The exact illustration needs only Python's standard library.
Optional independent numerical checks use NumPy:

```sh
python3 experiments/verify_identities.py
```

`experiments/identity_check.json` records 200 seeded identity checks and independent values for the exact example.
Both CE values are in nats; RPS is the normalized individual-rating score, with denominator K−1.
The original PDF is `../18_when_does_ce_beat_correlation_notebook_page_math_and_sweep_start_21Nov2026/MATH_11-21-2026.pdf`, SHA-256 `d929b23e8e397d45966ec663539a8ce1c29e961c5deae7d877b0d579d27edb27`.
The original derivation is preserved there unchanged.
