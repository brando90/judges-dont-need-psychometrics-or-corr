# Brier score is proper, in the style of the 6.437 notes (11-26-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/brier_proper_6437_style_11-26-2026/README.md>

**TLDR:** The log-loss properness proof from the MIT 6.437 notes, redone for the Brier score: expected Brier equals $\lVert q-p\rVert^2+(1-\lVert p\rVert^2)$, so the Brier score is strictly proper, but it is local only for two outcomes. The ranked probability score (RPS) inherits both facts. 8 of 8 checks PASS; the Lean 4 file compiles with no `sorry`. Synthetic and exact arithmetic only; no human ratings.

Part of experiment 35's extended checks ([`../README.md`](../README.md)); see also [`../DERIVATIONS.md`](../DERIVATIONS.md) (claims C1 to C3, C7, C8) and [`../../DERIVATION.md`](../../DERIVATION.md). The notes page whose margin asked for this, with the handwriting transcribed, is [`../../assets/TRANSCRIPTION_6437_PAGE_11-26-2026.md`](../../assets/TRANSCRIPTION_6437_PAGE_11-26-2026.md).

## Files

- [`BRIER_PROPER_6437_STYLE_11-26-2026.md`](BRIER_PROPER_6437_STYLE_11-26-2026.md): the note: statement, proof (5′) to (8′), locality, RPS corollary, intuition and worked example.
- [`check_brier_proper_6437_style.py`](check_brier_proper_6437_style.py): eight deterministic checks (numpy and fractions only), each printing PASS or FAIL with its numbers.
- [`results/check_brier_proper_6437_style.out`](results/check_brier_proper_6437_style.out): the captured output (8 PASS, 0 FAIL).
- [`Brier6437.lean`](Brier6437.lean): Lean 4.36.0 and Mathlib v4.36.0 proofs of the identity, strict propriety, the $A,B$ family, locality at $L=2$ and non-locality at $L=3$.
- [`lean_check.txt`](lean_check.txt): command, output, and toolchain versions for the Lean check.
- [`make_figures.py`](make_figures.py): draws the three figures.
- [`figures/fig1_k2_brier_vs_logloss.png`](figures/fig1_k2_brier_vs_logloss.png): $K=2$, true $p(1)=0.7$, expected Brier and expected log-loss against the report.
- [`figures/fig2_k3_brier_excess_simplex.png`](figures/fig2_k3_brier_excess_simplex.png): $K=3$, excess expected Brier $=\lVert q-p\rVert^2$ over the simplex.
- [`figures/fig3_locality_k3_vs_k2.png`](figures/fig3_locality_k3_vs_k2.png): Brier cost against $q(x)$, a cloud at $K=3$ and one curve at $K=2$.
- [`HOW_TO_READ_RESULTS.md`](HOW_TO_READ_RESULTS.md): data, metrics, differences, uncertainty and how reports were chosen.

## Reproduce

From this folder: `python3 -I check_brier_proper_6437_style.py` (under 1 s) and `python3 -I make_figures.py`. Lean: `cd ../lean && lake exe cache get && lake env lean ../brier_proper_6437_style_11-26-2026/Brier6437.lean` (about 35 s once Mathlib is built).
