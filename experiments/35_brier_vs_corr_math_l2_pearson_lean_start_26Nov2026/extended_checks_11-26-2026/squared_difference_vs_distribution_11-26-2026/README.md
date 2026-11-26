# Optimizing a difference of random variables does not match a distribution (11-26-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/squared_difference_vs_distribution_11-26-2026/README.md>

**TLDR:** The "next step" from page three of Brando's 13:01 derivation photographs: a mean difference, a squared difference and a correlation each constrain a few moments of a coupling, so optimizing them does not give distribution matching. The raw squared difference is $(\mu_X-\mu_Y)^2+(\sigma_X-\sigma_Y)^2+2\sigma_X\sigma_Y(1-\rho)$; a mean difference leaves $K-2$ free directions (four for $K=6$); a squared difference between independent draws rewards collapse to the mean; a free coupling does see the law, a fixed pairing does not. 10 of 10 deterministic checks PASS; the Lean 4 file compiles with no `sorry`. Synthetic and exact arithmetic only; no human ratings.

Part of experiment 35's extended checks ([`../README.md`](../README.md)). The handwriting is transcribed in [`../../assets/TRANSCRIPTION_DERIVATION_PAGES_ROUND2_11-26-2026.md`](../../assets/TRANSCRIPTION_DERIVATION_PAGES_ROUND2_11-26-2026.md). Companion: [`../brier_proper_6437_style_11-26-2026/`](../brier_proper_6437_style_11-26-2026/README.md).

## Files

- [`SQUARED_DIFFERENCE_VS_DISTRIBUTION_11-26-2026.md`](SQUARED_DIFFERENCE_VS_DISTRIBUTION_11-26-2026.md): the note: results R1 to R7, how they sit against the page, intuition and worked numbers.
- [`check_squared_difference_vs_distribution.py`](check_squared_difference_vs_distribution.py): ten deterministic checks (exact fractions, numpy, one scipy linear program), each printing PASS or FAIL with its numbers.
- [`results/check_squared_difference_vs_distribution.out`](results/check_squared_difference_vs_distribution.out): the captured output (10 PASS, 0 FAIL).
- [`MeanMatching.lean`](MeanMatching.lean): Lean 4.36.0 and Mathlib v4.36.0 proofs of the concrete facts.
- [`lean_check.txt`](lean_check.txt): command, output and toolchain versions for the Lean check.
- [`make_figures.py`](make_figures.py): draws the two figures.
- [`figures/fig1_equal_mean_different_laws.png`](figures/fig1_equal_mean_different_laws.png): the six-rating pair with equal means, and the $K=3$ simplex with the mean-1 line coloured by Brier distance.
- [`figures/fig2_standardized_and_independent_draws.png`](figures/fig2_standardized_and_independent_draws.png): $Y=2X$ has correlation one and a different law; expected squared difference between independent draws for seven reports.
- [`HOW_TO_READ_RESULTS.md`](HOW_TO_READ_RESULTS.md): data, metrics, differences, uncertainty and how reports were chosen.

## Reproduce

From this folder: `python3 -I check_squared_difference_vs_distribution.py` (about 1 s) and `python3 -I make_figures.py`. Lean: `cd ../lean && lake exe cache get && lake env lean ../squared_difference_vs_distribution_11-26-2026/MeanMatching.lean` (about 20 s once Mathlib is built).
