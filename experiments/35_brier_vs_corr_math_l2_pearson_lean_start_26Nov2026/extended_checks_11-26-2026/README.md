# Experiment 35, extended checks: Brier, squared error and correlation: what is proper and why training on correlation is not training on Brier, 11-26-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/README.md>

**TLDR:** Correlation is squared error after an oracle shift-and-stretch fitted on the evaluation data, so it is not a proper score and is blind to calibration and to everything beyond the conditional mean; exact counterexamples (machine-checked in Lean 4) show equal correlation with different Brier scores and a judge with higher correlation but strictly worse Brier and ranked probability score (RPS), and a synthetic training run shows the Pearson-trained judge never learns the level, scale or shape that Brier- and cross-entropy-trained judges recover. For a discrete rating, matching all K indicator means recovers the distribution (as next-token cross-entropy does for a language model), but the single mean rating does so only at K = 2.

**Status:** complete (11-26-2026). Synthetic data and exact arithmetic only; no human ratings, no model calls.

**Relation to the parent folder.** This subfolder extends experiment 35 ([parent README](../README.md)), which holds a smaller package: seeded synthetic checks and nine Lean theorems without Mathlib. The two packages agree where they overlap: the Brier score is strictly proper, scalar squared error returns the mean, and Pearson correlation is unchanged by a positive affine map. The parent's script prints a ranked probability score (RPS) that is not divided by K - 1. Every RPS in this subfolder is the normalized RPS on [0, 1], so do not compare the two RPS scales.

## Question

The question is whether correlation is close enough to squared error, and so to the proper Brier score, to be used front and centre as an evaluation or training objective. This folder checks the math: what correlation is in terms of squared error, whether it is proper, what a judge trained on it learns, and how that explains the binary (K = 2) control of experiment 29.

## Headline results

1. **Correlation is squared error with level and scale removed by an oracle.** Mean squared error equals squared mean difference plus squared standard-deviation difference plus two times the product of the standard deviations times one minus the correlation; the best affine recalibration on the evaluation data leaves the variance times one minus the squared correlation (checks N14, N17; Lean `mse_corr_form`, `standardized_mse`, `oracle_affine_mse`).
2. **Correlation is not proper.** It is invariant to positive affine maps (N15, Lean `corr_affine_invariant`); its maximizers are all positive affine images of the conditional mean (N16); and it is not an average of per-item scores (N18).
3. **Exact counterexamples.** Equal correlation 1, mean expected Brier 0.34 against 0.4696 (N19, Lean `N19_*`). Judge C with correlation 1 against judge D with correlation 0.976: Brier 0.481 for C against 0.405 for D (N20, Lean `N20_*`, figure 3). On six ordinal ratings: mean expected RPS 0.18195 for the correlation-1 judge against 0.12933 for the correlation-0.99926 judge (N21).
4. **Training.** The Pearson gradient has no component along level or scale (N25). On synthetic data with a well-specified linear-logit judge, at N = 100,000 training ratings the held-out mean total variation distance (TVD) to the true distribution is 0.0037 for the cross-entropy (CE)-trained judge, 0.0044 for the Brier-trained judge, and 0.420 to 0.524 for the Pearson-trained judge; held-out affine recalibration repairs its mean but leaves TVD at 0.342 (N22, N24, figure 2).
5. **Consistent with experiment 29.** With two categories, squared error on the rating is half the Brier score, so the squared-error-trained judge should tie with CE, and a recalibration should close most of the Pearson gap because the mean is the whole distribution. That is what experiment 29 reports (DERIVATIONS.md, Part B).
6. **Discrete means.** For a discrete rating the K indicator means are the probabilities, so matching all of them (what cross-entropy and Brier do, and what next-token cross-entropy does for a language model) recovers the distribution. The single mean rating does so only at K = 2 and leaves K - 2 shape directions open at K >= 3: two six-point laws with the same mean 5/2 and the same variance 5/4 differ by 0.40321 nats of excess cross-entropy (DM3). In a 50-token toy, the model trained on the correlation of the token number with the observed number stays at total variation distance (TVD) about 0.70 from the truth at N = 100,000 and at N = 10,000,000, while the cross-entropy-trained model falls from 0.0677 at N = 10,000 to 0.0022 at N = 10,000,000 (DM9a, DM9b). Correlation also leaves level and scale free at K = 2 (TVD 0.0988 against 0.0013 at N = 1,000,000 ratings, repaired to 0.0012 by one affine calibration fitted on held-out ratings, DM6d, DM6e). Two of the 24 checks, DM6a and DM8f, FAIL: they are optimizer-convergence diagnostics of the correlation objective, kept as printed (see the explainer).
7. **Sample size.** The rule of thumb that a per-item histogram of n ratings over K = 6 categories has excess cross-entropy (K - 1)/(2n) nats overstates the measured value at small n: measured over predicted is 0.651, 0.847, 0.964, 0.993 and 0.998 at n = 20, 50, 200, 1,000 and 5,000 (synthetic, Laplace-smoothed histogram, `excess_ce_rule_of_thumb_check.py`).

Compact background: Brier, CE and RPS are strictly proper and their empirical minimizers converge in N (N1 to N8; Lean `brier_strictly_proper`, `rps_strictly_proper`); squared error recovers the mean under skewed, bimodal and heavy-tailed noise without any Gaussian assumption, but not the distribution (N9 to N13; Lean `l2_decomposition`, `l2_unique_minimizer`, `l2_onehot_eq_brier`).

## What was not re-done

Experiment 33 already checks the standardized-score identity, the item-level MSE decomposition, the factorization of correlation through the conditional mean and the logit gradients; experiment 34 covers reliability coefficients. N14 to N16 recheck the first three only as quick sanity lines inside this script.

## Files

- `DERIVATIONS.md`: numbered claims C1 to C13 with statement, proof, plain-English intuition, check id and Lean theorem.
- `check_math.py`: all 25 checks; `results/check_math_output.txt` is its captured output (25 PASS, 0 FAIL), `results/checks.csv` the table, `results/q4_training_tvd_vs_n.csv` the training runs.
- `figures/fig1_propriety_curves`, `fig2_tvd_vs_n`, `fig3_corr_vs_brier_counterexample` (pdf and png).
- `lean/`: Lake project (Lean 4.36.0, Mathlib v4.36.0), `BUILD_LOG.txt` with the build log and `#print axioms` output.
- `discrete_means_EXPLAINER.md`: plain-language explainer with three verdicts (does matching expectations recover a discrete distribution; is a language model just doing correlation; is Brier the same as training on correlation). `discrete_means_check.py` runs 24 checks (22 PASS, 2 FAIL, both optimizer-convergence diagnostics explained in the explainer); `results/discrete_means_output.txt` is its captured output; `figures/discrete_means_fig1_tvd_vs_n` and `discrete_means_fig2_affine_freedom` (pdf and png).
- `excess_ce_rule_of_thumb_check.py` and `results/excess_ce_rule_of_thumb_output.txt`: Monte Carlo check of the (K - 1)/(2n) rule of thumb.
- [`../assets/TRANSCRIPTION_DERIVATION_PAGES_11-26-2026.md`](../assets/TRANSCRIPTION_DERIVATION_PAGES_11-26-2026.md): transcription of two handwritten pages that derive the squared error of standardized scores, 2(1 - correlation), and contrast it with the Brier divergence between two distributions; the photographs are kept beside it. The identity is the Lean theorem `standardized_mse` in `lean/ProperScoring/Basic.lean`; the parent folder's [`DERIVATION.md`](../DERIVATION.md) is a typeset version of the same pages.
- `HOW_TO_READ_RESULTS.md`, `LOG.md`.

Reproduce, from this folder: `python3 -I check_math.py` (about 80 s on 2 CPU cores); `python3 -I discrete_means_check.py` (about 30 s); `python3 -I excess_ce_rule_of_thumb_check.py`; `cd lean && lake exe cache get && lake build && lake env lean Axioms.lean`.

See also: [discrete_means_EXPLAINER.md](discrete_means_EXPLAINER.md) and the [parent README](../README.md).
