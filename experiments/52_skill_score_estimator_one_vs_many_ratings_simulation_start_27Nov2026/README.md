# Experiment 52: bias and variance of the skill-score estimator with one rating per item versus several

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/52_skill_score_estimator_one_vs_many_ratings_simulation_start_27Nov2026/README.md>

**TLDR:** A synthetic check of an estimator, not an empirical claim. With one rating per item, the plug-in skill score is essentially unbiased when the base rate is known (for judge J1 the bias is within 0.002 of the truth at every n from 25 to 5,000; for all four judges it is within 3 Monte Carlo standard errors), and its bias is of order 1/n when the base rate is estimated from the same ratings (-0.034 at n = 25, -0.010 at 100, -0.0006 at 1,000; normalized RPS, judge J1). The bigger problem is variance: the SD is 0.13 at n = 25 and 0.029 at n = 500, for a true skill of 0.20. At a fixed total number of ratings, one rating on each of many items beats five ratings on a fifth as many items (SD 0.029 vs 0.035 at 500 ratings), as predicted. All four preregistered predictions (S1 to S3, plus the pytest check) hold.

**As of:** 11-27-2026 (Friday, Pacific time). CPU, \$0. Protocol frozen first: [PROTOCOL.md](PROTOCOL.md), commit 757d462.

## Setup (from the protocol)

K = 5 ordinal classes; 200,000 synthetic items with a discretized-logistic human distribution $p^{*}_i$ (latent mean uniform on 0.5 to 4.5, scale uniform on 0.4 to 1.5). Four judges: J1 noisy but calibrated in scale, J2 overconfident, J3 shifted by +0.8, J4 the oracle $p^{*}$. The skill score and its true value:

```math
\mathrm{SS} = 1 - \frac{\mathbb{E}\,S(q, Y)}{\mathbb{E}\,S(b, Y)}, \qquad b = \text{population marginal}, \quad Y \sim p^{*}_i .
```

Estimators: E1 one rating per item, known base; E2 one rating, base estimated from the same n ratings (additive 0.5 per class); E3 one rating, leave-one-item-out base; E4 five ratings on the same n items; E5 five ratings on n/5 items (same total ratings as E2). 5,000 replicates per cell, seed `SeedSequence([20261127, 52])`.

## Results (normalized RPS, judge J1, true SS = 0.2017; bias / SD over 5,000 replicates)

| estimator | n = 25 | n = 100 | n = 500 | n = 1000 | n = 5000 |
|---|---|---|---|---|---|
| E1 one rating, known base | -0.0010 / 0.128 | -0.0020 / 0.064 | -0.0001 / 0.029 | +0.0002 / 0.020 | -0.0001 / 0.009 |
| E2 one rating, in-sample base | -0.0343 / 0.139 | -0.0101 / 0.065 | -0.0017 / 0.029 | -0.0006 / 0.020 | -0.0003 / 0.009 |
| E3 one rating, leave-one-out base | +0.0251 / 0.129 | +0.0056 / 0.064 | +0.0015 / 0.029 | +0.0010 / 0.020 | +0.0000 / 0.009 |
| E4 five ratings, same n items | -0.0130 / 0.070 | -0.0044 / 0.034 | -0.0007 / 0.015 | -0.0004 / 0.011 | -0.0001 / 0.005 |
| E5 five ratings, n/5 items | -0.0736 / 0.172 | -0.0182 / 0.080 | -0.0029 / 0.035 | -0.0016 / 0.024 | -0.0006 / 0.011 |

Monte Carlo SE of each bias = SD/sqrt(5000), for example 0.0018 at n = 25 for E1. Every judge and the Brier and CE versions are in [results/skill_estimator_tables.md](results/skill_estimator_tables.md); the pattern is the same for all four judges and all three scores.

- **S1 (unbiased numerators): supported.** The one-rating mean score is within 0.001 of its expectation at every n (row `numerator_one_rating` in the JSON), and `tests/test_skill_estimator.py` checks it exactly with fractions.
- **S2 (ratio bias of order 1/n): supported.** E2's bias is -0.0343 at n = 25 and -0.0006 at n = 1000, a ratio of about 57 (the rule asked for at least 10). A pytest case also checks that a ratio of two unbiased means is biased when the base score varies.
- **S3 (variance): supported.** At equal items, five ratings per item halves the SD (E4 0.070 vs E2 0.139 at n = 25). At an equal total number of ratings, one rating on many items is better (E2 0.029 vs E5 0.035 at 500 ratings; 0.065 vs 0.080 at 100).
- Side observation, synthetic only: the overconfident judge J2 has positive skill under normalized RPS (+0.094) but negative skill under Brier (-0.123) and CE (-0.535). Which score matches the label encoding changes the verdict.

## What this means for the paper

- The skill score is safe to report with one rating per item, but it needs a few hundred test items before it is informative (SD about 0.03 at 500 items). A judge's skill estimated on 25 to 100 items should not be ranked without an interval.
- Prefer a base rate estimated out-of-sample (other folds) or a known marginal. An in-sample base biases the skill downward by about 0.01 at n = 100.

**Proposed paper text (not inserted; another worker owns the appendix):** "With one rating per item the per-sample proper risk is unbiased, and the skill score inherits only a ratio bias of order $1/n$ (in simulation: below 0.002 at 500 items with an out-of-fold base rate). Its standard deviation is the binding constraint (about 0.03 at 500 items), and at a fixed rating budget, spreading single ratings over more items gives a lower variance than repeated ratings on fewer items."

## Reproduce

```
python scripts/skill_estimator_sim.py results --reps 5000      # about 1 minute on one CPU core
python scripts/summarize.py results figures                     # tables; figure kept on the box
python -m pytest -q tests/test_skill_estimator.py               # 4 passed, twice (results/pytest_run1.txt, pytest_run2.txt)
```

The figure `figures/skill_estimator_bias_sd.png` (bias and SD against n, `ax.grid(True)`) is not committed because the push tool only carries text. It is on the box at `/workspace/calib_followups/figures/e52/`, and `scripts/summarize.py` rebuilds it.
