# Experiment 33: checks of the moment derivation (what correlation and cross-entropy identify), 11-25-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/33_moment_derivation_checks_what_correlation_and_ce_identify_start_25Nov2026/README.md>

**TLDR:** One CPU script, `verify_moment_derivation.py`, checks every identity and bound of Appendix B of the 11-25-2026 technical memo on seeded random instances with exact finite sums: Pearson factors through the conditional mean, the cross-entropy (CE) gap bounds every bounded moment, the first K-1 moments determine a distribution, and the logit gradients of CE, Brier, ranked probability score (RPS), mean squared error (MSE) and Pearson are as stated. It samples no human rating, fits no judge and calls no model.

**Status:** complete (11-25-2026). The memo's derivation appendix quotes the printed output.

## Question

The memo's thesis is that a correlation objective constrains one conditional moment of the human rating distribution while CE constrains all of them. This folder verifies the algebra behind that statement numerically, so the appendix does not rest on hand derivation alone.

## What the script checks (K = 6 ratings, 0 to 5, seed 20261125)

| Memo subsection | Statement | Instances |
| --- | --- | --- |
| B.2 | `Corr(m(X), R) = Corr(m(X), mu(X)) * sqrt(Var mu / Var R)`; law of total variance; the affine ceiling; no random forecast beats it | 200 random 7-item populations, 300 random forecasts each |
| B.3 | Pinsker, `abs(E_p f - E_q f) <= width(f) * TV`, `abs(Var_p - Var_q) <= (K-1)^2 * TV`, and the population bounds with CE gap | 20,000 random pairs; 5,000 random 9-item populations |
| B.4 | Vandermonde determinant (exact integer, 34,560 for K = 6), conditioning, free directions left by matching moments | exact; 200 random distributions |
| B.5 | Exponential-family CE gradient, calibration-head gradients, logit gradients of CE, Brier, MSE, RPS, Pearson (finite differences); polynomial features reach every positive distribution | 100 to 200 random instances each |
| B.6 | Pearson as squared error on standardized scores; MSE decomposition | 200 random vector pairs |
| B.7 | Exact break-even prior for selecting a spread from one rating (rational arithmetic, `fractions`) | 5 mean offsets |

## Run

```bash
python3 -I experiments/33_moment_derivation_checks_what_correlation_and_ce_identify_start_25Nov2026/verify_moment_derivation.py
```

Needs only numpy. It runs in seconds and writes `results/verify_moment_derivation_output.txt` (the printed lines) and `results/verify_moment_derivation.json` (the numbers). See `HOW_TO_READ_RESULTS.md` for what each line means.

## Limits

The instances are Dirichlet draws, so the script tests identities and inequalities, not behavior on real ratings. The statements concern population optima and fixed forecasts; they do not predict how a trained network with shared parameters behaves, which experiments 08, 24 to 29 and 31 measure.
