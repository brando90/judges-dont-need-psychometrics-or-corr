# How to read the results of experiment 01

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** A synthetic study with a known true rating distribution. Each criterion **chooses** one of nine fixed candidate forecasts using validation ratings; the chosen candidate is then scored exactly against the true distribution on separate test items. Test ratings never enter fitting or selection.

## What is compared

Seven selection criteria: CE, Brier, RPS, Pearson, Spearman, Kendall and normalized mean squared error (nMSE). Each picks one of nine fixed candidate forecasts: the true distribution, four wrong mixtures with the same mean, a permuted control and three constants. Only the empirical constant is fitted, on training ratings.

## Which data each number uses

| Data | Role |
| --- | --- |
| Training items | 96 items. Used only to fit the empirical constant candidate. |
| Validation items | 48 items with 1, 2 or 3 sampled ratings each. Each criterion chooses a candidate here. |
| Test items | 48 items from separate parameter-band families. Scored exactly against the known true distribution, with no sampled test ratings. |
| Repetition | 256 independent draws of training and validation ratings. |

## The two kinds of tables

- **"Exact population test metrics":** each candidate's exact expected metric on the test items. No sampling and no selection are involved.
- **"Selection with sparse ratings":** "Oracle in selected set /256" counts how often the chosen set contains the true distribution. "Expected test CE" is the exact test CE of the chosen candidate, with ties broken uniformly at random, averaged over the 256 draws.

## Metrics (test items, against the true distribution)

| Metric | Definition | Range | Better |
| --- | --- | --- | --- |
| CE | Expected −ln q(rating) under the true distribution | 0 to ∞, nats | Lower |
| KL | CE minus the true distribution's entropy | 0 upward, nats | Lower |
| Brier | Σ over the six ratings of (q(r) − 1{rating = r})², expected under the truth | 0 to 2 | Lower |
| RPS | Expected (1/5) Σ_{k=0}^{4} (Q(k) − 1{rating ≤ k})² | 0 to 1 | Lower |
| nMSE / MAE | Forecast mean vs a future individual rating, on a 0–1 scale | 0 upward | Lower |
| Pearson / Spearman / Kendall | Correlation with the exact conditional mean (or a future rating, where stated) | −1 to 1 | Higher |

## Uncertainty

Student-t 95% Monte Carlo intervals over the 256 rating draws. Exact population metrics have no sampling interval. No p-values were computed.

**TLDR-end:** [judge objectives: experiment 01] Criteria choose on validation ratings; reported scores are exact test-item metrics against the true distribution.
