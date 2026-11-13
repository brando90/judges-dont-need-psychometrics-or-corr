# How to read the results of experiment 05

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/05_cross_entropy_vs_correlation_information_gap_start_13Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Every number is an exact expectation on known synthetic distributions: no train, validation or test split, no human ratings, no prior exposure.

| Item | Meaning |
| --- | --- |
| Data | Random distributions on scores 0–5 (Dirichlet, seed 0); experiment 04 construction A1 (11 artifacts, p_x = 0.9·Binomial(5, j/12) + 0.1·Uniform); experiment 00's synthetic 2,000 scores. Nothing is fitted or held out. |
| CE | Cross-entropy −Σ P log q in nats, 0 to ∞, lower is better; the true distribution gives its entropy (A1: 1.404). |
| KL / gap G | Kullback–Leibler divergence in nats, ≥ 0, lower is better; G = KL(P ‖ maximum-entropy forecast with P's mean) = excess CE of a mean-only forecast. Reference: log 6 = 1.792 is the largest possible entropy. |
| Pearson | Correlation of forecast means with human means, −1 to 1, higher is better; 1.000 for all three forecasts compared in A1. |
| TVD | 0.5·Σ|P − q|, 0 to 1, lower is better. |
| Differences | Always the same metric on the same data for two forecasts; here excess CE = CE of a forecast minus CE of the true distribution. |
| Uncertainty | None: exact values. The 2,000-draw summaries are descriptive ranges over random distributions, not intervals. |
