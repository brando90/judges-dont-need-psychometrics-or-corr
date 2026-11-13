# 01_synthetic_pearson_vs_kl_squashed_judge_0to5

**Objective:** show that correlation with human scores can barely separate a faithful judge from one whose score distribution is badly off, while KL(P_human ‖ P_judge) separates them clearly.

**How to read the results:** [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md) states which data (train, validation or held-out test), which metric and which comparison every reported number uses.

**Setting:** synthetic data with n=2000 human scores drawn uniformly from {0..5}, seed 0. Judge A = human score + noise in {-1,0,1}, clipped. Judge B = 5 if the human score is ≥3, otherwise 4 (order-preserving, squashed). KL is computed on Laplace-smoothed (alpha=1) marginal histograms. nMSE = mean((h-j)/5)^2.

**Run:** `pip install numpy scipy && python kl_vs_corr.py`

**Results (11-07-2026):**

| Judge | Pearson | Spearman | KL | nMSE |
|---|---|---|---|---|
| A noisy faithful | 0.900 | 0.900 | 0.001 | 0.023 |
| B squashed | 0.877 | 0.878 | 3.420 | 0.216 |

**Limitations:** the data is synthetic and this is a single seed with no confidence intervals. The pooled KL ignores item pairing.
