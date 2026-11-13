# How to read the results of experiment 02

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Every main number in [RESULTS_comparison.md](RESULTS_comparison.md) is measured on held-out test artifacts that were not used to fit that judge. "CE" and "Pearson" in a column or row name say how a judge was **trained**; both judges are then scored with the **same** metric on the **same** test artifacts.

## What is compared

Two ways of training the same judge: a mixture of the five cached judges' rating histograms, with nonnegative weights that sum to one.

- **CE-trained judge** ("CE fitting"): weights chosen to minimize cross-entropy against the human ratings of the training artifacts.
- **Pearson-trained judge** ("Pearson fitting"): weights chosen to maximize the Pearson correlation between the judge's expected rating and the human mean rating of the training artifacts.
- **Constant baseline:** every artifact gets the rating histogram of the training artifacts' human ratings.

## Which data each number uses

| Data | Role |
| --- | --- |
| Training folds | 4 of 5 task-family folds. Used only to fit the mixture weights. |
| Held-out test fold | The remaining fold's artifacts. Never used to fit that judge. All metric-panel numbers are computed here. |
| Repetition | 5 historical partitions × 5 folds. Each artifact is held out once per partition; out-of-fold predictions for all 75 artifacts are pooled, and the metric is averaged over the 5 partitions. |

The [historical failure diagnosis](RESULTS_comparison.md#historical-failure-diagnosis) table is different: it describes **training** behavior only (for example "Training Pearson"), with no held-out data.

**Prior exposure:** the 75 artifacts, their 195 expert ratings and the historical folds were used in earlier studies. "Held out" means held out from this fit, not a fresh, never-seen test set.

## Metrics (all on held-out test artifacts)

| Metric | Definition | Range | Better |
| --- | --- | --- | --- |
| CE (primary) | Mean over observed human ratings of −ln q(rating) | 0 to ∞, in nats | Lower |
| Ranked probability score (RPS) | Mean over observed human ratings of (1/5) Σ_{k=0}^{4} (Q(k) − 1{rating ≤ k})², with Q the judge's cumulative probabilities | 0 to 1 | Lower |
| TVD | 0.5 Σ_r abs(q(r) − p̂(r)), with p̂ the artifact's human rating histogram | 0 to 1 | Lower |
| Pearson / Spearman / Kendall | Correlation between the judge's expected rating and the human mean rating across test artifacts | −1 to 1 | Higher |
| MSE / MAE / bias | Judge's expected rating vs human mean rating, on a 0–1 scale; bias is judge minus human | — | Lower (bias: closer to 0) |

For scale on test RPS: CE-trained 0.139, Pearson-trained 0.144, constant 0.187.

## How to read a difference

The column "CE − Pearson" means: **the row's metric for the CE-trained judge minus the same metric for the Pearson-trained judge**, both on the held-out test artifacts. It never subtracts two different metrics.

Example: the main result, test CE of the CE-trained judge minus test CE of the Pearson-trained judge = −0.260 nats [−1.458, +0.198]. Negative means the CE-trained judge had lower (better) test CE; the interval includes 0, so no difference is established. On test RPS the same subtraction gives −0.0046 [−0.0356, +0.0100].

## Uncertainty

95% percentile intervals from 1,000 bootstrap draws over task families (all artifacts of a family move together), refitting both judges in each draw; 998 draws passed the numerical certificates. No p-values were computed.

**TLDR-end:** [judge objectives: experiment 02] All panel numbers are held-out test metrics; "CE − Pearson" subtracts one test metric between the CE-trained and the Pearson-trained judge.
