# How to read the results

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/38_cka_opd_invariance_ladder_start_26Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Every proper-score number is a held-out test risk. Linear CKA and normalized OPD in the transform table compare the test logit matrix with the test human histogram. The collapse table uses one row per test rating.

## Data

Prompt groups are split 60/20/20 with seed 0. A prompt's ratings all stay in one split.

| Set | What a prompt is | Items | Ratings | Test prompts |
| --- | --- | ---: | ---: | ---: |
| SummEval expert, one aspect | one summary, grades 1–5 stored as classes 0–4 | 1,600 | 4,800 | 320 |
| ChaosNLI SNLI | one premise–hypothesis pair, 100 votes, 3 classes | 1,514 | 151,400 | 304 |
| ChaosNLI MNLI-m | same, MNLI-matched | 1,599 | 159,900 | 321 |
| ChaosNLI alphaNLI | one abductive item, 2 options | 1,532 | 153,200 | 307 |
| DICES-990 | one dialogue item, `Q_overall`, quality filter applied | 990 | 64,886 | 198 |
| VeriBench FSC | one formalization item, grades 0–5 | 100 | 250 | 20 |

SummEval's split group is the source article, so all 16 systems of one article stay together; the four aspects are fit separately on that article split. ChaosNLI and DICES use the author item id. VeriBench uses `item_id`.

These public sets were already used in earlier experiments in this repository. This run refits a new head and does not reuse those weights. VeriBench scores were not in this repository; they are the native-main `final_score_0_5` column from trustworthy-theorem-judges experiment 09 at commit `460466dd10b735c5ff25f9e9cb21f0441fcf960b`. No file there is named `R_v1`. On DICES and VeriBench the same person can rate items that fall in different splits; no item's own ratings are split.

The collapse table is still the held-out test prompts. Its rows are the individual test ratings, not a second split.

## Metrics

Linear CKA is `||Xcᵀ Yc||_F² / (||Xcᵀ Xc||_F ||Ycᵀ Yc||_F)` after column centering. Higher means the centered columns are more aligned. On one column it equals squared Pearson correlation, which lies in `[0, 1]`.

Normalized OPD is `sqrt(max(0, 2 - 2 ||Xhatᵀ Yhat||_*))` after centering and scaling each matrix to Frobenius norm 1. Lower means a closer match after a rotation. Identical centered shapes have OPD 0. Two orthogonal shapes have OPD `sqrt(2) ≈ 1.414`.

CCA sum of squares is the sum of squared canonical correlations of the centered columns. It is unchanged by an invertible remix of either side. Higher means more shared linear dimensions.

Pearson of the expected rating is the ordinary Pearson correlation between the judge's mean class value and the mean human rating on the same test prompts. It lies in `[-1, 1]`. Higher means a stronger positive linear association.

Cross-entropy is `-log q(y)` in nats. Brier is `||q - e_y||²`, range `[0, 2]`. Ranked probability score (RPS) is the sum of squared gaps between the forecast CDF and the outcome CDF, range `[0, K-1]`. sqrt(RPS) is the square root of the unbiased mean RPS, not the mean of the square roots. Lower is better for all three. A reference: the uniform forecast on `K` classes has cross-entropy `log K` (about 1.609 for 5 classes, 1.099 for 3, 0.693 for 2).

The reported proper-score risk is the mean over test prompts of the mean loss inside the prompt. It is not a plug-in divergence against the test histogram.

## Differences

In `figures/ce_gap_heldout.png` the vertical axis is held-out test cross-entropy of the transformed logits minus held-out test cross-entropy of the identity logits. Positive means the transformed judge is worse.

In `figures/cka_abs_change_heldout.png` the vertical axis is the absolute change in held-out test linear CKA. Exact zeros are drawn at the display floor `1e-18`.

In `figures/collapse_cka_increase_heldout.png` the vertical axis is held-out linear CKA after deleting the orthogonal residual, minus held-out linear CKA with that residual kept. Positive means the collapsed scalar looks better to linear CKA.

## Uncertainty

Proper-score intervals are item-clustered percentile 95% intervals. The cluster is the test prompt. `B = 1000` draws, seed `20261126`. Linear CKA and OPD on a fixed matrix are deterministic; their "uncertainty" under sign flips and rotations is floating-point roundoff, not a confidence interval.

## How the judge was chosen

Search space: softmax linear head on 24 principal components of a 256-dimensional signed token hash, plus an intercept. Objective: mean cross-entropy against the train-item histogram, plus `0.5 * 0.01 * ||non-intercept weights||²`. Optimizer: L-BFGS-B from the zero vector, at most 40 iterations. Cross-entropy in the logits is convex; the iteration cap means the printed head is that run, not a certificate of the global minimizer past the cap. Temperature is chosen on validation cross-entropy from `{0.5, 0.75, 1, 1.25, 1.5, 2, 3}`. Test prompts are not used for the weights or the temperature.
