# How to read the results of experiment 08

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** A synthetic toy with four target rating distributions. Every judge is trained or tuned on n training tasks (3 ratings each) and scored on 500 fresh test tasks against the **true** distribution. Judge names say how each was **trained and tuned**; all are scored with the same test metrics.

## What is compared

One judge family (softmax over the 6 ratings of a linear map of 13 cosine/sine features of x), fitted by L-BFGS-B exactly as in experiment 13, plus a constant baseline. Judges differ in the training objective and in how the ridge strength c is chosen:

| Judge | Training objective | Ridge c | Formulation |
| --- | --- | --- | --- |
| constant | none: predicts the training rating histogram for every test task | none | baseline |
| ce_fixed | cross-entropy (CE) | 1 | ④ |
| ce_valrps | CE | chosen by validation ranked probability score (RPS) | ③ |
| rps_valrps | RPS | chosen by validation RPS | ③-style |
| pearson_fixed | Pearson | 1 | ② |
| pearson_valcorr | Pearson | chosen by validation Pearson | ① |
| pearson_valrps | Pearson | chosen by validation RPS | ② with ③-style selection |

## Which data each number uses

| Data | Role |
| --- | --- |
| Training tasks | n ∈ {4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048}, 3 sampled ratings each. Used to fit, and (by 4-fold cross-validation within them) to choose c from {0.01, 0.1, 1, 10, 100}. |
| Test tasks | 500 fresh tasks per seed, never used for fitting or choosing, scored against the true distribution. |
| Repetition | 100 seeds; scenarios A–D (README). |

## Metrics (all on the test tasks)

| Metric | Definition | Range | Better |
| --- | --- | --- | --- |
| RPS excess | Judge's RPS minus the true distribution's RPS = mean over 5 thresholds of (Q(k) − P(k))² | 0 (perfect) upward | Lower |
| CE excess | KL(true ‖ judge), natural log | 0 upward, nats | Lower |
| Test Pearson | Correlation between the judge's expected rating and the true mean rating across test tasks; undefined (constant judge) counted as 0 | −1 to 1 | Higher |
| Distance from constant | Mean total variation distance between the judge's test forecasts and the constant forecast | 0 to 1 | Descriptive |

## How to read a difference

"Judge a vs judge b" crossovers use **the same test metric for both judges on the same seed's tasks**: the smallest training size from which judge a is significantly better at every larger size. A difference never subtracts two different metrics.

## Uncertainty

95% percentile bootstrap intervals over the 100 seeds (10,000 draws). No p-values were computed. Synthetic toy that works by construction; not evidence about real ratings.

**TLDR-end:** [judge objectives: experiment 08] Same tasks for every judge; names say how each judge was trained and tuned; all scores are test scores against the true distribution.
