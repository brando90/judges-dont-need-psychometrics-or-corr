# How to read the results of experiment 08 (QA annotation 11-26-2026)

**QA correction (11-26-2026):** Current verified corrections, evidence limits and score-first companion panels are in [QA_CORRECTIONS_11-26-2026.md](QA_CORRECTIONS_11-26-2026.md). Original wording and scored tables below are retained as the historical record.

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
| RPS excess | Judge's RPS minus the true distribution's RPS = mean over 5 thresholds of (Q(k) − P(k))² | 0 to 1 (0 is perfect) | Lower |
| CE excess | Kullback–Leibler divergence KL(true ‖ judge), natural log | 0 upward, nats | Lower |
| Test Pearson | Correlation between the judge's expected rating and the true mean rating across test tasks; undefined (constant judge) counted as 0 | −1 to 1 | Higher |
| Distance from constant | Mean total variation distance between the judge's test forecasts and the constant forecast | 0 to 1 | Descriptive |

The original report counts undefined test correlations as zero; that is a reporting convention, not a defined Pearson value. The raw file retains these undefined values: all 4,400 constant-forecast rows, plus one row (C, n = 4, seed 9) for each of the three Pearson-trained variants. Thus that cell has 99 defined correlations and one substituted zero per Pearson variant; all distribution-loss values remain defined. The constant is omitted from the correlation plot.

## How the judges were fitted

Each fit uses at most 400 L-BFGS-B iterations and 600 function evaluations, starting from the log training histogram (bins floored at 0.001) plus seeded Gaussian weight noise with standard deviation 0.01. The penalty is (c/n)‖W‖². Four-fold cross-validation pools predictions from all held-out training folds; ties within 1e-12 choose the larger c. When every validation correlation is undefined, c = 1 is used and the row is flagged (one row: C, n = 4, seed 9). No global-optimum certificate was recorded.

The scenario descriptions use κ as a concentration parameter. Fixed κ in C and D still allows rating variance 5μ(1−μ)(κ+5)/(κ+1) to change with μ. A has a small nonzero mean trend, not literally “disagreement only”.

## How to read a difference

"Judge a vs judge b" crossovers use **the same test metric for both judges on the same seed's tasks**: the smallest training size from which judge a is significantly better at every larger size. A difference never subtracts two different metrics.

## Uncertainty

95% percentile bootstrap intervals over the 100 seeds (10,000 draws). No p-values were computed. Evidence is limited to the stated synthetic family; it does not establish performance on real ratings.

## Numerical QA note (11-16-2026)

The inherited `experiments/archive/13_*/scripts/toy.py::ce_loss_grad` clips and renormalizes probabilities before evaluating CE, but returns a gradient that is inconsistent when clipping is active. This is a real implementation defect, not a failure of CE propriety. A bounded check encountered it in experiment 08's validation search (C, n = 4, seed 9, c = 0.1); replacing it in memory with stable log-softmax CE left that row's selected metrics unchanged to numerical precision. Impact checked 11-16-2026: refitting every CE fit of experiments 07, 08 and 13 with an exact CE (log-sum-exp value, exact gradient, no floor) preserved the reported cell means at printed precision, crossovers and hypothesis checks, with the ridge-share exception below. Fixed-ridge CE metrics agree with the saved rows within 1e-9. Experiment 07's calibrated-head metrics also agree with saved rows within 1e-9; experiment 13's finite-sample heads agree within 1e-9 only between exact and frozen CE reruns on this machine. Its saved Pearson fits and heads came from a different machine and do not reproduce to that tolerance. In experiment 08, 338 of the 4,400 validation-tuned CE judges (337 of them at the weakest ridge, c = 0.01, where clipping is encountered) move by at most 0.00023 in test RPS excess, and one ridge choice changes (C, n = 256, seed 7: 0.01 to 0.1); no cell mean at the printed precision, crossover or hypothesis check changes, and one chosen-ridge share moves from 69% to 68%. All 95,291 exact-CE refits of experiment 08 converged (scipy's flag). Measurement sources and saved rows are unchanged. Evidence: [results/ce_gradient_check.json](results/ce_gradient_check.json), from [scripts/check_ce_gradient.py](scripts/check_ce_gradient.py). See the [stage-1 report](../../paper_latex_and_notes/notes/mega_qa_11-16-2026/qa1_codex.md).

The bootstrap intervals are pointwise and crossovers range only over the tested training sizes, not every possible sample size. They do not provide simultaneous 95% coverage across all comparisons. Scenarios A and B reuse earlier synthetic tasks; test tasks are excluded from each fit and validation selection.

**TLDR-end:** [judge objectives: experiment 08] Same tasks for every judge; names say how each judge was trained and tuned; all scores are test scores against the true distribution.
