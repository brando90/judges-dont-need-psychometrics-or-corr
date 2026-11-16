# How to read the results of experiment 06

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** A synthetic toy. Six judges from one family are trained on the same n training tasks (3 ratings each) with six objectives, then scored on the same 500 fresh test tasks against the **true** rating distribution. Objective names say how a judge was **trained**; every judge is scored with the same test metrics.

## What is compared

One judge family, q_W(r | x) = softmax(Wᵀφ(x))_r with 13 cosine/sine features, trained six ways:

- **Cross-entropy (CE)-trained (④), ranked probability score (RPS)-trained, Pearson-trained (②):** experiment 13's saved fits, reused unchanged.
- **Spearman-trained, Kendall-trained, ½ Pearson + ½ Spearman-trained:** new here. Training uses smooth surrogates of the rank correlations (temperature 0.1 rating points), defined in the [README](README.md).

## Which data each number uses

| Data | Role |
| --- | --- |
| Training tasks | n = 4 to 2,048 tasks, 3 sampled ratings each. Used only to train. Identical to experiment 13's training tasks for each (scenario, n, seed). |
| Test tasks | 500 fresh tasks per seed, never used in training, scored against the true distribution. Identical to experiment 13's. |
| Repetition | 100 seeds per n. |

## Metrics (all on the test tasks)

| Metric | Definition | Range | Better |
| --- | --- | --- | --- |
| Ranked probability score (RPS) excess | Judge's RPS minus the true distribution's RPS = mean over 5 thresholds of (Q(k) − P(k))² | 0 to 1 (0 is perfect) | Lower |
| Cross-entropy (CE) excess | Kullback–Leibler divergence KL(true ‖ judge), natural log | 0 upward, nats | Lower |
| Pearson / Spearman | Linear / rank correlation between the judge's expected rating and the **true** mean rating across test tasks | −1 to 1 | Higher |

## How to read a difference

"Objective minus CE-trained" is **the test excess of the judge trained with that objective minus the test excess of the CE-trained judge, on the same seed's tasks.** Positive means the CE-trained judge did better. It never subtracts two different metrics.

## How the judges were fitted

The three new objectives use exactly experiment 13's optimizer (L-BFGS-B, at most 400 iterations and 600 evaluations), ridge (1/n)‖W‖² and starting point (log training histogram plus seeded Gaussian weight noise of standard deviation 0.01), through smooth surrogates with temperature 0.1. They are not convex, so these results reflect this optimizer, start and budget. Convergence within the budget was not recorded.

## Uncertainty

95% percentile bootstrap intervals over the 100 seeds (10,000 draws). No p-values were computed. This is evidence about the stated synthetic family, not real ratings.

## Numerical QA note (11-16-2026)

The inherited `experiments/archive/13_*/scripts/toy.py::ce_loss_grad` clips and renormalizes probabilities before evaluating CE, but returns a gradient that is inconsistent when clipping is active. This is a real implementation defect, not a failure of CE propriety. A bounded check encountered it in experiment 08's validation search (C, n = 4, seed 9, c = 0.1); replacing it in memory with stable log-softmax CE left that row's selected metrics unchanged to numerical precision. Impact checked 11-16-2026: refitting every CE fit of experiments 07, 08 and 13 with an exact CE (log-sum-exp value, exact gradient, no floor) preserved the reported cell means at printed precision, crossovers and hypothesis checks, with the ridge-share exception below. Fixed-ridge CE metrics agree with the saved rows within 1e-9. Experiment 07's calibrated-head metrics also agree with saved rows within 1e-9; experiment 13's finite-sample heads agree within 1e-9 only between exact and frozen CE reruns on this machine. Its saved Pearson fits and heads came from a different machine and do not reproduce to that tolerance. In experiment 08, 338 of the 4,400 validation-tuned CE judges (337 of them at the weakest ridge, c = 0.01, where clipping is encountered) move by at most 0.00023 in test RPS excess, and one ridge choice changes (C, n = 256, seed 7: 0.01 to 0.1); no cell mean at the printed precision, crossover or hypothesis check changes, and one chosen-ridge share moves from 69% to 68%. All 95,291 exact-CE refits of experiment 08 converged (scipy's flag). Measurement sources and saved rows are unchanged. Evidence: [../08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/results/ce_gradient_check.json](../08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/results/ce_gradient_check.json), from [../08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/check_ce_gradient.py](../08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/check_ce_gradient.py). See the [stage-1 report](../../paper_latex_and_notes/notes/mega_qa_11-16-2026/qa1_codex.md).

The bootstrap intervals are pointwise and crossovers range only over the tested training sizes, not every possible sample size. They do not provide simultaneous 95% coverage across all comparisons. Scenarios A and B reuse earlier synthetic tasks; test tasks are excluded from each fit and validation selection.

**TLDR-end:** [judge objectives: experiment 06] Every number is a test-task score against the true distribution; objective names say how each judge was trained.
