# How to read the results of experiment 06

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** A synthetic toy. Six judges from one family are trained on the same n training tasks (3 ratings each) with six objectives, then scored on the same 500 fresh test tasks against the **true** rating distribution. Objective names say how a judge was **trained**; every judge is scored with the same test metrics.

## What is compared

One judge family, q_W(r | x) = softmax(Wᵀφ(x))_r with 13 cosine/sine features, trained six ways:

- **CE-trained (④), RPS-trained, Pearson-trained (②):** experiment 13's saved fits, reused unchanged.
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
| Ranked probability score (RPS) excess | Judge's RPS minus the true distribution's RPS = mean over 5 thresholds of (Q(k) − P(k))² | 0 (perfect) upward | Lower |
| Cross-entropy (CE) excess | KL(true ‖ judge), natural log | 0 (perfect) upward, nats | Lower |
| Spearman | Rank correlation between the judge's expected rating and the **true** mean rating across test tasks | −1 to 1 | Higher |

## How to read a difference

"Objective minus CE-trained" is **the test excess of the judge trained with that objective minus the test excess of the CE-trained judge, on the same seed's tasks.** Positive means the CE-trained judge did better. It never subtracts two different metrics.

## How the judges were fitted

The three new objectives use exactly experiment 13's optimizer (L-BFGS-B, at most 400 iterations and 600 evaluations), ridge (1/n)‖W‖² and starting point, through smooth surrogates with temperature 0.1. They are not convex, so these results reflect this optimizer, start and budget. Convergence within the budget was not recorded.

## Uncertainty

95% percentile bootstrap intervals over the 100 seeds (10,000 draws). No p-values were computed. This is a toy that works by construction; it is not evidence about real ratings.

**TLDR-end:** [judge objectives: experiment 06] Every number is a test-task score against the true distribution; objective names say how each judge was trained.
