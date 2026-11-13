# How to read the results of experiment 13

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** A synthetic toy. Judges are trained on n sampled training tasks, then every reported number is computed on 500 fresh test tasks against the **true** rating distribution, not sampled ratings. "CE fit" (cross-entropy), "RPS fit" (ranked probability score) and "Pearson fit" name how a judge was **trained**; all are scored with the same test metrics. (This guide was added on 11-13-2026; `results.md` is a frozen copy and is unchanged.)

## What is compared

One judge family (softmax of a linear map of 13 cosine/sine features of the task) trained four ways on the same training tasks:

- **CE fit (④):** minimize cross-entropy against the sampled training ratings.
- **RPS fit:** minimize RPS against the sampled training ratings.
- **Pearson fit (②, raw):** maximize Pearson between the judge's expected rating and the training tasks' human mean rating.
- **Pearson fit + calibrated head:** Pearson-trained score, then an ordinal head fitted by CE on that score.

## Which data each number uses

| Data | Role |
| --- | --- |
| Training tasks | n = 4 to 2,048 tasks, 3 sampled ratings each. Used only to train. |
| Test tasks | 500 fresh tasks per seed, never used in training. Scored against the true rating distribution, so there is no rating noise in the test score. |
| Repetition | 100 seeds per n; training and test tasks are redrawn each seed. |

## Metrics (all on the test tasks)

| Metric | Definition | Range | Better |
| --- | --- | --- | --- |
| RPS excess | Judge's RPS minus the true distribution's RPS = mean over 5 thresholds of (Q(k) − P(k))², Q judge and P true cumulative probabilities | 0 (perfect) upward | Lower |
| CE excess | KL(true ‖ judge), natural log | 0 (perfect) upward, nats | Lower |
| TVD | 0.5 Σ_r abs(q(r) − p(r)) against the true distribution | 0 to 1 | Lower |
| Spearman / Pearson | Correlation between the judge's expected rating and the **true** mean rating across test tasks | −1 to 1 | Higher |

The power-law fits and "tasks needed" counts are fitted to the seed-mean test excess as a function of n. They are not bounds.

## Added figure and crossover (11-13-2026)

[fig_excess_vs_n_without_head.png](fig_excess_vs_n_without_head.png) redraws the learning curves without the Pearson + calibrated head judge, with y-axes that name the metric (held-out test RPS excess; held-out test CE excess = KL(true ‖ judge) in nats). [scripts/plot_without_head_and_crossover.py](scripts/plot_without_head_and_crossover.py) makes it from the saved `results/raw.jsonl`, with no new fits, and prints paired per-seed differences (Pearson-trained minus CE-trained) with 95% bootstrap intervals over seeds. The CE-trained judge is significantly better on both test metrics at every n from 64 training tasks (192 ratings) in scenario A, and from 4 tasks (12 ratings) in scenario B. In scenario A the raw Pearson-trained judge is significantly better on test RPS up to 32 tasks (96 ratings).

## Uncertainty

95% bootstrap intervals over the 100 seeds. No p-values were computed. This is a toy that works by construction; it is not evidence about real ratings.

**TLDR-end:** [judge objectives: experiment 13] Every number is a test-task score against the true distribution; fit names say how each judge was trained.
