# Experiment 34: G-theory and D-theory versus cross-entropy, where they agree and where they fail

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026/README.md>

**TLDR:** A CPU simulation with known ordinal distributions. Cross-entropy and a mean-matched binomial projection agree when the conditional distribution is binomial. A lenient point mass with generalizability coefficient 1 loses to a constant forecast on ranked probability score (RPS), cross-entropy (CE), and total variation distance (TVD). A same-mean, same-variance shape change is invisible to both reliability coefficients and visible to CE. Results: [results.md](results.md).

**As of:** 11-25-2026. Started the same day. Local CPU, $0, no model calls.

## Question

Is generalizability theory (G-theory) or a decision study (D-theory) needed to learn a conditional human-rating distribution once the learner minimizes cross-entropy or another proper score?

The conjecture under test: cross-entropy learns every free probability of the conditional distribution. G-theory and D-theory decompose the mean and the variance into a generalizability coefficient Eρ² and a dependability coefficient Φ.

Experiment [04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026](../04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/README.md) already showed that matching the mean and variance does not identify the distribution, and that variance components can still choose a rater-versus-occasion design. This experiment trains the forecasts and asks when the coefficients and the proper scores rank them the same way.

## Evaluation setup

Four settings, all on ratings in {0, 1, 2, 3, 4, 5}:

1. Twelve binomial items. The mean identifies the distribution.
2. The same items, plus a point mass placed one category above the training mean.
3. Eight items whose truth matches a binomial mean and variance and differs in higher moments. The maximum RPS gap on the same-moment polytope is found by enumerating all 20 vertices.
4. Six raters on the twelve items: four neutral, one clipped +2, one clipped −2. The target is the mixture.

Learners, all closed form: add-one CE, binomial projection, pooled Laplace, and the lenient point mass. Selection at 16 ratings compares maximum Eρ² with minimum validation RPS. 400 replications, seed 20261125. Cuts are in [preregistered_criteria.json](preregistered_criteria.json).

## Result

At 256 binomial ratings per item, the projection minus add-one RPS is −0.000187 [−0.000194, −0.000181], inside ±0.01, and both beat the constant forecast by about 0.075. At 16 ratings the lenient point mass has Eρ² = 1, Pearson 0.961, and RPS 0.0492 [0.0477, 0.0506] worse than the constant forecast. At 256 ratings on the shaped items the same-moment projection remains 0.0205 [0.0205, 0.0205] RPS worse than add-one. Full tables: [results.md](results.md). Reading guide: [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md). Derivation: [derivation/g_d_vs_ce_derivation.pdf](derivation/g_d_vs_ce_derivation.pdf).

```bash
python3 experiments/34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026/run_experiment.py
```

**TLDR-end:** [judge objectives: experiment 34] Reliability agrees with cross-entropy for a binomial and for ranking noisy raters; a perfectly repeatable lenient point mass loses to a constant forecast, and a same-moment shape change does not move Eρ² or Φ.
