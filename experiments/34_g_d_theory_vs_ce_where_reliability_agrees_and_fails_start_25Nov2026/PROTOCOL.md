# Evaluation setup for experiment 34

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026/PROTOCOL.md>

**TLDR:** The pass/fail cuts were written in `preregistered_criteria.json` before the Monte Carlo output was read and were not changed afterward. They share this commit with the results because the experiment was committed once, after the run.

**As of:** 11-25-2026.

## What a match would mean

The measurement is the expected ranked probability score (RPS), smoothed cross-entropy (CE), and total variation distance (TVD) of a forecast under the known conditional distribution. The forecast is fit on training ratings only.

- **Oracle.** The true conditional distribution. It minimizes expected RPS and CE.
- **Intended binomial learner.** The binomial projection, which matches the mean. On binomial items it should approach the oracle. On shaped items it should stay a fixed RPS gap above the oracle.
- **Intended distribution learner.** Add-one CE. It should approach whatever the true six-category distribution is, more slowly than the binomial projection when the truth is binomial.
- **Degenerate forecasts.** The pooled Laplace forecast ignores the item. The lenient point mass is a spike. The spike should have judge-only reliability 1 and should lose to the pooled forecast on RPS if the one-category shift is large relative to the spread of the item means.
- **Literal use of a reliability coefficient.** Maximum Eρ² among a pool that contains a point mass selects the point mass, because a point mass has within-item variance 0. That prediction is a consequence of the definition. The empirical content is whether that choice loses on RPS.

## Cuts written before the Monte Carlo output

From `preregistered_criteria.json`:

- At 256 binomial ratings, the 95% interval for projection minus add-one RPS lies inside (−0.01, 0.01), and each of those learners beats the pooled forecast by more than 0.03.
- At 4 binomial ratings, the interval for projection minus add-one lies entirely below 0.
- At every training size in {4, 16, 64, 256}, the lenient point mass has Eρ² = 1 and its RPS interval versus the pooled forecast lies entirely above 0.
- At 256 ratings on the shaped items, projection minus add-one RPS lies entirely above 0.01.
- At 64 and 256 ratings per rater, the lenient point mass loses to the pooled forecast and the add-one learner on all raters beats it.
- Highest Eρ² selects the lenient point mass in at least 95% of replications. Validation RPS selects it in under 5%, and the selected forecast beats the pooled forecast.
- Among stochastic raters only, the population maximum Eρ² is a neutral rater, and finite-sample selection at 64 and 256 ratings picks a neutral rater at least 95% of the time.
- Sample RPS on 32 test ratings matches the exact expectation at 16 training ratings: the interval for the difference covers 0.

The outcomes are in [results.md](results.md). All 28 cuts passed.

## Seeds and leakage

`SeedSequence([20261125, scenario, n_train, replication, stream])`. Stream 1 is training, stream 2 is validation, stream 3 is the evaluator's test ratings. A fit function receives counts only. The exact risk uses the known distribution, not the test counts. The test-count RPS is reported only as the evaluator check.

**TLDR-end:** [judge objectives: experiment 34 setup] Cuts were fixed before the Monte Carlo was read; the run met all 28.
