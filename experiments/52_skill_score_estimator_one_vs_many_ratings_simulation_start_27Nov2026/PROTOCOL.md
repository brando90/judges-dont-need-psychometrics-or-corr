# Experiment 52 protocol: bias and variance of the skill-score estimator with one rating per item versus several (frozen 11-27-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/52_skill_score_estimator_one_vs_many_ratings_simulation_start_27Nov2026/PROTOCOL.md>

**TLDR:** A synthetic check of an estimator, not an empirical claim. With a known population of items, rating distributions and judges, compare the plug-in skill score 1 - S_hat(judge)/S_hat(base) computed from one rating per item with the same estimator from m ratings per item, at equal items and at equal total ratings. Frozen before the script exists.

**As of:** 11-27-2026 (Friday, Pacific time). CPU, $0.

## Known population (synthetic law)

- K = 5 ordinal classes. Item latent mean mu_i ~ Uniform(0.5, 4.5); human distribution p\*_i = discretized logistic (cut points at 0.5, 1.5, 2.5, 3.5) with scale s_i ~ Uniform(0.4, 1.5) (so items differ in agreement).
- Judges: J1 calibrated-but-noisy (its own mu with N(0, 0.5) error, correct scale), J2 overconfident (correct mu, scale 0.25), J3 shifted (mu + 0.8), J4 = p\* itself (oracle).
- Scores: normalized RPS (primary), multiclass Brier, CE.
- True skill SS = 1 - E[S(q, Y)]/E[S(b, Y)], with b = the population marginal; expectations computed exactly under p\* on 200,000 items.

## Estimators

- E1: one rating per item, n items, base = known marginal.
- E2: one rating per item, base = the marginal estimated from the same n ratings (in-sample plug-in).
- E3: same as E2 but the base for item i is estimated leaving item i out.
- E4: m = 5 ratings per item, n items (per-item mean score first, then over items), base estimated in-sample.
- E5: m = 5 at equal total ratings (n/5 items).
- n in {25, 50, 100, 200, 500, 1000, 5000}; 5,000 replicates per cell; seed SeedSequence([20261127, 52]).
- Reported: bias, SD, RMSE, and the Monte Carlo standard error of the bias.

## Predictions (analytical, checked numerically)

- S1: the numerator and denominator means are unbiased for their expectations in E1, E4 (per-sample proper risk is unbiased with one rating per item).
- S2: the ratio is biased with |bias| = O(1/n), so the bias at n = 25 is at least 10 times the bias at n = 1000 (within Monte Carlo error).
- S3: at equal items, E4 has lower variance than E2; at equal total ratings, E2 (more items, one rating) has lower or equal variance than E5.
- A pytest file checks S1 exactly on a small discrete law (exact enumeration) and S2 by fixed-seed Monte Carlo with a named tolerance.
