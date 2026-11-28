# Experiment 60 PROTOCOL: sampled-score judge versus conditional-mean judge on real rating distributions (oracle check, Figure 1 data)

Frozen 11-28-2026 (Pacific), in a commit that precedes every analysis run of this experiment. CPU only, no model calls, API spend $0.

## Purpose and label
This is an ORACLE CHECK of a theorem on empirical human rating laws. No language model is involved and nothing here is evidence about
real LLM judges. The theorem (Lean-checked for finite laws: `PlanAudit.Model.anti_proper` in
`experiments/35_*/extended_checks_11-26-2026/gemini_plan_claims_audit_11-26-2026/PlanClaims.lean`, port `Sampled.Model.anti_proper` in
`docs/notes/point2_support_27Nov2026/lean/Point2/SampledReadout.lean`; Lean 4.36.0, Mathlib fc4f59e5, zero `sorry`, standard axioms):
for a judge whose score S is drawn from p*(.|c) independently of the human rating Y given the item C, Pearson r(S, Y) = rho, where rho =
Var(mu(C))/Var(Y) and mu(c) = E[Y|c]; the judge that outputs mu(C) has r = sqrt(rho) > rho whenever 0 < rho < 1 (the conditions: independence
given the item, positive between-item variance of mu, positive within-item variance). Strictly proper scores rank the truthful
forecast ahead of the point-mass forecast at mu when mu is on the grid. This experiment shows the two correlations and the proper scores
on the empirical rating laws of three real sets, with the finite-rater noise handled by a split.

## What the analyst had seen before freezing
The theorem and its Lean statements (above), the Experiment 35/36 worked examples, and aggregate results of Experiments 46, 55, 56.
NOT seen: any correlation of a sampled or mean oracle with held-out ratings on these sets, or any rho estimate from them.

## Data (read-only)
Individual human ratings from `experiments/39_*/data/processed/study_a_bundle.npz` (loaded through `experiments/44_*/src/lib44.py`):
alphaNLI (400 items, 100 ratings each, binary), DICES-350 (350 items, 123 each, ordered No < Unsure < Yes coded 0, 1, 2), DICES-990
(400 items, 69 to 76 each, same coding). ChaosNLI SNLI and MNLI-m are EXCLUDED: their labels are unordered, so a Pearson correlation
with an arbitrary numeric coding is not meaningful.

## Design
For each set and each of R = 200 random splits (seed `SeedSequence([20261128, 60, set_index, split])`), the ratings of every item are
divided at random into half A (floor(n_i/2) ratings, builds the oracle judges) and half B (the rest, the held-out human ratings;
no rating is in both halves). Observations are (item, rating) pairs of half B.
- **Truthful (sampled) judge:** for every observation draws a score from the item's half-A empirical distribution p_A(i), independently.
- **Mean judge:** outputs mu_A(i), the half-A mean rating of the item.
- **Correlations:** Pearson r between a judge's output and the half-B ratings over all observations. For the sampled judge the expected
  correlation is computed in closed form from moments (E[S] = mean of mu_A over observations, E[S^2] from the second moments of p_A,
  E[S Y] = E[mu_A(i) Y]); for 20 of the 200 splits it is also estimated by one Monte Carlo draw per observation (seeded) as a check on the
  closed form. Spearman is not used (the theorem is about Pearson).
- **Proper-score side (matched rule: normalized RPS for DICES, Brier for alphaNLI):** the truthful forecast is p_A(i) floored at 1e-6;
  the collapsed forecast is the point mass (floored) at the grid label nearest mu_A(i) (halves round up), which is the theorem's point-mass
  forecast when mu is on the grid and a hard-label approximation otherwise; the base rate is the pooled half-A histogram. All are scored
  against half B with the unbiased per-sample risk; skill = 1 - risk/base risk.
- **Theory predictions with finite-rater noise handled:** rho_hat = Var_items(mu) / Var(Y) with Var_items(mu) estimated unbiasedly
  as Var of the item means of all ratings minus the mean of sigma_i^2/n_i (sigma_i^2 the within-item variance); the prediction for the
  sampled judge is rho_hat; for the mean judge, with finite half sizes, Var(mu)/sqrt((Var(mu) + mean sigma_i^2/nA_i) Var(Y)), also
  reported as the large-n limit sqrt(rho_hat).
- **Known-law synthetic check (labeled as a check of the code, not as evidence):** a finite law with known rho, where the closed
  forms must reproduce rho and sqrt(rho) to 1e-9 as the number of raters grows.

## Resampling
Item-clustered bootstrap over items (each item its own cluster), B = 4,000 draws, percentile 95 percent intervals, seed
`SeedSequence([20261128, 60, 99, set_index])`: sufficient statistics are computed per split and per item, item weights resample the
items, and the statistic is averaged over the 200 splits inside every draw. The interval therefore covers item sampling given the
rater split average, not the finite number of raters of the human population.

## Hypotheses (three; each with a falsification rule)
- **H1.** On each of the three sets, r(mean judge) minus r(sampled judge) is positive with a 95 percent interval that excludes 0.
  Falsified if any set fails. (Both share the numerator Cov(S, Y); the ratio is sqrt(Var S / Var of mean), so this is close to an
  identity on any rating law with within-item spread; it is stated to be checked, not because it could easily fail.)
- **H2.** The theory predictions reproduce the observed correlations: on at least 2 of the 3 sets, |r(sampled) - rho_hat| <= 0.02 and
  |r(mean judge) - its finite-half prediction| <= 0.02. Falsified otherwise, with the finite-sample explanation tested by the synthetic check.
- **H3.** The truthful forecast has lower matched-rule risk than the collapsed forecast on every one of the three sets, with a 95 percent
  interval of the difference (collapsed minus truthful, positive means truthful is better) that excludes 0. Falsified otherwise.

## Meaning of decision words
"Supported" and "falsified" are the rules above. Nothing here says correlation is wrong in general: the result is about Pearson
correlation of sampled scores under the stated conditions.

## Planned outputs
`results/sampled_vs_mean.csv`, `results/summary.md`, `results/hypotheses.json`, FIGURE 1 for the paper: `results/figure1_sampled_score.png`
and `.pdf` (3.3 inches wide, fonts at least 7 pt; panel (a) correlation against rho on [0,1] with the curves r = rho and r = sqrt(rho) and the
three sets' points with 95 percent bars; panel (b) matched-rule risk of the truthful versus the collapsed forecast per set, lower is
better; grid on every axis; axis labels name the metric and the held-out half B), `results/figure1_values.csv`,
`HOW_TO_READ_RESULTS.md`, `README.md`, `tests/test_exp60.py` (closed forms on a known finite law, the Monte Carlo agreement, determinism).
