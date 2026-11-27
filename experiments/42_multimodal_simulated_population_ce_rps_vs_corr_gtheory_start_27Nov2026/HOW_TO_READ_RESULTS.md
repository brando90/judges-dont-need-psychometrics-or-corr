# How to read experiment 42

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/42_multimodal_simulated_population_ce_rps_vs_corr_gtheory_start_27Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** The draws are simulated from known laws. The laws' base measures are real marginal counts. A number is either the exact risk of a forecast under a named law, or a Student-t interval over 400 replications of the training draw. Ranked probability score (RPS), cross-entropy (CE), and total variation distance (TVD) score the conditional distribution. Pearson, Spearman, Eρ², and Φ do not.

## Which data

Six bins with means 1.6, 2.0, 2.4, 2.8, 3.2, 3.6. Eight simulated items in each bin share that bin's law. Items 0–3 are train, 4–5 are validation, 6–7 are test. No item id is in two splits. Each simulated rater id is one draw and stays in one split.

The reported RPS, CE, and TVD are expectations of the train-fitted forecast under the known test law. They are not in-sample scores of the training ratings. The validation ratings are used only by the validation-RPS rule. A separate check draws 32 ratings on each test item and compares that sample mean with the exact risk.

Nothing in the Monte Carlo is a new human rating. The base measures were computed from public files already used by experiment 36. The tilted draws were not used in an earlier experiment.

Targets:

- **Polarized.** The deployment subpopulation. Low pole: reflected VeriBench histogram tilted to mean 0.5. High pole: VeriBench `native_main` histogram tilted to mean 4.5. The bin mean sets the mixture weight.
- **Strict.** Reflected histogram tilted to the bin mean minus 0.8. Used for subgroup calibration.
- **Shift mixture.** Weights 0.05, 0.05, 0.80, 0.10 on strict, lenient, polarized, chaos. Every component has the bin mean, so the mixture mean equals the bin mean.
- **Equal-moment panel.** Three fixed laws. Not a held-out-item split. The risk is the expectation under law U.
- **Native 3-class side table.** DICES-990 and ChaosNLI marginals tilted to means 0.7, 1.0, and 1.3 on their own three classes. No embedding into 0–5.

## Metrics

RPS is the mean over bins of E[(1/(K−1)) Σ_t (F(t) − 1{y ≤ t})²]. On the 0–5 laws, K−1 = 5, so the range is 0 to 1. Lower is better. The polarized oracle scores 0.221429. The Pearson binomial scores 0.281693 on that same law.

CE is E[−ln q(y)] in nats. Tables of point masses use the smoothed forecast (1 − K×10⁻⁶) q + 10⁻⁶. Unsmoothed CE of a forecast that puts zero mass on a score the truth uses is infinite. Lower is better.

TVD is the mean over bins of 0.5 Σ |q − p| against the known law, not against a finite test histogram. Range 0 to 1. Lower is better.

Bias is the forecast mean minus the true conditional mean, averaged over bins. Positive means the forecast is lenient relative to that target.

Pearson and Spearman compare the six bin means. A constant forecast is undefined. Eρ² = v_u / (v_u + v_e), with v_u the variance of the six bin means using divisor 6. For one source, Φ equals Eρ². In a two-source panel, joint Φ also charges the source main effect. A zero denominator is undefined. Higher Eρ² means more repeatable, not more accurate.

## What is subtracted

"Pearson binomial minus persona cross-entropy" is expected RPS of the binomial forecast minus expected RPS of the polarized persona's add-one forecast, on the polarized law, from the same training ratings. Positive means the persona cross-entropy forecast is better.

"Pooled cross-entropy minus persona cross-entropy" uses that same order. Positive means the persona forecast is better. At 4 and 16 ratings the sign is negative: the pooled forecast is better.

"Validation-chosen minus lenient point" is expected RPS of the forecast selected by validation RPS, minus expected RPS of the lenient point mass. Negative means the selected forecast is better.

## How the forecasts were fit

Persona cross-entropy: per persona and bin, (count + 1) / (n + 6). Closed form. Unique minimizer of categorical cross-entropy plus a Dirichlet(1) penalty. The polarized deployment uses only the polarized quota.

Persona RPS: the empirical frequency, counts / n. It minimizes the empirical normalized RPS. Closed form.

Pooled cross-entropy: the same add-one formula after summing the four persona quotas. It never sees the persona label.

Pearson binomial: Binomial(5, mean/5) at the pooled training mean. The clip to [0.02, 0.98] does not bind on these means. Closed form. It matches the mean.

Spearman point: midranks of the six pooled means, mapped linearly onto {0,…,5}, round-half-up, then a point mass. Closed form.

Lenient point: clip(round-half-up(pooled mean) + 1, 0, 5). Round-half-up is floor(x + 0.5). Closed form.

Maximum Eρ² chooses among persona cross-entropy, persona RPS, pooled cross-entropy, the Pearson binomial, and the lenient point. A tie breaks toward the earlier name. The Spearman point is scored and is not in that pool: every non-constant point mass has Eρ² = 1, so the coefficient does not choose among them.

Validation RPS draws a fresh quota-sized sample from the polarized law on the validation items only.

Maximum Φ picks the persona whose own-source Φ is highest and broadcasts that persona's add-one law as the forecast of the polarized population. For one source, Φ equals Eρ².

## Uncertainty

Monte Carlo intervals are Student-t 95% intervals over 400 replications, 399 degrees of freedom. The resampling unit is the replication: new training ratings, and new validation ratings when a rule uses them. Bins, laws, and the split stay fixed. A two-sided one-sample t test compares the mean difference with 0. A stored p-value of 0 is underflow of the survival function and is reported as p < 1e-300. A coefficient that is identical in every replication has p-val = n/a.

One item-clustered percentile interval uses 1,000 bootstrap draws of the 12 test items (2 per bin) for a single fitted forecast. That interval is for the sample mean of that fit, not for retraining.

## Figures

Every data axis has gridlines. The vertical direction of an RPS or TVD axis is "lower is better," and the axis title says so.
