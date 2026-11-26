# How to read experiment 34

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Every number is an expectation under a known synthetic conditional distribution, for a forecast fitted only on training ratings of the same fixed items. Ranked probability score (RPS), cross-entropy (CE), and total variation distance (TVD) score the conditional distribution. The generalizability coefficient Eρ² and the dependability coefficient Φ score repeatability. A difference subtracts the same metric for two forecasts on the same items.

## Which data

| Setting | Items | What is random | What is held out |
| --- | --- | --- | --- |
| Binomial agreement and leniency | 12 fixed items, θ = (j+1)/14, Binomial(5, θ) | Training ratings, and 16 validation ratings when a rule selects a forecast | No items. The metric is the expected score of a new rating from the true conditional distribution |
| Shape | 8 fixed items, j = 2…9. The truth matches a binomial mean and variance and differs in shape | Training ratings from the shaped distribution | Same as above. Prior exposure: none |
| Six raters | The same 12 items. Four neutral raters, one rater clipped +2 categories, one clipped −2. Target = the equal mixture of the six rater laws | Each rater's training ratings | The target is a new rating from a rater drawn from those six laws |

No human rating is used. No rating from this folder appeared in an earlier experiment.

## Metrics

RPS is the mean over items of E[(1/5) Σ_{k=0}^{4} (F(k) − 1{y ≤ k})²]. Range 0 to 1. Lower is better. On the 12 binomial items the true distribution scores 0.102440 and the pooled frequency scores 0.178049.

CE is E[−ln q(y)] in nats, after replacing q by (1 − 6×10⁻⁶) q + 10⁻⁶. Lower is better. Unsmoothed CE of a point mass that misses support is infinite.

TVD is the mean over items of 0.5 Σ \|q − p\| against the known p, not against a finite test histogram. Range 0 to 1. Lower is better.

Bias is the forecast mean minus the true conditional mean. Positive means lenient. Pearson, Spearman, and Kendall τ-b compare means across items. A constant forecast has undefined correlations.

Eρ² = v_u / (v_u + v_e), with v_u the variance of item means using divisor N items. Higher means more repeatable. Φ equals Eρ² for one source. In a two-source panel, Φ also charges the source main effect. A zero denominator is undefined.

## What is subtracted

"Binomial projection minus add-one" is expected RPS of the binomial-projection learner minus expected RPS of the add-one CE learner on the same items and the same training ratings. Positive means the add-one learner is better when the metric is RPS, smoothed CE, or TVD.

"Lenient point mass minus pooled Laplace" uses that same order. Positive means the constant forecast is better.

"Chosen forecast minus pooled Laplace" is the expected RPS of whichever forecast a selection rule picked, minus the pooled Laplace forecast from the same training ratings.

## How the forecasts were fit

Add-one CE: closed form (count + 1) / (n + 6), the unique minimizer of categorical cross-entropy plus a Dirichlet(1) penalty. Binomial projection: θ = clip(mean / 5, 0.02, 0.98), closed form. Pooled Laplace: add-one on all items together. Lenient point mass: clip(round-half-up(mean) + 1, 0, 5), or, in the six-rater setting, clip(round-half-up(lenient rater mean), 0, 5). Round-half-up is floor(x + 0.5). Validation RPS uses 16 ratings that are not in the training fit and not in the reported expectation. Maximum Eρ² uses only the forecast's own means and variances.

## Uncertainty

Student-t 95% intervals over 400 replications, null mean difference 0, two-sided one-sample t test. The replication redraws ratings. It does not redraw items. p-val=n/a when every replication has the same value. SciPy p-values that underflow to 0 are reported as p < 1e-300. Pass/fail cuts are only the ones in `preregistered_criteria.json`.

**TLDR-end:** [judge objectives: experiment 34] RPS, CE, and TVD are expectations under the known conditional distribution; Eρ² and Φ are repeatability of the forecast itself.
