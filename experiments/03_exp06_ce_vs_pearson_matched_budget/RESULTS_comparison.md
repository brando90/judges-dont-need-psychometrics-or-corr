# Matched cross-entropy versus Pearson: results

**Doc link:** [https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/03_exp06_ce_vs_pearson_matched_budget/RESULTS_comparison.md](https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/03_exp06_ce_vs_pearson_matched_budget/RESULTS_comparison.md) · [Google findings companion](https://docs.google.com/document/d/1f6AtTPIoHx-B3jaOy4ot8TNH__t_Kbiut5KwO0kKuX0/edit)

**Status, 11-09-2026:** comparison complete; independent primary-prediction and metric reconstruction passed. Paper integration remains a live dependency.

Experiment `03_exp06_ce_vs_pearson_matched_budget` continues historical experiment 06 in the sister repository. It diagnoses the old affine-softmax failure and reports a separate matched-simplex condition. It does not relabel the historical softmax fits as repaired or converged.

## Main result

Cross-entropy (CE) fitting minus Pearson fitting in held-out CE was **-0.260401 [-1.457729, 0.198472], p-val=n/a**. The 95% percentile interval resamples task families and refits both methods; 998/1000 draws passed all certificates. The interval includes zero: this cache does not establish a CE advantage.

CE fitting gave CE 5.491441, Pearson fitting 5.751842, and the training-human-marginal baseline 1.928001. Both mixture methods are poor probability forecasts relative to that baseline, despite useful score correlations. Sparse sampled-rating histograms restrict the common mixture family; this comparison does not repair that representational limitation.

Scope: 75 artifacts, 25 tasks, 19 task families, 195 expert ratings. Five historical partitions reuse these same artifacts. All 50 primary fits passed; 49204 bootstrap fits were attempted, with 2 failed certificates retained. There were no new model calls or provider inference charges. The recorded comparison elapsed time was 63.08 seconds on one local CPU thread.

## Historical failure diagnosis

The first historical failing receipt was selected in saved order, using only its training subset and the original initialization/settings. The reproduction hit the same iteration limit. The local SciPy version differs from the original, and function-call counts differ, so this is a configured-condition reproduction rather than a bitwise trajectory reproduction.

| Training-only condition | Iterations | Evaluations | Converged | Gradient infinity norm | Expected-score SD | Training Pearson | L2 penalty |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| exact_original | 300 | 334 | False | 0.019144332 | 0.010405734 | 0.959511 | 0.0043477277 |
| larger_budget_diagnostic | 3000 | 3342 | False | 0.13677418 | 0.00013078555 | 0.974352 | 0.0033656195 |
| no_l2_diagnostic | 300 | 321 | False | 3.3143473e-05 | 0.251782 | 0.973325 | 0 |

L2 means a squared-coefficient penalty; SD means standard deviation. Expected scores are normalized to 0–1. These deterministic diagnostics have no sampling confidence interval or hypothesis test (p-val=n/a).

The observed mechanism is ill-conditioned optimization near nearly constant expected scores, not a demonstrated analytic-gradient coding error. Pearson is insensitive to positive affine changes of expected scores; its gradient contains inverse prediction-spread terms. A coefficient penalty adds pressure toward smaller weights, while the softmax mean can collapse even with nonzero coefficients. The longer run decreased the objective but made score spread smaller and left a substantial gradient. Removing the penalty increased spread but still failed at the original budget: regularization is not established as the sole cause.

Scaling the final coefficient matrix does not preserve correlation, because softmax is nonlinear; the scaling diagnostic explicitly shows this. Thus coefficient-scale invariance must not be confused with Pearson’s invariance in score space. Common shifts of all class logits leave probabilities unchanged, confirming redundant stored coefficients; removing that redundancy alone is not demonstrated to solve the failure.

| Initial-coefficient scale | Best relative directional-gradient error across declared steps |
| ---: | ---: |
| 1 | 5.2301274e-10 |
| 0.01 | 2.5371675e-06 |
| 0.0001 | 3.9237392e-06 |
| 1e-06 | 5.5240841e-07 |
| 0 | 1.1427909e-06 |

The probability change under a common class-logit shift was 1.67e-16. Directional checks used three declared finite-difference steps and show step-size sensitivity near constant predictions. Full traces, rather than only selected endpoints, are in `results/diagnostic.json`.

## Corrected comparison and fair budgets

Both objectives fit global nonnegative weights summing to one over the same five once-smoothed candidate histograms. Data, candidate order, capacity, folds and resource ceilings match. There is one configuration per method and no validation search or test-dependent output calibration. This is a new probability family, with four free mixture dimensions; comparing its results directly against historical softmax as an objective-only effect would be invalid.

CE uses convex constrained optimization plus a bounded numerical refinement and an independent convex suboptimality bound. Pearson uses finite cone-projection face enumeration with fixed tie and constant-forecast rules. Matching resource ceilings does not mean that solver steps have equal computational cost. Objective-specific numerical checks establish optimization quality in each problem.

| Objective | Primary fits | Failed | Max function evaluations | Max linear solves | Total primary-fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| ce | 25 | 0 | 42 | 1 | 0.033963 |
| pearson | 25 | 0 | 0 | 31 | 0.021750 |

Common ceilings: 2,000 evaluations or linear solves and 1,000 iterative updates per fit; two hours for the complete comparison. Actual Newton refinement steps are separately recorded. Timing is descriptive single-run runtime, not a benchmark of solver speed.

## Full held-out metric panel

Every point estimate averages the pooled out-of-fold metric across the same five partitions. CE and Pearson intervals below include mixture refitting; the constant-baseline interval holds its original fold fits fixed. The paired contrast uses the same successful resamples for both methods. CE and forward Kullback–Leibler (KL) divergence use natural logs. Brier sums six categories; ranked probability score (RPS) averages five cumulative thresholds. Mean squared/absolute error compare predicted means to empirical human means, not to future individual ratings. Bias is judge minus human; positive means higher predicted ratings.

| Metric | CE fitting [95% refit interval] | Pearson fitting [95% refit interval] | CE − Pearson [95% refit interval] | Constant [95% fixed interval] |
| --- | --- | --- | --- | --- |
| ce | 5.491441 [4.136846, 7.019823] | 5.751842 [4.482476, 7.493935] | -0.260401 [-1.457729, 0.198472] | 1.928001 [1.835097, 2.056740] |
| kl | 4.925508 [3.577473, 6.446515] | 5.185910 [3.932453, 6.910109] | -0.260401 [-1.457729, 0.198472] | 1.362068 [1.213199, 1.568433] |
| brier | 0.889590 [0.762717, 1.051776] | 0.913693 [0.767128, 1.184614] | -0.024103 [-0.195676, 0.053535] | 0.869580 [0.844988, 0.900870] |
| rps | 0.139080 [0.108225, 0.174874] | 0.143634 [0.111255, 0.195868] | -0.004555 [-0.035602, 0.009965] | 0.187208 [0.153498, 0.235780] |
| weighted_pmf | 0.196227 [0.152980, 0.259540] | 0.202068 [0.161783, 0.275776] | -0.005841 [-0.055419, 0.015785] | 0.258201 [0.198428, 0.343399] |
| tvd | 0.512387 [0.429119, 0.601239] | 0.518075 [0.430043, 0.636010] | -0.005688 [-0.067752, 0.027772] | 0.696790 [0.657204, 0.739422] |
| pearson | 0.766432 [0.528040, 0.893263] | 0.753947 [0.474271, 0.895530] | 0.012485 [-0.095462, 0.141789] | -0.404177 [-0.588268, -0.196325] |
| spearman | 0.670349 [0.293075, 0.895752] | 0.687232 [0.318126, 0.886716] | -0.016884 [-0.125892, 0.127952] | -0.347285 [-0.543386, -0.139498] |
| kendall | 0.504410 [0.198483, 0.764381] | 0.526896 [0.222446, 0.754116] | -0.022486 [-0.121395, 0.113098] | -0.272850 [-0.435716, -0.114428] |
| mse | 0.041435 [0.025978, 0.059832] | 0.044864 [0.026505, 0.079747] | -0.003429 [-0.026015, 0.006380] | 0.089985 [0.054031, 0.142968] |
| mae | 0.160820 [0.116986, 0.201574] | 0.156112 [0.116372, 0.215676] | 0.004708 [-0.031219, 0.019378] | 0.249185 [0.183033, 0.342883] |
| bias | 0.065977 [0.005193, 0.137166] | 0.063496 [-0.014875, 0.147779] | 0.002482 [-0.048151, 0.044887] | 0.005811 [-0.102187, 0.139368] |
| human_mean | 0.525333 [0.397097, 0.627882] | 0.525333 [0.397097, 0.627882] | 0.000000 [0.000000, 0.000000] | 0.525333 [0.397181, 0.627812] |
| judge_mean | 0.591311 [0.484144, 0.694253] | 0.588829 [0.479987, 0.694993] | 0.002482 [-0.048151, 0.044887] | 0.531145 [0.521336, 0.538927] |
| human_sd | 0.286736 [0.220823, 0.353343] | 0.286736 [0.220823, 0.353343] | 0.000000 [0.000000, 0.000000] | 0.286736 [0.220868, 0.353326] |
| judge_sd | 0.275832 [0.207330, 0.350161] | 0.288848 [0.226545, 0.354505] | -0.013016 [-0.039038, 0.009162] | 0.026260 [0.020454, 0.029044] |
| entropy | 0.565932 [0.474796, 0.639616] | 0.565932 [0.474796, 0.639616] | 0.000000 [0.000000, 0.000000] | 0.565932 [0.474796, 0.639606] |

Weighted PMF is the human-frequency-weighted squared probability-vector distance; PMF means probability mass function. TVD is total variation distance. SDs describe across-artifact expected-score variation, not uncertainty of the mean. Every table entry has p-val=n/a; no confirmatory test was specified. Human entropy is descriptive and cancels in the CE-versus-Pearson KL contrast.

## Fold aggregation and uncertainty

| Objective | Mean within-fold Pearson | Mean within-fold prediction SD | Mean pooled out-of-fold Pearson |
| --- | ---: | ---: | ---: |
| ce | 0.701445 | 0.241067 | 0.766432 |
| pearson | 0.697107 | 0.259596 | 0.753947 |

These are distinct descriptive summaries; use the pooled-metric intervals above for the prespecified aggregate. Per-partition and per-fold records are in `results/results.json`. The constant model varies across training folds, so pooled correlation need not be undefined even though each individual fold predicts a constant.

Resampling unit: task family with all attached artifacts and multiplicities. Both mixture objectives are refitted within the fixed historical fold assignments for each draw. Empty test folds require no fit. The intervals include mixture fitting uncertainty, but not fresh human ratings, candidate-prompt selection or unseen solver/rater populations. The two excluded draws are retained and the reported intervals are conditional on passing numerical certificates; no replacement draws were used.

| Failed draw | Partition index | Fold | Objective | Convex gap | Required bound |
| ---: | ---: | ---: | --- | ---: | ---: |
| 169 | 3 | 1 | ce | 1.18330625298e-08 | 1e-08 |
| 530 | 0 | 2 | ce | 1.32971786915e-08 | 1e-08 |

Both failures were CE certificates just above the frozen tolerance despite optimizer success. They were not repaired or replaced after measurement. All Pearson bootstrap fits passed. The full compressed fit ledger and decompression/hash instructions are published alongside the results.

## Existing repeated-score variability

| Candidate | Cached ratings per artifact | Root mean square within-artifact sample SD (0–1) |
| --- | ---: | ---: |
| H0_opus | 3 | 0.054975 |
| H0_sol | 3 | 0.130639 |
| H1_sol | 3 | 0.111555 |
| H2_opus | 3 | 0.049889 |
| H2_sol | 3 | 0.084327 |

These are historical selector aliases, not a claim about newly executed model identities. They describe raw-score repeat variation; repeated batches of aggregated forecasts and new-rater variability were not measured.

## Verification, limits and continuation

Independent reconstruction checked 75 primary models and 1125 forecasts; maximum probability discrepancy was 1.11e-16. The independent implementation reproduced the full metric panel and paired interval endpoints. Bootstrap refits were not independently reexecuted. Five synthetic test groups passed, including 100 sparse-input certificate cases. The frozen protocol and sources are bound in `FREEZE.json`, committed before measurement at `0549cff`.

Measured environment: Python 3.11.6, NumPy 2.2.6, SciPy 1.16.3. Local execution followed Brando’s explicit fallback authorization; Rule 55 now records that standing remote-first/local-fallback preference. All three local Codex profiles have verified shared project/checkpoint pointers; native cross-account chat-history synchronization remains unverified. See `SESSION_ACCESS.md`.

This is retrospective evidence from a small, previously exposed cache. Sparse human empirical histograms are not known population rating distributions. Different solver/rater populations, new human ratings, unrestricted language-model training, all correlation objectives, and universal CE superiority are outside its scope. Correlation’s failure to identify a full probability distribution is distinct from whether a particular finite-sample CE learner improves a particular held-out metric.

The old softmax numerical failure remains unresolved as an optimization problem; the repaired scientific comparison avoids that confound by using a separately labeled family with numerical certificates. The paper remains unchanged. This experiment stays active because its measured outcome still needs a paper-integration decision.

Reproduction: run the frozen scripts with the private hash-bound dataset, new output directories and recorded arguments; then run `scripts/verify.py`. For published fit receipts, run `gzip -dk results/bootstrap_fit_receipts.json.gz` before verification. Reports can be regenerated with `python build_report.py` using only committed aggregate evidence.

Copy-ready message: I reproduced the Pearson optimizer failure and reran cross-entropy versus Pearson within the same probability-mixture family. The log-loss difference was -0.260 [-1.458, 0.198] with family resampling and refitting, so the cache does not establish an advantage; both methods also lost to a simple marginal-distribution baseline on log loss.

**TLDR-end:** [judge objectives: experiment 06] The historical failure was reproduced and a certified matched-family comparison completed. It does not establish CE superiority, and it exposes a substantial probability-modeling limitation in both mixture methods.
