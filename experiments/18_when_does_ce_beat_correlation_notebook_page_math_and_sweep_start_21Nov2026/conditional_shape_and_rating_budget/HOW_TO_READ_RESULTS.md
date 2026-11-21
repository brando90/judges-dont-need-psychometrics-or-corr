# How to read the conditional-shape and annotation-budget study | 11-21-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/18_when_does_ce_beat_correlation_notebook_page_math_and_sweep_start_21Nov2026/conditional_shape_and_rating_budget/HOW_TO_READ_RESULTS.md>

Date: 11-21-2026

**TLDR:** Read the held-out ranked probability score (RPS) in [results.md](results.md) for the distributional comparison. The accompanying tables expose the effects of rating budget, shape assumptions and use of distributional validation.

## What varies

The simulation crosses distinct training artifacts (8, 32 or 128) with ratings per training artifact (1, 4 or 16). An artifact has an angle x and an independent binary feature z. Across the two scenarios, the true expected rating is the same nonlinear function of x. In the neutral scenario, z does nothing. In the varying scenario it moves probability between extreme and central ratings while leaving the mean fixed. The full candidate family (four curves × three shape rules) includes either truth. A replication samples the true curve and the varying shape's sign. There are 256 independent replications. Within a replication each cell uses a nested training sample and the same disjoint set of 384 synthetic test artifacts, each with three ratings. Training and validation artifacts are also disjoint. Equal numbers of ratings within each split give every artifact equal weight. There are no human participants, old exposed ratings, solver holdouts or task-family holdouts.

## How the judges are fitted

The two main methods score every member of the same 12-candidate probability family on training data, with no iterative optimizer, initialization or hyperparameters. The cross-entropy (CE) fit minimizes individual-rating log loss. The Pearson fit maximizes sample correlation between forecast means and the mean rating per training artifact. All shape rules induce exactly the same forecast mean at every artifact. The default deterministic order favors a neutral shape, then positive, then negative. This is a tied-optimum assumption and influences the results in either direction. Both fits are globally optimal over the finite set up to 10⁻¹² objective tolerance. When Pearson is undefined on the training labels the pipeline records a fallback. A validation tie-break gives both methods the same additional n/4 artifacts × k ratings and minimizes validation RPS only among their training-optimal candidates. An additional diagnostic chooses the true shape after training a Pearson mean curve and cannot be deployed. The oracle knows all probabilities, and the constant comparator uses training-marginal frequencies smoothed by half a count per rating level.

## What the metrics mean

| Name in the machine-readable tables | Held-out test quantity | Range, direction and reference |
|---|---|---|
| rps | Normalized individual-rating ranked probability score (RPS), the mean squared difference between five forecast cumulative probabilities and indicators of the realized rating being at or below each threshold. | 0–1, lower is better. Even the true distribution has positive average loss when the outcome is random. |
| ce | Individual-rating cross-entropy (CE), average negative natural log probability of the observed rating. | 0 to infinity in nats, lower is better. Population reference is conditional entropy. |
| empirical_tvd | Total variation distance (TVD), half the sum of absolute differences from the three-rating histogram, averaged equally across held-out artifacts. | 0–1, lower is closer to the sample histogram. Even the oracle has positive empirical TVD. |
| truth_tvd | TVD from the true simulated conditional probabilities. | 0–1, lower is better, oracle value zero. |
| pearson_empirical, spearman_empirical, kendall_empirical | Pearson linear correlation, midrank Spearman correlation and tie-adjusted Kendall tau-b of forecast means with observed test-artifact mean ratings. | −1 to 1, higher is better. Repeated observed means are ties. Constant forecasts give undefined correlation. Even an oracle mean has imperfect agreement with three noisy ratings. |
| pearson_truth, spearman_truth | Pearson and Spearman correlations between forecast and true test-artifact means. | −1 to 1, higher is better, oracle value 1. |
| mean_absolute_error, signed_bias | Mean absolute difference and mean signed difference (forecast minus observed) between test-artifact mean ratings. | Absolute error 0–5, lower is better. Signed bias −5 to 5, zero is optimal on average. Positive is lenient. |
| forecast_mean, observed_mean | Average predicted and observed test-artifact mean ratings. | Rating points 0–5; descriptive. |
| curve_recovery, shape_recovery | Share of simulations that select the true curve or discrete shape rule, compared with the known generator. | 0–1, higher is better. Undefined for the constant baseline, which selects neither parameter. |

## How to interpret the comparison

In [summary.csv](results/summary.csv), a blank subtract_method means an absolute fit metric. When populated it gives the method whose value of the **same held-out metric** is subtracted. In the primary row, a positive difference means the Pearson-trained judge has higher held-out RPS than the CE-trained judge; lower is better. In [allocation.csv](results/allocation.csv) both columns being compared use 128 training ratings and the same method. A positive difference means lower held-out RPS at eight artifacts with 16 ratings each than at 128 artifacts with one rating each. Confidence intervals are pointwise 95% percentile intervals from 2,000 bootstrap resamples of the 256 complete independent simulations (fixed seed 20261122); each pair uses the same simulation. A cell contains 256 full training, validation and test draws, not 256 independent artifact clusters per split. Shared cells are dependent and the intervals do not adjust for comparisons across the grid. The prespecified direction rule uses zero, with no practical importance threshold or null-hypothesis test (p-val=n/a). Missing nonfinite values are counted explicitly in the defined denominator. A zero-width Monte Carlo interval records identical values in these sampled replications.

## Reproduce

From this directory, run with Python, NumPy, SciPy and Matplotlib. The exact simulation library versions appear in [run.json](results/run.json).

```sh
shasum -a 256 -c results/protocol.sha256
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 scripts/run.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 scripts/verify.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 scripts/summarize.py
```

The [protocol](PROTOCOL.md) was hashed before any measurement and moved byte-for-byte when the parallel local agent established the canonical experiment directory. Its historical header points to the originally planned directory. The [compressed raw records](results/raw.csv.gz) give one row per fitted procedure and cell, including selections, fallback flags and per-replication metrics. The run receipt hashes the uncompressed data and simulation script. Verification checks probability sums, positivity, analytic means, scalar loss calculations, independent Pearson and CE choices, validation tie selection, the undefined-target fallback, the exact population RPS shape penalty, grid completeness, range checks, rank ties, the data hash and replay of both endpoint replications. Test empirical means are computed from raw integer draws so identical totals receive identical ranks.
