# Matched cross-entropy versus Pearson comparison

**Status, 11-09-2026: DRAFT, empirical execution not authorized by a freeze yet.** Review the training-only historical reproduction before accepting this new condition. A hash manifest, `FREEZE.json`, must bind this protocol, configuration and executable sources before any new held-out measurement. Historical test exposure makes the comparison retrospective even after this freeze.

## Question and distinction from the old condition

Compare cross-entropy (CE) against Pearson optimization within the same family of probability forecasts. The primary contrast is held-out CE under CE fitting minus held-out CE under Pearson fitting. Negative values favor CE on this metric; neither sign is required. Report all other metrics as tradeoffs, without changing the primary contrast after observation.

The original affine-softmax fits remain historical evidence. `scripts/diagnose.py` reproduces the first recorded failing Pearson fit in saved receipt order using its original training subset, initialization and numerical settings. It saves every iteration's objective components, gradient norm and prediction variability. A longer-budget run and a no-penalty run are explicitly diagnostic ablations, not fair-comparison measurements. Directional finite differences at shrinking weights test the analytic gradient near constant predictions. Scaling a fitted coefficient vector tests the interaction between shrinkage and correlation; adding a common class-logit shift checks parameter redundancy. No held-out labels or outcomes are used to select the diagnostic fit.

The proposed repair uses a **new model-family condition**, not a claim that the old softmax results have become converged. The November 8 sister-repository work compared weighted squared probability loss with Pearson in this family. This condition instead compares CE with Pearson and must not reuse that study's weighted-loss result as a CE result.

## Shared data and feasible predictions

Use the hash-bound historical cache and candidate order in `protocol.json`. Each candidate contributes its histogram of cached ratings on categories 0 through 5. Smooth each candidate probability vector once as `(1-6*epsilon)*q + epsilon`, with the frozen epsilon. Human targets are unsmoothed empirical histograms. This fits an equal-artifact mean of individual-rating log losses: artifacts have equal weight despite different numbers of ratings.

Both methods predict the same global simplex mixture of the same candidate probability vectors: nonnegative weights summing to one. There is no intercept, train-only output remapping, per-fold test calibration, or objective-specific feature set. Four free mixture dimensions for five candidates are distinct from the old affine-softmax capacity. No new model outputs or ratings are collected.

Use all five historical partition seeds and five outer task-family folds per seed. Fit outside each test fold. No hyperparameters are selected; both methods have one fixed configuration. Also report the training-human-marginal constant probability baseline. Repeated partitions reuse the same artifacts and are never counted as independent data.

## Solvers, limits and identification

Both objectives have the same ceiling of 2,000 function/gradient evaluations or linear face solves, 1,000 iterative updates, one configuration and one CPU thread. The entire comparison has a two-hour ceiling. Record actual evaluation counts, linear solves, iterations, runtime, failure status and probability variance; these ceilings are resource allowances, not an assertion that unequal algorithmic steps cost the same.

CE minimizes the convex mean negative log likelihood with Sequential Least Squares Programming (SLSQP), analytic gradients, one uniform start, fixed bounds and a sum-one equality constraint. The stopping tolerance is `1e-15`. A fixed continuation performs at most ten Newton refinements on the solver-selected simplex face, using a sum-zero basis, backtracking and the same evaluation/iteration budgets. Synthetic sparse-input checks showed that objective-rounding stops can precede the gradient certificate; this refinement addresses that numerical issue without loosening the certificate. Independently compute the simplex Frank–Wolfe gap `gradient @ weights - min(gradient)`, which upper-bounds objective suboptimality for this convex problem. A gap at most `1e-8` is a successful certificate, irrespective of the optimizer's own success flag. Retain both flags. The deterministic solver's solution is the CE tie convention; no secondary held-out metric selects among tied distributions.

Pearson uses the preserved finite active-face cone-projection solver and its deterministic minimal-support/lexicographic tie rule. For positive attainable correlation, the projection certificate verifies the cone solution and its correlation identity. For all nonpositive covariances, use the best nonconstant vertex with the declared index tie rule. Constant targets or all-constant candidates receive the explicit uniform-mixture fallback and undefined-correlation flag. Five candidates require at most 31 nonempty faces; no random restarts or tuning are granted. Preserve the inherited feasibility, constant-norm and certificate tolerances in its receipts. Do not loosen these tolerances after seeing empirical failures.

An optimality certificate is numerical evidence for the stated finite problem, not statistical identification of a human-rating distribution. Pearson only identifies features of expected ratings and may leave multiple probability forecasts tied. This is one declared Pearson fitting procedure, not a test of every correlation objective or every tie rule.

## Measurements and expected controls

`scripts/run_comparison.py::metrics` reports the complete panel. Scores use natural logarithms and normalized expected rating `rating/5`.

| Metric | Definition | Exact-distribution control | Constant or invalid input |
| --- | --- | --- | --- |
| CE | Mean `-sum(p*log(q))` | Human empirical entropy | Uniform forecast: `log(6)`; nonpositive candidate probabilities rejected before fitting |
| Forward Kullback–Leibler divergence | CE minus empirical entropy | Zero | Nonnegative within numerical error |
| Brier score | Expected six-class squared error | `mean(1-sum(p*p))` | Range 0–2 |
| Ranked probability score | Mean expected squared cumulative error over five thresholds | `mean(Fp*(1-Fp))` | Range 0–1 |
| Weighted probability loss | `mean(sum(p*(p-q)^2))` | Zero | Range 0–1; descriptive empirical-histogram loss |
| Total variation | Mean half the probability-vector absolute difference | Zero | Range 0–1 |
| Pearson, Spearman, Kendall | Expected-score correlations, separately | One for nonconstant exact predictions | Undefined when the required variation is absent |
| Mean squared/absolute error, bias | Expected-score error; bias = judge minus human | Zero | Report both score means and standard deviations |

Synthetic controls cover known-mixture recovery, two-candidate dense grids, CE gradients, duplicate candidates and undefined constant forecasts, plus 100 fixed-seed sparse-input certificate checks. Preserve failures during implementation; no empirical data are opened by these tests. Before freezing, run all controls and syntax checks. A mismatch blocks empirical interpretation until explained.

For empirical results, report each partition's pooled out-of-fold metrics separately from fold metrics. The primary aggregate averages pooled metrics across the five partitions. Fold correlations are diagnostics and are not interchangeable with correlations after pooling independently fitted folds. Report cached raw-repeat variability and repetition counts from the existing data only.

## Uncertainty and failed cells

Use 1,000 paired resamples of the historical task families, retaining attached artifacts and family multiplicities. Report percentile 95% intervals with the resampling unit and valid/invalid counts. There is no confirmatory null-hypothesis test: `p-val=n/a`.

Compute fixed-prediction intervals, then refit both mixture objectives for every nonempty test fold under each resampled family draw. Keep the original partition assignments fixed. These refit intervals include mixture-weight fitting variation conditional on the cached candidates and fixed folds. They exclude fresh-rater variation, upstream prompt selection and new-agent generalization.

Save every attempted fit receipt. A primary certificate failure remains in the primary point estimate with a visible failure flag; do not describe its fitted optimum as certified. For a refit draw containing any failed certificate or insufficient training rows, retain its receipts and mark the draw invalid for the paired refit interval. Continue the remaining declared cells; do not replace a draw. Report that intervals excluding numerical failures are conditional on successful draws and inspect whether exclusions materially limit interpretation. Undefined correlations remain missing for that metric even when other metrics are finite.

## Verification and privacy

`scripts/verify.py` independently reconstructs every saved primary prediction from weights, candidate histograms and freshly reconstructed family folds. It computes the full metric panel using a separate implementation and SciPy's correlation routines, recomputes interval endpoints, checks denominator/receipt reconciliation and rechecks primary CE certificates. It does not claim independent rerunning of every bootstrap fit.

Raw human ratings, per-artifact forecasts, learned weights, native session paths and resamples stay under ignored private storage. Publish aggregate results, all numerical fit receipts, the resampling failure ledger, source hashes and verification. No paper claim changes until results and limitations have been verified. Keep this folder active while its results are an unresolved paper-integration dependency.
