# Conditional binomial distribution identification: fixed procedure

**Status:** specified before generating synthetic ratings or computing study metrics. Source hashes are in `FREEZE.json`; its Git commit precedes measurement.

The question is whether scalar association identifies an item-conditioned rating distribution. This is a controlled synthetic identification study, not a claim that any selection objective improves every finite-sample test metric or generalizes to human judgments. No human data, language-model forecasts, or private artifacts enter this experiment.

## Population and candidates

For each synthetic item with parameter t, R | t follows Binomial(5,t), with ratings 0 through 5. Twelve families divide |t-0.5| into twelve intervals of width 1/30. Each family has eight midpoint distances, each appearing with both signs. Thus every split is exactly symmetric under t -> 1-t, and every family has sixteen items. Parameters lie strictly inside [0.1,0.9]. The fixed manifest, not random sampling of items, defines the finite population.

Train families are {0,2,4,7,9,11} (96 items), validation families {1,5,8} (48 items), and test families {3,6,10} (48 items). Families and items are disjoint. Family-held-out here means synthetic parameter bands; it does not establish transfer to natural task families.

The identical nine-candidate pool is available to every selection objective:

1. Five distributions q_lambda = (1-lambda) p + lambda ((1-t) delta_0 + t delta_5), lambda in {0,.25,.5,.75,.95}. All have mean 5t, and lambda=0 is the oracle.
2. Permuted conditional distribution p(r | 1-t). It has exactly the same aggregate histogram as the oracle in each symmetric split, but assigns the wrong distribution to each item.
3. Constant Binomial(5,.5).
4. Constant uniform distribution on the six ratings.
5. Constant empirical train-rating distribution with Laplace alpha=1. This is the only fitted candidate; its fitting uses train ratings only, separately for each replication and rating count.

Every distribution has positive probability at every rating. There is no additional smoothing or clipping. The permutation control is a declared known bad conditional forecast, not a train/test data shuffle. Oracle knowledge of the analytic population is intentionally part of the constructed candidate class. It is not an implementable human judge learned from sparse labels.

## Metrics and mathematical predictions

Cross-entropy (CE) uses natural logs. The Brier score sums six category errors, range [0,2]. The ranked probability score (RPS) averages five cumulative-threshold errors, range [0,1]. These conventions match the public theorem-judge experiment06. Let P and Q denote cumulative probabilities. Expected excess losses are respectively KL(p||q), sum_r (p_r-q_r)^2, and mean_k (P_k-Q_k)^2. Each is strictly positive when the distributions differ, so each proper loss uniquely identifies p in the population.

Individual-rating CE is -log q(R|t), meaningful with one rating per item. Brier and RPS also average individual-rating losses. The empirical frequencies from one, two, or three ratings are not declared known population distributions. For selection, empirical probabilities are merely algebraic weights implementing that individual-rating average. No empirical-human KL is computed.

Pearson, Spearman, and Kendall tau-b use the candidate's conditional mean and each item's sampled rating mean. Normalized mean squared error (nMSE) averages ((candidate mean - R)/5)^2 over individual ratings. Selecting with squared error to the sampled item mean would give the same ordering. With these scalar summaries, all five mixture candidates are tied on every possible rating realization, not just in expectation. Their exact mean is supplied analytically to avoid floating-point rank artifacts; a test verifies agreement with their numerical probability means.

| Reported quantity | Oracle/intended/literal analytic forecast | Changed-shape mixtures | Permutation | Constants |
| --- | --- | --- | --- | --- |
| Expected CE / KL | entropy / 0 | CE larger, KL > 0, monotonically increasing with lambda | CE larger, KL > 0 | CE larger, KL > 0 |
| Expected Brier / RPS | irreducible outcome variance | larger, excess proportional to lambda squared | larger | larger |
| Expected nMSE / MAE to individual ratings | nonzero outcome uncertainty | exactly identical to oracle | nMSE larger; MAE expected larger | no distribution identification |
| Mean MSE / mean MAE | 0 / 0 | 0 / 0 | positive / positive | positive / positive |
| Bias and judge/human means | 0; both means 0.5 normalized | same | 0 by symmetry, despite wrong conditional forecasts | fixed constants 0; train-fitted bias may differ |
| Pearson / Spearman / Kendall to population conditional mean | 1 / 1 / 1 | same | -1 / -1 / -1 | undefined |
| Pearson to a future individual rating | positive below 1 | same | negative | undefined |
| Sample correlations and proper losses | noisy across rating replications | sample scalar measures exactly equal; proper measures can differ | usually worse; no finite-sample guarantee | sample correlations undefined |
| Oracle selection | unique minimizer of population proper losses | scalar objectives cannot identify lambda | wrong condition detectable conditionally | finite-sample selection can err |
| Empty/non-PMF input | invalid, not silently normalized | not applicable | not applicable | not applicable |

There are no prompts in this synthetic study: literal compliance is the specified analytic forecast. The tests check probability normalization and the scorer directly. A trace stores synthetic examples across split/rating-count combinations. The old marginal-histogram toy is reproduced separately with its original code and unchanged historical outputs.

## Sampling, selection and common evaluation

Generate 256 independent rating replications using NumPy's PCG64 default generator, seed sequences [20261108, replication, split-index]. Each split draws three independent ratings per item; the one/two/three-rating conditions use prefixes, making comparisons paired. Split-index order is train=0, validation=1, test=2. Fixed manifest order is preserved.

Train only the constant empirical candidate. For each replication and rating count, select over all nine candidates using validation CE, Brier, RPS, Pearson, Spearman, Kendall, or nMSE. Every criterion has the same data, candidate count, one exhaustive search and no tuning. This study concerns selection among prespecified distributions; it is separate from training a parameterized judge under different objectives. No optimization comparison or train loss superiority is implied.

Undefined constant-input correlations are ineligible for correlation selection, not set to zero. If all candidates are undefined, retain a failed selection in the denominator. A tie is any objective within absolute 1e-12 of the best value (zero relative tolerance). Enumerate the complete tied set. Report the min, equally weighted mean, and max test metric across that set. The equal-weight result is the expectation of uniform random tie-breaking, not the score of an ensemble, and it is not the only permissible tie policy. Also report oracle-in-tie and oracle probability under uniform tie-breaking. No index-order choice may be used to claim superiority.

Only after selection, evaluate every selected system and every candidate on the same test metrics. Exact population CE/KL/Brier/RPS/nMSE/MAE/bias and mean agreement use the known test distribution. Report Pearson/Spearman/Kendall against the population conditional mean as separate diagnostics, plus exact Pearson against a future individual rating. Report sampled-test proper losses, all three correlations, nMSE, MAE, bias, judge mean and sampled rating mean. Never confuse the population mean correlations with noisy human-rating agreement. Test data are not used for fitting, selection, or revision.

The full denominator is 256 x 3 = 768 simulation conditions, 768 x 7 = 5,376 selections, and 768 x 9 = 6,912 common candidate evaluations. Preserve errors and missingness, rather than dropping or retrying selections.

## Uncertainty, budget and scope

The test population is known, so deterministic population expectations receive no Monte Carlo confidence interval. Repeat-average performance of selection depends on sampled train/validation ratings. Report its 95% Student-t Monte Carlo interval across 256 independent replication seeds, alongside the standard deviation and 2.5/97.5 percentiles of replication outcomes. The interval concerns the mean over repeated synthetic ratings on fixed items, not task/family generalization, rater heterogeneity, or future human populations. Repeated rating counts are paired, not independent datasets. No hypothesis test or superiority threshold is declared: p-val=n/a throughout.

An ordinary CPU analysis uses at most eight threads and 1,800 seconds. The study makes zero provider calls and incurs $0 provider charges. External shell timeout and an internal deadline bound execution. No optional conditions are admitted after results are seen. Reproduction checks may deterministically recompute summaries from committed rows and run analytic tests; they do not create a second inferential sample.
