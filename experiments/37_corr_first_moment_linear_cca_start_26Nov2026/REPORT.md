# Pearson correlation, first moments, and canonical correlation

**Data.** Held-out test items from a prompt-level 60/20/20 split, seed 37, unless a row says full collection. Validation items were removed from the fit and were not used to choose a hyperparameter. SummEval, ChaosNLI, and DICES-990 were already used in experiments 28, 29, and 36; this is a new split of those corpora, not a new corpus. VeriBench functional semantic correctness (FSC) had not been scored in this repository.

**Judge.** Cross-entropy softmax on 8 principal components of a 64-dimensional signed token hash, plus an intercept. Components are fit on train items only. L2 penalty 0.01 on the non-intercept weights. L-BFGS-B from the zero vector. The logit loss is convex and the solver returned success in 7 to 19 iterations, under the cap of 200. VeriBench text is the task folder and candidate filename, not the Lean source.

**Uncertainty.** Percentile 95% interval, item resample, B = 1000, seed 37. A p-value is the two-sided percentile bootstrap probability of a draw on the other side of 0. `p-val=< 0.0010` means none of the 1000 draws crossed 0.

## Answers

**1. Sanmi had the point.** The sentence "correlation only captures first moments" is too strong. Across prompts, Pearson correlation is a standardized linear second-moment statistic. It deletes a global shift, it is unchanged in the square by a nonzero rescaling, and it is exactly zero for the deterministic curve `Y = X^2` on `{-1, 0, 1}`.

What remains, and what the law of total covariance proves, is narrower. If the judge's draw is independent of the human given the prompt, the covariance equals the covariance of the two conditional means. Within a prompt, the numerator uses only those conditional first moments. Within-prompt variance is not in the numerator. It is added in the denominator, `Var(X) = Var_c(μ_X) + E_c[Var(X|c)]`, and any positive conditional variance strictly shrinks the squared correlation when the conditional means vary.

Reply:

> You were right that I overstated it. Pearson correlation is not "the first moment." Across prompts it is a linear second-moment statistic: a constant shift drops out, scale drops out of the squared correlation, and `Y = X^2` on `{-1, 0, 1}` has correlation exactly 0. The part that survives is conditional. When the judge is independent of the human given the prompt, the covariance equals the covariance of the conditional means, so the numerator sees only those means, and spread inside the prompt only enlarges the denominator. Pearson elicits an affine function of the conditional mean and punishes conditional variance. It does not see a general first moment, and it does not see a nonlinear relationship.

**2. Pearson captures linear association, not "the first moment."** On the held-out human pairs the association is nearly linear: Spearman sits within about 0.02 of Pearson, and adding the square of one rating raises in-sample R² by at most 0.0045 (DICES-990). The large gap is the within-prompt noise. One raw pair on ChaosNLI SNLI+MNLI-matched has Pearson 0.4788 [0.4123, 0.5455]; the Pearson of the two half means on the same test items is 0.9783. DICES-990 moves from 0.2278 [0.0844, 0.3671] to 0.8957. That is the denominator penalty in the ratings, not a curved conditional mean.

**3. CCA is the vector version of the same object, and it is not a judging metric.** Canonical correlation is the largest correlation between linear projections of two views. For an invertible matrix `A` and a shift `b`, the projection `uᵀ(Ax+b)` is the projection `(Aᵀu)ᵀx` plus a constant, and `Aᵀ` only renames the coefficient. Sign flips, label permutations, and orthogonal mixes therefore leave every canonical correlation unchanged. One-hot and probability-vector CCA do see more than the scalar mean: two prompts with the same mean and different probability mass functions are invisible to scalar Pearson and visible to CCA. Symmetry, label permutation, and the reward for deleting within-prompt noise remain. On SummEval expert test items the 4-dimensional CCA of expected grades is 0.248, while the 20-dimensional CCA of the four probability vectors is 0.505, and an aspect roll, a sign flip, and an orthogonal mix of the 4-dimensional view change the canonical correlations by at most 6.5e-15.

## VeriBench FSC

There is no file named `R_v1` in this repository or in the trustworthy-theorem-judges tree cloned for this run, and there is no 343-item table with about five experts. The public experiment 09 export has 262 numeric `final_score_0_5` ratings on 106 items, 7 rater ids, median 2 ratings per item (minimum 2, maximum 4). Forty-eight items have at least 3 ratings. Two items have at least 4, so a half-split is not available. The loader is `load_veribench` in `run_real.py`. It reads that CSV directory and does not impute the missing table.

On the 24 held-out FSC items the filename-hash head has test cross-entropy 1.1245 [0.7249, 1.5686], against a uniform reference of `log(6) = 1.7918`. Reversing the six class probabilities leaves the canonical correlations unchanged (maximum absolute difference 6.550e-15) and moves test cross-entropy to 4.3581 [4.1534, 4.5388], Brier from 0.5634 [0.3515, 0.7862] to 1.5284 [1.4975, 1.5613], and `sqrt(RPS)` from 0.6378 [0.4903, 0.7706] to 1.9032 [1.8364, 1.9627]. The Pearson correlation of the head's expected grade with the item-mean grade on those 24 items is -0.4714 [-0.7508, -0.0402]. That is a small test of a filename hash, not a content model of the Lean files.

Leave-one-rater-out on the 13 held-out items with at least 3 ratings raises Pearson by 0.1283 [0.0718, 0.3564]. The Brier and cross-entropy differences against the add-one forecast have intervals that contain 0. On all 48 such items, Pearson rises by 0.0251 [0.0133, 0.0601] and cross-entropy of the floored point mass exceeds add-one by 3.0820 [1.4431, 4.7946]. Brier does not clear 0. Ranked probability score moves the other way: the point mass is lower (better) by 0.5107 [0.3622, 0.6378] in the order "point mass minus add-one," reported in the table as -0.5107 [-0.6378, -0.3622].

## Held-out proper scores of the cross-entropy head

Uniform cross-entropy is `log(K)`: 1.6094 on SummEval (5 classes), 1.0986 on ChaosNLI 3-way and DICES-990, 0.6931 on ChaosNLI abductive NLI, 1.7918 on FSC. Lower is better. Total variation distance (TVD) is the plug-in `0.5 * sum |q - p_hat|` against the item's own empirical distribution.

| Dataset | Test items | CE | Brier | RPS | sqrt(RPS) | TVD | Pearson of expected grade vs item mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SummEval expert | 1280 | 1.2442 [1.2055, 1.2801] | 0.6351 [0.6145, 0.6544] | 0.5764 [0.5530, 0.6009] | 0.7592 [0.7437, 0.7752] | 0.5149 [0.5043, 0.5240] | 0.2799 [0.2273, 0.3296] |
| SummEval crowd | 1280 | 1.3098 [1.2826, 1.3373] | 0.6993 [0.6903, 0.7095] | 0.4993 [0.4821, 0.5171] | 0.7066 [0.6943, 0.7191] | 0.5031 [0.4906, 0.5166] | 0.0237 [-0.0246, 0.0736] |
| ChaosNLI SNLI+MNLI-m | 624 | 1.0516 [1.0379, 1.0653] | 0.6359 [0.6274, 0.6446] | 0.3914 [0.3817, 0.4017] | 0.6256 [0.6178, 0.6338] | 0.3593 [0.3484, 0.3708] | 0.1207 [0.0442, 0.1907] |
| ChaosNLI abductive | 307 | 0.6961 [0.6924, 0.6996] | 0.5029 [0.4993, 0.5065] | 0.2515 [0.2496, 0.2532] | 0.5015 [0.4996, 0.5032] | 0.3782 [0.3628, 0.3922] | -0.0536 [-0.1618, 0.0554] |
| DICES-990 | 198 | 0.7426 [0.7069, 0.7777] | 0.4423 [0.4145, 0.4692] | 0.3948 [0.3691, 0.4198] | 0.6283 [0.6076, 0.6479] | 0.1798 [0.1641, 0.1961] | 0.1363 [0.0079, 0.2652] |
| VeriBench FSC export | 24 | 1.1245 [0.7249, 1.5686] | 0.5634 [0.3515, 0.7862] | 0.4068 [0.2404, 0.5939] | 0.6378 [0.4903, 0.7706] | 0.4524 [0.3565, 0.5520] | -0.4714 [-0.7508, -0.0402] |

The abductive head is within about 0.004 nats of the uniform value 0.6931. A two-class reversal is the same map as a one-step label roll, and it does not raise the proper scores there. On every other set, reversal raises cross-entropy, Brier, and `sqrt(RPS)`. On every set, including the abductive set, the sign flip and the orthogonal mix leave the canonical correlations unchanged to at most 4.2e-14 and make cross-entropy infinite because the realized class no longer has positive mass. Algebraic Brier of the raw sign-flipped vector is about 2.1 to 2.7, against original Brier between 0.44 and 0.70. The full grid is `tables/cca_invariance.md`.

## Collapse of within-prompt spread

Population statement, proved and checked exactly: on the two-prompt binary witness, squared correlation rises from 1/16 to 1/4 when the conditional variance is removed, Brier rises from 3/8 to 1/2, and the one-threshold ranked probability score rises from 3/16 to 1/4. The point mass puts zero probability on a class that still occurs, so cross-entropy is not finite.

Held-out half-splits, where both halves have at least two ratings:

| Dataset | Items | Pearson, collapsed mean minus raw | Brier, point mass minus add-one | CE, point mass minus add-one | RPS, point mass minus add-one |
| --- | --- | --- | --- | --- | --- |
| SummEval crowd | 1280 | 0.0805 [0.0592, 0.1013] p-val=< 0.0010 | 0.1030 [0.0700, 0.1370] p-val=< 0.0010 | 3.6146 [3.3243, 3.9063] p-val=< 0.0010 | -0.0221 [-0.0457, 0.0016] p-val=0.0700 |
| ChaosNLI 3-way | 624 | 0.2013 [0.1911, 0.2118] p-val=< 0.0010 | 0.2522 [0.2317, 0.2749] p-val=< 0.0010 | 3.9026 [3.7022, 4.1139] p-val=< 0.0010 | 0.1132 [0.1038, 0.1221] p-val=< 0.0010 |
| ChaosNLI abductive | 307 | 0.1598 [0.1439, 0.1779] p-val=< 0.0010 | 0.0623 [0.0485, 0.0753] p-val=< 0.0010 | 1.3941 [1.1989, 1.5814] p-val=< 0.0010 | 0.0311 [0.0243, 0.0386] p-val=< 0.0010 |
| DICES-990 | 198 | 0.2229 [0.1952, 0.2486] p-val=< 0.0010 | 0.5719 [0.4787, 0.6597] p-val=< 0.0010 | 5.8821 [5.2150, 6.5738] p-val=< 0.0010 | 0.2173 [0.1874, 0.2499] p-val=< 0.0010 |

SummEval expert has three ratings per item, so the half-split is empty. Leave-one-rater-out on 1280 test items raises Pearson by 0.0578 [0.0520, 0.0631] and Brier by 0.2081 [0.1626, 0.2532], and cross-entropy by 4.4677 [4.1100, 4.8012]. Ranked probability score, point mass minus add-one, is -0.1093 [-0.1426, -0.0761]. With only two predictor ratings, add-one spreads mass across all five grades, and the ordinal score can prefer a point mass at the mean. That comparison is not the population Bayes act in the Lean witness. On ChaosNLI and DICES-990, where the predictor half is large, Brier, cross-entropy, and ranked probability score all get worse together.

## What was machine-checked

Lean 4.36.0, `lake build`, no `sorry`, axioms only `propext`, `Classical.choice`, and `Quot.sound`. The package does not depend on Mathlib, so `lake exe cache get` reports `unknown executable cache` and Mathlib was not compiled. Theorems: `law_total_covariance`, `law_total_variance`, `spread_shrinks_sqcorr`, `quadratic_sample_cov_matches_zero`, `sqCorr_affine`, `cov_proj_affine`, `shear_preimage`, `sign_flip_preimage`, `swap_preimage`, `mix_preimage`, `cca_sq_spread`, `cca_sq_collapsed`, `cca_sq_strictly_increase`, `brier_point_aspect1`, `rps_point_aspect1`, `point_ce_not_finite_aspect1`, and the aspect-2 twins. Python `proofs_numeric.py` checks the same claims on 200 exact random mixtures and checks the false claim "the quadratic has nonzero covariance" by showing that it fails. Pytest runs that script twice with identical output.

Not machine-checked as a theorem: the decision to hash tokens, the imposed class order used by ranked probability score on ChaosNLI, and the claim that any particular human pair is linear. Those last numbers are the test-set estimates above.
