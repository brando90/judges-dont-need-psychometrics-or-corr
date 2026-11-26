# Protocol — correlation, first moments, and canonical correlation

Frozen before the real-data measurement in this folder. Toy checks do not use these splits.

## Question

Across prompts, is Pearson correlation a first-moment functional, or a linear second-moment functional? Within a prompt, what does conditional independence imply for the covariance? Does canonical correlation analysis (CCA) repair that, or does it stay invariant to the distortions that make a proper score useless?

## Data

- SummEval expert and crowd, four aspects (coherence, consistency, fluency, relevance), grades 1–5. Split group is the source article.
- ChaosNLI SNLI pooled with MNLI-matched (3 classes) and ChaosNLI abductive NLI (2 classes). Each item is its own group. Ranked probability score uses the imposed orders already stated in experiment 36.
- DICES-990 `Q_overall`, quality-filter raters removed as in experiment 36. Order No < Unsure < Yes.
- VeriBench functional semantic correctness (FSC) human ratings from trustworthy-theorem-judges experiment 09, column `final_score_0_5`, integers 0–5. Split group is `task_folder`. The file name `R_v1` and the 343-item, about-5-experts description from experiment 36 are recorded as found or not found. No substitute table is invented.

Prompt-level split, seed 37, fractions 0.60 / 0.20 / 0.20. Validation is not used to choose a hyperparameter. Test groups are unused in the fit.

## Judge

Cross-entropy softmax on 8 principal components of a 64-dimensional signed token hash, plus an intercept. The principal components are fit on train prompts only. L2 penalty 0.01 on the non-intercept weights. L-BFGS-B from the zero vector, at most 200 iterations. The logit loss is convex; if the iteration cap binds, the head is that run's result and is not claimed to be the exact minimizer.

## Metrics

Unbiased test risk: for each test prompt, average the loss over that prompt's ratings, then average prompts. Losses: cross-entropy (nats, lower better), Brier (lower better), ranked probability score (RPS, lower better), and the square root of that RPS. Plug-in total variation distance (TVD) is `0.5 * sum |q - p_hat|` against the item's own empirical distribution, labeled as plug-in. Pearson and Spearman are secondary.

Uncertainty: percentile 95% interval, item (prompt) resample, B = 1000, seed 37. A two-sided percentile bootstrap p-value for a difference uses the null value 0 on those same draws.

## Adversarial maps, fixed before looking at test losses

Applied to the fitted probability vector on test prompts, human view held fixed:

1. Reversal of the class axis (an ordinal sign flip that stays on the simplex).
2. A one-step roll of the class labels (a permutation matrix).
3. `A q + b` with `A` a random orthogonal matrix from a Gaussian QR (seed 37) and `b` a standard normal vector. This need not be a probability. Cross-entropy is infinite when the coordinate of the realized class is not positive. Algebraic Brier of the raw vector is still reported.

CCA is computed on the raw transformed vectors, before any repair.

## Within-prompt collapse

Half-split, seed 37: items with at least 4 ratings, both halves size at least 2. Leave-one-rater-out, same seed: items with at least 3 ratings, used when a half-split is not available (expected for FSC and for SummEval expert). The distributional forecast is an add-one probability mass function of the predictor half. The collapsed forecast is a point mass at the nearest class to that half's mean, floored at `1e-6` so cross-entropy is finite. Both are scored on the held-out half.

## Not done

No new human ratings. No model API calls. No GPU. Synthetic checks are labeled synthetic.
