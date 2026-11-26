# Experiment 35: Brier score, scalar L2, and Pearson correlation

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/README.md>

**TLDR:** On seeded synthetic laws, the Brier score is strictly proper and its empirical minimizer is the vector of frequencies, while scalar squared error returns only the mean. Pearson correlation is unchanged by a positive affine map, so it ties forecasts whose mean squared error (MSE), Brier score, and ranked probability score (RPS) differ. `BrierCorr.lean` proves the nine algebraic theorems in Lean 4.36.0 with no Mathlib and no `sorry`.

**Status:** verification complete (11-26-2026). No human ratings and no model calls. Cost `$0`. Read `HOW_TO_READ_RESULTS.md` before quoting a number. The four-panel figure is `figures/brier_vs_corr.png`.

**Google Doc:** <https://docs.google.com/document/d/1AkNeSm86_bk1twd_w-TidjOHqx0gOv1lRI7WS2ADO_8/edit>

## Derivation

The notebook derivation is in [DERIVATION.md](DERIVATION.md): standardized squared error equals \(2(1-\operatorname{Corr}(X,Y))\), unstandardized mean squared error (MSE) adds the squared bias and the squared scale gap, and excess Brier risk is the squared distance between distributions. That note names the Lean theorems `standardized_mse_corr`, `mse_decomposition`, `pearson_affine_invariant`, `brier_decomposition`, and `binary_brier_eq_two_l2`.

## Question

When the target is a conditional rating distribution `p*(y | x)`, which functionals identify it? This folder checks three facts that the training experiments use:

- The Brier score is strictly proper: its population minimizer is `p*` itself, and the excess risk is the squared Euclidean distance `||q - p*||^2`.
- Scalar L2 loss, `E[(Y - c)^2]`, is minimized at the mean for every distribution. A bimodal ordinal law and a binary law make the distinction concrete. On a binary outcome the mean is the distribution, and the two-class Brier score is exactly twice the scalar squared error. On `{0, 1, 2, 3, 4, 5}` the mean is not the distribution.
- Pearson correlation is invariant under `f -> a f + b` for `a > 0`. Two ordinal forecasts with the same conditional mean can therefore share a correlation and a scalar MSE while their Brier scores and RPS differ.

## Run the numerical checks

From the repository root, with Python 3, NumPy, SciPy, and Matplotlib:

```bash
python3 experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/experiments.py
```

The script resolves paths from `__file__`. It writes `results.txt` and `figures/brier_vs_corr.png` in this directory. The generator is `numpy.random.default_rng(0)`. This run used Python 3.12.3, NumPy 2.5.5, SciPy 1.18.1, and Matplotlib 3.11.2. Every numeric line matched the attached stdout stored in `assets/results_source_attachment_11-26-2026.txt`. The only stdout change is the figure path, now `figures/brier_vs_corr.png`.

The attached originals are kept unchanged under `assets/`:

- `assets/experiments_source_attachment_11-26-2026.py`
- `assets/results_source_attachment_11-26-2026.txt`
- `assets/brier_vs_corr_source_attachment_11-26-2026.png`

`experiments.py` is the runnable script. It adds path handling, `ax.grid(True)` on every panel (faint minor gridlines on the log-log panel), and axis labels that name the metric. The random draws and the printed arithmetic are the attached script's.

## Verify the Lean proofs

Toolchain pin: `lean-toolchain` contains `leanprover/lean4:v4.36.0`. No Mathlib and no `lakefile`. From this directory:

```bash
elan run leanprover/lean4:v4.36.0 lean BrierCorr.lean
```

Checked here with Lean 4.36.0 (`x86_64-unknown-linux-gnu`, commit `611c117bfaf7`, Release). Exit code 0. The captured stdout is `lean_check.txt`. Each of the nine theorems depends only on `propext`, `Classical.choice`, and `Quot.sound`. The file contains no `sorry`.

The proofs are the attached file, plus two `#print axioms` lines for `brier_proper` and `l2_decomposition` so the receipt lists all nine theorems. No proof term was edited.

| Theorem | Statement |
| --- | --- |
| `brier_decomposition` | If the coordinates of `p` sum to 1, then `E_p[BS(q)] - E_p[BS(p)] = ||q - p||^2`. |
| `brier_proper` | `p` minimizes expected Brier score. |
| `brier_strictly_proper` | Expected Brier score equals the Bayes value if and only if `q = p` coordinatewise. |
| `l2_decomposition` | `E[(Y - c)^2] = E[(Y - mu)^2] + (mu - c)^2`, with `mu = E[Y]`. No Gaussian assumption. |
| `l2_mean_minimizer` | The scalar L2 risk is minimized if and only if `c = mu`. |
| `binary_brier_eq_two_l2` | `(q - y)^2 + ((1 - q) - (1 - y))^2 = 2 (q - y)^2`. |
| `standardized_mse_corr` | If `sum_i x_i^2 = sum_i y_i^2 = n`, then `(1/n) sum_i (x_i - y_i)^2 = 2 (1 - (1/n) sum_i x_i y_i)`. |
| `mse_decomposition` | MSE `= (mean x - mean y)^2 + (s_x - s_y)^2 + 2 s_x s_y (1 - rho)`, with `s_x` and `s_y` positive square roots of the two variances. |
| `pearson_affine_invariant` | For `a > 0`, Pearson correlation of `a x + b` with `y` equals Pearson correlation of `x` with `y`. |

Standard deviations are positive rationals whose squares equal the variances, because this development does not use a square root on `Rat`.

## Numerical takeaways

All figures below are copied from `results.txt`. Differences of two forecasts use the printed precision. `p-val=n/a`. Definitions and the in-sample versus population distinction are in `HOW_TO_READ_RESULTS.md`.

**Part 1. Strict properness on one fixed synthetic law.** The decomposition `E[BS(q)] = E[BS(p*)] + ||q - p*||^2` holds for six forecasts, with absolute gaps `0` or about `1e-16`. Softmax BFGS returns a forecast whose maximum absolute coordinate error against `p*` is `1.9e-08`. Over `200` replications, the mean L2 distance of the empirical frequencies to `p*` falls from `0.26256` at `N = 10` to `0.00246` at `N = 100,000`, against the multinomial reference `sqrt(sum_k p_k (1 - p_k) / N)` of `0.27386` and `0.00274`. Population expected Brier of those frequencies moves from `0.82840` to `0.75001`, toward the Bayes value `0.75`. The in-sample Brier at `N = 10` is `0.66950`, below the Bayes value, because the frequencies are scored on the sample that defined them.

**Part 2. The mean is not an ordinal distribution.** For `P(Y = 0) = P(Y = 5) = 1/2`, the L2 minimizer is `2.5`, the variance is `6.25`, and the Brier minimizer is `(0.5, 0, 0, 0, 0, 0.5)`. Reading `2.5` as a two-point law on `{2, 3}` puts all mass where `p*` is zero and raises the Brier score from `0.5000` to `1.5000`. On `100,000` draws the sample mean is `2.4838` and the Gaussian maximum likelihood fit (sample mean and `ddof = 0` standard deviation) is `mu = 2.484`, `sigma = 2.500`, so that Gaussian density peaks at a point with true probability zero. For binary outcomes the Brier/L2 ratio is `2` at every printed `(q, y)`, and the L2 minimizer at `P(Y = 1) = 0.3` is `0.300000`.

**Part 3a. One seeded sample of `n = 10,000` pairs.** Sample correlation `0.492509`. Standardized identity gap `2.2e-16`. MSE decomposition gap `0`. Printed terms: squared bias `2.3793`, squared scale gap `0.1882`, correlation term `4.9551`.

**Part 3b. Two hundred synthetic items, one seeded draw.** The conditional mean, `0.1 E[Y|x] + 4`, and `5 E[Y|x] - 10` all have Pearson correlation `1` with `E[Y|x]` and the same correlation with one rating (`0.4281537449`) and with twenty ratings (`0.3642743739`). Their population expected MSE values are `2.5526`, `6.0788`, and `9.2940`. Expected MSE of `0.1 E[Y|x] + 4` minus expected MSE of `E[Y|x]` is `3.5262`. The same difference for `5 E[Y|x] - 10` is `6.7414`. Positive means the affine forecast has larger squared error. Brier and RPS in this panel score the two-point distribution with that scalar mean, clipped to `[0, 5]`, not a full conditional distribution. A judge that outputs `p*(y | x)` has expected Brier `0.7184` and unnormalized expected RPS `0.8447`. The printed population ceiling `sd(E[Y|x]) / sd(Y) = 0.3758` is the maximum correlation of any forecast of `x` with a fresh rating. The one-sample correlation `0.4281537449` is one finite draw and can sit above that ceiling.

**Part 3c. Same conditional mean, different spread.** On the next `200` items from the same generator, the true conditional distribution and its two-point collapse have identical means (maximum absolute gap `0`). Printed correlation with one rating is `0.385696` for both, and population expected MSE is `2.9474` for both. Expected Brier is `0.6058` versus `1.4706`. Expected unnormalized RPS is `0.8692` versus `1.4007`. On one item that is uniform on `{0, 4}`, the true forecast has Brier `0.500` and unnormalized RPS `1.000`; a point mass at the shared mean `2` has Brier `2.000` and unnormalized RPS `2.000`.

## What this does not show

These checks do not train a network, do not select a judge on validation data, and do not use human ratings. They do not estimate a finite-sample win rate for cross-entropy against Pearson training. That comparison is in the numbered training experiments indexed from `experiments/README.md`. The unnormalized RPS in this script must not be quoted as the project's normalized individual-rating RPS.
