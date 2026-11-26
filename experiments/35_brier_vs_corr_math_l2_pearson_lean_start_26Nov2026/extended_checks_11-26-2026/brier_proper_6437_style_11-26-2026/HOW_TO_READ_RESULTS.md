# How to read the results of the Brier properness note (6.437 style), 11-26-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/brier_proper_6437_style_11-26-2026/HOW_TO_READ_RESULTS.md>

**Data.** Every number is analytic, exact rational arithmetic, or computed on synthetic distributions drawn with fixed seeds in `check_brier_proper_6437_style.py` (numpy `default_rng` seeds 1 to 7 and 100 to 104) and `make_figures.py` (seed 7). No human ratings are used, so no train, validation or test split applies and there is no prior exposure. The "true distribution" $p$ is either a fixed vector written in the script (for example $(0.2,0.5,0.3)$ or $(0.7,0.3)$) or a draw from a flat Dirichlet distribution.

**Metrics.**

- Brier score: $\sum_a (q(a)-\mathbb{1}\{x=a\})^2$, the squared distance from the report to the one-hot vector of the observed rating. Range 0 to 2. Lower is better. Its expected value under $p$ has floor $1-\lVert p\rVert^2$ (0.42 in the worked example).
- Excess expected Brier: expected Brier of $q$ minus expected Brier of $p$, both under the same $p$; equals $\lVert q-p\rVert^2$. Range 0 to 2. Lower is better; 0 means $q=p$.
- Log-loss: $-\log q(x)$ in nats. Range 0 to infinity. Lower is better. Its expected value has floor the entropy of $p$ (0.6109 nats in the worked example); the excess is $\mathrm{KL}(p\Vert q)$.
- Ranked probability score (RPS): $\sum_{k=0}^{K-2}(F_q(k)-\mathbb{1}\{x\le k\})^2$, the squared gaps of the cumulative distributions. The checks use this unnormalised form (range 0 to $K-1$); the repository's normalised RPS divides by $K-1$ (range 0 to 1). Lower is better. The identity is checked in both forms.
- $\mathbb{E}\lVert f-p\rVert^2$: mean squared Euclidean distance between the empirical frequency vector of $N$ synthetic ratings and $p$, compared against the reference $(1-\lVert p\rVert^2)/N$; the printed ratio should be near 1.
- "Max gap": the largest absolute difference between a quantity computed from its definition and the closed-form identity, over all random instances; values near $10^{-15}$ are floating-point rounding.

**What is subtracted.** Every difference subtracts the same metric on the same distribution $p$: the expected score of a report $q$ minus the expected score of the truthful report $p$. No two different metrics are ever subtracted. Figure 1 shows Brier and log-loss on separate axes and does not compare their values.

**Uncertainty.** Only check (iv) is Monte Carlo: for each $N$ it uses 5 seeds times 2,000 multinomial replications, and prints the ratio per seed so the spread is visible (seed ratios range from 0.978 to 1.037). The pass threshold is a pooled ratio within 5 percent of 1. No confidence intervals are claimed. All other checks are exact identities or exhaustive comparisons.

**How the report $q$ was chosen.** No judge is trained. Reports are either written by hand (the worked example and the counterexamples), drawn at random (Dirichlet draws and tangent perturbations), or found by exhaustive search over a grid on the $K=3$ simplex (step 1/200 for check iii, step 1/400 for check vii), which is guaranteed to return the best grid point. The water-filling minimiser in check (vii) is the analytic Karush-Kuhn-Tucker solution, found by bisection on its threshold $c$ and cross-checked against the grid.
