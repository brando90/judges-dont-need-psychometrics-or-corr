# How to read these results

**As of 11-26-2026.**

1. **Data.** Human numbers are held-out test prompts or tasks. The split is by group (SummEval: source article; ChaosNLI, DICES, POPQUORN: item/instance id; VeriBench FSC `R_v1` and `native_main`: task/task_folder so all samples of a task stay on one side), fractions 0.6/0.2/0.2, seed 0. Raters of one item are never placed on two sides of the split. A within-prompt oracle is labeled as such and is not a held-out judge. CertJudge panels (`E63`, `E39`, `Exp 67`) are read-only verified evaluations of real LLM judges from `cert-judge`.

2. **Metrics.** Cross-entropy is \(-\log \hat q(y)\) in nats, lower better, range \([0,\infty)\), reference \(\log K\) for a uniform forecast. One-hot Brier is \(\Vert\hat q-e_y\Vert_2^2\), lower better, range \([0,2]\). Ranked probability score (RPS) is \(\sum_{t=0}^{K-2}(F(t)-1_{y\le t})^2\), lower better, range \([0,K-1]\). The matrix column sqrt(RPS) is the square root of the unbiased mean of that score. Normalized RPS divides by \(K-1\) and is the project's primary proper score; it is in the tables next to sqrt(RPS). Pearson r, Spearman rho, and Kendall tau are higher-better association diagnostics.

3. **The test estimator.** For a proper loss the reported risk is the mean over test prompts of the mean over that prompt's ratings. A plug-in \(D_B(\hat p_{test}\Vert\hat q)\) is biased high by \((1/k_c)(1-\Vert p^\ast(c)\Vert_2^2)\) in expectation when \(\hat p\) is the empirical frequency. That plug-in is not the reported test risk.

4. **Differences.** A comparison is the same metric on the same split. "Held-out test CE of the Pearson-trained head minus held-out test CE of the CE-trained head" is positive when the CE-trained head is better.

5. **Intervals.** Percentile 95% intervals from an item-clustered bootstrap, B=1000 (B=40 only if `SANMI36_SMOKE=1`). The unit is the prompt/item. The interval is for this fixed trained head, not for retraining.

6. **How the head was chosen.** Signed 256-dimensional token hash, 24 principal components fit on train prompts only, plus an intercept. L2 penalty \(10^{-2}\) on non-intercept weights. Cross-entropy, Brier, RPS, and Pearson: L-BFGS-B from the zero vector, at most 35 iterations. Cross-entropy of a linear softmax is convex in the weights; the others are not, and the result is this start and this budget. MSE is ridge regression (closed form, global for that penalized linear problem) read out as the adjacent two-point distribution with the same mean. Validation did not choose these penalties.

7. **Direction and scale.** Proper scores: lower is better. Correlations: higher is better. A uniform forecast on K classes has cross-entropy \(\log K\).
