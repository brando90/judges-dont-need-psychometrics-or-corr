# Cross-entropy versus correlation: the information gap

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/05_cross_entropy_vs_correlation_information_gap_start_13Nov2026/DERIVATION.md>

**TLDR:** Cross-entropy (CE) training is optimal only at the human rating distribution, and that optimum also maximizes correlation. The converse fails: correlation fixes only the mean of each forecast, so every distribution with the right mean is correlation-optimal. The gap is exactly the entropy deficit of the human distribution against the maximum-entropy distribution with the same mean.

Setting (Brando's Impromptu Call, 11-13-2026, action item 1). Scores y ∈ {0,…,K−1}, K = 6. For an item x, the human distribution is P_x; a judge forecasts q_x and its scalar rating is the forecast mean m_q(x) = E_{q_x}[Y]. Pearson and Spearman correlation read only these means.

## Result 1: CE is the log score, KL plus an irreducible entropy

CE(P, q) = −Σ_y P(y) log q(y) = H(P) + KL(P ‖ q). So CE is minimized, over all forecasts, only at q = P (Gibbs inequality), and the whole excess over H(P) is KL(P ‖ q). Used as a training loss against a one-hot human score (one rating per item), its expectation over raters is exactly this CE, so the learner is pushed towards P_x, not towards its mean.

## Result 2: correlation is maximized by many forecasts, CE by one

Over all functions m(X), Pearson's ρ(E[Y|X], m(X)) is maximal at m(X) = E[Y|X], up to an affine map (Pearson ignores scale and offset), with ρ² = Var(E[Y|X]) / Var(Y) = 1 − E[Var(Y|X)] / Var(Y). Any forecast whose mean is E_{P_x}[Y] attains it. For K = 6 that is a (K−2) = 4-dimensional family of forecasts per item, all correlation-optimal, including P_x. So:

- CE-optimal ⇒ correlation-optimal (q = P gives the right mean).
- Correlation-optimal ⇏ CE-optimal. Example below: a two-point forecast on {0, 5} with the human mean has Pearson 1.0 and CE 5.04 nats, against 1.40 nats for the true distribution (exact values, `results/gap_check.json`).
- The same holds for Spearman: only the ordering of means matters.

## Result 3: the information gap is an entropy deficit

Let q_m be the maximum-entropy distribution on {0,…,K−1} with mean m: q_m(y) ∝ exp(λ y), a tilted uniform. It is the only forecast the mean alone determines without extra assumptions. Because log q_m is affine in y and P and q_m share the same mean,

KL(P ‖ q_m) = H(q_m) − H(P) ≥ 0 (moment-matching identity; the Pythagorean identity for the exponential family with sufficient statistic y).

Define the information gap of an item as G(P) = H(q_{m(P)}) − H(P). Properties:

1. G(P) = the excess CE of the mean-only judge, in nats, and G(P) ≤ log K − H(P) ≤ log K.
2. G(P) = 0 iff P is itself a tilted uniform; any other shape (binomial, bimodal, skewed with the same mean) has G > 0.
3. Minimax: min over q of max over {P : mean(P) = m} of E_P[−log q] equals H(q_m) and is attained at q_m. So G is the regret of a judge that is told only the human mean, against an adversarial choice of shape (Topsøe, 1979; Grünwald and Dawid, 2004: maximum entropy is the minimax-optimal log-loss code).
4. Pinsker: total variation distance (TVD) between P and q_m is at most sqrt(G / 2).
5. Without the maximum-entropy choice the gap is unbounded: among forecasts with the correct mean, KL(P ‖ q) runs from 0 to ∞ (put near-zero mass on a point P supports).

## Result 4: when there is no gap

If the judge family is exponential with sufficient statistic y (softmax over logits affine in the score, θ_y = a·y + b), CE minimization is moment matching, so the CE-optimal member has the human mean and CE and Pearson select the same judge. The gap exists only for judges that can shape more than the mean (free logits, or a prompt that changes the spread). Checked numerically: best affine-softmax slope −0.4274 on artifact 3 gives mean 1.37494 against the human 1.37500.

## What this does not show

- It is a population-level statement about forecasts. It does not say a CE-trained judge generalizes better with 195 ratings; that is an empirical question (experiment 02: CE minus Pearson held-out log loss −0.26 [−1.46, 0.20], interval includes zero).
- Real raters are sparse (two or three per artifact), so P_x is estimated; G computed from two or three ratings is noisy and biased upward in entropy deficit.
- Spearman, Kendall and ICC differ in detail (ranks, offsets) but share the property that they do not read the shape of q.

## Sources

- Cover and Thomas, *Elements of Information Theory*, chapter 12 (maximum entropy; KL identity for matched moments), and chapter 2 (CE = H + KL).
- Topsøe (1979), "Information-theoretical optimization techniques", Kybernetika 15(1); Grünwald and Dawid (2004), "Game theory, maximum entropy, minimum discrepancy and robust Bayesian decision theory", Annals of Statistics 32(4). Cited from memory, not re-opened in this run.
- Experiment 00 (squashed judge: Pearson 0.877, KL 3.42) and experiment 04 construction A1 for the numbers below.
