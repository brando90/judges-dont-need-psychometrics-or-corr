# 05: information gap — results

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/05_cross_entropy_vs_correlation_information_gap_start_13Nov2026/results.md>

**Status, 11-13-2026:** COMPLETE for the derivation and its numerical check. Not archived: the real-data comparison (plan 06) and paper integration are open.

All numbers are exact computations on known distributions (no sampling, no held-out split, no ratings). Source: `results/gap_check.json`, produced by `gap_check.py` (seed 0 for random distributions).

| Check | Result |
| --- | --- |
| CE = H(P) + KL(P‖q), 2,000 random pairs (K = 6) | max abs error 3.6e-15 |
| KL(P‖q_m) = H(q_m) − H(P), 2,000 random P | max abs error 4.5e-13 |
| Gap G on 2,000 random P (Dirichlet, K = 6), nats | min 0.0018, mean 0.321, max 1.596 (log 6 = 1.792) |
| Minimax: E_P[−log q_m] = H(q_m) for P with mean ≈ 2.2 (42 draws within 0.02) | 1.7743–1.7783 against H(q_m) = 1.7763 (tolerance from the 0.02 mean window) |
| Experiment 04 construction A1 (11 artifacts, known p_x): mean G | 0.117 nats; item range 0.020 to 0.210; TVD to the maximum-entropy forecast 0.055 to 0.275 |
| A1, forecast = true distribution: Pearson of judge mean against human mean; mean CE | 1.000; 1.404 nats |
| A1, forecast = maximum entropy with the human mean: Pearson; mean CE | 1.000; 1.521 nats (excess = mean G = 0.117) |
| A1, two-point forecast on {0, 5} with the human mean: Pearson; mean CE; mean KL | 1.000; 5.039 nats; 3.635 nats |
| Experiment 00 (n = 2,000, seed 0): Pearson of noisy-faithful judge A; squashed judge B | 0.900; 0.877 (KL 3.42 nats in experiment 00) |
| Affine-softmax family on A1 artifact 3: CE-best mean against human mean | 1.37494 against 1.37500 (no gap inside the family) |

**Reading:** three forecasts with identical Pearson 1.000 against the human means differ by 0.117 to 3.635 nats of KL, so correlation cannot rank them and CE can. This is a population statement about forecasts, not evidence that a CE-trained judge is better on real ratings: experiment 02 found CE minus Pearson-trained held-out log loss −0.26 [−1.46, 0.20] (interval includes zero).

**Limitations:** K = 6 only; the maximum-entropy completion is one defined reference, not the unique correlation-optimal forecast; A1 is synthetic.

**TLDR-end:** [judge objectives: 05_cross_entropy_vs_correlation_information_gap_start_13Nov2026] Correlation fixes only the forecast mean; CE fixes the whole distribution; the gap is the entropy deficit against the maximum-entropy forecast (0.117 nats on average in the A1 construction).
