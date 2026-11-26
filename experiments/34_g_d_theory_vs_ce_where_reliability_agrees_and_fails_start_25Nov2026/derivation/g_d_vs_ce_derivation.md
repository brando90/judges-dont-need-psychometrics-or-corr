# What generalizability coefficients score, and what cross-entropy scores

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026/derivation/g_d_vs_ce_derivation.md>

**TLDR:** The generalizability coefficient and the one-source dependability coefficient depend only on item means and variances. Cross-entropy and ranked probability score (RPS) are minimized at the true conditional distribution. Typeset derivation: [g_d_vs_ce_derivation.pdf](g_d_vs_ce_derivation.pdf).

**As of:** 11-25-2026.

## What the result does and does not show

The coefficients do not identify a six-category conditional distribution. They can rank calibrated noisy sources in the same order as a proper score, and a decision study can still allocate raters. The numbers below are the synthetic population values and Monte Carlo means from experiment 34, not human-rating estimates.

## Definitions

Ratings are in {0, 1, 2, 3, 4, 5}. The normalized individual-rating RPS is

RPS(q, y) = (1/5) Σ_{k=0}^{4} (F_q(k) − 1{y ≤ k})²,

with F_q the forecast cumulative distribution. Range 0 to 1. Lower is better. The excess over the true distribution is (1/5) Σ_k (F_q(k) − F_p(k))², so the expectation is minimized uniquely at q = p. Cross-entropy E[−ln q(y)] has the same unique minimizer among positive forecasts, and the excess is KL(p ‖ q).

With item means m_i and variances v_i, and divisor N,

v_u = (1/N) Σ (m_i − mean)²,    v_e = (1/N) Σ v_i,    Eρ² = v_u / (v_u + v_e).

One-source Φ equals Eρ². A point mass has v_e = 0, so Eρ² = Φ = 1 when the locations are not all equal. In a two-source panel, Φ also divides by the source main effect, so a constant level shift gives Φ < Eρ². The judge-only coefficient does not include that source effect.

## Three calculations

**Binomial.** The mean identifies Binomial(5, θ). At the true mean the projection's expected RPS equals the oracle, 0.102440. The pooled frequency scores 0.178049. At 4 ratings, projection minus add-one RPS is −0.017067 [−0.017622, −0.016512]. At 256 ratings it is −0.000187 [−0.000194, −0.000181].

**Leniency.** The point mass one category above the rounded mean has population RPS 0.226209, bias +1.011905 (means 3.333333 and 2.321429), Pearson 0.974928, and judge-only coefficient 1. The pooled frequency scores 0.178049. At 16 ratings the RPS gap is +0.049183 [0.047743, 0.050622]. The joint human-and-judge coefficients fall to Eρ² 0.764852 and Φ 0.681426.

**Same moments.** The vertex of the floor-1/100 same-moment polytope with the largest RPS excess has mean excess 0.020576 (shaped truth 0.083473, matching binomial 0.104049). The coefficients match. At 256 ratings, projection minus add-one is +0.020508 [0.020476, 0.020541] in RPS and +0.689846 [0.689685, 0.690007] in smoothed cross-entropy.

**Six raters.** Population own-source coefficients: neutral 0.617978, lenient 0.587170, harsh 0.536109, lenient point mass 1. Expected RPS against the mixture: neutral 0.143804, constant 0.191923, lenient facet 0.288927, point mass 0.347638.

**TLDR-end:** [judge objectives: experiment 34 derivation] Coefficients use means and variances; proper scores use the conditional distribution; a point mass can score 1 on reliability and worse than a constant forecast on RPS.
