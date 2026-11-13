# How to read the results of experiment 04

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Three kinds of evidence: exact synthetic counterexamples (Phase A), a simulated test of decision-study design allocation (Phase B) and a retrospective check on old human ratings (Phase C). Reliability numbers say how repeatable a score is; RPS, CE and TVD say how well a forecast matches human ratings. Never read one as the other.

## Which data each number uses

| Phase | Data | Split / what is held out | Prior exposure |
| --- | --- | --- | --- |
| A (exact) | 11 synthetic artifacts with known rating distributions p_x = 0.9·Binomial(5, j/12) + 0.1·Uniform | Nothing is fitted, so there is no train/test split; every value is an exact expectation over the known distribution. A finite-sample Monte Carlo (4,000 replications) is labeled as such | None (synthetic) |
| B (simulation) | Gaussian crossed objects × raters × occasions model with known variance components (configs C1, C2) | Each of 256 replications calibrates on 64 objects × 8 × 8 and is evaluated on 256 **fresh** objects with fresh raters, occasions and residuals | None (synthetic) |
| C (retrospective) | Historical VeriBench judge study: 75 artifacts, 25 tasks, 2–3 expert ratings each; 5 cached judges × 3 repeat calls | Primary: the historical **test split**, 30 artifacts / 10 tasks / 78 expert ratings. Secondary: all 75 (in-sample for H1/H2 prompt development; validation was used for historical judge selection) | **Yes.** All ratings and the test split were used in earlier work. Read Phase C as a retrospective illustration, not fresh held-out evidence |

## Metrics

| Metric | Definition | Range / units | Better | Reference value |
| --- | --- | --- | --- | --- |
| RPS (primary distribution metric) | Ranked probability score: (1/5) Σ_{k=0}^{4} (Q(k) − 1{y ≤ k})², averaged over ratings within an artifact, then equally over artifacts | 0–1 | Lower | A: true distribution 0.1167 (not 0: humans disagree). C: constant train-marginal forecast 0.180 |
| RPS excess | Expected RPS of a forecast minus expected RPS of the true distribution | ≥ 0 | Lower | 0 for the true distribution |
| CE | Cross-entropy −ln q(y), no floor unless stated | nats, 0 to ∞ | Lower | A: true distribution 1.404. C: constant forecast 1.80; judges use the experiment 02 floor ε = 1e-6 |
| TVD | 0.5 Σ_y \|q(y) − p(y)\|: population p in A; observed expert frequencies in C (noisy, sparse) | 0–1 | Lower | 0 for the true distribution |
| MAE, bias | Mean absolute difference and mean signed difference (judge minus human) of mean ratings | 0–5 scale | MAE lower; bias near 0 (positive = lenient judge) | — |
| Eρ² | Relative generalizability coefficient v_u / (v_u + v_δ): how well scores **rank** objects across repeats/raters | 0–1 | Higher = more repeatable, **not** more accurate | C: two experts agree at 0.80 |
| Φ | Absolute dependability coefficient v_u / (v_u + v_Δ): also counts rater/occasion level shifts | 0–1 | Higher = more repeatable levels | Φ ≤ Eρ² always |
| v_δ, v_Δ (B) | Expected relative / absolute squared error of an object's averaged score against its universe score | score units² | Lower | C1 equal design 4×4: v_Δ 0.3875 |
| Pearson, Spearman, Kendall τ-b; ICC(A,1), ICC(C,1) | Agreement between judge mean and expert mean; ICC(A,1) also penalizes offsets | −1 to 1 | Higher | — |

"Undefined" means a zero denominator (for example a constant judge), never perfect reliability.

## What is subtracted

- **A1/A1_max, A2:** "excess" always means the same expected metric for a forecast minus that metric for the true distribution, on the same artifacts.
- **A1 finite sample:** the q-panel estimate minus the p-panel estimate, paired by replication (same sampled objects).
- **Phase B:** "chosen minus equal" is the held-out absolute error of the design chosen from calibration data minus the error of the equal 4×4 design **in the same replication**; negative means the chosen design is better. "Predicted minus true" is the calibration-estimated error minus the known population error of the chosen design.
- **Phase C:** "H0_opus minus H0_sol" is the test RPS (or Eρ², MAE) of the most repeatable judge minus that of the lowest-RPS judge, on the same task-bootstrap resample; positive RPS means the more repeatable judge predicts expert ratings worse. "Judge minus human Eρ²" compares a judge's test–retest repeatability with the two experts' inter-rater agreement: different facets (calls versus people), reported only as context.

## Uncertainty

- Phase A exact values carry no interval. The A1 finite-sample table uses Student-t 95% intervals over 4,000 replications (paired); SD ratios use a 2,000-draw replication bootstrap.
- Phase B: Student-t 95% intervals over 256 independent replications; Wilson 95% intervals for "chose the oracle design" counts. Formula checks use a 3.5-standard-error tolerance.
- Phase C: 95% percentile intervals over 10,000 task-bootstrap draws (2,000 for the all-75 set). Raters and repeat calls are held fixed, so intervals ignore rater and call sampling.
- No confirmatory hypothesis test was specified: p-val=n/a throughout. Post-hoc diagnostic reruns are labeled and never replace frozen results.

**TLDR-end:** [judge objectives: experiment 04] A and B are constructed or simulated evidence with known truth; C is a retrospective check on previously exposed ratings. Reliability coefficients measure repeatability; RPS, CE and TVD measure agreement with the human rating distribution.
