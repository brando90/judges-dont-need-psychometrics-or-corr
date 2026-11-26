# How to read the results of experiment 04 (QA annotation 11-26-2026)

**QA correction (11-26-2026):** Current verified corrections, evidence limits and score-first companion panels are in [QA_CORRECTIONS_11-26-2026.md](QA_CORRECTIONS_11-26-2026.md). Original wording and scored tables below are retained as the historical record.

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Three kinds of evidence: exact synthetic counterexamples (Phase A), a simulated test of decision-study design allocation (Phase B) and a retrospective check on old human ratings (Phase C). Reliability numbers say how repeatable a score is; ranked probability score (RPS), cross-entropy (CE) and total variation distance (TVD) say how well a forecast matches human ratings. Never read one as the other.

## Which data each number uses

| Phase | Data | Split / what is held out | Prior exposure |
| --- | --- | --- | --- |
| A (population calculations) | 11 synthetic artifacts with known rating distributions p_x = 0.9·Binomial(5, j/12) + 0.1·Uniform; the separate offset control uses 5 artifacts | No train/test split: forecasts are constructed using the known population. Moment identities use exact rational arithmetic; CE and the tempered family use floating-point evaluation of population expectations. The finite-sample Monte Carlo study (4,000 replications) is separate | None (synthetic) |
| B (simulation) | Gaussian crossed objects × raters × occasions model with known variance components (configs C1, C2) | Each of 256 replications calibrates on 64 objects × 8 × 8 and is evaluated on 256 **fresh** objects with fresh raters, occasions and residuals | None (synthetic) |
| C (retrospective) | Historical VeriBench judge study: 75 artifacts, 25 tasks, 2–3 expert ratings each; 5 cached judges × 3 repeat calls | Primary: the historical **test split**, 30 artifacts / 10 tasks / 78 expert ratings. Secondary: all 75 (in-sample for H1/H2 prompt development; validation was used for historical judge selection); secondary three-expert human subsets: 18 test artifacts (6 tasks) and 45 of 75 artifacts (15 tasks) rated by all three experts (amendment 3) | **Yes.** All ratings and the test split were used in earlier work. Read Phase C as a retrospective illustration, not fresh held-out evidence |

## Metrics

| Metric | Definition | Range / units | Better | Reference value |
| --- | --- | --- | --- | --- |
| RPS (primary distribution metric) | Ranked probability score: (1/5) Σ_{k=0}^{4} (Q(k) − 1{y ≤ k})², averaged over ratings within an artifact, then equally over artifacts | 0–1 | Lower | A: true distribution 0.1167 (not 0: humans disagree). C: constant train-marginal forecast 0.180 |
| RPS excess | Expected RPS of a forecast minus expected RPS of the true distribution | ≥ 0 | Lower | 0 for the true distribution |
| CE | Cross-entropy −ln q(y), no floor unless stated | nats, 0 to ∞ | Lower | A: true distribution 1.404. C: constant forecast 1.80; judges use the experiment 02 floor ε = 1e-6 |
| TVD | Total variation distance: 0.5 Σ_y \|q(y) − p(y)\|, against population p in A and observed expert frequencies in C (noisy, sparse) | 0–1 | Lower | 0 for the true distribution |
| MAE, bias | Mean absolute error and mean signed difference (judge minus human) of mean ratings | Rating points: MAE 0–5; bias −5 to 5 | MAE lower; bias near 0 (positive = lenient judge) | — |
| Eρ² | Relative generalizability coefficient v_u / (v_u + v_δ): consistency of differences between objects across repeats/raters | Population 0–1; raw estimates need not obey this bound | Higher = more repeatable, **not** more accurate | C: two experts agree at 0.80 |
| Φ | Absolute dependability coefficient v_u / (v_u + v_Δ): also counts rater/occasion level shifts | Population 0–1; raw estimates need not obey this bound | Higher = more repeatable levels | Φ ≤ Eρ² for nonnegative population components and positive denominators |
| v_δ, v_Δ (B) | Expected relative / absolute squared error of an object's averaged score against its universe score | score units² | Lower | C1 equal design 4×4: v_Δ 0.3875 |
| Pearson, Spearman, Kendall τ-b; ICC(A,1), ICC(C,1) | Association between judge mean and expert mean; intraclass correlation ICC(A,1) also penalizes offsets while ICC(C,1) measures consistency | Correlations: −1 to 1; raw variance-component ICC estimates can fall outside population bounds | Higher | — |

"Undefined" means a zero denominator (for example a constant judge), never perfect reliability.
Raw analysis-of-variance (ANOVA) component estimates are retained in the finite-sample and historical analyses. For example, the historical test H0_opus call-variance estimate is negative, so its estimated Φ exceeds its estimated Eρ²; this does not contradict the population inequality.

## What is subtracted

- **A1/A1_max, A2:** "excess" always means the same expected metric for a forecast minus that metric for the true distribution, on the same artifacts.
- **A1 finite sample:** the q-panel estimate minus the p-panel estimate, paired by replication (same sampled objects).
- **Phase B:** "chosen minus equal" is the held-out absolute error (mean squared deviation from the object's universe score, expectation v_Δ) of the design chosen from calibration data minus the error of the equal 4×4 design **in the same replication**; negative means the chosen design is better. "Best minus chosen" (in V3-S2) subtracts the chosen design's held-out error from the shifted-population optimal design's error; negative means the chosen design is worse. "Predicted minus true" is the calibration-estimated error minus the known population error of the chosen design. "Predicted minus realized" (in V4) is the calibration-predicted Φ from dependent occasions minus the realized independent-occasion deployment Φ; positive means upward bias.
- **Phase C:** "H0_opus minus H0_sol" is the test RPS (or Eρ², MAE) of the most repeatable judge minus that of the lowest-RPS judge, on the same task-bootstrap resample; positive RPS means the more repeatable judge predicts expert ratings worse. "Judge minus human Eρ²" compares a judge's test–retest repeatability with the two experts' inter-rater agreement: different facets (calls versus people), reported only as context.

## How forecasts, designs and judges were chosen

- **Phase A (`A1_max`):** maximize a convex quadratic RPS objective over a polytope with three linear moment constraints and a 0.01 probability floor. Exact rational vertex enumeration covers all C(6, 3) = 20 triples of free categories after subtracting the floor (`scripts/check_counterexamples.py`). A convex function attains a maximum at some vertex of this compact polytope, so the exhaustive enumeration guarantees a global maximum; this is not a linear objective or a local optimizer. Other Phase A forecasts (`A1`, `A2`, `A3`) are closed-form constructions with zero fitted parameters.
- **Phase B:** estimate six nonnegative variance components and a residual component bounded below by `1e-12` on each 64 × 8 × 8 calibration panel. The objective is the balanced Gaussian restricted maximum likelihood (REML) negative log-likelihood, minimized by the limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm with bounds (`L-BFGS-B`). Each initial component is the larger of its raw ANOVA estimate and `1e-3 * MS_e`, where `MS_e` is the residual mean square. The limits are 20,000 iterations and 40,000 function evaluations, with `ftol=1e-15` and `gtol=1e-10`. The variance-scale objective is not globally convex. In all 256 base replications of each configuration, the raw ANOVA solution is feasible and attains each stratum's individual minimum at its observed mean square; this supplies a global-optimality certificate for those fits. Other fits reflect this optimizer, start and budget unless similarly checked. Design choice exhaustively evaluates the three candidate allocations at budget 16 (`8×2`, `4×4`, `2×8`), minimizing estimated absolute error variance on calibration data only.
- **Phase C:** no judge fitting in experiment 04; all five pre-existing cached prompt-by-model judges imported from experiment 02 are evaluated (`H0_sol`, `H1_sol`, `H2_sol`, `H0_opus`, `H2_opus`). Historically, `H1`/`H2` prompts were developed using the training split and validation was used in judge comparisons. The most-repeatable and lowest-RPS judges in the reported pair are identified on the same historical test data being summarized, then held fixed across bootstrap draws. This is a descriptive comparison, not a validated judge-selection procedure.

## Uncertainty

- Phase A exact values carry no interval. The A1 finite-sample table uses Student-t 95% confidence intervals (CIs) over 4,000 replications (paired); standard deviation (SD) ratios use a 2,000-draw replication bootstrap.
- Phase B: Student-t 95% intervals over 256 independent replications; Wilson 95% intervals for "chose the oracle design" counts. Formula checks use a 3.5-standard-error tolerance.
- Phase C: 95% percentile intervals over 10,000 task-bootstrap draws for every set (the all-75 set used 2,000 until freeze amendment 3). Raters, repeat calls and the selected pair of judges are held fixed. Intervals ignore rater/call sampling and are not adjusted for selecting the pair from these same test results. An interval excluding zero does not establish a confirmatory selection advantage.
- No confirmatory hypothesis test was specified: p-val=n/a throughout. Post-hoc diagnostic reruns are labeled and never replace frozen results.

## Limits and clarifications (Mega QA, 11-14-2026)

The Mega quality assurance (QA) reviewers' corrections are applied in the report generator and the regenerated `RESULTS_reliability_vs_accuracy.md`; these limits apply to its numbers:

- **Moment matching:** the A1 and A1_max forecasts match the first two conditional moments only. A1 changes the third raw moment by `6 * epsilon`; on this finite rating support, equality of all moments would identify the distribution. The sufficiency counterexample concerns the population second-moment components of the specified independent-given-artifact panels, not every possible reliability statistic or equality of finite-sample estimates.
- **Monte Carlo misses:** the frozen Phase B tolerance check passed 35/36 in-model comparisons. Its larger diagnostic rerun supports sampling variability as the explanation but does not convert the frozen miss into a pass or prove its cause. Similarly, the Phase A diagnostic supports a sampling explanation for its missed interval; those six intervals share panels and estimates, so no joint miss probability is computed. The reported large-sample standard deviation is an estimate, not a known true standard deviation.
- **Distribution forecasts and sampled histograms:** the figure's sharpening claim concerns moving from the true distribution toward sharper forecasts; moving from the flatter `T=2` forecast toward `T=1` improves RPS. The historical RPS evaluates a smoothed three-call histogram, not a judge's latent distribution known without sampling error.
- **Shared offset:** the pooled control samples rater types from a population with human/judge probabilities one half each, with type shared across artifacts for a rater. The offset contributes population rater variance `1/4`. This is not an ANOVA estimate from a fixed panel containing exactly one human and one judge.
- **Secondary protocol coverage:** the protocol's secondary three-expert human analysis was not in the frozen code; it was implemented during the Mega QA (freeze amendment 3), after the other results were seen, on its own random stream so the primary test-split results are unchanged. The same amendment raised the all-75 bootstrap from 2,000 to the protocol's 10,000 draws; its point estimates are unchanged.

**TLDR-end:** [judge objectives: experiment 04] A and B are constructed or simulated evidence with known truth; C is a retrospective check on previously exposed ratings. Reliability coefficients measure repeatability; RPS, CE and TVD measure agreement with the human rating distribution.
