# 04: Reliability versus distribution accuracy — status

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/results.md>

**Status:** COMPLETE — Phases A, B and C ran on 11-13-2026. Not archived: paper integration and the Google companion remain open.
**Last updated:** 11-13-2026.

Full report: [RESULTS_reliability_vs_accuracy.md](RESULTS_reliability_vs_accuracy.md). Reading guide: [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md). Plan: [PLAN.md](PLAN.md). Frozen protocol and predictions: [protocol.json](protocol.json), hashes in [FREEZE.json](FREEZE.json) (commit `579915a`, before any measurement; two recorded code-only amendments).

| Phase | Status | Evidence |
| --- | --- | --- |
| Published source identification and reading | DONE | [SOURCES.md](SOURCES.md), [source_manifest.json](source_manifest.json) |
| Formula checks | DONE: 945/945 | `results/test_metrics.json` |
| A: exact counterexamples and controls | DONE: 14/16 predictions matched; 2 mismatches explained (Monte Carlo chance; binomial mode ties) | `results/phase_a.json` |
| B: decision-study allocation (256 replications) and four assumption violations | DONE: 5/6 matched; 1 of 36 formula checks out (skewed-error chance, confirmed by an 8,192-replication rerun) | `results/phase_b.json`, `results/phase_b_*.csv` |
| C: retrospective historical ratings (test split, 30 artifacts / 10 tasks) | DONE: 1/1 matched | `results/phase_c.json` (aggregates only; per-item data stays private) |
| Independent recomputation | DONE: 134/134 | `results/verification.json` |

**Result:** Matching means, variances and reliability coefficients does not identify the human-rating distribution. In the largest-gap exact construction (A1_max), all 11 artifacts tie on every moment and coefficient, yet mean population total variation distance (TVD) is 0.446 and expected ranked probability score (RPS) excess is 0.0167. In the old judge data (test split, previously exposed), the most repeatable judge (`H0_opus`, one-call test–retest Eρ² 0.983) had the worst test RPS (0.273). Its paired gap against the lowest-RPS judge (`H0_sol`) is +0.082 [0.004, 0.168], from a 10,000-draw task bootstrap. Allocating rating effort by variance components chose the best design in 256/256 replications for both configurations. It failed only under declared assumption violations: correlated raters, heterogeneous task groups, population shift and dependent calibration occasions.

Zero model calls, zero new ratings, $0; local CPU, about 31 s in total. Synthetic results establish counterexamples and in-model behavior, not deployment frequencies.

**Google synchronization:** pending; see [SOURCES.md](SOURCES.md) for the destination. No document write is claimed.

**TLDR-end:** [judge objectives: 04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026] Experiment complete: reliability and moment matching are insufficient for human-rating distribution accuracy, while variance-component design allocation works when its assumptions hold.
