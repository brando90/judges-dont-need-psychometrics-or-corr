# 04: Reliability versus distribution accuracy — status (QA annotation 11-26-2026)

**QA correction (11-26-2026):** Current verified corrections, evidence limits and score-first companion panels are in [QA_CORRECTIONS_11-26-2026.md](QA_CORRECTIONS_11-26-2026.md). Original wording and scored tables below are retained as the historical record.

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/results.md>

**Status:** COMPLETE — Phases A, B and C ran on 11-13-2026. Not archived: reliability appendix added to the AAAI-27 workshop paper on 11-14-2026 (`9-appendix-reliability.tex`); main-body cross-reference remains optional.
**Last updated:** 11-14-2026.

Full report: [RESULTS_reliability_vs_accuracy.md](RESULTS_reliability_vs_accuracy.md). Reading guide: [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md). Plan: [PLAN.md](PLAN.md). Frozen protocol and predictions: [protocol.json](protocol.json), hashes in [FREEZE.json](FREEZE.json) (commit `579915a`, before any measurement; two pre-measurement code-only amendments plus freeze amendment 3 on 11-14-2026 adding declared secondary coverage).

| Phase | Status | Evidence |
| --- | --- | --- |
| Published source identification and reading | DONE | [SOURCES.md](SOURCES.md), [source_manifest.json](source_manifest.json) |
| Formula checks | DONE: 945/945 | `results/test_metrics.json` |
| A: exact counterexamples and controls | DONE: 14/16 predictions matched; 2 mismatches explained (sampling variability supported by a diagnostic; binomial mode ties) | `results/phase_a.json` |
| B: decision-study allocation (256 replications) and four assumption violations | DONE: 5/6 matched; 1 of 36 formula checks out (sampling variability supported by an 8,192-replication diagnostic rerun) | `results/phase_b.json`, `results/phase_b_*.csv` |
| C: retrospective historical ratings (test split, 30 artifacts / 10 tasks) | DONE: 1/1 matched; protocol's secondary three-expert analysis and 10,000 all-75 draws added 11-14-2026 (freeze amendment 3, primary unchanged) | `results/phase_c.json` (aggregates only; per-item data stays private) |
| Independent recomputation | DONE: 140/140 | `results/verification.json` |

**Result:** Matching means, variances and reliability coefficients does not identify the human-rating distribution. In the largest-gap exact construction (A1_max), all 11 artifacts tie on their first two conditional moments and population reliability coefficients, yet mean population total variation distance (TVD) is 0.446 and expected ranked probability score (RPS) excess is 0.0167. In the old judge data (test split, previously exposed), the most repeatable judge (`H0_opus`, one-call test–retest Eρ² 0.983) had the worst test RPS (0.273). Its paired gap against the lowest-RPS judge (`H0_sol`) is +0.082 [0.004, 0.168], from a 95% percentile interval over 10,000 task-bootstrap draws (most-repeatable judge minus lowest-RPS judge; positive means worse RPS). The pair was identified on these same test results and held fixed in the bootstrap, so the interval is descriptive and does not adjust for selection. Allocating rating effort by variance components chose the best design in 256/256 replications for both configurations. The in-model error-formula check passed 35/36 comparisons. Separate tests of correlated raters, heterogeneous task groups, deployment shift and dependent calibration occasions exposed prediction or allocation limits. The [reading guide](HOW_TO_READ_RESULTS.md#limits-and-clarifications-mega-qa-11-14-2026) records limits, clarifications and freeze amendment 3's secondary protocol coverage.

Zero model calls, zero new ratings, $0; local CPU, about 31 s in the initial 11-13-2026 run (about 104 s including freeze amendment 3's 10,000-draw secondary bootstraps). Synthetic results establish counterexamples and in-model behavior, not deployment frequencies.

**Google companion:** synced 11-13-2026. Section "Experiment 04 results: reliability is not distribution accuracy" added before References in [Trustworthy LLM Judges: Research Notes and Experimental Framing](https://docs.google.com/document/d/1WI-raRtTBnQETjkyMzg23PdhgDLgn33A9kQY1EU0zyE/edit); read back and verified (styles, bullets, links).

**TLDR-end:** [judge objectives: 04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026] Experiment complete: reliability and moment matching are insufficient for human-rating distribution accuracy, while variance-component design allocation works when its assumptions hold.
