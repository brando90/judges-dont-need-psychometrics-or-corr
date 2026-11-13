# 04: Reliability versus distribution accuracy — checkpoint

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/CKPT_g_theory_plan.md>

**TLDR:** All three phases ran on 11-13-2026 under a protocol frozen before measurement. Reliability and moment matching are insufficient for human-rating distribution accuracy; design allocation by variance components works when its assumptions hold. Remaining: paper integration.

Created: 11-13-2026 11:38 PST
Last updated: 11-13-2026 (execution session)

## Goal
- Test sufficiency of reliability and variance matching for human-rating distribution accuracy, separately from evaluation-design usefulness.
## What ran
- Planning: Pi / gpt-6-astra, xhigh (source reading, plan).
- Execution: Claude Code, claude-opus-5-5, desktop app; local CPU, about 31 s; zero model calls, zero new ratings, $0. Freeze commit `579915a`; two code-only amendments recorded in FREEZE.json.
## Results
- 20/23 frozen predictions matched; 3 explained mismatches (two Monte Carlo chance events confirmed by post-hoc reruns, one binomial mode tie). Formula checks 945/945; recomputation 134/134.
- A1_max: identical moments and coefficients, mean TVD 0.446, RPS excess 0.0167. Phase C test split: most repeatable judge H0_opus (Eρ² 0.983) worst RPS 0.273; paired vs H0_sol +0.082 [0.004, 0.168].
- Phase B: REML-guided allocation chose the oracle design 256/256 in both configs; violations V1–V4 behaved as predicted.
## Stuck / blocked
- None. Google companion synced 11-13-2026 (https://docs.google.com/document/d/1WI-raRtTBnQETjkyMzg23PdhgDLgn33A9kQY1EU0zyE/edit).
## Next steps
- [p 7/10] Decide whether to add a short reliability-vs-accuracy paragraph to the AAAI-27 workshop paper (its sources: RESULTS_reliability_vs_accuracy.md).
- [p 5/10] Harder Phase B condition (close configurations, few calibration levels) to measure allocation error rates with weak signals.
## Brando to-do
- None required.

## Log
11-13-2026 11:38 PST -- Created the public-theory plan, source page map and status record; excluded unpublished material and preserved all existing results. Experimental execution remains pending.
11-13-2026 -- Froze protocol and scripts (`579915a`), ran Phases A–C, verified (945 formula, 134 recomputation checks), generated report, figure and reading guide.
11-13-2026 -- Emailed verified results (Gmail id 1a1d688dcc090e5f, delivery checked) and synced the Google companion section.
