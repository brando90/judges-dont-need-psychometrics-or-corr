# 04: Reliability versus distribution accuracy — checkpoint

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/CKPT_g_theory_plan.md>

**TLDR:** All three phases ran on 11-13-2026 under a protocol frozen before measurement. Reliability and moment matching are insufficient for human-rating distribution accuracy; design allocation by variance components works when its assumptions hold. Remaining: optional main-body cross-reference to the 11-14-2026 AAAI-27 reliability appendix.

Created: 11-13-2026 11:38 PST
Last updated: 11-14-2026 (Mega quality assurance (QA) stage 3)

## Goal
- Test sufficiency of reliability and variance matching for human-rating distribution accuracy, separately from evaluation-design usefulness.
## What ran
- Planning: Pi / gpt-6-astra, xhigh (source reading, plan).
- Execution: Claude Code, claude-opus-5-5, desktop app; local CPU, about 31 s initial run (about 104 s with amendment 3); zero model calls, zero new ratings, $0. Freeze commit `579915a`; three amendments recorded in FREEZE.json (two pre-measurement code-only, one secondary-coverage addition).
## Results
- 20/23 frozen predictions matched; 3 explained mismatches (two Monte Carlo misses consistent with sampling variability in post-hoc reruns, one binomial mode tie). Formula checks 945/945; recomputation 140/140.
- A1_max: identical first two conditional moments and population coefficients, mean total variation distance (TVD) 0.446, ranked probability score (RPS) excess 0.0167. Phase C test split: most repeatable judge H0_opus (Eρ² 0.983) worst RPS 0.273; paired vs H0_sol +0.082 [0.004, 0.168].
- Phase B: restricted maximum likelihood (REML) allocation chose the oracle design 256/256 in both configs; violations V1–V4 behaved as predicted.
## Stuck / blocked
- None. Google companion synced 11-13-2026 (https://docs.google.com/document/d/1WI-raRtTBnQETjkyMzg23PdhgDLgn33A9kQY1EU0zyE/edit).
## Next steps
- [p 7/10] Decide whether to add a main-body cross-reference in the AAAI-27 workshop paper to `9-appendix-reliability.tex` (added 11-14-2026 from `RESULTS_reliability_vs_accuracy.md`).
- [p 5/10] Harder Phase B condition (close configurations, few calibration levels) to measure allocation error rates with weak signals.
## Brando to-do
- None required.

## Log
11-13-2026 11:38 PST -- Created the public-theory plan, source page map and status record; excluded unpublished material and preserved all existing results. Experimental execution remains pending.
11-13-2026 -- Froze protocol and scripts (`579915a`), ran Phases A–C, verified (945 formula, 134 recomputation checks), generated report, figure and reading guide.
11-13-2026 -- Emailed verified results (Gmail id 1a1d688dcc090e5f, delivery checked) and synced the Google companion section.
11-14-2026 -- Added AAAI-27 reliability appendix (`03ada1e`) and ran Mega QA stage 1 (`qa1_carbon`: clarified Phase B prediction-vs-selection wording, fixed `4-proposal.tex` sentence-per-line check, and expanded first-use acronyms in figure/report/guide).
11-14-2026 -- Mega QA stage 2 (`qa2_codex_vals`, Codex CLI / gpt-6-astra / ultra): reproduced all required checks, preserved frozen outputs, corrected appendix/statistical interpretation and stage 1 mathematical-report errors; see `qa/mega_qa_11-14-2026/qa2_codex_vals.md`.
11-14-2026 -- Builder, between Mega QA stages 2 and 3: freeze amendment 3 adds the protocol's secondary three-expert analysis and 10,000 all-75 draws (primary test results byte-identical); report generator wording corrected per QA2-F1/F2/F3; verification 140/140.
11-14-2026 -- Mega QA stage 3 (`qa3_carbon`, Antigravity CLI / Gemini 4 Carbon (High) / high): verified all scripts, Phase C primary invariance across `be39e2a`, 945/945 formula and 140/140 recomputation checks; synced post-amendment-3 status/anchors/acronyms and V4 subtraction gloss; see `qa/mega_qa_11-14-2026/qa3_carbon.md`.
