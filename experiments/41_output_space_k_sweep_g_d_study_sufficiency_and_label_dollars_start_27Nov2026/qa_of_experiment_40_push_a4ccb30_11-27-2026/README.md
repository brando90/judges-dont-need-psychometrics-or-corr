# QA of Cursor's Experiment 40 push (commit a4ccb30)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/qa_of_experiment_40_push_a4ccb30_11-27-2026/README.md>

**TLDR:** An independent reviewer (Claude Opus at maximum effort, about 25 minutes, CPU only) re-ran Cursor's Experiment 40 (`experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/`). The code reproduces exactly, the six copied sister-repository files are byte-identical to their source, no rater name or identifier leaks, and the four general scoring-rule theorems (Brier and ranked probability score decomposition and strict properness) are real proofs with only the standard axioms. The headline claims do not hold: the one "detectable misallocation" (VeriBench, budget 64) stops being significant once resampled by task or corrected for multiple comparisons, the "agreement once the budget is large" paragraph is a hard-coded string that contradicts the folder's own results file, the dollar frontier is met by a constant forecast that ignores the text, and the Lean "G-theory blindness" witness uses a table cell that no rating law on its own scale can have.

**As of:** 11-27-2026 (Friday, Pacific time). Local CPU, $0, no model calls.

## What is in this folder

```text
REVIEW_by_independent_opus_subagent_11-27-2026.md   the full review: verdict, eleven findings (severity, location, evidence, fix), what was verified OK, what could not be checked, comparison with Experiment 41
scripts/                                            the reviewer's fifteen check scripts and one Lean file (ReviewChecks.lean)
outputs/                                            their saved outputs
```

The two run logs in `outputs/` end in `.txt` (the saved reviewer output was `.log`; this repository's root `.gitignore` drops `*.log`), and the one pointer to `outputs/06_run_repro` in the review text was updated to match.

The scripts contain hard-coded paths of the review machine (`/tmp/e40_cursor/...`, `/tmp/opus_review_exp40c/...`) and need a checkout of commit a4ccb30 plus the sister-repository files at the pinned commits. They are evidence of what was run, not a turnkey reproduction.

## The findings in one line each (numbering follows the review)

1. **Blocker: the VeriBench budget-64 headline is not robust, and "sealed" is wrong.** The 40 test artifacts come from 10 tasks, and resampling by task gives an interval of minus 0.028 to 0.170 that covers zero. The two-sided p-value is 0.039 for this cell, 0.83 after Holm correction over 41 comparisons and 0.197 over the 5 largest-budget ones. Across 200 alternative training draws the difference has a standard deviation of 0.056, larger than the interval's own standard error. The "equal" budget is not equal (54 ratings for the oracle cell, 64 for the D-study cell, 40 for k = 1). The same test prompts were already used in sister-repository Experiments 27 and 28.
2. **Blocker: the "agreement" paragraph contradicts the results file and is hard-coded.** 21 of 47 intervals exclude zero, all with the D-study allocation worse, and 13 survive Holm correction. At budgets of 256 and 512 the spread of test cross-entropy across k is 0.006 to 0.087 nats, not "a few thousandths". The budget-4096 rows are not equal-budget comparisons (some k use 594 of 4,096 ratings). The executive text is a string literal in `run_all.py`.
3. **Important (blocker for paper use): the dollar frontier prices a base rate, not a judge.** The head beats a constant train-marginal forecast by 0.004 to 0.032 nats on five public sets and is 0.0008 nats worse on SummEval crowd, all below the 0.05-nat tolerance; a constant forecast from 32 ratings meets the tolerance at budget 32 on five sets. 12 of the 14 dollar thresholds equal the smallest grid budget, so they are left-censored.
4. **Important: the Lean G-theory witness does not formalize what the README says.** Its G coefficients are closed constants, not functions of the two laws, so "same coefficients" holds by construction. Cell (1,1) has mean 1/4 and variance 5/4, and no law on 0 to 3 can have both (the largest variance is 11/16; the reviewer proved this in Lean as `cell11_infeasible`). The cross-entropy step is numerics only. `dstudy_depends_only_on_k` proves two numbers.
5. **Important: the real-data rebin table is mostly infinite.** The squared-error head's cross-entropy is infinite in 52 of 69 rows, and 8 rows still print a "95% interval" built from 11 to 390 of the 1,000 bootstrap replicates.
6. **Important: the synthetic K = 2 crossover comes from a bug in the competitor.** Its Gaussian spread is estimated from k-rating means, so it shrinks from 26.3 to 5.1 as k grows. With the spread of individual ratings, all five K = 2 verdicts flip, and for K of 3 or more the gaps shrink 1.3 to 4.5 times.
7. **Important: the "not recomputable" note on the copied GPU results is false.** The copied rating files hold all 175 items; normalized RPS recomputes exactly (0.1482 and 0.2145), while the quoted cross-entropy (3.589 and 5.375) differs from the raw mean of minus log q (11.97 and 57.60), so an undocumented smoothing applies.
8. **Important: the intervals under-cover.** They resample test prompts with the head held fixed, while SummEval and VeriBench split by article or task; SummEval crowd at budget 512 goes from [0.0065, 0.0229] by item to [minus 0.0022, 0.0315] by article.
9. **Minor: figures and captions.** The allocation figure drops capped cells (including the k = 10 cell that is both the D-study pick and the oracle on POPQUORN offensiveness), the Pareto figures do not say that the y-axis is a test-chosen minimum over k, the k\* figure is a flat line, and no figure has a caption.
10. **Minor: provenance.** "Three scores" is wrong for 30 of the 75 old rows, and there is no sha256 receipt for the six copied files.
11. **Minor: process.** The protocol and results arrived in one commit, so "written before" cannot be checked; the $15,000 to $75,000 researcher range has no source; eight fits stop at the 35-iteration cap; one test assertion always passes.

## What I re-checked myself (the rest rests on the reviewer's saved scripts and outputs)

1. **Interval counts (finding 2):** from Cursor's `results.json`, 47 D-study-minus-oracle intervals, 6 of them the same cell (an automatic interval of zero to zero), 21 excluding zero, all 21 with the D-study allocation worse. Matches the review.
2. **The impossible cell (finding 4):** the largest variance of any law on the interval from 0 to 3 with mean 1/4 is (3 minus 1/4) times (1/4) = 11/16, and the witness table needs 5/4; the cell definitions are in `theory/GTheory.lean` lines 495 to 523. I also recompiled the reviewer's Lean file on my own toolchain (core Lean 4.36.0 on a copy of Cursor's `GTheory.lean`): `cell11_infeasible`, `cell11_second_moment_bound`, `rps4Terms_eq_rps` and `cell11_table_values` all compile and depend only on `propext`, `Classical.choice` and `Quot.sound` ([output](outputs/ReviewChecks_recompiled_by_me_11-27-2026.txt)). Matches the review.
3. **Infinite cross-entropy rows and the GPU-bridge recomputation (findings 5 and 7):** I ran the reviewer's scripts 13 and 14 (first part) and got the same output.

**Not re-run by me:** the task-clustered and retraining bootstraps and Holm corrections (finding 1), the constant-forecast frontier (finding 3), the synthetic-spread check (finding 6) and the group-clustered intervals (finding 8).

## How this relates to Experiment 41

The two studies agree where they overlap: single-rating reliability (SummEval expert 0.719, SummEval crowd 0.492, ChaosNLI 0.485 exactly; the crossed-facet sets within 0.008) and the ratings per item needed for reliability 0.80 (within my intervals everywhere). In Cursor's own cells where the budget really binds, one rating per item beats the D-study choice in 23 of 35, which supports Experiment 41's finding that one rating per item is better or tied in most equal-budget cells. The dollar figures differ because the criteria differ (Experiment 41, penalty chosen on validation: the smallest training size after which no larger one adds more than 10% of the full-training gain, or a 0.001 normalized RPS margin, \$6.40 to \$192 at \$0.05 per rating depending on the definition; with the fixed 0.01 penalty an earlier draft said \$6.40 to \$25.60, which was an artifact of that penalty; Cursor: 0.05 nats of a test-chosen frontier, \$1.60 to \$6.40). Cursor's heads learn very little beyond a constant forecast, which its README does not say; Experiment 41's heads learn 0.002 to 0.015 normalized RPS on the four sets where they learn anything, and Experiment 41's README states both. The reviewer also notes that the VeriBench `native_main` ratings (100 items, 250 ratings) can be loaded through the public sister repository, so Experiment 41 could add them; with about 20 test prompts they could not resolve matched gaps near 0.002, and their prior exposure would have to be disclosed. Not done here.

## Privacy

Counts only, patterns built at run time: the reviewer found no rater name, rater identifier or email address in Cursor's committed files, and my own scan of this folder (the reviewer's scripts and outputs) found none either. The copied rating files carry no per-rater column.
