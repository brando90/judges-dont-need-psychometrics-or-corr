# QA of this experiment's first draft (before the fixes), with the status of every finding

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/qa_of_this_experiment_draft_before_fixes_11-27-2026/README.md>

**TLDR:** An independent reviewer (Claude Opus 5.5 at maximum effort, a separate subagent that had not seen the work being produced) reviewed the first draft of this experiment, which was then numbered 40. The matched-gap headline, the Lean and the numerics held up. The sufficiency and dollar answers did not: they were artifacts of a fixed 0.01 ridge penalty, and several sentences misstated numbers, preregistration status, the calibration control, the "published" crowd rate and the unreported MSE-trained head. All 18 findings are now fixed or labeled; the table below gives the status of each, and the review itself is unedited.

**As of:** 11-27-2026 (Friday, Pacific time). Local CPU, \$0, no model calls.

## What is in this folder

```text
REVIEW_by_independent_opus_subagent_11-27-2026.md   the full review (verdict, 18 findings with location, evidence and fix, what was verified OK, what could not be checked)
scripts/                                            the reviewer's five probe scripts (single-view probes at other ridge penalties; they read copies, nothing in the experiment was edited)
outputs/                                            their saved outputs
lean/                                               the reviewer's compiler runs
```

The review quotes the draft's file and line numbers (the draft's folder was then called `40_...`); those lines no longer exist because the README was rewritten.

## Status of each finding

1. **Blocker, conclusions depend on the fixed 0.01 penalty (saturation points and dollars at saturation): fixed.** The K-sweep and the G-study and D-study analyses were rerun with the penalty chosen on validation prompts (`k_sweep_l2_sensitivity_*`, `gd_study_l2_validation_chosen*`, `sufficiency_and_dollars_*_l2_validation_chosen*`), the fixed-penalty files were kept under a `_preregistered_l2_0.01` tag, and the README says the earlier saturation points are artifacts. Remaining caveat: between 21% and 49% of the sensitivity fits chose an end of the six-point grid.
2. **"Preregistered" margin and power rule: fixed.** The README states that the margins, the power rule, the equal-budget implementation and the validation-chosen penalties are not preregistered and that their scripts were not committed beforehand.
3. **Number error behind the 10% rule: fixed.** The sentence was removed; the gain over the constant is now read at the largest training size and reported with intervals.
4. **Wrong range of the signal over the constant: fixed.** 0.0177, 0.0116 and 0.0060 on POPQUORN politeness and 0.0004 to 0.0027 on the other nine views with signal (preregistered run).
5. **Unreported MSE-trained head: fixed.** MSE rows in the K-sweep section, a rewritten MSE-plus-L2 section ("refined, not disproved") and a separate note; the penalty form of the MSE head was also corrected for the sensitivity runs.
6. **"One shared calibration map": fixed.** Each head has its own map of the same family; the README, HOW_TO_READ_RESULTS, the table header and the code comments now say so.
7. **"Calibration, not shape" used a statistic that cannot see shape: fixed.** The README uses the CE-trained head's own distribution minus its recalibrated mean. It is at or above zero on 14 of 16 views with the fixed penalty (no shape learning) and at or below zero on 14 of 16 with the validation-chosen penalty (a small amount, at most 0.0013). The K = 2 residual on POPQUORN politeness is described as a ranking difference. The preregistration file still contains the wrong P1 rationale because that file is frozen; this table is where it is noted.
8. **Raw-gap magnitudes, the P2 exception and the P5 failure depend on the penalty: fixed.** Both runs are reported side by side; the README says the size and the exceptions depend on the penalty and the sign on views with signal does not.
9. **P7 reported more favorably than P5, one cell dropped silently: fixed.** "Not supported as written" in the preregistered run (fails in one cell); the DICES cell that could not be formed and the k = 8 cap for POPQUORN offensiveness are stated; cells on sets with no signal are separated.
10. **A risk-estimation formula used to explain a training result: fixed.** The cost-optimal ratings per item are labeled a test-side precision statement, and the training result is stated separately (20 of 20 evaluable cells with validation-chosen penalties, 19 of 20 preregistered).
11. **"\$0.05 per rating" is not a published price: fixed.** Called a derived rate everywhere, including the figure legend and the script header.
12. **Estimation-rate correction generalized too far: fixed.** Restricted to the plug-in estimate of one distribution; the wording "wrong" became "valid but loose"; the numerics line was reworded and rerun twice with byte-identical output.
13. **Lean statements weaker than the prose: fixed.** The file header now lists which theorems are general and which are instances; the pooled ceiling, the tower step, the pooled attainment and decision-accuracy properness were added (23 theorems).
14. **Weak or tautological numeric checks: fixed.** E3, F3, D3 and G3 were rewritten to enumerate exact joint laws; checks for the tower step, decision-accuracy properness and chi-squared were added (33 claims).
15. **Reporting-standard violations: partly fixed.** The tagged figures name the metric and the split on every axis, spell out the ranked probability score, carry minor gridlines on log axes, show uncertainty bands on the learning curves and say in the caption how the judge was chosen; the learnable gain is no longer selected on the test minimum; the strata-figure bar is explained in HOW_TO_READ_RESULTS. Not changed: the dollar figure still draws its lines from a 1e-4 baseline on a log axis.
16. **Stale or inaccurate text: fixed.** The experiment number, the stop-rule description, "proportional" for the MSE and Brier relation at K = 2, and the "for training a proper-loss head" qualifier in the closing line.
17. **P3 decision rule made concrete without saying so: fixed.** The README and the scoring JSON state the reading used and the clause not applied; neither changes a verdict.
18. **Folder changed during the review (the allocation files): fixed.** The README references them; they were later checked (17 Lean theorems, 0 errors, standard axioms only; 19 numerical claims, all PASS, identical on two runs).

## What this review could not check (from the review's own list), and what happened since

1. Full reruns of the two heavy scripts: since then both were rerun in full with the validation-chosen penalty.
2. VeriBench views (HTTP 404): still not run.
3. A penalty sweep over all views and seeds: since then done (sensitivity run, 80 fits per head).
4. The sister-repository inputs (4.0 and 37.3 minutes per rating): still only the pinned aggregates.
5. The journal citations in the positioning section: the three classification-calibration references were checked against publisher or repository pages on 11-27-2026.

## Privacy

A counts-only scan over this folder (rater full names, name parts, identifiers, email addresses, home paths, handles) found zero matches. The review's own process note says one directory listing of the sister checkout printed rater-bearing file paths into its tool log; nothing from it went into any file.

**TLDR-end:** [qa of this experiment: draft before fixes] The first review's verdict (headline, Lean and numerics hold; sufficiency and dollars were fixed-penalty artifacts; several misstatements) led to a validation-chosen rerun and a rewritten README; all 18 findings are fixed or labeled, with one cosmetic item left (the dollar figure's log-axis baseline).
