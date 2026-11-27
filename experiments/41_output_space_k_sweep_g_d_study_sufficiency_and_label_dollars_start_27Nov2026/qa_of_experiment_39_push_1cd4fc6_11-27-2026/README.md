# QA of Cursor's Experiment 39 push (commit 1cd4fc6)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/qa_of_experiment_39_push_1cd4fc6_11-27-2026/README.md>

**TLDR:** An independent reviewer (Claude Opus at maximum effort, no access to my conclusions) re-ran Cursor's Experiment 39 from scratch. Everything reproduces exactly (all six input hashes, 42,451 summary values, a byte-identical Study B table, 25 of 25 checks), so nothing is fabricated. Three of the headline interpretations do not hold as written: the "modal RPS oracle prefers more items" statistic comes from an undeclared tie-break, both D-study "selectors" are constant rules so their wins hold by construction, and the proper-score selector uses 4.5 to 7.7 times more ratings than the pilot it is said to use. What survives is narrower: for this recalibration of frozen judges, designs with more items and fewer ratings per item score slightly lower held-out ranked probability score (RPS) than designs with more raters, by 0.4% to 2.6% of the total RPS.

**As of:** 11-27-2026 (Friday, Pacific time). Local CPU, $0, no model calls.

## What is in this folder

```text
REVIEW_by_independent_opus_subagent_11-27-2026.md   the full review: verdict, twelve numbered findings (severity, location, evidence, fix), what was verified OK
scripts/                                            the reviewer's eleven check scripts (each reads a copy of the experiment; nothing in the original was edited)
outputs/                                            their saved outputs
lean/VDelta_independent_check_by_reviewer.lean      the reviewer's independent core-Lean check of Study B's closed forms (Lean 4.36.0, no Mathlib)
lean/VDelta_independent_check_by_reviewer_compile_output.txt   compiler output with #print axioms (only propext, Classical.choice, Quot.sound)
```

## The findings in one line each (numbering follows the review)

1. **Blocker for the oracle statistics.** The ranked-probability-score oracle is mostly decided by an undeclared "more items" tie-break: in 75% of comparable replications at least two designs tie exactly for the best test RPS. The reported "U1 agrees with the RPS oracle in 0.00 to 0.36" and "the modal RPS oracle is the max-items design" are artifacts of that rule. Tie-neutral comparison of max-items against the U1 design: strictly better 22%, exactly equal 65%, strictly worse 13%.
2. **Both D-study selectors are constant rules.** U1 chose max raters in 150 of 150 replications of all 30 cells and U2 chose max items in 150 of 150 replications of all 20 comparable cells, so the item-mean and grand-mean MSE wins follow from sampling variance (the analytic ordering holds in 20 of 20 cells). The experiment shows that the optimal allocation depends on the target; it does not test whether proper-score design replaces D-theory.
3. **The proper-score selector uses far more rating data than the pilot it is described as using.** Validation targets are the full rating histograms of the held-out pilot items. A protocol-faithful rerun (validation on the 16-rating pilot histograms) ties the original in 20 of 20 cells; the fixed max-items rule beats U1 in 12 of 20 cells and the selector in 10 of 20.
4. **Intervals leave out test-item sampling and have no multiplicity correction.** A two-way bootstrap widens the paired intervals by a median factor of 1.14 (9 of 20 cells stay significant instead of 10); absolute-level intervals are 7 to 16 times too narrow as statements about new items. Holm keeps 3 of the 10 "PS beats U1" cells, Benjamini-Hochberg 6.
5. **The pre-declared decision criterion was not met, and the freeze cannot be verified.** 10 of 20 cells is not a "pre-declared majority"; every file was first committed 17 hours after the run, README and REPORT before PROTOCOL.
6. **The documented RPS formula is not the one computed.** The code computes the individual-rating RPS averaged over each item's ratings (it includes the forecast-independent item term); the documents state the distribution-to-distribution form. The significant gaps are 0.37% to 2.6% of total RPS and 2.2% to 11.7% of the reducible part.
7. **A fresh clone cannot reproduce or verify the run as documented** (no download code or pins; `verify.py` needs an uncommitted 7.8 MB CSV).
8. **Study B wording.** "Falls to near chance" is a coin flip between two designs, not three-way chance; the "small pilot hurts" cells are driven by four-level facets and none survives Holm; the "facets harder to tell apart" premise is false.
9. **Documentation that contradicts the code** (eight items, e.g. sampling with versus without replacement, "Laplace" versus 0.5 smoothing, "held-out" budget axis label).
10. **U2's "max items" prescription assumes no rater main effect** (nested raters); with crossed raters the optimum is interior.
11. **The repository's proof rule is not met** for the formal claims; the reviewer's core-Lean file proves the closed forms (v_Delta for 8x2, 4x4 and 2x8, the crossovers 4/15 and 11/15, uniqueness on the open interval).
12. **Nits** (two rounding errors in the Study B table, 1,017 `NaN` tokens in a "JSON" file, 7 unreported REML non-convergences, dead code, alphaNLI has 9 distinct transforms not 27).

## What this review did and did not cover

- **Covered:** the Experiment 39 folder at commit 1cd4fc6 (inputs, data preparation, both studies, `verify.py`, every table and headline number in README and REPORT, the Study B mathematics, privacy of the committed files by counts).
- **Not inspected by that reviewer:** three origin commits that landed after it started: 593bf45 and 6b3f8f4 (documentation-only edits to Experiment 39's README and results.md: the Google Doc link, then restoring closing sections that the first edit truncated) and a4ccb30 (Cursor's Experiment 40, reviewed separately in the Experiment 41 report).
- **My own additions, in the Experiment 41 folder rather than here:** `lean/AllocationAndClosedForms.lean` (17 theorems; closed forms for the 8x2, 4x4 and 2x8 designs, the crossovers, uniqueness, the cost-optimal ratings per item, `reliability_iff`, `free_items_one_rating_optimal`) and `scripts/check_allocation_and_closed_forms.py` (19 numerical claims, byte-identical across two runs). They agree with the reviewer's independent Lean file on the Study B closed forms.
- **Not machine-checked anywhere:** the empirical statements (who beats whom on held-out RPS); they depend on the data and are reproduced, not proved.

## Privacy

The reviewer scanned the committed files with patterns built at run time (counts only): no rater identifiers, no email addresses, no home-directory paths and no handles other than the repository owner's inside documentation links. A second scan of this folder, run by me before commit, is recorded in the Experiment 41 report.
