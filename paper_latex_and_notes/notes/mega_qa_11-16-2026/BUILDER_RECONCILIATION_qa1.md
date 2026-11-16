# Builder reconciliation after stage 1 (qa1_codex)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/paper_latex_and_notes/notes/mega_qa_11-16-2026/BUILDER_RECONCILIATION_qa1.md>

**TLDR:** Stage 1's one open major finding (QA1-F1, the frozen cross-entropy (CE) gradient) is resolved by a full exact-gradient refit: no number reported in the paper, results files or one-page summary changes. Stage 1's other fixes are accepted, with one wording refinement. The builder also fixed four issues of its own. The appendix page-limit question (QA1-F11) stays open for Brando.

Builder: Claude Code, `claude-opus-5-5`, 11-16-2026 10:40–11:20 PST. Stage 1 report: [qa1_codex.md](qa1_codex.md) (unchanged). Stage 1's edits and the builder's edits below are committed together as the commit after `60f71a5`.

## Stage 1 findings

| ID | Builder decision | Evidence |
| --- | --- | --- |
| QA1-F1 (major, open in stage 1) | **Resolved: no reported number changes.** | New post-hoc check `experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/check_ce_gradient.py` swaps in an exact CE (log-sum-exp value, gradient softmax(z) − p, no floor) in memory and refits every CE fit of experiments 07, 08 and 13 with the saved data, starts, ridge and budgets. Output: `results/ce_gradient_check.json` of that experiment (161 s on 8 processes). Fixed-ridge CE fits (4,400 rows of experiment 08, which equal experiment 13's CE fits to 4e-14), experiment 07's CE fits under both budgets, and every calibrated head (experiments 07 and 13) agree to within 1e-9. Of 4,400 validation-tuned CE judges (formulation ③), 338 move by at most 0.00023 in test RPS excess; 337 of them had chosen the weakest ridge, c = 0.01, where the 1e-6 floor flattens the objective, and one ridge choice changes (C, n = 256, seed 7: 0.01 → 0.1). Recomputed with identical bootstrap draws, none of 88 printed cell means, 20 crossovers or 12 hypothesis-check lists changes; one of 44 chosen-ridge shares moves from 69% to 68% (C, n = 256). All 95,291 exact-CE refits of experiment 08 converged (scipy's flag). Only 3 of 2,692,104 exact-CE evaluations had an observed training rating below 1e-6. Experiment 13's saved Pearson rows came from another machine and do not reproduce bit for bit here (a known, documented property of the non-convex fits), so its CE effect is measured as exact versus frozen code on one machine. Frozen measurement code and saved rows are unchanged; the "unresolved" caveats added in stage 1 are replaced by this result in the paper, results files, reading guides, figure notes, table headers and one-page summary. |
| QA1-F2 – F10, F12 | Accepted. | Spot-checked against the saved outputs and the cited sources; build and tables reverified (below). |
| QA1-F5 refinement | Wording fixed. | Bellemare et al. define the Cramér distance as the square root of the integrated squared CDF difference; the appendix and one-page summary now say the excess is one fifth of "the square of the Cramér distance". |
| QA1-F11 (minor, open) | **Open for Brando.** | Call for papers, read 11-16-2026: position papers "2–4 pages"; "References do not count toward the page limit" appears only under full papers; appendices are not mentioned. The body ends on page 4 (Conclusion on page 4; References start page 5); the PDF has 8 pages. |

## Builder's own findings (independent of stage 1, recorded before reading its report)

| ID | Severity | Location | Fix |
| --- | --- | --- | --- |
| B1 | major (rule) | experiment 08 `results.md` and one-page summary: "② worse than the constant forecast from 128 tasks in A, 8 in B, 16 in C and 256 in D (+0.016, +0.028, +0.025, +0.007)" was printed by no script | `make_report.py` now prints these (separate bootstrap stream; every earlier table line unchanged); the values match; `results.md` cites the table and the intervals |
| B2 | minor (selective) | `5b-learning-study.tex`: tuning sentence reported only A's crossover (96 ratings) | Now gives 96 in A, 48 in D, 12 in the other two, and that tuned CE was significantly worse than fixed-ridge Pearson at 12–48 ratings in A and 12 in D (tables.md H2 lines) |
| B3 | minor (overclaim) | `5b-learning-study.tex`: "their fits rarely converge" | "many of their fits stop at the iteration limit" (Kendall fits converge 97–100%) |
| B4 | nit | `make_paper_figure.py`: PDF not byte-reproducible | `metadata={"CreationDate": None}`; two reruns give the same MD5 |
| B5 | minor (call requirement) | The call requires matched compute budgets for system comparisons | Setup now says every fixed-ridge fit, whatever its objective, uses the same penalty and L-BFGS-B budget |
| B6 | submission | Anonymization | Already fixed by stage 1 (QA1-F8) |

## Other builder changes

- Stage 1's page renders (`qa1_artifacts/*.png`, 4.7 MB) moved out of git to `/Users/brandomiranda/.local/state/judges/mega-qa-1116/qa1_renders/`; the report links only scripts and JSON receipts, which stay. The paper README no longer says the renders are in the folder.
- `latex_sentence_lines.py 5b-learning-study.tex` (run without `--check`) rewrote that file in place; its rewrite is whitespace-only. One sentence ending "in B and C." was then rephrased because the tool reads "C." as an initial.

## Verification after reconciliation

- `make -C paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL`: exit 0; 0 errors; 0 undefined citations or references; 0 overfull and 5 underfull boxes; 8 pages; Conclusion ends on page 4.
- `latex_sentence_lines.py --check` on every paper `.tex`: pass.
- `make_report.py` of 06, 07 and 08 and the experiment 13 plot script: rerun; `tables.md` differs from stage 1's version only in the header sentence about CE (and experiment 08's new last table).
- `make_paper_figure.py`: prints 192 / 12 / 12 / 12 ratings and ratios 36.5–45.8 (A) and 8.8–25.5 (B), as the paper states.
- `check_ce_gradient.py 8`: exit 0, results above.

**TLDR-end:** [judges-corr-vs-loss: mega QA reconciliation after stage 1] The CE gradient defect changes no reported number, so the empirical comparison stands; the remaining open item is whether the appendices fit the workshop's page limit.
