# Builder reconciliation after stage 3 (qa3_codex) and final verification

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/paper_latex_and_notes/notes/mega_qa_11-16-2026/BUILDER_RECONCILIATION_qa3.md>

**TLDR:** All six stage 3 fixes are accepted; its Cramér-distance correction reverses an error the builder introduced after stage 1. The builder closed the open acronym finding (QA3-F8) within the four-page body. The only open item is the workshop's page-limit rule for appendices, which needs Brando or the organizers.

Builder: Claude Code, `claude-opus-5-5`, 11-16-2026 12:20–12:45 PST. Stage 3 report: [qa3_codex.md](qa3_codex.md) (unchanged).

## Stage 3 findings

| ID | Builder decision | Check |
| --- | --- | --- |
| QA3-F1 (minor) | Accepted; builder error | Bellemare et al. (2017), section 4.1, read in the arXiv PDF: "The Cramér distance between P and Q is l₂²(P, Q) := ∫ (F_P(x) − F_Q(x))² dx. Note that as written, the Cramér distance is not a metric proper. However, its square root is". Stage 1's wording was right; the builder's "square of the Cramér distance" (and stage 2's acceptance of it) was wrong. The appendix and one-page summary now say "the Cramér distance as defined by Bellemare et al." |
| QA3-F2 (major) | Accepted | The resolved-caveat text overstated the check: one printed chosen-ridge share changes (C, n = 256: 69% → 68%), experiment 13's head comparison isolates the CE change on one machine, the dense-grid head's Spearman correlation moves by about 1e-5, and experiment 13's head and dense-grid convergence flags were not recorded. The reworded claims match `results/ce_gradient_check.json`. |
| QA3-F3, F5 (minor) | Accepted | In-sample dense-grid references are no longer called test results; the one-page summary's synthetic findings are scoped to the tested targets and sizes. |
| QA3-F4 (minor) | Accepted | Checked against `experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/results/summary.json`: with one validation rating per item, the truth is chosen 249/256 times by CE and 245/256 by RPS and Brier (256/256 with two or three); mean test CE is 1.3315 nats choosing by CE and 1.9949 choosing by Pearson. |
| QA3-F6 (minor) | Accepted | Narrow-screen overflow fix in the one-page summary's CSS. |
| QA3-F7 (minor, open) | Open for Brando | Same item as QA1-F11 and QA2-F6: the call gives position papers 2–4 pages, excludes references only under full papers, and does not mention appendices. |
| QA3-F8 (minor, open in stage 3) | **Fixed by the builder** | First body uses now read: "the expected Kullback--Leibler (KL) divergence" (`4-proposal.tex`), "normalized mean squared error (nMSE)" (table caption, `5-toy-counterexample.tex`), "mean absolute error" (its only use), and "the limited-memory quasi-Newton method L-BFGS-B" (`5b-learning-study.tex`). To keep the body at four pages, two setup details documented in experiment 08's README were cut from the paper: the ridge tie-break and fallback rule, and the clause noting that A and B reuse experiment 13's tasks (interval scope is kept as "Intervals are pointwise"). |

## Further builder edits

- One-page summary: "is the one possible explanation" → "is one possible explanation" (an earlier edit implied uniqueness).
- Paper README: the stale sentence about who owns review and publication now points to this folder; the build receipt says four underfull boxes.
- Stage 3's page renders (`qa3_artifacts/*.png`) moved out of git to `/Users/brandomiranda/.local/state/judges/mega-qa-1116/qa3_renders/`, as for stage 1; its report links only scripts and JSON receipts.

## Final verification (builder, 11-16-2026)

- Paper: `make` exit 0; 0 errors, 0 undefined citations or references, 0 overfull boxes, 4 underfull boxes; 8 pages; the Conclusion ends on page 4 and References start page 5 (pages 3 and 4 inspected as images); `latex_sentence_lines.py --check` passes on every `.tex` file.
- Every learning-study number in the paper is unchanged from the version stage 3 verified; the edits after stage 3 change wording only.
- One-page summary, experiment results and scripts: verified by stage 3 (`qa3_artifacts/verification_commands.json`); the builder's later one-page edit is a two-word wording fix.

**TLDR-end:** [judges-corr-vs-loss: mega QA final reconciliation] All three stages' fixes are verified and the four-page body is preserved; only the workshop appendix rule remains open for Brando.
