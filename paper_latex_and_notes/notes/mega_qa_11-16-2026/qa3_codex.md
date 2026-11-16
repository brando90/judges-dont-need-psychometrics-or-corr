# Mega QA stage 3: paper, learning study and results page

**TLDR:** Accept with minor open issues. Six findings are fixed, including an overstatement of the exact cross-entropy refit check and an incorrect attribution of the Cramér-distance definition. The paper's numerical claims survive verification, its body still ends on page 4, and the frozen measurements are unchanged. Submission packaging/page-limit eligibility and older first-use acronym omissions remain open.

## Reviewer, scope and execution

- Actual harness/model/effort: Codex CLI 0.169.0, `gpt-6-astra`, `ultra`; verified from this session's runtime record. Account profile: `CODEX_HOME=/Users/brandomiranda/.codex-vals`. Approval policy `never`, sandbox `danger-full-access`.
- Stage: `qa3_codex`, begun 11-16-2026 at 19:43:26 UTC. Checkout HEAD: `a3efff2`; reviewed the scoped changes from `af810f5`, including stages 1–2 and builder reconciliation. This stage leaves edits uncommitted.
- Verification is analytic/model reading plus executable Python, browser and LaTeX checks. It is **not Lean-checked**. No Deep Think (DT), additional reviewer, agent, paid model run or full measurement sweep was launched.
- Independent inspection preceded reading the earlier reports. I subsequently read both reports, both builder reconciliations and the available stage markers. Earlier reports remain unchanged; corrections to them are recorded below.
- Work stayed in the owned checkout, except the required completion marker. No commit, push, email, publication or main-checkout edit. The shared-policy refresh was blocked by an existing edit in the home clone; I preserved it and used the local index/rules. Shared state was read, but no shared log was written because this brief restricts writes to the checkout and completion marker.
- Environment: Python 3.11.6, NumPy 2.2.6, SciPy 1.16.3, macOS 27.2, arm64. [Environment receipt](qa3_artifacts/environment.json).

Paths below use `P` for `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL`, `H` for `paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/judge-objectives-status.html`, and `E06`, `E07`, `E08`, `E13` for the experiment folders named in [COMMON.md](COMMON.md). Line numbers describe the final stage-3 working tree.

## Reading coverage

| Area | Coverage and limits |
| --- | --- |
| Instructions | COMMON and stage-3 brief in full; repository AGENTS in full and byte equality with CLAUDE; result-clarity Cursor rule; shared hard rules and triggered rules; assistant, scientific-terminology and review-paper skills. |
| Paper | Entire main source and sections/appendices 1–9, bibliography, README and `rps_appendix_examples.py`; all eight rendered PDF pages inspected. Appendix 9 and older sections outside the edit list were read for consistency and left unchanged. The six new bibliography entries were checked against primary-source metadata; the substantive Cramér, survival-function and alternative-score passages were checked in their papers, not represented as end-to-end literature reviews. |
| E06–E08 | Full preregistration READMEs, results narratives, reading guides, measurement/sweep/report scripts, paper-figure script and CE diagnostic. All raw rows parsed and grid completeness checked. All numeric table rows checked programmatically; selected rows and every paper-linked result inspected directly. Six complete small/large cells recomputed with the existing functions. |
| E13 | Full `toy.py`, added plotting script and reading guide; setup, saved finite-sample rows, dense-grid references and relevant historical result tables read. All saved rows parsed. Frozen upstream files were not edited. |
| Rules/guides/indexes | Full requested guides for 00, 01, 02, 06, 07, 08 and archive 11–14; experiment/archive indexes and migration verifier. Experiments 04/05 remained outside the review's measurement/edit scope. |
| Human-rating comparisons | Tracked experiment 02 comparison and archive 12/14 result records, including the actual JSON outputs. Seven forest-plot rows checked against those records. Private original ratings were unavailable by design; no claim of independently recomputing the human-rating analyses. |
| One-page summary | HTML source, formulas, chart data, captions and image mapping; desktop and mobile browser rendering; all eight referenced images inspected. All 45 math blocks rendered. Published Version 23 was not modified. |
| Earlier stages | Full `qa1_codex.md`, `qa2_carbon.md`, both builder reconciliations and stage markers, after independent review. Material findings and decisions are reconciled individually below. |

## Data and metric contract

The learning experiments fit 78-weight softmax judges on sampled synthetic tasks with three ratings per training task. E06/E13 cover A/B; E08 covers A–D, with 100 seeds per scenario/size. E07 uses 30 seeds per scenario/size. Evaluation uses 500 held-out synthetic tasks per seed against the known conditional distribution; those test tasks do not enter fitting or validation selection. A/B reuse earlier synthetic tasks, so this is not previously unseen research evidence. E13's separate 4,000-point grid is fitted and scored in-sample.

Normalized ranked probability score (RPS) excess is the mean squared cumulative-probability gap to the true distribution, on [0,1], with zero optimal and lower better. Cross-entropy (CE) excess is Kullback–Leibler divergence from truth to the judge, in nats, with zero optimal and lower better. Pearson/Spearman compare predicted and true mean ratings, on [−1,1], with higher better when defined. A difference below always compares the same metric on the same tasks. Reproduction errors are absolute differences between reruns, not improvements of one training objective over another; they have no sampling confidence interval.

Unless explicitly labeled otherwise, reported synthetic uncertainty is a paired 95% percentile bootstrap over seeds with 10,000 draws. A crossover requires a positive lower endpoint for comparator test RPS minus CE-trained-judge test RPS at that and every larger **tested** size. These are pointwise intervals, not simultaneous coverage over sizes. The fixed fits use the same ridge and limited-memory Broyden–Fletcher–Goldfarb–Shanno optimizer with bounds (L-BFGS-B), perturbed log-histogram start and 400-iteration/600-evaluation budget. E07 changes starts/budgets; E08 selects ridge on training-task folds. Exact CE plus ridge is convex. Other fitted objectives have no global-optimum certificate; the findings reflect the specified optimizer, starts and budgets.

## Findings

Counts are distinct findings, not changed files: 0 critical, 1 major, 7 minor, 0 nit; six fixed, two minor open.

| ID | Severity | Final location | Evidence | Resolution |
| --- | --- | --- | --- | --- |
| QA3-F1 | minor | `P/8-appendix-rps.tex:132`; `H:502` | The builder's refinement says Bellemare defines Cramér distance as the square root. Section 4.1 actually calls the integrated **squared** cumulative-distribution gap the Cramér distance and separately identifies its square root as a metric. | **Fixed.** Say “the Cramér distance as defined by” Bellemare. Constants and formulas remain unchanged. This disagrees with the stage-1 builder refinement and stage-2 acceptance. |
| QA3-F2 | **major** | E06/E07/E08 guides `:44/:41/:56`; E13 guide `:43`; `H:154,157,385`; E08 report script `:144,152,173`; `P/5b-learning-study.tex:30,38` | “No reported number changes,” “every head matches saved fits” and blanket convergence claims exceed the diagnostic. One reported ridge share changes; E13 head comparisons isolate the CE change on one machine; dense-grid head rank correlation changes slightly; E13 head/grid convergence flags were not recorded. The frozen operation clips **and renormalizes** probabilities. | **Fixed reporting.** Preserve the supported printed means/crossovers and paper numbers, name the ridge-share exception and comparison scope, qualify convexity as unclipped/exact CE, and remove unsupported convergence claims. The diagnostic's two inaccurate “about 1e-3” docstrings are corrected; its executable behavior is unchanged. Frozen code and measurements remain unchanged. |
| QA3-F3 | minor | `H:221`; E13 guide `:5,49`; archive index `:19` | The head's dense-grid RPS was still called “test” RPS; the guide/index said every reported score was held out, despite the in-sample dense-grid fit. | **Fixed.** Distinguish finite-sample test curves from in-sample grid references in all three locations. |
| QA3-F4 | minor | `H:176,217` | Experiment 01 selects truth 249/256 times by CE but 245/256 by RPS with one validation rating per item. Mean test CE for CE selection is 1.331453, not 1.328 (the latter applies with two/three ratings). The tied Pearson value is a uniform average over tied candidates. | **Fixed.** Separate the two selection counts and specify the one-rating comparison, 1.331 nats versus 1.995 nats, with its actual tie treatment. |
| QA3-F5 | minor | `H:119,370,376` | Residual “wherever the mean carries information” and “realistic numbers of ratings” claims generalized four constructed targets to human data. | **Fixed.** Scope the finding to B/C/D and the tested synthetic sizes; human small-sample gains remain untested. |
| QA3-F6 | minor | `H:101–102` | A 390-pixel viewport had a 769-pixel page scroll width because long code paths and an inline formula escaped their containers. | **Fixed.** Wrap long code paths and allow inline formula scrolling on narrow screens. Final page width is 390 pixels, without page overflow; tables/figures retain their own scrolling. |
| QA3-F7 | minor | `P/README.md:13`; `P/00_main_aaai_2027.tex:83`; provenance comments in multiple `.tex` files | The official call does not establish an appendix exemption for position papers. The PDF has four body pages and eight total pages. Appendix 9 still has no body pointer; internal source comments contain provenance paths/commits. | **OPEN, inherited QA1-F11/QA2-F6.** Builder must determine the actual submission packaging rule. The rendered PDF and bibliography pass the identity scan; the working source tree is not certified as an anonymous source-upload package. Appendix 9 is read-only in this stage. |
| QA3-F8 | minor | `P/4-proposal.tex:6`; `P/5-toy-counterexample.tex:26,58`; `P/5b-learning-study.tex:24` | KL appears in the body before its appendix expansion; normalized mean squared error (nMSE), mean absolute error (MAE), and L-BFGS-B are not fully expanded at their first paper uses. Stage 2 fixed appendix-local KL use, not the whole-paper first use. | **OPEN.** Carry forward for a coordinated paper edit, including older sections outside this stage's edit list, while preserving the four-page body. No numerical effect. |

## Exact-CE diagnostic: what the evidence establishes

I read the implementation and reran `python3 E08/scripts/check_ce_gradient.py 8`. It exited successfully in 240 seconds. The fresh JSON agrees with the tracked diagnostic in every field except runtime; the tracked runtime-only change was restored. [Fresh receipt](qa3_artifacts/ce_gradient_check_rerun.json), [run log](qa3_artifacts/ce_gradient_check.txt).

1. **E08:** all 95,291 exact-CE fits returned SciPy success. Of 2,692,104 exact-CE objective evaluations, 285,141 had some category below 1e-6, across 1,907 of 4,400 rows; only three evaluations in three rows had observed rating mass in such a category. These counters describe the **exact-CE optimization trajectories**, not an instrumented census of the original frozen trajectories.
2. All 4,400 fixed-ridge CE metric rows agree with saved results within 1e-9. Among 4,400 validation-tuned CE rows, 338 have some metric change above 1e-9. The largest absolute change in held-out test RPS excess is 0.0002299998 on its [0,1] scale. One selected ridge changes: C, 256 tasks, seed 7, from 0.01 to 0.1. These are paired refit comparisons, not independent new experimental observations.
3. The report replay checks **424 quantities**: **423 unchanged, one changed**. There are 44 mode/share quantities and 380 other quantities; stage 2's phrase “423 non-mode items” is incorrect. The changed share is C/256, ridge 0.01: 69% to 68%. Printed cell means, crossovers and hypothesis checks survive. This supports the paper's limited “every number reported here” statement, not a universal “no reported number changes” statement.
4. **E07:** CE under both budgets and calibrated heads under all three conditions reproduce saved metrics within 1e-9, with unchanged success flags.
5. **E13:** finite-sample CE reproduces the 2,200 saved rows within 1e-9. Exact-versus-frozen **same-machine** finite-sample reruns agree within 1e-9, including heads. The saved Pearson/head fits are platform-sensitive: 2,198/2,200 paired rows have some frozen-rerun discrepancy above 1e-9. Across those rows, the largest absolute head test-RPS-excess difference is 0.400364 and head test-CE-excess difference is 10.0102 nats. Those are not effects of changing CE and must not be described as universally “about 1e-3.”
6. In E13's in-sample dense-grid reference, CE/RPS metrics agree within 1e-9 between exact and frozen CE on this machine, but head Spearman differs by about 9.6e-6. The diagnostic does not record E13 head/grid convergence flags. Shared finite-sample CE fits are covered by E08's convergence check.

The underlying clipping-gradient defect is real and remains in frozen upstream code. The diagnostic establishes its limited impact on the reported comparisons; it does not repair that code, certify every nonconvex solution, or establish cross-platform stability. Future reuse should use a separately versioned exact-CE implementation rather than silently treating the frozen implementation as corrected.

## Reconciliation of earlier findings and builder decisions

| Earlier item | Stage-3 determination |
| --- | --- |
| QA1-F1, builder exact-CE closure | **Empirical impact checked, wording corrected by QA3-F2.** The paper's comparisons survive. The implementation defect is not patched; one report share changes, and saved-head reproduction needs the distinctions above. |
| QA1-F2 | **Correctly fixed.** E06's prediction is only partly supported. At B/2,048 training tasks, mean held-out test RPS excess is 0.0623/0.0330/0.0851 for Spearman/Kendall/mixed training, versus 0.0951 for Pearson training and 0.0037 for CE training. These are 100-seed means on the test panel defined above; rank training can improve on Pearson while remaining worse than CE. |
| QA1-F3 | **Correctly fixed.** E07 tests one optimizer family, 400/600 versus 2,000/4,000 iteration/evaluation caps, and selection by penalized training objective. CE has one long start; each head is fitted once to its selected score. Saved best starts match the training minima. RPS crossovers persist; this is not optimizer independence. |
| QA1-F4 | **Correctly fixed.** The sparse-histogram KL direction, cumulative-probability variance divided by the rating count, and individual-rating disagreement floor are distinguished. The reverse KL remains CE minus histogram entropy; cancellation in expectation does not eliminate observed paired noise. |
| QA1-F5 and builder refinement | **Constants/identities correct; attribution refinement wrong.** Independently verified cumulative, absolute-error and energy identities and the finite-rating floor correction. QA3-F1 restores the source's actual Cramér naming. |
| QA1-F6 | **Correctly fixed in live artifacts.** Concentration is not variance; scenario A has a nearly constant mean. Frozen preregistration prose remains historical. |
| QA1-F7 | **Extended by QA3-F3.** Plot labels were fixed, but residual HTML/guide/index statements were not. Dense-grid optimization is in-sample with ridge 1/(4,000 × 10^6), not a certified population optimum. The historical illustrative feature values are −0.7, 0 and 0.5. |
| QA1-F8 / builder B6 | **Rendered/bibliography fix verified.** No identifying-string hits in the current PDF scan. Source-comment provenance remains the conditional packaging issue QA3-F7; earlier blanket source-cleanliness wording was too strong. |
| QA1-F9 | **Verified with later corrections.** Metric/split labels, training details, gridlines, uncertainty and prior exposure are present. E08 keeps undefined raw correlations; its historical zero substitution is disclosed, including 99 defined plus one undefined seed per Pearson variant at C/4. |
| QA1-F10 | **Verified.** Eight total pages/four body pages; synthetic formulation status and unpublished source status are stated; unsupported numerical examples were removed. Regenerated paper output prints ratio ranges 36.5–45.8 for A and 8.8–25.5 for B. These compare held-out test RPS excess of rank/mixed-trained judges to CE-trained judges at 2,048 tasks, not two different metrics. |
| QA1-F11 | **Still open as QA3-F7.** A later PASS does not establish the organizer's appendix policy. |
| QA1-F12 | **Paper fix verified; HTML extended by QA3-F5.** Scope is four synthetic targets, three with informative means, one judge family; rank/mixed comparisons cover A/B only. |
| Builder B1 | **Verified against generated output.** Pearson-trained-judge test RPS minus constant-forecast test RPS at 2,048 tasks is +0.0157/+0.0281/+0.0252/+0.0074 in A/B/C/D; positive favors the constant. Sustained tested-size crossovers are 128/8/16/256 tasks. The table contains paired 95% seed-bootstrap intervals. Stage 2's different four values are incorrect; the builder's second reconciliation corrects them. |
| Builder B2 | **Verified.** Tuned-CE versus fixed-Pearson RPS crossovers are 96/12/12/48 training ratings in A/B/C/D; the paper also states the tested smaller sizes where tuned CE loses. |
| Builder B3 | **Verified.** “Many” nonconvex fits hit the budget; not all objectives rarely converge. The Kendall convergence evidence is from E07, not E06. For E07 Pearson at n≥32, original fits succeed in 0/360 cases and long fits hit a budget in 326/360 cases. |
| Builder B4 | **Verified.** Paper-figure PDF regenerated byte-identically with creation-date metadata disabled. |
| Builder B5 | **Verified within its stated scope.** Fixed-ridge per-fit optimizer caps/penalty match. Validation search and restart conditions are separately described; this is not a claim that every whole procedure uses equal total compute. |
| QA2-F1 | **Verified after QA3 footer edit.** E08's narrow figure still uses six footer lines. All five active report figures' figure-level text and legends fit the canvas; all eight page images were visually inspected. |
| QA2-F2 / builder reconciliation 2 | **Mechanism/replay fix verified, report description corrected.** Actual code uses `marginal_W0`, `val_scores`, `pick`, the saved seeded streams and only `maxiter`/`maxfun` optimizer overrides. The invented function/constant names in stage 2 are not evidence. The 424-quantity replay and both clipping counters are real; see QA3-F2 for scope and denominators. |
| QA2-F3 | **Appendix-local fix verified; whole-paper omission remains QA3-F8.** |
| QA2-F4 | **Verified.** HTML distinguishes synthetic weight fitting from mixtures of fixed human-data judges, includes formulation ①, and expands the cited first-use acronyms. Additional numerical/scope defects are QA3-F4/F5. |
| QA2-F5 | **Verified.** The three specified guide/result acronym expansions are present. |
| QA2-F6 | **Still open as QA3-F7.** No new organizer policy or anonymous source package was established. |

Stage 2 also names `results/summary.json` for human-data experiments where the actual checked files are E12 `results/results.json` and E14 `results/open/results.json` / `results/all/results.json`. I checked the actual outputs, not those nonexistent paths. Earlier reports and builder notes are preserved as audit history, not silently rewritten.

## Verification commands and observed outputs

All COMMON verification commands ran successfully. [Command receipt](qa3_artifacts/verification_commands.json) records exact paths, exit codes and durations. The stage-specific scripts below are retained with their receipts.

| Command or check | Observed result |
| --- | --- |
| `make -C P` and PDF/log inspection | Exit 0; zero LaTeX errors, undefined references/citations or overfull boxes. Eight pages; Conclusion on page 4, References begin page 5. All pages viewed. [PDF receipt](qa3_artifacts/pdf_check.json). |
| `python3 P/rps_appendix_examples.py` | Exit 0; printed examples and sparse-rating checks agree with appendix. Independent 1,000-distribution identity checks have maximum absolute residual 1.78e-15. |
| `python3 E08/scripts/make_paper_figure.py` | Exit 0; fixed-ridge CE-versus-Pearson RPS crossovers 192/12/12/12 training ratings in A/B/C/D; ratio endpoints above reproduced. Figure PDF byte-identical. |
| `python3 E06/scripts/make_report.py` | Exit 0; `tables.md` byte-identical; unchanged figures. |
| `python3 E07/scripts/make_report.py` | Exit 0; `tables.md` byte-identical; unchanged figure. |
| `python3 E08/scripts/make_report.py` before editing it | Exit 0; table byte-identical. [Receipt](qa3_artifacts/unchanged_report08.json). |
| Same E08 report after the explanatory-text correction | Exit 0; only header/footer wording changes. All 194 numeric table rows unchanged; two figures regenerated from source. |
| `python3 E13/scripts/plot_without_head_and_crossover.py` | Exit 0; both generated figures byte-identical. |
| `python3 experiments/archive/verify_migration.py` | PASS: 195 preserved files, 194 byte-identical and one navigation-only exception; 68 Python and 51 JSON checks passed. |
| `python3 E08/scripts/check_ce_gradient.py 8` | Exit 0; 240 seconds; diagnostic identical except runtime. Detailed limitations above. |
| `python3 qa3_artifacts/check_numerics.py` | Six sampled cells reproduce all shared numeric fields exactly: E06 A/4/11 and B/1,024/17; E07 A/4/11 and B/512/17; E08 A/4/11 and D/2,048/17, shown as scenario/tasks/seed. Six analytic-gradient checks away from clipping/degeneracy have maximum absolute finite-difference error ≤4.32e-10. Selection tie/fallback checks pass. [Receipt](qa3_artifacts/numeric_checks.json). |
| `python3 qa3_artifacts/audit_saved.py` | Complete unique grids: E06 2,200, E07 540, E08 4,400, E13 2,200. Numeric table rows unchanged: 156/104/194 for E06/07/08. Seven HTML forest rows and E01 selection claims verified. Twenty-nine designated frozen files unchanged. AGENTS and CLAUDE identical. [Receipt](qa3_artifacts/saved_audit.json). |
| `python3 qa3_artifacts/check_plot_bounds.py` | PASS: figure-level text and legends within all five E06–E08 plot canvases; no output artifacts modified. [Receipt](qa3_artifacts/plot_bounds.json). |
| `python3 qa3_artifacts/check_html.py` | PASS: 45/45 math blocks, 8/8 images, no JavaScript or KaTeX errors, no desktop/mobile page overflow. Viewports 1,100×1,000 and 390×844. [Receipt](qa3_artifacts/html_check.json). |
| `python3 ~/agents-config/scripts/latex_sentence_lines.py --check P/*.tex` | Exit 0, all paper `.tex` files. |
| `git diff --check`, status and full tracked diff inspection | Pass; only scoped corrections and stage-owned artifacts. No raw data, frozen preregistration or frozen measurement source changed. |

The numeric harness was corrected during development to handle E07's nested record shape and pre-amendment E08 rows without fallback metadata; these were harness/schema assumptions, not saved-measurement mismatches. The final complete run is logged separately in `qa3_artifacts/numeric_checks_final.txt`. The early log remains for transparency. Missing fallback fields in 208 historical E08 rows are the documented pre-amendment schema, not dropped observations; the sampled old row's new fallback flags are false.

Additional numerical care: gradient checks above do not prove exactness at clipping or near-constant Pearson predictions. A bounded inspection of the frozen Pearson denominator epsilon found a small derivative discrepancy at four sampled fitted vectors; its largest component was 3.69e-7 in the smallest cell, versus a fitted gradient norm about 1.01e-3. This does not establish a new reported-result change. The exploratory receipt is [pearson_epsilon_check.json](qa3_artifacts/pearson_epsilon_check.json); the frozen implementation was not changed.

## Sources and submission limits

The [official workshop call](https://eval-ai-scientists.github.io/aaai27/) says position papers are 2–4 pages. Its reference exclusion appears under full papers; it does not specify an appendix exemption. I also opened the linked OpenReview group, but it did not expose an additional definitive rule in the available rendering. Therefore this stage verifies four body pages and eight total pages, not eligibility of the complete eight-page submission.

The Cramér attribution is checked against [Bellemare et al., section 4.1](https://arxiv.org/html/1705.10743). The survival-function connection was checked in [Schaeffer et al.](https://arxiv.org/html/2406.04391), and the alternative-score discussion against [Galdran's equation 7](https://arxiv.org/html/2309.08701). Those checks support the specific cited passages; the RPS identities were additionally verified independently in Python. Metadata checks used the publisher records for [Epstein](https://journals.ametsoc.org/view/journals/apme/8/6/1520-0450_1969_008_0985_assfpf_2_0_co_2.xml), [Matheson and Winkler](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.22.10.1087), and [Székely and Rizzo](https://www.sciencedirect.com/science/article/pii/S0378375813000633).

## Files changed and handoff

Eighteen existing files changed:

- Paper: `5b-learning-study.tex`, `8-appendix-rps.tex`, rebuilt `00_main_aaai_2027.pdf`.
- E06 and E07: each `HOW_TO_READ_RESULTS.md` and `results.md`.
- E08: `HOW_TO_READ_RESULTS.md`, `results.md`, `scripts/check_ce_gradient.py` (docstrings only), `scripts/make_report.py`, generated `results/tables.md`, `fig_complexity_vs_n.png`, `fig_complexity_correlation_vs_n.png`.
- Archive: E13 `HOW_TO_READ_RESULTS.md` and `experiments/archive/README.md`.
- One-page summary: HTML source and its README.

New stage-owned files are this report and `qa3_artifacts/` verification scripts, logs, receipts and render evidence. The required JSON completion marker is written last under the designated runtime directory. No benchmark reference, raw observation, frozen executable, preregistration or sealed data changed. No online page was republished.

The assigned builder should reconcile and commit these uncommitted changes, include the rebuilt PDF and regenerated page images, then resolve QA3-F7/F8 before submission. Preserve the published/source distinction until the page is republished. No further model-review round is requested here. If a separate packaging check is commissioned, the recommended configuration is Codex CLI, `gpt-6-astra`, `ultra`.

**Snapshot:**

- `qa3_codex.md`: six fixed findings and two minor open findings.
- `qa3_artifacts/verification_commands.json`: all COMMON commands succeeded.
- `qa3_artifacts/ce_gradient_check_rerun.json`: exact-CE impact check, with its one ridge-share exception.
- `qa3_artifacts/numeric_checks.json`: six complete sampled cells reproduced exactly.
- `qa3_artifacts/saved_audit.json`: saved grids, numeric tables and frozen-file checks.
- `qa3_artifacts/pdf_check.json`: four body pages, eight total, clean build.
- `qa3_artifacts/html_check.json`: 45 formulas, eight images, no page overflow.

## Verdict

**Accept with minor open issues.** No open critical or major finding remains within the reviewed scope. This is acceptance of the corrected scientific reporting and verified artifacts, not a claim that the frozen CE implementation has been repaired or that the organizers allow an eight-page submission.

**TLDR-end:** [judges-corr-vs-loss: qa3 final Mega QA] Six findings are fixed and required checks pass; frozen measurements and the four-page body are preserved. Submission packaging and older acronym hygiene remain open for the builder.

```text
VERDICT: FIXED
CRITICAL_ISSUES: 0
MAJOR_ISSUES: 0
FIXES_APPLIED: 6
STRUCTURAL: IMPROVED
SUMMARY: Six findings fixed and all required checks pass; the paper's numerical claims and four-page body are preserved. Two minor submission and acronym issues remain open.
```
