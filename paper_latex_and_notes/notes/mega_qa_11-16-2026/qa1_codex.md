# Mega QA stage 1: learning study, experiments 06–08 and paper

**TLDR:** Review complete; **reject pending resolution of one major numerical issue**. The six sampled runs reproduce the saved measurements exactly, but the inherited cross-entropy loss and gradient disagree when probability clipping activates. Ten findings were fixed in the paper, reporting code, figures, reading guides and HTML; the upstream numerical defect and the workshop's appendix-length policy remain open.

**Reviewer:** Codex CLI, actual runtime model `gpt-6-astra`, reasoning effort `ultra`, account profile `CODEX_HOME=/Users/brandomiranda/.codex-vals`. Runtime settings were checked in this session's `turn_context`, rather than inferred from the profile default. Full access: `danger-full-access`, approval `never`. No delegated reviews or provider experiment calls were launched. This is a model reading supported by deterministic Python and numerical checks, **not Lean-checked**.

**Review date:** 11-16-2026. Started at `2026-11-16T18:06:34Z`. Reviewed `af810f5..233d504` within the paths in COMMON.md. The starting checkout HEAD was `60f71a5d17008b202cc2c9eadf95506a30ca362e` (the coordinator's brief commit); scientific sources matched `233d504`. Changes remain uncommitted in the owned checkout. The completion time is recorded in the marker.

**Snapshot:**

- Six small/large sampled runs: all shared numeric fields match saved rows exactly.
- Experiments 06/07/08: complete unique grids of 2,200 / 540 / 4,400 rows.
- Frozen measurement scripts, preregistrations, raw rows and run logs: unchanged.
- Generated tables: all 156 / 104 / 188 Markdown table rows preserve baseline values.
- Paper: four body pages, eight total pages; no TeX errors, undefined citations/references or overfull boxes.
- HTML: 45/45 math blocks rendered; 8/8 referenced images loaded; no page or KaTeX errors.
- Archive preservation: 195 mapped files verified, including 194 byte-identical source files.
- Open: one major clipping-gradient defect; one minor submission-length clarification.

## Reading coverage and evidence

I read COMMON.md and BRIEF_qa1_codex.md in full, the project instructions, the refreshed shared index and relevant hard/research/reporting rules, and the assistant and scientific-terminology skills. I read the shared priorities and coding state. The stage-specific restriction to this checkout takes precedence over routine publication and shared-log actions.

| Scope | Coverage |
| --- | --- |
| Paper | Whole current numbered paper, main file, bibliography and README; all of the new learning section, RPS appendix and example script; the relevant base-to-target diffs. All eight rendered pages inspected. Appendix 9 was read for consistency and left unchanged; its experiment was not independently revalidated. |
| Experiments 06–08 | Full measurement/objective/runner/report scripts, preregistrations and amendment, results and reading guides. Generated tables recomputed; every raw row parsed and checked for grid completeness, duplicate keys and nonfinite fields. Raw JSON was checked programmatically, not manually line by line. |
| Archive 13 | Full upstream toy implementation and relevant setup/results, sampled saved rows and population reference; full added plot script and reading guide. Frozen upstream sources and measurements were read only. |
| Historical human evidence | Tracked experiment 02 comparison and the relevant aggregate results of archive 11, 12 and 14; relevant solver/report routines and all added reading guides. Private individual ratings were unavailable and were not reconstructed. |
| HTML | Whole source, equations, embedded chart data, image mapping and relevant diff. Browser execution and screenshots checked the math and loaded assets. The published page was not changed. |
| Rules and navigation | Result-clarity contract in AGENTS.md/CLAUDE.md and the Cursor rule; all requested reading guides (00, 01, 02, 06, 07, 08 and archive 11–14); both experiment indexes and the full migration verifier. Rows 04/05 and their measurements were left alone. |
| Citations | Primary-source metadata and claim-relevant passages for the six named RPS references; not a claim to have read every cited paper end to end. Sources are linked below. |

For synthetic comparisons here, the held-out data are **500 test tasks per seed**, excluded from fitting and validation selection, with 100 seeds in 06/08 and 30 in 07. A/B reuse earlier synthetic tasks and therefore are not fresh independent replications. Ranked probability score (RPS) excess is the mean squared cumulative-probability gap to truth, in [0,1], with zero optimal; cross-entropy (CE) excess is Kullback–Leibler divergence from truth to forecast, in nats, zero optimal. Both are lower-is-better. Bootstrap intervals are paired 95% percentile intervals over seeds with 10,000 draws, pointwise rather than simultaneous over sizes. The numerical checks below are deterministic reproduction/gradient diagnostics and have no statistical interval.

Path abbreviations in the findings table refer to the unique experiment folders numbered E06, E07, E08 and archive E13. `P/` is `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/`; `H` is `paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/judge-objectives-status.html`. Line numbers refer to the reviewed/final files; the original text is recoverable from `233d504`.

## Findings

| ID | Severity | File:line | Evidence | Fix or open |
| --- | --- | --- | --- | --- |
| QA1-F1 | **major** | E13 `scripts/toy.py:40`; P `5b-learning-study.tex:30` | CE clips and renormalizes softmax probabilities for its value, but returns `Q-P` as the logit gradient. A finite-difference counterexample has maximum gradient discrepancy 0.999999. Instrumenting an existing E08 validation fit encounters clipping below 1e-6 on observed target mass. | **OPEN.** Upstream source is explicitly frozen and outside the allowed edit scope. Added visible limitations to paper, HTML, results and guides; no measurements changed. Details below. |
| QA1-F2 | major | E06 `results.md:35`; P `5b-learning-study.tex:45` | The preregistered prediction included rank/mixed losses being at least as large as Pearson after crossover. At B, 2,048 tasks, mean test RPS excess is 0.0623 / 0.0330 / 0.0851 for Spearman/Kendall/mix versus 0.0951 for Pearson. All remain above CE's 0.0037. | **Fixed.** Verdict now says the prediction holds only in part. Paper no longer implies rank objectives cannot improve on Pearson. Preregistration unchanged. |
| QA1-F3 | major | E07 `results.md:7`; E07 `HOW_TO_READ_RESULTS.md:11`; H `:153` | Only one optimizer family was tested. CE's “best” equals its single long start; the head is refitted once on the selected Pearson score. Budgets are 400/600 to 2,000/4,000 iterations/evaluations, not uniformly fivefold. Restarts select the penalized objective. | **Fixed.** Claims limited to tested starts/budgets and RPS crossovers; CE/head exceptions and budgets stated. No claim of optimizer independence or a global optimum. |
| QA1-F4 | major | P `8-appendix-rps.tex:110`; H `:483` | Empty empirical bins make KL(forecast ∥ histogram) infinite, but the reverse direction is finite for positive observed-rating probabilities and equivalent to CE up to histogram entropy. Empirical cumulative-distribution variance is P(1−P)/m; the individual-rating RPS floor is P(1−P). Cancellation in expectation does not remove observed paired noise. | **Fixed.** Corrected direction, sampling factor and interpretation; retained CE and categorical Brier as valid sparse-rating losses. Added finite-sample checks to the example script. |
| QA1-F5 | major | P `8-appendix-rps.tex:129`; H `:498` | Normalized excess is one fifth of integrated squared CDF distance and one tenth of the stated energy-distance expression. “Root RPS” must distinguish excess from observed loss. Constant mean vectors have undefined correlation. With multiple ratings, the disagreement floor has an unbiased estimator. | **Fixed.** Constants, definitions, constant-vector caveat and n/(n−1) correction are explicit. Removed an unsupported claim about whether a standard named metric exists. |
| QA1-F6 | minor | P `5b-learning-study.tex:14`; E08 report scenario titles; H `:326` | For BetaBin(5, μκ, (1−μ)κ), variance is 5μ(1−μ)(κ+5)/(κ+1). Fixed concentration in C/D does not make variance constant. A's mean is nearly, not exactly, constant. | **Fixed.** Live figures/prose distinguish concentration from variance and qualify the target descriptions. Frozen preregistration descriptions remain historical evidence. |
| QA1-F7 | major | E13 `scripts/plot_without_head_and_crossover.py:107`; E13 reading guide; H `:409` | `population_limit` fits and scores on the same 4,000-point grid with near-zero ridge. It is not an unseen test set, a certified global solution or a proved n→∞ limit. Figure 4 uses hand-selected illustrative features. | **Fixed.** New plot and HTML label the dense-grid reference as in-sample; Figure 4 says illustrative feature values. Corrected calibrated-head correlations from the saved grid reference to 0.9987 and 0.9946. Frozen upstream output untouched. |
| QA1-F8 | major | P `00_main.bib:19` | Companion entries exposed a contributor/repository identity, username, URLs and revision pins; the rendered bibliography also contained the submitting author's name in a cited author list. | **Fixed.** Companion citations anonymized; ordinary Schaeffer citation retained in third person with `and others`. Original provenance remains in Git history. No identifying-string hits in current TeX/bib or PDF scan. |
| QA1-F9 | minor | E06/E07/E08 report scripts; P figure caption `5b-learning-study.tex:35`; all requested reading guides | Axes/captions lacked full metric/split or fitting details; guides omitted search guarantees and optimizer bounds. E08 uses a disclosed zero substitution for undefined correlations, which needed explicit missingness counts in its plot/HTML. | **Fixed.** Added metric definitions, labels, fit/selection details, interval method/draws, prior exposure and scope of guarantees; regenerated figures/tables. E08 retains all raw missingness and states 99 defined plus one undefined seed for each Pearson variant in C,n=4. Historical zero-substitution convention unchanged and labeled. |
| QA1-F10 | minor | P `README.md:3`; one-pager README and H | README described six pages and an obsolete three-page render check. HTML had a stale “formulation III not run” statement despite E08, an unsupported fixed MSE example without a specified score panel, and an unreceipted numerical-identity claim. | **Fixed.** Current page/build receipt, synthetic-versus-human formulation status, and source-versus-published version distinguished. Removed unsupported numerical claims; retained the correct algebraic example. Corrected double-rounded ratio endpoints to the script's 8.8–25.5 and 36.5–45.8, and the stale importer experiment number in the archive index. |
| QA1-F11 | **minor** | P `README.md:3`; workshop call | Current PDF has four body pages and eight total pages. Call says 2–4 pages for position papers, excludes references explicitly under full papers, and does not establish an appendix exemption. | **OPEN.** Current length is disclosed. Builder must establish the submission rule or prepare a compliant submission arrangement; this stage cannot certify appendix eligibility. |
| QA1-F12 | major | P `1-abstract.tex:16`, `2-introduction.tex:21`, `7-discussion.tex:19`; H | Universal “whenever the mean is informative” language exceeded four targets/one judge family. Introduction implied rank objectives were tested on all four targets, although they were tested on A/B only. A small-sample result did not identify the best possible strategy. | **Fixed.** Scope is the three informative-mean targets among four tested; rank comparisons are two-target evidence; small-sample interpretation and optimizer dependence are qualified. |

Finding totals: **0 critical, 8 major, 4 minor, 0 nit**. **Ten findings fixed; one major and one minor remain open.** The numerical caveat is not a refutation of CE propriety and does not, by itself, show that the reported crossovers change.

## Open numerical issue: exact evidence and required resolution

The frozen function computes a clipped CE value and a gradient inconsistent with that value when clipping is active. With logits `[0,-30,-2,1,-1,-3]` and all target mass on category 1, its reported gradient there is approximately −0.999999, while the central finite difference of its returned loss is numerically zero. The maximum discrepancy is 0.999999000001. Ordinary nonclipping checks of all six objectives had maximum discrepancy 8.04e-11, so this defect is localized rather than a general failure of the gradient derivations.

The bounded E08 reproduction for scenario C, n=4, seed=9 encountered one clipped CE evaluation inside a validation fit with **three training tasks** and ridge c=0.1: minimum unmodified softmax probability 3.136071708858948e-7, with target histogram mass 1/3 on the clipped component. That fit's final minimum probability was 0.02050646. Replacing CE **in memory only** by stable log-softmax loss and its correct gradient changed the selected row's numeric metrics by less than 1e-12 and did not change the fallback. This demonstrates an actual exercised inconsistency without claiming material harm in that row. It does not bound the effect on all saved fits or dense-grid references.

Reproduction: [check_clipping.py](qa1_artifacts/check_clipping.py), [clipping_probe.json](qa1_artifacts/clipping_probe.json), and [numeric_checks.json](qa1_artifacts/numeric_checks.json). The builder needs an authorized, versioned correction/adaptor for the frozen upstream CE objective, an amendment preserving the original evidence, and a justified impact check before accepting the empirical comparison. COMMON.md reserves full-sweep decisions for the builder; none was launched here.

## Verification commands and observed results

Every command required by COMMON.md was run before changes and after relevant changes. Exact commands, exit codes, elapsed times and log filenames are in [baseline_commands.json](qa1_artifacts/baseline_commands.json) and [final_commands.json](qa1_artifacts/final_commands.json). Unchanged report scripts initially reproduced all three tables byte-for-byte. After reporting edits, every original numeric table row remains identical; additions are explanatory text and corrected scenario names.

| Command (from checkout root; experiment abbreviations as above) | Observed result |
| --- | --- |
| `make -C paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL` | Exit 0. Eight pages; body ends on page 4. Zero `!` errors, undefined references/citations or overfull boxes; four underfull warnings. All eight page images inspected. |
| `python3 P/rps_appendix_examples.py` | Exit 0. All original example numbers reproduced; identity and finite-sample CDF checks pass. |
| `python3 E08/scripts/make_paper_figure.py` | Exit 0. Fixed-ridge RPS crossover at 64 tasks/192 ratings in A and 4 tasks/12 ratings in B/C/D; large-size ratios reproduce the paper. |
| `python3 E08/scripts/make_report.py` | Exit 0; both images regenerated; 188 baseline table rows unchanged. |
| `python3 E07/scripts/make_report.py` | Exit 0; figure regenerated; 104 baseline table rows unchanged. |
| `python3 E06/scripts/make_report.py` | Exit 0; both images regenerated; 156 baseline table rows unchanged. |
| `python3 E13/scripts/plot_without_head_and_crossover.py` | Exit 0; both permitted images regenerated; dense-grid reference labeled correctly. |
| `python3 experiments/archive/verify_migration.py` | PASS: 195 mapped files, 194 byte-identical, one preserved navigation exception; 68 Python files parsed and 51 JSON files parsed. Source-repository removal/private inputs were not checked in this stage. |
| `python3 …/qa1_artifacts/check_numerics.py` | All six objective gradients pass away from clipping; six sampled runs exactly match shared saved numeric fields; clipped-gradient counterexample recorded. |
| `python3 …/qa1_artifacts/check_clipping.py` | Actual clipping reproduced; stable-CE counterfactual changes selected metrics by <1e-12. No source/output measurement edits. |
| `python3 …/qa1_artifacts/check_saved_evidence.py` | Complete unique saved grids; frozen files unchanged; tables' numeric rows unchanged; restart selection and fallback checked; paper hygiene and AGENTS/CLAUDE equality checked. |
| `python3 ~/agents-config/scripts/latex_sentence_lines.py --check <each paper .tex>` | Every paper TeX file passes, including unchanged appendix 9. |
| `python3 …/qa1_artifacts/check_html.py` | Chromium: 45/45 math blocks rendered, 8/8 mapped images loaded, zero page/KaTeX errors. Local asset mapping follows the one-pager README. |
| `git status --short`, `git diff --stat`, `git diff --check` | Changes restricted to the allowed scope plus this stage's report/evidence; whitespace check passes. No commit or push. |

Fresh sampled calls used each experiment's own `run_one` function, three ratings per training task, and 500 test tasks. These are bounded reproduction checks, not new estimates of comparative performance:

| Experiment / function | Sampled (scenario, training tasks, seed) | Comparison with saved numeric fields |
| --- | --- | --- |
| 06 / `rank_objectives.run_one` | (A,4,0), (B,2048,0) | Exact for both; includes CE reproduction check. |
| 07 / `restarts.run_one` | (A,4,0), (B,1024,0) | Exact for both, including recorded convergence/selection fields. |
| 08 / `complexity.run_one` | (C,4,9), (D,2048,0) | Exact for both; C row reproduces the declared undefined-validation-correlation fallback. |

Environment: Python 3.11.6, NumPy 2.2.6, SciPy 1.16.3. Saved E07 rows confirm **0/360** original Pearson fits succeeded at n≥32 and **326/360** long fits hit the budget. All best-start records select the minimum recorded penalized training objective. E08's one fallback is C,n=4,seed=9. Its raw undefined correlations remain present; no failed or undefined row was silently removed from the measurement record.

## Sources and page-limit check

- [Workshop call for papers](https://eval-ai-scientists.github.io/aaai27/): position-paper range checked; references exclusion is written under full papers; no appendix exemption found. Thus four body pages alone do not establish submission compliance for the eight-page PDF.
- [Epstein (1969), publisher record](https://journals.ametsoc.org/view/journals/apme/8/6/1520-0450_1969_008_0985_assfpf_2_0_co_2.xml) and [Matheson & Winkler (1976), publisher record](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.22.10.1087): bibliographic metadata and scoring-rule attribution checked.
- [Bellemare et al.](https://arxiv.org/html/1705.10743), especially the Cramér definition and sample-gradient discussion, supports distinguishing the integrated squared distance from its square root and its normalization here.
- [Székely & Rizzo](https://www.sciencedirect.com/science/article/pii/S0378375813000633): energy-distance attribution/metadata checked; the integer-scale identity was also checked numerically in the local example script.
- [Galdran](https://arxiv.org/html/2309.08701), proposed score in section 2.3: formula compared with the local counterexample; the proposed variant's smaller value for a false forecast was reproduced.
- [Schaeffer et al.](https://arxiv.org/html/2406.04391), relevant survival-function and distribution-comparison passages: supports the narrowly stated connection; it does not establish the RPS appendix's propriety claims.

## Files changed and handoff

**43 previously tracked files changed**, listed exactly in [tracked_files_changed.json](qa1_artifacts/tracked_files_changed.json): the permitted paper sections/bibliography/README/example script/PDF and figure; experiment 06–08 report scripts, results, tables, figures and guides; the permitted archive 13 plotting script/two figures/guide; the other requested reading guides and indexes; HTML source and its README. This report and `qa1_artifacts/` add logs, checks and renders. No measurement/preregistration/raw/log edits, no appendix-9 edits and no change to AGENTS.md/CLAUDE.md were needed. The [saved-evidence receipt](qa1_artifacts/saved_evidence_check.json) records the preservation checks.

**Verdict: reject.** QA1-F1 prevents acceptance of the empirical comparison until its full impact is resolved; QA1-F11 is also required before submission. The remaining mathematical, claim-scope, provenance and reporting findings are fixed and deterministically checked. The public one-pager still needs the builder's authorized publication step with the regenerated images.

For the already planned next stage: Claude Code CLI, `claude-fable-5-1`, effort `max`, fresh context, per COMMON.md; final fresh reviewer: Codex CLI, `gpt-6-astra`, `ultra`. These are handoff settings, not dispatched jobs. A collaborator-ready status sentence: “The first QA pass reproduced the sampled measurements and fixed the paper's math and reporting issues; one exercised CE clipping-gradient defect remains unresolved, so I have not accepted the empirical comparison yet.”

**TLDR-end:** [judges-corr-vs-loss: qa1_codex Mega QA] Completed the independent first-stage review, applied ten fixes and left reproducible evidence. Acceptance remains blocked by the frozen CE gradient defect; appendix-length eligibility also needs resolution.

```text
VERDICT: FAIL
CRITICAL_ISSUES: 0
MAJOR_ISSUES: 1
FIXES_APPLIED: 10
STRUCTURAL: IMPROVED
SUMMARY: Ten findings fixed and required checks completed; six sampled runs reproduce saved measurements exactly. Acceptance remains blocked by an exercised clipping-gradient defect in frozen upstream code, with submission appendix eligibility also unresolved.
```
