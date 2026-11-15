# Mega QA Stage 1 Report — `qa1_carbon` (Experiment 04 & AAAI-27 Reliability Appendix)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/qa/mega_qa_11-14-2026/qa1_carbon.md>

**TLDR:** Stage 1 clean-eyes review (`Gemini 4 Carbon`, effort `High`) audited 100% of experiment 04 and the AAAI-27 reliability appendix (`9-appendix-reliability.tex`). All core mathematics, variance-component and proper-scoring-rule formulas, frozen measurement scripts (`FREEZE.json`), and reported numeric values match exact hand derivations and saved JSON outputs (`945/945` formula checks, `134/134` independent recomputations). We fixed 1 `major` claim-versus-measurement overstatement on Phase B assumption violations, 4 `minor` issues (`4-proposal.tex` sentence-per-line check failure, unexpanded `RPS` and missing draw count on plot axes, ambiguous "absolute error" / missing optimizer-and-choice context in appendix and reading guide, and an amendment-diff bug in `scripts/freeze.py`), and 1 `nit` status drift across READMEs/checkpoint, leaving only 2 out-of-scope `nit` items open. **Verdict: accept with minor open issues** (`0` open critical, `0` open major, `0` open in-scope minor).

## Reviewer identity and session bounds

- **Stage:** `qa1_carbon` (Stage 1 of 3, clean-eyes first reviewer; sole writer during stage)
- **Actual model:** `Gemini 4 Carbon`
- **Actual effort:** `High`
- **Owned checkout:** `/Users/brandomiranda/.worktrees/judges-dont-need-psychometrics-or-corr/mega-qa-04-1114` (detached `HEAD` at `03ada1e`, base `d9d1c1d` on `origin/main`)
- **Date:** 11-14-2026 (`started_at`: `2026-11-14T19:21:01-08:00`)

## Files read (coverage: 100% of in-scope files and source article)

| Group | File | Coverage read |
| --- | --- | --- |
| QA briefs & repo rules | `experiments/04_.../qa/mega_qa_11-14-2026/COMMON.md` | 57/57 lines (100%) |
| QA briefs & repo rules | `experiments/04_.../qa/mega_qa_11-14-2026/QA1_CARBON.md` | 9/9 lines (100%) |
| QA briefs & repo rules | `CLAUDE.md`, `experiments/README.md` | 95/95, 49/49 lines (100%) |
| Shared policy | `~/agents-config/INDEX_RULES.md`, `rules/writing.md`, `skills/review-paper/SKILL.md`, `skills/scientific-terminology/SKILL.md`, `writing/ml_research/ml_research_writing.md`, `scripts/latex_sentence_lines.py` | 100% of each file |
| Experiment 04 docs & protocol | `RESULTS_reliability_vs_accuracy.md` (187/187), `PLAN.md` (188/188), `protocol.json` (161/161), `FREEZE.json` (45/45), `HOW_TO_READ_RESULTS.md` (45/45), `README.md` (69/69), `results.md` (25/25), `CKPT_g_theory_plan.md` (30/30), `SOURCES.md` (55/55), `source_manifest.json` (23/23) | 100% of every file |
| Experiment 04 scripts | `gtheory_lib.py` (292/292), `test_metrics.py` (240/240), `check_counterexamples.py` (448/448), `run_d_study.py` (349/349), `run_historical.py` (213/213), `verify_results.py` (219/219), `make_figure.py` (86/86), `make_report.py` (320/320), `freeze.py` (41/41) | 100% of all 9 scripts |
| Experiment 04 outputs | `results/phase_a.json` (3,974/3,974), `results/phase_b.json` (3,717/3,717), `results/phase_c.json` (2,156/2,156), `results/verification.json` (701/701), `results/test_metrics.json`, `results/phase_b_*.csv`, `results/fig_reliability_vs_rps.png` (visual) | 100% of JSON/PNG outputs + CSV summaries |
| Source article (read-only) | `sources/local/shavelson_webb_rowley_1989_generalizability_theory.pdf` (11/11 pages, printed pp. 922–932, visually inspected), `.txt` (708/708 lines), `page_925.png` (visual) | 100% (all 11 pages) |
| AAAI-27 paper | `00_main_aaai_2027.tex` (83/83), `1-abstract.tex` (19/19), `2-introduction.tex` (22/22), `3-problem.tex` (22/22), `4-proposal.tex` (27/27), `5-toy-counterexample.tex` (66/66), `6-case-study.tex` (26/26), `7-discussion.tex` (17/17), `8-appendix-rps.tex` (150/150), `9-appendix-reliability.tex` (50/50), `00_main.bib` (61/61), `Makefile` (18/18), `README.md` (15/15), `_build/00_main_aaai_2027.log` & `.blg`, `00_main_aaai_2027.pdf` (6/6 pages visual + metadata scan) | 100% of every file and rendered page |
| Cross-experiment reference | `experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/README.md` & `HOW_TO_READ_RESULTS.md` (public files only; `private/` untouched except through `run_historical.py`) | 100% |

## Independent mathematical, code, and hand-recomputation audit

1. **Hand recomputation of `A1` (`j = 6`, $\theta_6 = 1/2$) and general $(6/5)\varepsilon^2$ identity:**
   - $p_6 = \frac{9}{10}\mathrm{Binomial}(5, 1/2) + \frac{1}{10}\mathrm{Uniform}\{0,\dots,5\} = \left(\frac{43}{960}, \frac{151}{960}, \frac{143}{480}, \frac{143}{480}, \frac{151}{960}, \frac{43}{960}\right)$, summing to $1$, with mean $\mu(p_6) = \frac{5}{2} = 2.5$ and variance $\mathrm{Var}(p_6) = \frac{9}{10}\cdot\frac{5}{4} + \frac{1}{10}\cdot\frac{35}{12} = \frac{17}{12}$.
   - Third-difference direction $\delta = (-1, 3, -3, 1, 0, 0)$ satisfies $\sum_{r=0}^5 \delta_r = 0$, $\sum_{r=0}^5 r\delta_r = 3 - 6 + 3 = 0$, and $\sum_{r=0}^5 r^2\delta_r = 3 - 12 + 9 = 0$, while $\sum_{r=0}^5 r^3\delta_r = 3 - 24 + 27 = 6 \ne 0$. With $\varepsilon = \min_r p_6(r)/6 = \frac{43}{5760}$, $q_6 = p_6 + \varepsilon\delta$ is strictly positive ($\min_r q_6(r) = \frac{43}{1152}$) with identical mean and variance.
   - Cumulative perturbation $\Delta_k = \sum_{r=0}^k \delta_r = (-1, 2, -1, 0, 0)$ for $k=0,\dots,4$ gives exact expected ranked probability score (RPS) excess $\frac{1}{5}\sum_{k=0}^4 (\varepsilon\Delta_k)^2 = \frac{\varepsilon^2}{5}(1 + 4 + 1) = \frac{6}{5}\varepsilon^2 = \frac{1849}{27648000} \approx 6.6876\times 10^{-5}$, and total variation distance (TVD) $\frac{1}{2}\sum_r |\varepsilon\delta_r| = 4\varepsilon = \frac{43}{1440} \approx 0.02986$. Matches `results/phase_a.json` (`A1.per_artifact[5]`) to the exact rational.
2. **Audit of `A1_max` vertex enumeration (`check_counterexamples.py:81-116`):**
   - Shifting $q = f + \tilde{q}$ with $f = \frac{1}{100}(1,\dots,1)$ leaves a 3-equality linear constraint system $(\sum \tilde{q}_r = 0.94,\ \sum r\tilde{q}_r,\ \sum r^2\tilde{q}_r)$ on $\tilde{q} \ge 0$. Because expected RPS excess $\frac{1}{5}\sum_{k=0}^4 (Q_k - P_k)^2$ is strictly convex in $q$, its global maximum over a bounded polytope is attained at an extreme point (support of $\tilde{q}$ of size $\le 3$). Since the Vandermonde matrix on any 3 distinct indices $i < j < k$ in $\{0,\dots,5\}$ is nonsingular, enumerating all $\binom{6}{3} = 20$ triples in `Fraction` arithmetic finds every vertex exactly.
3. **Hand recomputation of `A4` (`8x2`, `4x4`, `2x8`) and Phase B population errors (`C1` `4x8` and `V1`):**
   - For C1 $(v_u, v_a, v_b, v_{ua}, v_{ub}, v_{ab}, v_e) = (1, \frac{3}{5}, \frac{1}{10}, \frac{3}{5}, \frac{1}{10}, \frac{1}{10}, \frac{1}{2})$, single-rating variance is $1 + \frac{6}{5} + \frac{1}{5} + \frac{3}{5} = 3$, and $v_\Delta(n_a, n_b) = \frac{6/5}{n_a} + \frac{1/5}{n_b} + \frac{3/5}{n_a n_b}$:
     - `8x2`: $v_\Delta = \frac{3}{20} + \frac{1}{10} + \frac{3}{80} = \frac{23}{80} = 0.2875$, $\Phi = \frac{80}{103} \approx 0.777$.
     - `4x4`: $v_\Delta = \frac{3}{10} + \frac{1}{20} + \frac{3}{80} = \frac{31}{80} = 0.3875$, $\Phi = \frac{80}{111} \approx 0.721$.
     - `2x8`: $v_\Delta = \frac{3}{5} + \frac{1}{40} + \frac{3}{80} = \frac{53}{80} = 0.6625$, $\Phi = \frac{80}{133} \approx 0.602$.
     - C2 swaps $(a \leftrightarrow b, ua \leftrightarrow ub)$, reversing `8x2` and `2x8` (`23/80` vs `53/80`) with identical single-rating variance `3` and true oracle-minus-equal gap $\frac{23}{80} - \frac{31}{80} = -0.100$.
   - Phase B `C1` at `4x8` ($n_a=4, n_b=8$): $v_\Delta(4,8) = \frac{6/5}{4} + \frac{1/5}{8} + \frac{3/5}{32} = \frac{55}{160} = 0.34375$ and $v_\delta(4,8) = \frac{3/5}{4} + \frac{1/10}{8} + \frac{1/2}{32} = \frac{57}{320} = 0.178125$, matching `results/phase_b.json` (`formula_checks.C1.4x8_abs/rel`).
   - Phase B `V1` correlated rater pairs at `8x2` (4 pairs with correlation $\rho$): $v_\Delta^{\mathrm{V1}}(8,2;\rho) = \frac{1+\rho}{8}(v_a + v_{ua}) + \frac{v_b + v_{ub}}{2} + \frac{1+\rho}{16}v_{ab} + \frac{v_e}{16} = \frac{25(1+\rho)+21}{160}$, giving $\frac{117}{320} = 0.365625$ at $\rho=0.5$ and $\frac{71}{160} = 0.44375$ at $\rho=1.0$, matching `run_d_study.py:249-256` and `phase_b.json`.
4. **Audit of EMS, balanced REML likelihood, shared-offset pooled $\Phi$, and Phase C privacy/bootstrap:**
   - `gtheory_lib.py` expected mean squares, stratum degrees of freedom, and exact balanced Gaussian REML log-likelihood $\frac{1}{2}\sum_{s \ne \text{grand mean}} \mathrm{df}_s [\ln\lambda_s + \mathrm{MS}_s/\lambda_s]$ are algebraically exact for the orthogonal two-facet crossed design; whenever raw ANOVA estimates are non-negative (`256/256` runs in C1 and C2), $\lambda_s = \mathrm{MS}_s$ minimizes each convex term individually.
   - Shared $+1$ offset (`check_counterexamples.py:251-271`): two-point rater population $\{0, +1\}$ has population rater variance $\frac{1}{4}$ (or finite-facet factor $\frac{1}{2}$ under `df = 1`), giving exact single-rating $\Phi = \frac{1/2}{1/2 + 1/4 + 1/2} = \frac{2}{5} = 0.400$ vs $E\rho^2 = \frac{1/2}{1/2 + 1/2} = \frac{1}{2}$ (and finite-facet variant $1/3$), both reported transparently in `phase_a.json`.
   - `run_historical.py` uses the historical test split (`30` artifacts, `10` tasks, `78` ratings) as primary, applies the `1e-6` floor from experiment 02, resamples tasks with replacement (`10,000` draws), and writes only aggregate summaries into `results/phase_c.json` (zero per-item ratings, artifact names, or rater IDs leaked).
   - `FREEZE.json` SHA-256 verification passed (`6/6` frozen files match Amendment 2). Zero bugs were found in any frozen file, so `FREEZE.json` required no amendment.

## Findings

| ID | Severity | File:line | Evidence | Fix |
| --- | --- | --- | --- | --- |
| `QA1-F1` | `major` | `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/9-appendix-reliability.tex:48-49`; `experiments/04_.../scripts/make_report.py:55-59`; `experiments/04_.../README.md:48`; `experiments/04_.../results.md:19` | `9-appendix-reliability.tex:48` claimed *"The choice failed only when an assumption the theory itself warns about was violated: correlated raters, heterogeneous task groups, a shifted artifact population, and dependent occasions in the calibration study."* In `scripts/run_d_study.py:240-345` and `results/phase_b.json:3349-3636`, V1, V2, and V4 held the design fixed (`8x2` or `2x8`) and tested **error/coefficient prediction**, not design selection (`8x2` also remains analytically optimal in V1 and in both subgroups of V2). Under V3-S1 ("a shifted artifact population", $v_u: 1\to 0.25$), design selection was run and `8x2` **remained the optimal design** (`best_design_under_shift = "8x2"`, chosen `256/256`; only predicted $\Phi$ failed, `0.775` vs `0.465`). Design selection itself reversed only under V3-S2 ($v_{ub}: 0.1\to 1.2$, chosen `8x2` in `256/256` when `4x4` had lower deployment squared error by `-0.147` [`-0.198`, `-0.097`]), which the appendix omitted. Similar wording conflating prediction failure with design-selection failure appeared in `make_report.py:58`, `README.md:48`, and `results.md:19`. | Fixed in `9-appendix-reliability.tex:48-50` (including exact V2, V4, V3-S1, and V3-S2 numbers from `phase_b.json`), `make_report.py:58-59`, `README.md:48`, and `results.md:19`; rebuilt `00_main_aaai_2027.pdf` and `RESULTS_reliability_vs_accuracy.md`. |
| `QA1-F2` | `minor` | `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/4-proposal.tex:19` | Builder commit `03ada1e` left two sentences on line 19 (`where $Q_j=\sum_{k=0}^j q_k$. The appendix on the ranked probability score...`), causing mandatory check `python3 ~/agents-config/scripts/latex_sentence_lines.py --check 4-proposal.tex` (`COMMON.md:36`) to exit `1`. | Split `4-proposal.tex:19-20` onto two lines and rebuilt `00_main_aaai_2027.pdf`; `latex_sentence_lines.py --check` now exits `0` on all in-scope `.tex` files. |
| `QA1-F3` | `minor` | `experiments/04_.../scripts/make_figure.py:52,76,78`; `results/fig_reliability_vs_rps.png`; `scripts/make_report.py:98` | `CLAUDE.md:45,47,48` requires spelling out `"ranked probability score (RPS)"` on plot titles/axis labels, reporting bootstrap draw counts (`10,000` draws), and stating in figure captions how shown forecasts/judges were chosen. `make_figure.py` used bare `RPS` in Panel A y-axis, Panel B title, and Panel B y-axis and omitted `10,000 draws`; `make_report.py` caption lacked the one-line non-fitted/cached-judge note. | Updated `make_figure.py:52,76,78`, regenerated `results/fig_reliability_vs_rps.png` (visually verified centered two-line labels with zero clipping/overlap), and added the one-line choice summary to `make_report.py:98-100`. |
| `QA1-F4` | `minor` | `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/9-appendix-reliability.tex:44-47`; `experiments/04_.../HOW_TO_READ_RESULTS.md:5,22-55`; `scripts/make_report.py:81-248` | `9-appendix-reliability.tex:47` wrote `"held-out absolute error"` for Phase B's mean squared deviation around the universe score ($v_\Delta$, potentially confused with L1 MAE elsewhere in the paper) and omitted the subtraction direction gloss (`negative means lower error`), calibration/evaluation sizes (`64x8x8` / `256` fresh artifacts), and that constrained REML equalled closed-form ANOVA in all `256/256` runs (`CLAUDE.md:44-48`). `HOW_TO_READ_RESULTS.md` lacked a `How forecasts, designs and judges were chosen` section (`CLAUDE.md:48`), and first-use acronyms (`RPS`, `CE`, `TVD`, `REML`, `ANOVA`, `CI`, `SD`, `MAE`, `ICC`) were unexpanded in `HOW_TO_READ_RESULTS.md` and `make_report.py`. | Clarified `9-appendix-reliability.tex:44-47`, added `## How forecasts, designs and judges were chosen` plus V3-S2 subtraction gloss and first-use acronym expansions in `HOW_TO_READ_RESULTS.md`, and expanded first-use acronyms in `make_report.py` (`RESULTS_reliability_vs_accuracy.md`). |
| `QA1-F5` | `minor` | `experiments/04_.../scripts/freeze.py:30` | In `--amend` mode, `scripts/freeze.py:30` diffed current hashes against initial `old['hashes']` rather than `old['amendments'][-1]['hashes']`, causing Amendment 2 in `FREEZE.json:19-22` to list `scripts/check_counterexamples.py` in `changed_files` even though its SHA-256 (`64a1a56...`) did not change from Amendment 1. | Fixed `scripts/freeze.py:30-31` to compare against `old['amendments'][-1]['hashes']` when prior amendments exist. Historical `FREEZE.json` was preserved unchanged (`COMMON.md:33`). |
| `QA1-F6` | `nit` | `experiments/README.md:16,34`; `experiments/04_.../README.md:7,67,69`; `results.md:5-6`; `CKPT_g_theory_plan.md:5,8,22` | `experiments/README.md:16` still read `(results commit follows)` after `a78a626`, and all four files still stated `"paper integration open"` after `03ada1e` (`11-14-2026`) added `9-appendix-reliability.tex` to `00_main_aaai_2027.tex`. | Updated all four files to record results commit `a78a626` and appendix commit `03ada1e` (`11-14-2026`), noting that a main-body cross-reference remains optional, and appended one `11-14-2026` log line in `CKPT_g_theory_plan.md`. |
| `QA1-F7` | `nit` | `experiments/04_.../source_manifest.json:22` | Retains `"experiment_status": "planned_not_run"` from the 11-13-2026 source-PDF retrieval receipt (`SOURCES.md:13`). `CLAUDE.md:12` keeps source receipts unchanged and `COMMON.md:16` omits `source_manifest.json` from editable scope. | `"open"` |
| `QA1-F8` | `nit` | `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/README.md:3,13`; `8-appendix-rps.tex`; `3-problem.tex:18` | Out-of-scope paper files: `README.md:3,13` still describes a five-page PDF with only `8-appendix-rps.tex` (now 6 pages including `9-appendix-reliability.tex`); `8-appendix-rps.tex` fails `latex_sentence_lines.py --check`; and `3-problem.tex:18` does not yet cross-reference `9-appendix-reliability.tex` (optional). All three files lie outside `COMMON.md:18,32` Scope. | `"open"` |

## Files changed

1. `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/4-proposal.tex`
2. `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/9-appendix-reliability.tex`
3. `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/00_main_aaai_2027.pdf` (rebuilt via `make`)
4. `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/make_figure.py`
5. `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/results/fig_reliability_vs_rps.png` (regenerated via `make_figure.py`)
6. `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/make_report.py`
7. `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/RESULTS_reliability_vs_accuracy.md` (regenerated via `make_report.py` against frozen `results/*.json`, preserving exact SHA-256 hashes and runtimes)
8. `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/HOW_TO_READ_RESULTS.md`
9. `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/freeze.py`
10. `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/README.md`
11. `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/results.md`
12. `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/CKPT_g_theory_plan.md`
13. `experiments/README.md`
14. `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/qa/mega_qa_11-14-2026/qa1_carbon.md` (this report)

## Verification commands and observed outputs

All experiment commands were run from `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/`:

1. `python3 scripts/test_metrics.py`
   - Observed: `{"n_checks": 945, "n_fail": 0}` (exit `0`; `results/test_metrics.json` byte-identical to `HEAD`).
2. `python3 scripts/check_counterexamples.py`
   - Observed: `{"contract_checks": 16, "failed": ["A1.finite_sample", "A3.sampled_reliability"], "runtime_seconds": 2.25}` (exit `1`, expected per `COMMON.md:43`; `git diff results/phase_a.json` confirmed agreement on every field except `runtime_seconds`, then restored to `HEAD` per `COMMON.md:34`).
3. `python3 scripts/run_d_study.py`
   - Observed: `{"contract_checks": 6, "failed": ["B.formula"], "runtime_seconds": 1.65}` (exit `1`, expected per `COMMON.md:44`; all `results/phase_b_*.csv` byte-identical to `HEAD` and `results/phase_b.json` identical apart from `runtime_seconds`, then restored to `HEAD`).
4. `python3 scripts/run_historical.py`
   - Observed: `{"contract_checks": 1, "failed": [], "runtime_seconds": 40.4}` (exit `0`; `results/phase_c.json` identical to `HEAD` apart from `runtime_seconds`, then restored to `HEAD`).
5. `python3 scripts/verify_results.py`
   - Observed: `{"n_checks": 134, "n_fail": 0, ...}` (exit `0`, run both before and after restoration/edits; `results/verification.json` byte-identical to `HEAD`).
6. `python3 scripts/make_figure.py`
   - Observed: wrote `results/fig_reliability_vs_rps.png` (exit `0`; visually inspected rendered PNG to verify two-line axis labels, expanded `ranked probability score (RPS)`, `10,000 draws`, gridlines, and zero text overlap or clipping).
7. `python3 scripts/make_report.py`
   - Observed: `wrote RESULTS_reliability_vs_accuracy.md` (exit `0`; `git diff RESULTS_reliability_vs_accuracy.md` verified only intended prose/acronym clarifications changed while all 5 SHA-256 hashes and runtimes `A 1.57 s, B 1.18 s, C 27.84 s` remained identical to `HEAD`).
8. `FREEZE.json` SHA-256 integrity check against Amendment 2 hashes:
   - Observed: `{'protocol.json': True, 'scripts/check_counterexamples.py': True, 'scripts/gtheory_lib.py': True, 'scripts/run_d_study.py': True, 'scripts/run_historical.py': True, 'scripts/test_metrics.py': True}`.
9. `cd paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL && make`
   - Observed: exit `0`; `00_main_aaai_2027.pdf` built cleanly (6 pages, 296,705 bytes); `_build/00_main_aaai_2027.log` had `0` lines starting with `!`, `0` undefined citations/references, and `0` overfull boxes; BibTeX `.blg` had `0` warnings; visually inspected all 6 rendered PDF pages and confirmed `Anonymous submission` and zero identifying author/repo/path metadata.
10. `python3 ~/agents-config/scripts/latex_sentence_lines.py --check 00_main_aaai_2027.tex 4-proposal.tex 9-appendix-reliability.tex`
    - Observed: exit `0` (empty output; all in-scope `.tex` files pass).

## Open issues

- **`QA1-F7` (`nit`, out of editable scope):** `experiments/04_.../source_manifest.json:22` retains `"experiment_status": "planned_not_run"` as a historical 11-13-2026 retrieval receipt (`CLAUDE.md:12`, omitted from `COMMON.md:16`).
- **`QA1-F8` (`nit`, out of `COMMON.md` Scope):** `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/README.md:3,13` still describes a five-page PDF with only `8-appendix-rps.tex`; `8-appendix-rps.tex` has multi-sentence lines under `latex_sentence_lines.py --check`; and adding a main-body pointer from `3-problem.tex:18` to `9-appendix-reliability.tex` remains an optional author choice after the QA chain.
- Open `critical`: `0`. Open `major`: `0`. Open in-scope `minor`: `0`.

## Verdict

**`accept with minor open issues`** — All mathematics, frozen measurement scripts, and numeric claims in experiment 04 and `9-appendix-reliability.tex` are verified, all in-scope `major`/`minor`/`nit` issues (`QA1-F1` through `QA1-F6`) are fixed and re-verified, and only two out-of-scope `nit` observations (`QA1-F7`, `QA1-F8`) remain open for the builder.

## Note on macOS APFS case-insensitive filename collision (preserved Stage 1 brief)

On this macOS APFS case-insensitive volume, `QA1_CARBON.md` and `qa1_carbon.md` in `qa/mega_qa_11-14-2026/` resolve to the same directory entry (likewise for `QA2_CODEX_SU.md` / `qa2_codex_su.md` and `QA3_CARBON.md` / `qa3_carbon.md`). The directory entry has been cased as `qa1_carbon.md` per `COMMON.md:53`, and the original 9-line `QA1_CARBON.md` brief is preserved verbatim below:

```markdown
# Stage 1 brief: qa1_carbon

**TLDR:** You are the first, clean-eyes reviewer. Read COMMON.md in full, inspect everything in scope independently, fix what you can prove wrong, verify, and write `qa1_carbon.md` plus the marker `qa1_carbon.json`.

- Start by stating your actual model and effort.
- Read `COMMON.md` (binding), then the builder's report `RESULTS_reliability_vs_accuracy.md`, `PLAN.md`, `protocol.json`, every script, and the appendix `.tex`. Read the source paper sections that define variance components and coefficients (printed pages 924–926).
- Recompute at least: one A1 artifact by hand (moments, RPS excess = (6/5)ε²), the A4 table entries for 8×2 and 2×8, and the Phase B population error for one design; check them against the code and outputs.
- Then run all verification commands in COMMON.md, review claims and clarity, and fix.
- No earlier stage exists; do not wait for one.
```

**TLDR-end:** [judge objectives: mega QA `qa1_carbon`] Verified 100% of experiment 04 and `9-appendix-reliability.tex` (`945/945` formula checks, `134/134` recomputations, hand derivations exact); fixed Phase B assumption-violation wording (`QA1-F1`), `4-proposal.tex` sentence-per-line failure (`QA1-F2`), figure/report/guide acronyms and selection notes (`QA1-F3`, `QA1-F4`), `freeze.py` amendment diff (`QA1-F5`), and status links (`QA1-F6`). Verdict: accept with minor open issues (`0` open critical/major).
