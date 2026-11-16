# Builder reconciliation after stage 2 (qa2_carbon)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/paper_latex_and_notes/notes/mega_qa_11-16-2026/BUILDER_RECONCILIATION_qa2.md>

**TLDR:** All five stage 2 fixes are accepted after checking the code, outputs and renders. Its sharper account of why 338 tuned CE judges moved (the floor clips unobserved categories and renormalizes) is correct and supersedes the observed-rating count quoted after stage 1. Its report's description of the check script's internals and four of its quoted numbers are wrong; the report is kept unchanged and corrected here.

Builder: Claude Code, `claude-opus-5-5`, 11-16-2026 11:40–11:55 PST. Stage 2 ran on Gemini 4 Carbon (High) in the Antigravity CLI, a substitute for the planned Claude Fable stage (see `COMMON.md`, Roles). Stage 2 report: [qa2_carbon.md](qa2_carbon.md) (unchanged).

## Stage 2 findings

| ID | Builder decision | Check |
| --- | --- | --- |
| QA2-F1 (major) | Accepted | The footer of experiment 08's correlation figure (Figure 9 of the one-page summary), added in `beed30a`, overflowed the 6-inch figure. The wrapped six-line footer is fully visible in the regenerated PNG; the other figure and `results/tables.md` are byte-identical. |
| QA2-F2 (minor) | Accepted | The added counter and the extended `reported()` were read line by line: `reported()` replays `make_report.py`'s single `default_rng(0)` bootstrap stream in the same order (six crossover pairs, three "tuned CE worse" lists, H1, two H2 lists per scenario, then the figure loops). The saved JSON shows 285,141 of 2,692,104 exact-CE evaluations in 1,907 of 4,400 experiment 08 rows with some category below 1e-6 (none in experiments 07 and 13), 3 with an observed rating below it, and of 424 reported quantities only the chosen-ridge share for C, n = 256 changes (69% → 68%). The mechanism therefore is renormalization after clipping unobserved categories at weak ridge, as stage 2 says; `BUILDER_RECONCILIATION_qa1.md` quoted only the observed-rating count. |
| QA2-F3 (minor) | Accepted | The appendix again spells out the Kullback–Leibler (KL) divergence at its first use. |
| QA2-F4 (minor) | Accepted | The one-page summary's notation section now covers ① and the synthetic 78-weight judge; the new line that ① is unrun on human ratings matches `experiments/archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026/README.md` ("No validation split (③ and ① are not run)"). |
| QA2-F5 (minor) | Accepted | Acronym expansions at first use in three reading guides and results files. |
| QA2-F6 (minor, open) | Open for Brando | (1) Whether references and appendices may follow a 4-page position paper; (2) the reliability appendix (another session's) has no pointer from the body; (3) `% Provenance:` comments in the `.tex` sources name repository folders and commits, which matters only if sources are ever uploaded (the submission is a PDF). |

## Corrections to the stage 2 report (report kept as written)

- Its audit of `check_ce_gradient.py` names functions and constants that do not exist in this repository: `cx.scenario_data`, `toy.init_weights`, `cx.select_ridge_ce`, `cx.eval_Lock`, `toy.sample_split`, `fit_head`, random streams `10_000 + seed * 97 + …` and `20_000 + seed * 97 + idx`, L-BFGS-B options `ftol=1e-12, gtol=1e-8`, a head penalty `1e-4 / n` and a dense-grid penalty `1e-6`. The actual code uses `default_rng(1000003 * seed + 7919 * n + code)` for data, `toy.marginal_W0` for starts, a fold permutation from `default_rng([seed, n, code, 99])`, `cx.val_scores` and `cx.pick` for ridge selection, `default_rng([seed, n, r, scenario == "B"])` for restarts, only `maxiter` and `maxfun` as optimizer options, a CE head fitted by `toy.fit` with ridge `1/n`, and a dense-grid ridge of `1/(4000 × 10^6)`. The conclusions it drew (the check reproduces each saved fit's data, start, ridge and budget) are nevertheless correct: the builder's and stage 1's reruns reproduce the saved rows exactly.
- It quotes the constant-forecast differences at 2,048 tasks as +0.0165, +0.0280, +0.0251 and +0.0072; `make_report.py` prints +0.0157, +0.0281, +0.0252 and +0.0074 (results.md rounds them to +0.016, +0.028, +0.025, +0.007).
- It attributes the Kendall convergence rates (97–100%) to experiment 06; they come from experiment 07.

**TLDR-end:** [judges-corr-vs-loss: mega QA reconciliation after stage 2] Stage 2's fixes are verified and accepted; its report's description of the check script's internals is unreliable and is corrected here, and the appendix page-limit question remains Brando's.
