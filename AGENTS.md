# Project: judges-dont-need-psychometrics-or-corr

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/AGENTS.md>

**TLDR:** Shared agent policy lives in agents-config. Read it first, then the project facts below. `AGENTS.md` and `CLAUDE.md` are identical.

## Current priority — user clarification, 11-09-2026

The first practical priority is the best judge for the intended human-rating task, owned by [trustworthy-theorem-judges](https://github.com/brando90/trustworthy-theorem-judges). Pursue method 2 (correlation training) and method 4 (proper-loss training) fairly, with matched data, model family, harness and search budgets, validation-based hyperparameter selection, and a common held-out metric panel. This repository tests whether correlation-based optimization is inferior or an insufficiently justified default; its title and the user's hypothesis do not establish the outcome. Pearson measures linear association between predicted and human ratings; it does not constrain a neural judge or its input-to-rating mapping to be linear. High correlation can coexist with substantial signed bias and absolute error. Preserve scientific pins, frozen evidence, existing tests and active ownership; inspect the [completed experiment 03](experiments/03_exp06_ce_vs_pearson_matched_budget/RESULTS_comparison.md) and its [checkpoint](experiments/03_exp06_ce_vs_pearson_matched_budget/ckpt.md) before scheduling further work.

## Start here

At each new non-trivial task, refresh and read the home-level index `~/agents-config/INDEX_RULES.md`, and follow its triggered rules:

```bash
if [ -d "$HOME/agents-config/.git" ]; then git -C "$HOME/agents-config" pull --ff-only
else git clone https://github.com/brando90/agents-config.git "$HOME/agents-config"; fi
```

If you are offline, read <https://raw.githubusercontent.com/brando90/agents-config/main/INDEX_RULES.md>.

## Research workflow: no branches

This is a research project. Each experiment lives in its own numbered experiment folder (`experiments/NN_<setup_name>/`). Commit and push straight to `main`. Do not create branches or pull requests. If a leftover branch turns up, merge it into `main` (no force-push) and delete it. If two experiments pick the same number, keep the one already on `main` and renumber the new folder to the next free number, updating its internal references.

## Research precision (keep near the top)

Read every setting, definition, number and citation from this repo (experiment READMEs, scripts, run records, the paper) before writing it. Write `unknown` only after a documented search, and never guess. **Never fabricate results or citations.** Every number in the README or paper must be printed by a script in `experiments/`. Planned work stays marked as planned.

## Thesis

The supported thesis is that correlation alone does not identify a desired conditional human-rating distribution. Fit/evaluate probability forecasts with proper losses such as individual-rating cross-entropy, Brier score and ranked probability score; retain correlations as scalar diagnostics. Mean squared error identifies a conditional mean, not a distribution. One observed rating per item supports conditional log-loss evaluation; sparse empirical histograms are not known population distributions. The synthetic study does not establish human generalization, universal finite-sample superiority, or that every psychometric method is unnecessary. Original discussion: <https://fathom.video/calls/893310061>; preserve its historical notes.

## Layout (agents-config Rule 49: root buckets only)

- `experiments/NN_<setup_name>/`: one home per experiment, named after its setup (Rule 39), with code, data and results colocated. Index: `experiments/README.md`. Finished strands go to `experiments/archive/`.
- `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/`: paper dir. Main file `00_main_aaai_2027.tex`, sections `1-*.tex` … `7-*.tex`, bib `00_main.bib`, macros `math_commands.tex`, and the AAAI-27 kit `aaai2027.sty`/`.bst`. Notes go in `paper_latex_and_notes/notes/`.
- `src/` (create when needed): shared code and judge prompts.

## Paper (agents-config Rule 35)

Venue: AAAI-27 workshop "Effective and Efficient Evaluation for AI Scientists" (<https://eval-ai-scientists.github.io/aaai27/>). It is a position paper of 2–4 pages excluding references, in the AAAI-27 two-column kit, anonymized, submitted through OpenReview, due 11-20-2026 06:00 UTC. The bundled AAAI-27 style requires pdfTeX, so Tectonic is incompatible. Build with `cd paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL && make` (pdfLaTeX and BibTeX; or `make latexmk` when available). Aux files go to the gitignored `_build/`. Commit the rebuilt `00_main_aaai_2027.pdf` in the same commit as any semantic `.tex`/`.bib` change.

## Experiments

Python 3.10+ with numpy/scipy. Run with `python experiments/<folder>/<script>.py`. Scores are on a 0–5 integer scale unless stated otherwise. Follow each fixed experiment’s scorer definitions. The historical marginal toy uses Laplace alpha=1; the new conditional study uses strictly positive analytic forecasts and individual-rating proper losses. Fix seeds.

## Sister repo

Sister repo: https://github.com/brando90/trustworthy-theorem-judges (concrete application: theorem/formalization judges). Keep claims and terminology consistent across both; cross-link rather than duplicate.

Origin: this project arose from the trustworthy-theorem-judges study's question about choosing and training judges by correlation with human ratings. Pearson measures linear association of the resulting ratings; Spearman measures their ordering. Neither definition requires a linear input-to-rating mapping.
