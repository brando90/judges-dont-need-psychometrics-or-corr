# Project: judges-dont-need-psychometrics-or-corr

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/AGENTS.md>

**TLDR:** Shared agent policy lives in agents-config. Read it first, then the project facts below. `AGENTS.md` and `CLAUDE.md` are identical.

## Start here

At each new non-trivial task, refresh and read the home-level index `~/agents-config/INDEX_RULES.md`, and follow its triggered rules:

```bash
if [ -d "$HOME/agents-config/.git" ]; then git -C "$HOME/agents-config" pull --ff-only
else git clone https://github.com/brando90/agents-config.git "$HOME/agents-config"; fi
```

If you are offline, read <https://raw.githubusercontent.com/brando90/agents-config/main/INDEX_RULES.md>.

## Research precision (keep near the top)

Read every setting, definition, number and citation from this repo (experiment READMEs, scripts, run records, the paper) before writing it. Write `unknown` only after a documented search, and never guess. **Never fabricate results or citations.** Every number in the README or paper must be printed by a script in `experiments/`. Planned work stays marked as planned.

## Thesis

LLM/agent judges should be fit and evaluated by matching the human score distribution, i.e. minimizing E[KL(p_human(r|y,y*) ‖ p_θ(r|y,y*))], not by maximizing Pearson/Spearman correlation or applying psychometric validity frameworks. For human-readable reporting on bounded scores, use MSE on normalized scores. Source: Fathom call <https://fathom.video/calls/893310061>, with notes in `paper_latex_and_notes/notes/fathom_call_11-07-2026/notes.md`.

## Layout (agents-config Rule 49: root buckets only)

- `experiments/NN_<setup_name>/`: one home per experiment, named after its setup (Rule 39), with code, data and results colocated. Index: `experiments/README.md`. Finished strands go to `experiments/archive/`.
- `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/`: paper dir. Main file `00_main_aaai_2027.tex`, sections `1-*.tex` … `7-*.tex`, bib `00_main.bib`, macros `math_commands.tex`, and the AAAI-27 kit `aaai2027.sty`/`.bst`. Notes go in `paper_latex_and_notes/notes/`.
- `src/` (create when needed): shared code and judge prompts.

## Paper (agents-config Rule 35)

Venue: AAAI-27 workshop "Effective and Efficient Evaluation for AI Scientists" (<https://eval-ai-scientists.github.io/aaai27/>). It is a position paper of 2–4 pages excluding references, in the AAAI-27 two-column kit, anonymized, submitted through OpenReview, due 11-20-2026 06:00 UTC. Build with `cd paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL && make` (tectonic, or `make latexmk`). Aux files go to the gitignored `_build/`. Commit the rebuilt `00_main_aaai_2027.pdf` in the same commit as any semantic `.tex`/`.bib` change.

## Experiments

Python 3.10+ with numpy/scipy. Run with `python experiments/<folder>/<script>.py`. Scores are on a 0–5 integer scale unless stated otherwise. Laplace-smooth histograms (alpha=1) before computing KL. Fix seeds.
