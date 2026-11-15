# Mega QA of experiment 04 and its paper appendix: common brief

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/qa/mega_qa_11-14-2026/COMMON.md>

**TLDR:** Brando requested a sequential Mega QA on 11-14-2026: Carbon (Antigravity) → Codex → a fresh Carbon. Each stage reviews experiment 04 and the new paper appendix for correctness first, fixes what it can prove wrong, verifies, and writes a report plus a runtime marker.

## Roles and checkout

- Builder: Claude Code (`claude-opus-5-5`). Reviewers: stage 1 `qa1_carbon` (Antigravity CLI, `gemini-4-carbon-high`, effort high); stage 2 `qa2_codex_vals` (Codex CLI, `gpt-6-astra`, effort ultra, full access; Brando asked for the Stanford account, its login had expired and the personal account had hit its usage limit, so the Vals account runs this stage with his approval: "Any account is fucking fine"); stage 3 `qa3_carbon` (fresh Antigravity session, same model).
- Work only in the owned checkout `/Users/brandomiranda/.worktrees/judges-dont-need-psychometrics-or-corr/mega-qa-04-1114` (detached HEAD; base `d9d1c1d` on `origin/main`; builder commit `03ada1e` adds the appendix). You are the only writer during your stage.
- Do not commit, push, send email, launch other agents or QA, or touch the main checkout `/Users/brandomiranda/judges-dont-need-psychometrics-or-corr` (another agent works there). The builder commits and lands after the chain.
- Runtime directory for markers and logs: `/Users/brandomiranda/.local/state/judges/mega-qa-04-1114/`.
- This macOS volume is case-insensitive: stage briefs are `BRIEF_<stage>.md` and reports are `<stage>.md` in this folder. The builder commits each finished stage's changes locally, so `git log d9d1c1d..HEAD` shows one commit per stage.

## Scope

A. `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/`: `PLAN.md`, `protocol.json`, `FREEZE.json`, `scripts/` (gtheory_lib, test_metrics, check_counterexamples, run_d_study, run_historical, verify_results, make_figure, make_report, freeze), `results/` (JSON, CSV, PNG), `RESULTS_reliability_vs_accuracy.md`, `HOW_TO_READ_RESULTS.md`, `README.md`, `results.md`, `CKPT_g_theory_plan.md`, `SOURCES.md`; plus the experiment 04 rows in `experiments/README.md`.

B. Paper `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/`: new `9-appendix-reliability.tex`, the one-line change in `4-proposal.tex`, the new `shavelson1989generalizability` entry in `00_main.bib`, the `\input` in `00_main_aaai_2027.tex`, and the rendered `00_main_aaai_2027.pdf` (appendix pages). See `git diff d9d1c1d..HEAD`.

Source paper (read-only, git-ignored local copy): `experiments/04_.../sources/local/shavelson_webb_rowley_1989_generalizability_theory.pdf` and its extracted `.txt` (OCR artifacts in equations; page 925 rendered as `page_925.png`).

## What to check, in priority order

1. **Mathematics:** expected-mean-square coefficients and strata degrees of freedom; D-study error variances `v_delta`, `v_Delta` and coefficients `Erho2`, `Phi`; the stratum form of the balanced REML likelihood; ranked probability score (RPS), cross-entropy (CE), total variation distance (TVD), Brier definitions and their expected-excess identities; the A1 third-difference construction and the A1_max vertex enumeration; the population variance-component claims for i.i.d.-given-artifact panels; the shared-offset pooled-rater calculation. Check against the source paper where it applies.
2. **Code:** bugs, wrong estimands, seeds and random streams, finite-sample versus population confusion, tie handling, Phase C data handling (historical test split as primary set, task-level bootstrap, probability floor 1e-6, no per-item output).
3. **Numbers and claims:** every number in `RESULTS_reliability_vs_accuracy.md`, `README.md`, `results.md`, `HOW_TO_READ_RESULTS.md`, `CKPT_g_theory_plan.md` and `9-appendix-reliability.tex` matches the saved outputs; verdicts follow the frozen contract honestly; amendments and post-hoc diagnostics are labeled; nothing overclaims (synthetic counterexamples are not prevalence; Phase C is retrospective with prior exposure).
4. **Result clarity** (repo `CLAUDE.md`, "Reporting results", items 1–5): split, metric definition with range, direction and reference value, subtraction order in words, uncertainty type/level/unit/draws, and how a judge or design was chosen.
5. **Paper appendix:** correct against the results; anonymized (no author, account, repository or URL that identifies the authors in the rendered text); one sentence per line; acronyms spelled out at first use; terminology consistent with the main text (judge, rating, artifact, conditional rating distribution, proper probability losses); citation entry fields correct; builds cleanly.

## Rules for fixes

- Make minimal, evidence-backed fixes to any file in scope; leave a disagreement you cannot prove as an open finding.
- Files hashed in `FREEZE.json` are frozen measurement code. Change one only to fix a real bug; then run `python3 scripts/freeze.py --amend "<reason>"`, rerun the affected scripts, regenerate the report with `python3 scripts/make_report.py`, and rerun `python3 scripts/verify_results.py`. Never hand-edit `results/*.json`, `results/*.csv` or `RESULTS_reliability_vs_accuracy.md`; they are generated.
- Rerunning scripts rewrites `runtime_seconds` fields and the output hashes in the report. If your stage did not change measurement code, confirm the regenerated outputs agree apart from those fields, then restore them with `git checkout -- experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/results experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/RESULTS_reliability_vs_accuracy.md`.
- Phase C reads `experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/private/` (a git-ignored symlink to private historical ratings with prior exposure). Read it only through the scripts. Never copy per-item data, artifact names or rater columns into any file, report or terminal summary.
- Paper changes: rebuild with `cd paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL && make`; confirm `_build/00_main_aaai_2027.log` has no lines starting with `!` and no undefined citations or references; check `python3 ~/agents-config/scripts/latex_sentence_lines.py --check <file.tex>`. The AAAI style sets `secnumdepth` to 0, so a `\ref` to a section prints nothing; name sections in words.
- Do not strengthen a scientific claim beyond the evidence. Spell every date MM-DD-YYYY in prose.

## Verification commands (from the experiment folder)

```bash
python3 scripts/test_metrics.py          # expect 945 checks, 0 failures
python3 scripts/check_counterexamples.py # exits 1: 2 of 16 frozen predictions miss (explained in the report)
python3 scripts/run_d_study.py           # exits 1: B.formula 35/36 (explained in the report)
python3 scripts/run_historical.py        # about 30 s
python3 scripts/verify_results.py        # expect 134 checks, 0 failures
python3 scripts/make_figure.py
python3 scripts/make_report.py
```

## Deliverables

1. Report `experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/qa/mega_qa_11-14-2026/<stage>.md` with a `# Title`, `**TLDR:**`, your actual model and effort, what you read (files and how much of each), a findings table (id, severity critical/major/minor/nit, file:line, evidence, fix or "open"), files changed, verification commands with observed outputs, open issues, and a verdict (accept / accept with minor open issues / reject).
2. Marker `/Users/brandomiranda/.local/state/judges/mega-qa-04-1114/<stage>.json` with: `stage`, `status` (complete or incomplete), `model`, `effort`, `started_at`, `finished_at`, `findings_by_severity`, `open_critical`, `open_major`, `files_changed`, `verification` (command → result), `verdict`. Write the marker last; it signals completion.
3. Time limit: 75 minutes. Write the report and marker before it expires, marking anything unfinished.

**TLDR-end:** [judge objectives: mega QA of 04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026] Three sequential reviewers check experiment 04 and its paper appendix for correctness, fix what they can prove, and leave a report and marker each.
