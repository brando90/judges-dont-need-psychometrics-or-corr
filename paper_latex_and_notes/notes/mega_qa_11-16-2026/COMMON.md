# Mega QA of the learning study, experiments 06–08 and the paper: common brief

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/paper_latex_and_notes/notes/mega_qa_11-16-2026/COMMON.md>

**TLDR:** Brando requested a Mega QA on 11-16-2026 ("mega qa, go"). Three sequential reviewers (Codex, then Claude, then a fresh Codex) check the work one Claude Code session did from 11-13-2026 to 11-14-2026: the paper's new learning-study section, RPS appendix and edits; synthetic experiments 06, 07 and 08; a new experiment 13 plot script; the one-page results summary; and the result-clarity rules. Each stage fixes what it can prove wrong, verifies, and writes a report plus a marker.

## Roles and checkout

- Builder and coordinator: Claude Code, `claude-opus-5-5` (desktop app session). The builder commits, republishes the one-page summary and lands on `main` after the chain.
- Stage 1 `qa1_codex`: Codex CLI, `gpt-6-astra`, reasoning effort `ultra`, full access.
- Stage 2 `qa2_claude`: Claude Code CLI, `claude-fable-5-1`, effort `max`, full access, fresh context (the builder's model family).
- Stage 3 `qa3_codex`: Codex CLI, `gpt-6-astra`, `ultra`, a fresh context.
- Work only in the owned checkout `/Users/brandomiranda/.worktrees/judges-dont-need-psychometrics-or-corr/mega-qa-1116` (detached HEAD at `233d504`, which is `origin/main` on 11-16-2026). You are the only writer during your stage.
- Do not commit, push, send email, publish, launch other agents or reviews, or touch the main checkout `/Users/brandomiranda/judges-dont-need-psychometrics-or-corr` (other agents work there). Leave your edits uncommitted; the builder commits each finished stage locally, so `git log 233d504..HEAD` shows one commit per stage.
- Runtime directory for markers and logs: `/Users/brandomiranda/.local/state/judges/mega-qa-1116/`.
- This macOS volume is case-insensitive: briefs are `BRIEF_<stage>.md` and reports are `<stage>.md` in this folder.
- Read the repository instructions first: `AGENTS.md` (identical to `CLAUDE.md`), especially "Research precision", "Reporting results: name the data, the metric and the comparison (Brando, 11-13-2026)" and "How we choose (learn) a judge". Shared policy lives in `~/agents-config/INDEX_RULES.md`; this brief already applies the parts that bind you.

## Base and scope

Base commit: `af810f5` (the state before the session's first commit). Review `git diff af810f5..233d504 -- <paths below>` plus the files themselves. Commits by other sessions are interleaved in that range; only the paths below are in scope.

A. **Paper** `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/` (AAAI-27 workshop position paper, due 11-20-2026 06:00 UTC):
   - New `5b-learning-study.tex` (section "Learning a Judge from Sparse Ratings") and its figure `figures/fig_ratings_needed.pdf`.
   - New `8-appendix-rps.tex` (ranked probability score (RPS) appendix) and `rps_appendix_examples.py`, which prints its example numbers.
   - Edits in `1-abstract.tex`, `2-introduction.tex`, `6-case-study.tex` (paragraph "Matched comparisons on the same ratings"), `7-discussion.tex`, `00_main_aaai_2027.tex`, `00_main.bib` (entries `epstein1969scoring`, `matheson1976scoring`, `schaeffer2024elusive`, `bellemare2017cramer`, `galdran2023performance`, `szekely2013energy`), `README.md`, and the rendered `00_main_aaai_2027.pdf`.
   - Read the whole paper for consistency. `9-appendix-reliability.tex` and its bib entry belong to another session and passed its own Mega QA on 11-14-2026: read it for consistency and report problems, but do not edit it.
B. **Experiments** (synthetic learning study; each folder's `README.md` was committed before measurement and is its pre-registration):
   - `experiments/06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026/` (Spearman, Kendall and Pearson–Spearman mix objectives).
   - `experiments/07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026/` (optimizer budget and restarts).
   - `experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/` (four target distributions, validation-tuned ridge), including `scripts/make_paper_figure.py`, which writes the paper figure and prints the numbers the paper quotes.
   - In `experiments/archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026/`: only the new `scripts/plot_without_head_and_crossover.py`, the two figures it writes and `HOW_TO_READ_RESULTS.md`. The rest of experiment 13 is the frozen upstream study the others build on: read it (setup, `results/raw.jsonl`), do not edit it.
C. **One-page results summary** `paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/judge-objectives-status.html` (source of a published page, Version 23, prepared for a meeting with Sanmi Koyejo) and its `README.md`, which maps each image to its experiment folder. Images are referenced by relative path and are not in that folder.
D. **Rules and reading guides:** the "Reporting results" section of `AGENTS.md`/`CLAUDE.md` (the two files must stay identical), `.cursor/rules/result-clarity.mdc`, every `HOW_TO_READ_RESULTS.md` under `experiments/` that this session added (00, 01, 02, 06, 07, 08 and archive 11–14), `experiments/README.md` (rows other than 04 and 05), `experiments/archive/README.md` and `experiments/archive/verify_migration.py`.

Out of scope: experiments 04 and 05 and their paper appendix (another session's work), the human-rating collection, and `~/agents-config`.

## What to check, in priority order

1. **Paper claims and numbers.** Every number in the in-scope paper text matches a printed script output or saved result: `make_paper_figure.py` output, `experiments/0{6,7,8}_*/results/tables.md`, `rps_appendix_examples.py`, and the experiment 02, 12 and 14 result files cited in `6-case-study.tex`. Statements follow from the evidence and say what the evidence is (synthetic data, one input feature, one judge family, 100 seeds or 30 seeds, paired 95% bootstrap over seeds, "from n at every larger size"). Check the RPS appendix mathematics: the cumulative (survival-function) form, propriety, the expected-excess identity, the Cramér-distance and energy-distance relations, the absolute-error form, the sum-of-survival-function identity for the mean, and every worked example. Check the figure and its caption against the data.
2. **Code correctness** in experiments 06, 07 and 08, the experiment 13 plot script and `make_paper_figure.py`: data generation matches experiment 13; objectives (cross-entropy (CE), RPS, Pearson, smooth Spearman and Kendall surrogates, mix); ridge; L-BFGS-B settings and starts; cross-validation and the ridge-selection fallback; the constant forecast; test metrics against the true distribution; paired bootstrap; crossover definitions; random-number streams and seeds; finite-sample versus population confusions. The full sweeps take up to an hour each: do not rerun them. Recompute a few cells (small and large n) with the scripts' own functions and compare them with `results/raw.jsonl`.
3. **Results files.** `results.md`, `results/tables.md`, `README.md` and `HOW_TO_READ_RESULTS.md` of 06, 07 and 08 match the saved outputs; verdicts follow each pre-registration honestly; amendments and post-hoc analyses are labeled as such; nothing overclaims.
4. **Result clarity** (repository `AGENTS.md`, "Reporting results", items 1–5, plus its plot rules) in the paper, results files, figures and the one-page summary: data split, metric name with a one-line definition, range, direction and reference value; RPS always spelled out at first use; differences subtract the same metric for two named judges, with the order in words; uncertainty type, level, unit and draws; how each judge was chosen (search space, objective, data, optimizer, start, budget, whether the optimum is guaranteed), in captions too; axis labels name metric and split; gridlines.
5. **One-page summary.** Its numbers and claims agree with the repository and the paper; the mathematics renders and is correct; figure numbers and captions match the images; nothing is stated that the repository does not support.
6. **Paper hygiene.** Anonymization for double-blind review: no author name, account, username, repository URL or other identifying string in the rendered PDF or the submitted sources, including bib fields that do not print (for example the companion-repository entry) and self-citations written in the first person. Page limit: read the call for papers at <https://eval-ai-scientists.github.io/aaai27/> and determine whether references and appendices count toward the 2–4 page limit; report how many pages the body uses now. One sentence per line (`python3 ~/agents-config/scripts/latex_sentence_lines.py --check <file.tex>`). Acronyms expanded at first use. One term per concept (`~/agents-config/skills/scientific-terminology/SKILL.md`). Clean build, no undefined citations or references.

## Rules for fixes

- Make minimal, evidence-backed fixes to in-scope files; leave a disagreement you cannot prove as an open finding. Do not rewrite working prose for taste.
- Measurement scripts of 06, 07 and 08 (`rank_objectives.py`, `run_rank_sweep.py`, `restarts.py`, `run_restarts_sweep.py`, `complexity.py`, `run_sweep.py`) are frozen with their pre-registrations. Change one only to fix a real bug; then record a dated amendment in that experiment's `README.md`, state which saved rows the bug affects, and report it. Do not rerun a full sweep; the builder decides on reruns.
- Never hand-edit generated outputs (`results/raw.jsonl`, `results/tables.md`, `results/run_log.txt`, figures). Report scripts (`make_report.py`, `make_paper_figure.py`, the experiment 13 plot script) may be fixed and rerun. A rerun that changes nothing but noise (for example PNG bytes) should be restored with `git checkout -- <file>`; confirm `tables.md` is byte-identical after a rerun of an unchanged report script.
- The private historical ratings of experiments 02, 12 and 14 are not in this checkout. Check the case-study numbers against the tracked result files only.
- Paper: rebuild with `cd paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL && make`; confirm `_build/00_main_aaai_2027.log` has no lines starting with `!` and no undefined citations or references. The AAAI style sets `secnumdepth` to 0, so a `\ref` to a section prints nothing; name sections in words. Keep one sentence per line.
- One-page summary: edit the HTML source only; the builder republishes it. Keep its structure.
- Do not strengthen a scientific claim beyond the evidence. Write dates in prose as MM-DD-YYYY.

## Verification commands (from the checkout root)

```bash
cd paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL && make && grep -c '^!' _build/00_main_aaai_2027.log; grep -i 'undefined' _build/00_main_aaai_2027.log; pdfinfo 00_main_aaai_2027.pdf | grep Pages; cd -
python3 paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/rps_appendix_examples.py
python3 experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/make_paper_figure.py
python3 experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/make_report.py > /dev/null
python3 experiments/07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026/scripts/make_report.py > /dev/null
python3 experiments/06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026/scripts/make_report.py > /dev/null
python3 experiments/archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026/scripts/plot_without_head_and_crossover.py
python3 experiments/archive/verify_migration.py
git status --short; git diff --stat; git diff --check
```

## Deliverables

1. Report `paper_latex_and_notes/notes/mega_qa_11-16-2026/<stage>.md` with: a `# Title`; a `**TLDR:**` line; your actual model, effort and account profile; what you read (files, and how much of each); a findings table (id such as `QA1-F1`, severity critical/major/minor/nit, file:line, evidence, fix or "open"); files changed; verification commands with observed outputs; open issues; a verdict (accept / accept with minor open issues / reject); then this block:

   ```
   VERDICT: PASS | FAIL | FIXED
   CRITICAL_ISSUES: [count open]
   MAJOR_ISSUES: [count open]
   FIXES_APPLIED: [count]
   STRUCTURAL: PASS | IMPROVED | SKIP
   SUMMARY: [1-2 sentences]
   ```
2. Marker `/Users/brandomiranda/.local/state/judges/mega-qa-1116/<stage>.json` with `stage`, `status` (complete or incomplete), `model`, `effort`, `account`, `started_at`, `finished_at`, `findings_by_severity`, `open_critical`, `open_major`, `files_changed`, `verification` (command → result) and `verdict`. Write the marker last; it signals completion.
3. Time limit: 100 minutes. Work in the priority order above. Write the report and marker before the limit, marking any coverage you did not finish.

**TLDR-end:** [judges-corr-vs-loss: mega QA of the learning study and paper] Three sequential reviewers check the paper's learning study, RPS appendix and edits, experiments 06–08 and the one-page summary for correctness, fix what they can prove, and leave a report and marker each.
