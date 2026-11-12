# Project: judges-dont-need-psychometrics-or-corr

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/AGENTS.md>

**TLDR:** Shared agent policy lives in agents-config. Read it first, then the project facts below. `AGENTS.md` and `CLAUDE.md` are identical.

## Human-rating identity and collection — Brando, 11-11-2026

- Every active personal human-rating Google Sheet and comma-separated values (CSV) form/export begins with `rater_name`, then `rater_id`. Prefill the actual name from the verified roster on every assigned row; keep the stable ID for joins and duplicate detection. Apply this to every participant.
- Keep Google Sheets and their CSV exports consistent in column names, order and values. Warm-up and main forms use the same schema within a study, with different assigned rows. Import/export by column name; verify identities and that scores, notes and other source fields survive unchanged.
- [trustworthy-theorem-judges experiment 09](https://github.com/brando90/trustworthy-theorem-judges/tree/main/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026) owns the shared VeriBench collection. Read its [identity/import workflow](https://github.com/brando90/trustworthy-theorem-judges/blob/main/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/support/IDENTITY_AND_IMPORT.md) and current form amendment before preparing or importing shared ratings. Link to that collection and its named files rather than starting a competing collection workflow here.
- Track the authorized named rating artifacts in that collection; do not anonymize names, ignore the files or move them into `private/` by default. Keep original imports, source receipts and delivered archives unchanged; make schema revisions as new versions. Copies are not new observations. Preserve consent, provenance, prior exposure and sealed splits; credentials, private correspondence, consent records and sealed data remain protected.
- Names are roster metadata, not ratings. Never invent scores, rationales, assistance disclosures or human attestations to fill gaps. Collection changes do not authorize collaborator emails or change frozen evaluation protocols.

## Current priority — user clarification, 11-09-2026

The first practical priority is the best judge for the intended human-rating task, owned by [trustworthy-theorem-judges](https://github.com/brando90/trustworthy-theorem-judges). Pursue method 2 (correlation training) and method 4 (proper-loss training) fairly, with matched data, model family, harness and search budgets, validation-based hyperparameter selection, and a common held-out metric panel. This repository tests whether correlation-based optimization is inferior or an insufficiently justified default; its title and the user's hypothesis do not establish the outcome. Pearson measures linear association between predicted and human ratings; it does not constrain a neural judge or its input-to-rating mapping to be linear. High correlation can coexist with substantial signed bias and absolute error. Preserve scientific pins, frozen evidence, existing tests and active ownership; inspect the [completed experiment 03](experiments/03_exp06_ce_vs_pearson_matched_budget/RESULTS_comparison.md) and its [checkpoint](experiments/03_exp06_ce_vs_pearson_matched_budget/ckpt.md) before scheduling further work.

**Prospective evaluation panel — user decision, 11-09-2026:** normalized individual-rating ranked probability score (RPS) remains primary and the common validation-selection criterion; cross-entropy (CE) and total variation distance (TVD) are secondary held-out metrics. TVD is `0.5 * sum_r abs(q_i(r) - p_hat_i(r))` against each artifact's empirical human rating distribution, averaged equally across artifacts. It is a sparse-rating distribution diagnostic, not TVD mutual information or a proper individual-rating loss. Use the same test artifacts and retain failures and missingness. See the [evaluation definitions](experiments/03_exp06_ce_vs_pearson_matched_budget/EVALUATION_CRITERION_11-09-2026.md). Preserve completed experiments' frozen protocols, endpoints, executable sources and results.

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

## How we choose (learn) a judge: the formulations (Brando, 11-09-2026)

**Any agent that prepares or dispatches the human-rating task packet reads this first and keeps the packet (instructions, rating scale, splits, what the ratings are used for) consistent with it.** Full notes and sources: judge paper Q&A Doc, Part 3 (<https://docs.google.com/document/d/1zB-gYk-zU1wToS_E_carGSRGtUcF1NgZ-jWF5t1SJws/edit>).

Brando's four formulations, from his handwritten pages (Θ = the judge search space; S^tr, R^tr = train tasks and ratings; S^val, R^val = validation tasks and ratings; p^H = the human rating distribution; p^AJ = the judge's):

- ① argmax over judge in Θ of Corr(p^AJ, p^H), train data helps, Corr evaluated on S^val, R^val. "This is what people are doing" (his TODO: check by deep research).
- ② argmax over judge in Θ of Corr(p^AJ, p^H), using S^tr, R^tr only (e.g. optimizing the system with an agent or prompt optimization).
- ③ argmin over judge in Θ of Loss(p^AJ, p^H) on the validation set S^val, R^val, train data helps.
- ④ argmin over judge in Θ of Loss(p^AJ, p^H) on S^tr, R^tr only. "What we do in practice": train (or tune) on the train set only, choose hyperparameters on validation or historical hyperparameters.

His conjectures (not tested): (a) ③ gives better evaluation results than ④ (it uses general machine learning knowledge instead of a hack on human correlations); (b) ③ is more principled, or at least easier to think about, since it uses the successful machine learning framework. He does not think ② is needed. His question: do both experiments, and which is best? He also wrote that "the two main formulations" are "② & ④"; whether he meant ③ and ④ or ② and ④ is being clarified. Until he answers, drop neither reading: his words, ② and ④ (develop a judge by training correlation vs by training loss; the collection note `experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/support/JUDGE_FORMULATIONS_11-09-2026.md` in trustworthy-theorem-judges reads it this way), and our reading, ③ and ④ (select on validation vs train only), need the same independent raw ratings and sealed splits, so keep all four formulations evaluable.

Historical recommendation (an unrun idea, superseded for the prospective human-rating panel by the RPS-primary, CE/TVD-secondary decision above): run ③ and ④ in one repeated 5-fold cross-validation loop over tasks with an untouched test fold; test cross-entropy is the primary metric; ③ wins only if the paired 95% interval of the difference excludes 0 and the gain exceeds a margin fixed beforehand; ties go to ④. No such experiment has been run.

What the literature check found (about 90 papers read by abstract and key passages, none end to end; 6 re-checked by the author; 11-09-2026): the usual justification of an LLM judge is agreement with humans on a test set (accuracy, kappa, Pearson/Spearman/Kendall). I did not find a paper that states formulation ① or ② as its principle. In the 21 Chinese-company papers opened (Seed1.5-Thinking, MiniMax-M1, LongCat-Flash-Thinking, DeepSeek-GRM, Kimi k1.5 and others) I did not find one choosing a judge by maximum correlation; they report accuracy on a human-labeled test set. Baidu ERNIE 4.5 could not be opened. The ③/④ direction (a loss against the human rating distribution) appears in 2025-26 papers, each also reporting correlation or accuracy; two papers report that correlation alone selects poorly (arXiv 2510.09738; arXiv 2503.05965, "as much as 31%" worse judges). I did not find a source comparing ③ against ④ at about 25 tasks, so conjecture (a) is neither supported nor refuted.

For packets: keep the validation and test splits by task family, solver and rater; keep test ratings out of judge selection; record which ratings were available to the judge search (train, validation, test) so ③ and ④ can be run later without re-collecting; use "rating" for the data and "judge" for the model (one term per concept).

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
