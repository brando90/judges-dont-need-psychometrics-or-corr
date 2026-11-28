# Project: judges-dont-need-psychometrics-or-corr

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/AGENTS.md>

**TLDR:** Shared agent policy lives in agents-config. Read it first, then the project facts below. `AGENTS.md` and `CLAUDE.md` are identical.

## Notebook and screenshot sources — 11-21-2026

When Brando sends research images for an experiment, save the supplied image bytes and a dated transcription in both the canonical GitHub experiment folder and the project Google Drive folder. Verify the copies against the attachments. Keep private correspondence in restricted Drive storage and use a recoverable encrypted archive for those bytes in a public repository. Distinguish uncertain handwriting and overlapping screenshots in the transcription.

## Figures — Brando, 11-21-2026

Matplotlib figures: always `ax.grid(True)` (Brando, 11-21-2026). Apply it to every axis of every figure made or regenerated in this project.

## Proofs and conference empirical standard — Brando, 11-26-2026

Two standing requirements for this repository:

1. Every mathematical theorem or formal claim is checked both in Lean 4 and in Python. The Lean file has zero `sorry`, uses only the standard axioms (`propext`, `Classical.choice`, `Quot.sound`), and is checked with `#print axioms`. The compiler output is saved with the proof. A claim that has not been compiled is not called a proof.
2. Every paper or conference-facing empirical claim is backed by experiments on real multi-annotator human judging data. A synthetic law may verify a proof against a known target distribution, and is labeled as that check. The empirical report uses the unbiased per-sample proper risk at test time, a prompt-level train/validation/test split (no rater of a prompt in two splits), and item-clustered bootstrap 95% confidence intervals. The bar is a top machine-learning conference (NeurIPS, ICML, ICLR).

## Three-pillar research standard — Brando, 11-27-2026

The always-on rule `.cursor/rules/ml-research-standards.mdc` states this bar in one place; it extends the two sections above and below it.

1. **Lean 4.** General theorem, pinned toolchain, zero `sorry`, zero custom axioms, `#print axioms` (standard three only), compiler output saved. Mathlib files: `lake exe cache get`, never rebuild Mathlib. Machine-check only what compiles; both forms of anti-properness are Lean-checked: the squared form (`rho^2 < rho`, `anti_proper_from_parts` in experiment 36) and the square-root form (`r_collapsed = sqrt(rho) > rho = r_truthful` for a finite joint law, `PlanAudit.Model.anti_proper` in [`PlanClaims.lean`](experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/gemini_plan_claims_audit_11-26-2026/PlanClaims.lean), rebuilt 11-27-2026 with Lean 4.36.0 and Mathlib `fc4f59e5`, zero `sorry`, standard axioms only); accuracy/kappa arguments are "analytical" unless a zero-`sorry` lemma exists.
2. **pytest.** Every theorem has a deterministic pytest check (fixed seed, exact fractions or a named tolerance, claimed-false statements tested as failing cases, identical output twice).
3. **Pre-registered, leak-free real-data experiments.** `PROTOCOL.md` is committed before the analysis script runs (hypotheses, primary proper score matched to the label encoding, secondary metrics, resampling unit, paired tests, the meaning of "beats"). Real multi-annotator ratings, item-level splits with no rater of an item in two splits, the unbiased per-sample proper risk, item-clustered bootstrap 95% intervals (at least 1,000 draws), Holm-Bonferroni across declared cells, failures reported. "Beats" = lower mean held-out proper score and a significant pre-registered paired test.

## Human-rating identity and collection — Brando, 11-11-2026

- Every active personal human-rating Google Sheet and comma-separated values (CSV) form/export begins with `rater_name`, then `rater_id`. Prefill the actual name from the verified roster on every assigned row; keep the stable ID for joins and duplicate detection. Apply this to every participant.
- Keep Google Sheets and their CSV exports consistent in column names, order and values. Warm-up and main forms use the same schema within a study, with different assigned rows. Import/export by column name; verify identities and that scores, notes and other source fields survive unchanged.
- [trustworthy-theorem-judges experiment 09](https://github.com/brando90/trustworthy-theorem-judges/tree/main/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026) owns the shared VeriBench collection. Read its [identity/import workflow](https://github.com/brando90/trustworthy-theorem-judges/blob/main/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/support/IDENTITY_AND_IMPORT.md) and current form amendment before preparing or importing shared ratings. Link to that collection and its named files rather than starting a competing collection workflow here.
- Track the authorized named rating artifacts in that collection; do not anonymize names, ignore the files or move them into `private/` by default. Keep original imports, source receipts and delivered archives unchanged; make schema revisions as new versions. Copies are not new observations. Preserve consent, provenance, prior exposure and sealed splits; credentials, private correspondence, consent records and sealed data remain protected.
- Names are roster metadata, not ratings. Never invent scores, rationales, assistance disclosures or human attestations to fill gaps. Collection changes do not authorize collaborator emails or change frozen evaluation protocols.

## Current priority — user clarification, 11-09-2026

The first practical priority is the best judge for the intended human-rating task, owned by [trustworthy-theorem-judges](https://github.com/brando90/trustworthy-theorem-judges). Pursue method 2 (correlation training) and method 4 (proper-loss training) fairly, with matched data, model family, harness and search budgets, validation-based hyperparameter selection, and a common held-out metric panel. This repository tests whether correlation-based optimization is inferior or an insufficiently justified default; its title and the user's hypothesis do not establish the outcome. Pearson measures linear association between predicted and human ratings; it does not constrain a neural judge or its input-to-rating mapping to be linear. High correlation can coexist with substantial signed bias and absolute error. Preserve scientific pins, frozen evidence, existing tests and active ownership; inspect the [experiment 02 comparison](experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/RESULTS_comparison.md) and its [checkpoint](experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/ckpt.md) before scheduling further work.

**Prospective evaluation panel — user decision, 11-09-2026:** normalized individual-rating ranked probability score (RPS) remains primary and the common validation-selection criterion; cross-entropy (CE) and total variation distance (TVD) are secondary held-out metrics. TVD is `0.5 * sum_r abs(q_i(r) - p_hat_i(r))` against each artifact's empirical human rating distribution, averaged equally across artifacts. It is a sparse-rating distribution diagnostic, not TVD mutual information or a proper individual-rating loss. Use the same test artifacts and retain failures and missingness. See the [evaluation definitions](experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/EVALUATION_CRITERION_11-09-2026.md). Preserve completed experiments' frozen protocols, endpoints, executable sources and results.

## Start here

At each new non-trivial task, refresh and read the home-level index `~/agents-config/INDEX_RULES.md`, and follow its triggered rules:

```bash
if [ -d "$HOME/agents-config/.git" ]; then git -C "$HOME/agents-config" pull --ff-only
else git clone https://github.com/brando90/agents-config.git "$HOME/agents-config"; fi
```

If you are offline, read <https://raw.githubusercontent.com/brando90/agents-config/main/INDEX_RULES.md>.

## Research workflow: no branches

This is a research project. Each experiment lives in its own numbered experiment folder (`experiments/NN_<setup_name>/`). Commit and push straight to `main`. Do not create branches or pull requests. If a leftover branch turns up, merge it into `main` (no force-push) and delete it. If two experiments pick the same number, keep the one already on `main` and renumber the new folder to the next free number, updating its internal references.

Prefer a numbered experiment folder on `main` to a long-lived branch or a GitHub issue (Brando, 11-26-2026). The folder is the record: design, code, numerical results, figures, and proofs live in `experiments/NN_<setup_name>_start_<DDMonYYYY>/` and are committed on `main`. A branch or a GitHub issue is not a substitute for that folder, and it is not the place to leave an experiment for later consolidation.

## Merge to main, then a light QA — Brando, 11-25-2026

Merge to main (Brando, 11-25-2026): work goes to `main`. If in doubt, try to merge it, followed by a light QA (re-read the merged files on a fresh clone, run the quick tests, check the numbers you touched). No branches or pull requests.

## Check every proof with Lean and quick numerics — Brando, 11-26-2026

Brando, 11-26-2026: "you are checking your proofs with lean (and simple/quick numerical experiments) right? ... I'm asking cuz I think some of the stuff I sent you is wrong I think." Before any proof, derivation, identity or counterexample reaches Brando (chat, a note, the paper, slides), check it both ways and say which checks were run:

1. **Lean.** State the claim as a theorem and compile it with the pinned toolchain (`lean-toolchain`; `lake env lean` or `lake build`; agents-config Rule 20). Prove the general statement, not only an instance; if only an instance is proved, say so. Add one concrete instance showing the hypotheses can hold. Keep the `.lean` file, the observed compiler output and `#print axioms` (only `propext`, `Classical.choice`, `Quot.sound`; no `sorry`) in the experiment folder.
2. **Quick numerics.** A deterministic script (fixed seed; exact fractions where the quantity is rational, a named tolerance otherwise) that tests the same claim on many random instances and tests every statement claimed false as a failing case. It prints PASS or FAIL per claim, exits nonzero on a failure and is run twice with identical output. When a check fails, find out whether the claim or the check is wrong and report which.
3. **Audit Brando's inputs too.** Handwritten pages, printed notes and reports relayed from other agents (Cursor, Grok, ChatGPT) are claims, not facts: check each line the same way and report the wrong and imprecise ones first, with the corrected statement. Never silently repair his statement or my transcription of it.
4. **Say what was not machine-checked.** Label a claim "numerics only" (calculus facts), "cited" (a published theorem) or "not checkable" (a question, a modelling choice). Never write "verified" or "proved" for a claim that was only argued by hand.
5. **Order and fallback.** Run both checks before the claim is presented as established. If Lean is not available, install it (Rule 20) or mark the claim unchecked; never substitute "looks correct".

Reference implementation: [`claims_audit_11-26-2026/`](experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/claims_audit_11-26-2026/README.md) in experiment 35.

## Every proof in Lean 4, with a quick numerical check first — Brando, 11-27-2026

Brando, 11-27-2026: "always do proofs in lean", and "for every claim, a quick numerical experiment ... is run in parallel with, or before, the Lean proof". This extends the two sections above; where they differ, this one wins.

1. **Scope.** Every mathematical claim stated as a theorem, proposition, lemma, corollary or proof in the paper, the notes (`docs/notes/`), the docs or an experiment README is machine-checked in Lean 4 (Mathlib allowed). Cite the Lean file path and theorem name next to the claim (in the text, a footnote or a status table).
2. **Numerics first.** Before or alongside the Lean proof, run a quick seeded numerical experiment on the same statement: an exact-fraction or Monte Carlo script, plus a randomized counterexample search over finite distributions. Save the script, the seed and the output next to the claim's support files. A numerical check guides the work and catches false claims early; it never replaces Lean.
3. **Status labels.** Use exactly one of: "Lean-checked" (file and theorem cited), "numerically checked, Lean pending" (numerics passed, no Lean yet), "paper proof, not yet Lean-checked" (no machine check of either kind) or "Lean-checked special case" (state the case). Every claim that is not "Lean-checked" is listed in [`docs/LEAN_TODO.md`](docs/LEAN_TODO.md) with its location and what is missing.
4. **Exact scope.** The label states precisely what the Lean theorem proves: the readout (expected or sampled), the form (for example squared form `rho^2 < rho` versus square-root form `sqrt(rho) > rho`), the number field (`Rat` or `Real`), finite versus general laws, and any hypothesis the paper statement omits. If Lean proves less than the text says, change the text or the label.
5. **Build hygiene.** Lean files compile with no `sorry`, no `admit` and no new `axiom`; `#print axioms` lists only `propext`, `Classical.choice`, `Quot.sound`. Record the toolchain (`lean-toolchain`), the Mathlib revision (`lake-manifest.json`) and the exact build command next to the file, and save the compiler output. Fetch Mathlib with `lake exe cache get`; never rebuild it.

## Research precision (keep near the top)

Read every setting, definition, number and citation from this repo (experiment READMEs, scripts, run records, the paper) before writing it. Write `unknown` only after a documented search, and never guess. **Never fabricate results or citations.** Every number in the README or paper must be printed by a script in `experiments/`. Planned work stays marked as planned.

## Reporting results: name the data, the metric and the comparison (Brando, 11-13-2026)

Every reported number, table, plot and caption (results files, READMEs, the paper, slides, artifacts and chat) must let a reader answer these questions without opening code:

1. **Which data:** train, validation or held-out test; what was held out (tasks, task families, raters, seeds) and how many; and whether the held-out ratings were seen in earlier work (prior exposure). Label in-sample numbers as in-sample.
2. **Which metric:** its name, a one-line definition, its range or units, and whether lower or higher is better. Give a reference value (a baseline or typical value) so the size of a difference can be judged. Spell out every acronym at its first use in each document, table and plot axis title; in particular always write "ranked probability score (RPS)", because RPS is not a standard machine-learning metric. Its full definition, intuition and the reasons it is this project's primary metric are in the paper's appendix (`paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/8-appendix-rps.tex`).
3. **What is subtracted:** a difference always subtracts the **same metric on the same data** for two methods; never subtract two different metrics. Name methods by how they were trained or chosen ("Pearson-trained judge", "CE-trained judge"), not by a bare metric name, and write the order in words, for example "test RPS of the Pearson-trained judge minus test RPS of the CE-trained judge; positive means the CE-trained judge is better". Never write a bare "CE minus Pearson".
4. **Uncertainty:** interval type and level, resampling unit and number of draws.
5. **How the judge was chosen:** the search space, the objective and the data it was computed on, the optimizer or search procedure, its starting point and budget, and whether the best judge is guaranteed (exhaustive search, a convex objective or a checked certificate). When the objective is not convex, say that the result reflects that optimizer, start and budget. Put a one-line version in every figure caption that shows fitted or selected judges.

Plots: every axis label itself names its metric and split (for example "Held-out test RPS excess over the true distribution"), never only the panel title or caption; a difference axis also states its order; data plots with numeric axes have gridlines (major, plus faint minor lines on log axes), following agents-config's plotting guidance in `workflows/expts-and-results.md`; the plot or its caption says which direction is better and gives the scale of the underlying metric. Frozen or archived result files are not rewritten: every experiment folder has a `HOW_TO_READ_RESULTS.md` that answers these questions for its results, and a new experiment adds one when its first results appear. Cursor carries the same rule in `.cursor/rules/result-clarity.mdc`.

**Why:** Brando, 11-13-2026, on how judges were fitted: "It should be clear!" and "the plot we see could be a function of the 'optimizer'". Also on a results plot: "it needs to be clear if it's train or held out (unseen) test set ... I don't even know what the horizontal axis means, what is +0.05? What metric? train or test?" and, on a "CE minus Pearson" label, "I don't know what it means to subtract two different metrics."

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

## Paper (agents-config Rules 35 & 89)

Venue: AAAI-27 workshop "Effective and Efficient Evaluation for AI Scientists" (<https://eval-ai-scientists.github.io/aaai27/>). It is a position paper of 2–4 main body pages excluding references, in the AAAI-27 two-column kit, anonymized, submitted through OpenReview, due 11-20-2026 06:00 UTC. Per Trigger Rule 89, keep main-body conclusions on Page 4, keep the appendices (`8-*`, `9-*`, `9b-*`) roughly `1.5x` the main body (`~5.5–6.0` pages) with no forward `\ref` into supplementary labels, and place overflow tables/details in `10-supplementary.tex` after `\appendix` behind `\ifsupplementary`. The bundled AAAI-27 style requires pdfTeX, so Tectonic is incompatible. Build both `00_main_aaai_2027.pdf` (venue submission, no supplementary) and `00_main_arxiv.pdf` (arXiv edition via `00_main_arxiv.tex`, with supplementary) with `cd paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL && make` (pdfLaTeX and BibTeX). Aux files go to the gitignored `_build/`. Commit both rebuilt PDFs (`00_main_aaai_2027.pdf` and `00_main_arxiv.pdf`) in the same commit as any semantic `.tex`/`.bib` change.

## Experiments

Python 3.10+ with numpy/scipy. Run with `python experiments/<folder>/<script>.py`. Scores are on a 0–5 integer scale unless stated otherwise. Follow each fixed experiment’s scorer definitions. The historical marginal toy uses Laplace alpha=1; the new conditional study uses strictly positive analytic forecasts and individual-rating proper losses. Fix seeds.

## Sister repo

Sister repo: https://github.com/brando90/trustworthy-theorem-judges (concrete application: theorem/formalization judges). Keep claims and terminology consistent across both; cross-link rather than duplicate.

Origin: this project arose from the trustworthy-theorem-judges study's question about choosing and training judges by correlation with human ratings. Pearson measures linear association of the resulting ratings; Spearman measures their ordering. Neither definition requires a linear input-to-rating mapping.
