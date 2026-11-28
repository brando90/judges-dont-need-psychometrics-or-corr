# Corrected Claude Code prompt for Paper 1 (27 Nov 2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/prompts/claude_code_paper1_prompt_corrected_27Nov2026.md>

**TLDR:** Archive copy of the Gemini-written Claude Code prompt (main prompt, Stages 0 to 3, plus the "generality beyond RPS" addendum) with eight factual and process corrections applied inline, so it can be pasted into Claude Code as written. As of 27 Nov 2026, Brando is already running a corrected version of this prompt in Claude Code; this file is the record, not a new task.

## Status and provenance

- **Running copy.** Brando is running a corrected version in Claude Code as of 27 Nov 2026. That session's own text is authoritative for what it was actually told; this file is the archived, corrected reference.
- **Source.** The main prompt and the addendum were written by Gemini and pasted by Brando in chat on 27 Nov 2026 (messages t105u and t107u).

## Corrections applied

| # | Correction | Where applied |
| --- | --- | --- |
| a | Commit directly to `main`, not a feature branch (Brando's rule for this repository: `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/merge-to-main.mdc`, `.cursor/rules/experiment-folders-on-main.mdc`). Fetch latest `main` right before each push; never force-push; never rewrite others' changes. | Section 1, Stage 0.2, Stage 3 |
| b | Li et al., "Judging with Confidence" ([arXiv:2510.00263](https://arxiv.org/abs/2510.00263)), is **Findings of EMNLP 2026**, not ICLR 2026 (the ICLR 2026 OpenReview forum still reads "Submitted to ICLR 2026"). It applies Brier and log rewards to pairwise preference distributions and leaves ordinal/Likert scales to future work; that point stands. | Section 1, Stage 1, Stage 2, addendum |
| c | The G-theory paper "From LLM Judge Scores to Evaluation Design" (OpenReview [YawwpIWqTi](https://openreview.net/forum?id=YawwpIWqTi)) is a **withdrawn ICLR 2027 submission**. Do not cite it as published. The `Gtheory4LLM` CRAN package may be mentioned. | Section 1, Stage 1, Stage 2 |
| d | Alpha Parity, "Correlation Is Not Agreement" (OpenReview [67Z2nGMzNI](https://openreview.net/forum?id=67Z2nGMzNI)), is an **ACL ARR May 2026 submission**. Only the title and submission status are confirmed; its details (such as 3.06M judgments) are unverified. | Section 1, Stage 1, Stage 2 |
| e | Exp 43 is now **49 papers**: 29 of 30 scored/benchmark papers report correlation; 13 of 13 pairwise papers report agreement, accuracy or kappa; 43 of 49 have no item-level proper score. Lyu et al. ([2610.37868](https://arxiv.org/abs/2610.37868)), Li et al. and Choi et al. are already coded, so Stage 1.1 is mostly verification. The old "22/22" and "11/11" counts are replaced. | Stage 1, Stage 2, Stage 3, addendum |
| f | The "psychometric rebound" is an **emerging thread**: 1 of 49 coded papers reports IRT or G-theory reliability. Correlation and majority-label agreement remain dominant. | Section 1, Stage 1, Stage 2 |
| g | The thesis is **general**: a strictly proper score matched to the label space, plus a base-rate skill score `Skill_S = 1 - S(judge) / S(base-rate)` for CE, Brier and RPS. RPS is the ordinal flagship; CE and Brier are primary for unordered categories such as ChaosNLI. Accuracy against the majority label is proper but **not strictly proper** (it rewards only picking the mode). Alpha, kappa and IRT are agreement or reliability measures, not scoring rules at all. | Section 1, Stage 1, Stage 2, addendum |
| h | The withdrawn **arXiv:2606.19544** (Norman et al., "Reliability without Validity", withdrawn by arXiv admin on 3 Nov 2026) must not be cited anywhere. | Section 1, Stage 1, Stage 2, Stage 3 |

Other notes:

- The merged Google Doc (reference 1) is Gemini's synthesis, used as the merged answer but not edited. Where it disagrees with the corrections above, the corrections and [`experiments/43_llm_judge_validation_practice_survey_start_27Nov2026/README.md`](../../experiments/43_llm_judge_validation_practice_survey_start_27Nov2026/README.md) win. That README lists each fix: Li et al. venue, 37-paper counts, WMT25 soft pairwise accuracy (system level) versus tie-calibrated accuracy (segment level), LAGER (NeurIPS 2025), the scoring-bias paper (DASFAA 2026), RewardBench 2 (ICLR 2026), the truncated Lyu link, and the "rebound" wording.
- The venues given for Elangovan et al. (ICLR 2025), Guerdan et al. (NeurIPS 2025) and Chen et al. (NeurIPS 2025) match `papers.py` in Exp 43. The "Wang et al. (Findings EMNLP 2025)" entry has no matching row in `papers.py`; confirm it from the primary source before citing.
- Every number in a paper must come from a committed artifact. Do not use the "65.9%" figure or the "91-study medical review" from the Gemini doc; both are unverified.

---

# THE PROMPT (paste into Claude Code inside `~/judges-dont-need-psychometrics-or-corr`)

**Recommended model and effort.** Claude Opus at high effort for the full run, especially Stages 0 to 2 (audit, Exp 39 fix, Exp 43 and Lean updates, `paper_latex/` rewrite). If the main run is already finished and only the addendum is being applied as a follow-up, Claude Sonnet at medium effort is enough.

You are updating the repository `https://github.com/brando90/judges-dont-need-psychometrics-or-corr` and its local LaTeX manuscript (`paper_latex/`) to execute our finalized, Sanmi-aligned Paper 1 strategy. Working title: "Mirage for Ordinal AI Judges: Breaking the Broken Ruler of Correlation, Majority-Label Agreement, and Symmetric Psychometrics", **to be generalized per the framing update at the end of this prompt** (the thesis is a strictly proper score matched to the label space, not "RPS only").

## 0. CONTEXT AND REFERENCE LINKS (READ FIRST)

Read the local repo (`README.md`, `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/`, `paper_latex/`, and `experiments/`, especially Exps 4, 23, 26, 28, 29, 31, 33, 35, 36, 39, 41, 42 and 43).
Also consult these Google Docs and references (via Drive/MCP/browser if available, or cross-check against their canonical repo mirrors):

1. **Merged Literature Check and Final Strategy (Master Spec, 27 Nov 2026; Gemini's synthesis, unedited; see the corrections table above, and treat Exp 43's README as canonical where they differ):**
   https://docs.google.com/document/d/1Pt4NFmrYn3aclbQqx6_YGIUXS9bQappZsY08IEEBlNo/edit
2. **SK–BMG Alignment Draft (Top 5 Questions, incorporating Sanmi's Nov 26 email):**
   https://docs.google.com/document/d/1sH1WHv4qcskd6rPml9HZb8SxBjt1xMLZz_IUhN_q4Uw/edit
3. **SK–BMG Detailed Repo Audit Draft (Exp 39 review + Exp 41 exact numbers):**
   https://docs.google.com/document/d/1KWGwp-a3DRjW2OQuvk5jSGwPmhnOioOdiTch5IwrLtc/edit
4. **Nov 25 SK–BMG Meeting Transcript:**
   https://docs.google.com/document/d/16Rd9aLAAcHLS79d_GZY46q2r45qx75H-6PhPYyULMA4/edit?tab=t.0
5. **Most Important Goal of Corr vs. CE Paper:**
   https://docs.google.com/document/d/1VE92EHMwjT3YXzUOrMEcAHLS5SkyMZscf9VYOufPFw0/edit?tab=t.0
6. **Exp 39 Google Doc (needs a correction note after the CPU fix and re-run):**
   https://docs.google.com/document/d/1WtzpWNuUbkf3eGLOERMIvcyI7Sjec1hLfIZ1rJKfKvE/edit
7. **Exp 41 Simulated Multi-Modal Population Report:**
   https://docs.google.com/document/d/1WJQVKxCmCwppfAwHvdQ46OUFqcfWPor35i-SdBpA24A/edit
8. **Current Results Slide Deck (11-25-2026):**
   https://docs.google.com/presentation/d/1Q0FoiIzbt3uTG5LDE-rrK4iSLp-ppaW2J6icAfufg1w/edit
9. **Live Working Blog Draft (check and strip the Gemini prompt and share link from the appendix if a local copy exists):**
   https://cs.stanford.edu/people/brando9/2026/11/25/bitter-judges-lesson.html

## 1. HARD CONSTRAINTS AND OPERATING RULES

- **$0 API spend / CPU-only execution first:** Do NOT make paid API calls or launch expensive GPU sweeps without asking. All code fixes, Exp 39 re-runs, Exp 43 updates, base-rate skill (CE, Brier, and RPS) / agreement-stratification / eval-power computations, Lean 4 builds, and LaTeX builds run on CPU/saved artifacts.
- **Zero fabrication & strict provenance:** Every number, CI, and table cell in `paper_latex/` MUST come verbatim from a verified, committed experiment artifact (`results.md`, JSON/CSV) in this repo.
- **Exact notation lock (apply uniformly across paper, code, and figures):**
  - Prompt/context/item: `c` (do NOT overload `X` as both judge score and input item!)
  - Judge rating RV: `X ~ q_hat(. | c)` (predicted PMF `q_hat(y | c)`)
  - Human rating RV: `Y ~ p*(. | c, expert pool, rubric)` (conditional mean `mu_Y(c) = E[Y | c]`)
  - Centered RVs: `X_tilde := X - mu_X`, `Y_tilde := Y - mu_Y`
  - Standardized RVs: `z_X := X_tilde / sigma_X`, `z_Y := Y_tilde / sigma_Y`
  - Excess Brier divergence: `D_B(p* || q_hat) := sum_y (p*(y | c) - q_hat(y | c))^2`
  - Ordinal proper score: `RPS(q_hat, y)` (squared error on cumulative threshold indicators `1[y <= k]`), framed explicitly as **within-item ordinal distance-sensitive error ("how wrong you are")**, NEVER as cross-item ranking.
  - Chance/base-rate adjustment: **skill score** `Skill_S = 1 - S(judge) / S(base-rate forecast)` on the same held-out items, for each matched proper rule `S` (CE, Brier, and RPS). The RPS skill score is the ordinal case.
- **Mandatory Paper 1 Framing Guardrails (DO NOT VIOLATE):**
  1. Target **"correlation, majority-label agreement, and symmetric psychometrics (alpha, weighted kappa, IRT, G-theory)"** — NEVER claim "correlation is the only ruler for all LLM judges" (explicitly distinguish direct-score/ordinal judges [correlation, 29/30 scored/benchmark papers in Exp 43] from pairwise/reward models [accuracy/agreement/kappa, 13/13 pairwise papers in Exp 43]).
  2. Respect the 3 known weak spots: (a) pairwise judges use majority-label accuracy (same mode-collapse failure, different theorem; accuracy is proper but not strictly proper), (b) system-level leaderboards summarize rankings (note Chatbot Arena BT uses binary CE and Arena-Hard uses system-level Pair Rank Brier), (c) objective-answer tasks (JudgeBench, No Free Labels, Judge's Verdict) have near-point-mass truth where accuracy is close to proper.
  3. **Honest evaluation vs. training separation (Sanmi's Nov 26 guidance + Exp 41/23 reality):**
     - Lead Paper 1 with the airtight **Mirage-style evaluation protocol** (explicit construct `p*(y | c, expert pool, rubric)` + unbiased single-rating proper score `S` matched to the label space + base-rate skill `Skill_S` for CE, Brier, and RPS + human-agreement stratification + downstream system-level ranking). RPS is the flagship score on ordinal Likert scales; CE and Brier are primary on unordered categories such as ChaosNLI.
     - State the training takeaway honestly: **"Evaluate with a strictly proper score matched to the label space, and train — or at minimum calibrate — with one."** Explicitly report BOTH the large raw calibration gap (up to `0.128` normalized RPS in Exp 41; `7/7` raw CE-LoRA wins in Exp 29) AND the honest post-calibration linear-head result (`<= 0.0024` normalized RPS gap between CE and calibrated Pearson across 16 views; regularized MSE + calibration ties CE within `0.0004` in Exp 41; calibrated Pearson beats direct CE-fit on DICES at some budgets in Exp 28; raw Pearson got higher system-level Spearman `0.877` vs `0.835` on SummEval in Exp 23).
     - Hold strong **scaling-law** and **Bitter Lesson** claims out of Paper 1's main claim (frame end-to-end training beyond linear heads, fixed-dollar allocation, and cross-generation drift as Paper 2 / future work).
  4. **Bibliography hygiene:**
     - **Do not cite withdrawn `arXiv:2606.19544`** (withdrawn by arXiv Admin Nov 3, 2026) in ANY file, `.bib`, `.tex`, or markdown doc. A short note that it was withdrawn and must not be cited is allowed; remove any bibliographic citation.
     - Cite **Li et al. (`arXiv:2510.00263`, "Judging with Confidence")** as **Findings of EMNLP 2026** (not ICLR 2026; the ICLR 2026 OpenReview forum still reads "Submitted to ICLR 2026"). It treats pairwise binary preferences and explicitly leaves ordinal Likert scales to future work.
     - Cite and differentiate cleanly from **Lyu, Do, & Pergola (Oct 27, 2026, `arXiv:2610.37868`, "Human-Residual Steering")** (steering-method paper that uses single-pass logits + RPS/CE while still touting Spearman gains `0.449 -> 0.579`, without formal anti-properness proofs, psychometric failure analysis, base-rate skill, or judge-selection reversal protocol).
     - Cite as allies: Pavlick & Kwiatkowski (2019), ChaosNLI (Nie et al., 2020), Uma et al. (2021), Plank (2022), DICES (2023), Elangovan et al. (ICLR 2025, `2410.03775`), Guerdan et al. (NeurIPS 2025, `2503.05965`), Chen et al. (NeurIPS 2025, `2505.12301`), Wang et al. (Findings EMNLP 2025, `2503.03064`; confirm this entry from the primary source first — it has no matching row in Exp 43's `papers.py`).
     - The 2026 psychometric turn is an **emerging thread** (1 of 49 coded papers in Exp 43 reports IRT or G-theory reliability). Cite Choi et al. (ICML 2026 IRT-GRM, `2602.00521`) and Rao & Callison-Burch (`2606.00093`) as such. Alpha Parity / *Correlation Is Not Agreement* (OpenReview `67Z2nGMzNI`) is an ACL ARR May 2026 submission; only the title and submission status are confirmed, so do not cite unverified details. The G-theory paper (OpenReview `YawwpIWqTi`) is a **withdrawn ICLR 2027 submission** — do not cite it as published. The `Gtheory4LLM` CRAN package (Nov 2026) may be mentioned.

=====================================================================
STAGE 0: REPO AUDIT & RULES INSTALLATION
=====================================================================
1. Inspect git status, current branch, `paper_latex/` structure, `.cursor/rules/`, `CLAUDE.md`, `AGENTS.md`, and the exact state of Exps 4, 23, 26, 28, 29, 31, 33, 35, 36, 39, 41, and 43.
2. Work and commit directly on `main` (do not create a feature branch such as `paper1-mirage-ordinal-eval-sync`). Fetch the latest `origin/main` right before each push; never force-push; never rewrite another session's changes.
3. Ensure `.cursor/rules/ml-research-standards.mdc` (`alwaysApply: true`) enforces our 3-pillar bar (zero-`sorry` Lean 4 proofs, automated `pytest` numerical theorem checks, and leak-free real-data experiments with unbiased per-sample proper risk and item-clustered bootstrap CIs) and is referenced from `CLAUDE.md` and `AGENTS.md` without clobbering any existing repo rules.

=====================================================================
STAGE 1: REPO DATA, AUDIT FIXES & LEAN 4 HYGIENE (CPU ONLY)
=====================================================================
Execute these concrete repo updates in order:

1. **Update Exp 43 (`experiments/43_llm_judge_validation_practice_survey_start_27Nov2026/`):**
   - Stage 1.1 is mostly verification: Lyu et al. (`2610.37868`), Li et al. (`2510.00263`, Findings of EMNLP 2026), and Choi et al. (`2602.00521`) are already coded. Confirm `python3 tally.py` reports **49 papers**, **29/30** scored/benchmark papers reporting correlation, **13/13** pairwise papers reporting agreement/accuracy/kappa, and **43/49** with no item-level proper score (1 of 49 reports IRT or G-theory reliability). Add a missing verified 2026 paper only from a primary source.
   - Do **not** add the G-theory paper (OpenReview `YawwpIWqTi`) to `papers.py` as a published paper: it is a withdrawn ICLR 2027 submission. The `Gtheory4LLM` CRAN package may be mentioned.
   - Do **not** add Alpha Parity / *Correlation Is Not Agreement* (OpenReview `67Z2nGMzNI`) as a verified coded paper; it is an ACL ARR May 2026 submission whose details are unverified.
   - Ensure withdrawn `arXiv:2606.19544` is not cited anywhere in `papers.py` or `README.md`. A short note that it was withdrawn and must not be cited is allowed.
   - Re-run `python3 tally.py` twice to verify deterministic output, and update `README.md` with the refreshed counts and the exact 3-regime verdict (direct-score correlation, pairwise accuracy/kappa, 2026 psychometric turn as an **emerging thread**, not a rebound).

2. **Fix and re-run Exp 39 (`experiments/39_proper_score_vs_d_study_budget_allocation_start_26Nov2026/`):**
   - Read the independent review of Exp 39 in the repo and implement all 4 fixes cleanly on CPU:
     (a) Declare a deterministic, explicit tie-break rule for the RPS oracle.
     (b) Fix the constant D-study selectors so D-study actually adapts from pilot variance components (or explicitly label constant baselines separately).
     (c) Remove the hidden 4.5–7.7x pilot rating leakage in the proper-score selector so every selector operates at strictly matched total rating budget `B`.
     (d) Apply Holm–Bonferroni multiple-testing correction across the 20 dataset-budget cells.
   - Re-run Exp 39 on CPU, regenerate its figures/tables, update `results.md` with the corrected numbers, and write a concise `CORRECTION_NOTE_EXP39.md` ready to paste into the Exp 39 Google Doc (`https://docs.google.com/document/d/1WtzpWNuUbkf3eGLOERMIvcyI7Sjec1hLfIZ1rJKfKvE/edit`).

3. **Add missing Paper 1 eval-protocol diagnostics on existing cached forecasts/datasets (CPU only, $0):**
   - On saved held-out datasets/forecasts (DICES-350/990, ChaosNLI, SummEval, CertJudge/VeriBench if local), compute and save a clean JSON/CSV + `results.md` under a new dated experiment folder (e.g., `experiments/44_ordinal_eval_protocol_skill_stratification_power_start_27Nov2026/`):
     (a) **Base-rate skill scores** `Skill_S = 1 - S(judge) / S(base-rate)` for **CE, Brier, and RPS**, with item-clustered bootstrap 95% CIs (directly answering kappa/alpha pushback per Sanmi's email), clearly labeling which proper rule matches each dataset's label-space geometry (CE/Brier primary on unordered categories such as ChaosNLI; RPS primary on ordinal scales).
     (b) **Human-agreement stratification** (low-entropy consensus vs. high-entropy disputed items, using independent rater splits for stratifying vs. scoring where multi-rater data permits), verifying proper distributional judging stays peaky on consensus items and calibrated on disputed items.
     (c) **Eval-set statistical power / Minimum Detectable Difference (MDD) curves** sweeping `(N_test, k_test)` via item-clustered bootstrap on existing benchmarks, quantifying whether current test set sizes already separate judges cleanly under the matched unbiased single-rating proper score (CE, Brier, and RPS).
     (d) **Explicit real-benchmark winner reversal summary table** pulling exact paired CIs for DICES-990, ChaosNLI MNLI-m (Exp 28), and G-theory reliability vs. the matched proper score (RPS, and CE/Brier as applicable) (Exps 4 & 41).

4. **Lean 4 check & optional accuracy lemma (`experiments/33`, `35`, `36` or `theory/`):**
   - Run `lake exe cache get` (if Mathlib is used; NEVER rebuild Mathlib from scratch) and `lake build` to confirm all existing Lean 4 theorems (`brier_strictly_proper`, `mse_decomposition`, `standardized_mse_corr`, `pearson_affine_invariant`, `mean_not_identifying`, `ordinal_same_mean_different_law`, and the squared anti-properness `rho^2 < rho` lemma) compile with **zero `sorry`** and zero custom axioms.
   - Optional quick win (if clean and <= 50 lines): add a zero-`sorry` Lean 4 proof that hard 0–1 accuracy (`argmax`) and sampled expected accuracy (`sum_y q(y) p*(y)`) are maximized by the deterministic mode `delta_{argmax p*}` (showing that accuracy is proper but not strictly proper / overconfidence). If not formalized in Lean, explicitly label the accuracy/kappa mode-collapse argument as analytical (not Lean-checked) in the paper.

5. **Repo-wide hygiene check:**
   - Grep the entire repo for `2606.19544`. Do not cite it. A short note that it was withdrawn and must not be cited is allowed; remove any bibliographic citation.
   - Run `pytest` across all verification and metric unit tests.

=====================================================================
STAGE 2: UPDATE THE PAPER (`paper_latex/`)
=====================================================================
Inspect `paper_latex/` thoroughly (existing `.tex` files, `.bib`, figures, style/venue template, macros) and execute a surgical, top-ML-conference revision aligned with our finalized Paper 1 architecture:

2. **Literature and differentiation:**
   - Use the Merged Literature Check doc (including Exp 43's updated reproducible tally: 49 papers; 29/30 scored/benchmark papers report correlation; 13/13 pairwise papers report agreement/accuracy/kappa; 43/49 have no item-level proper score).
   - Differentiate crisply from Elangovan et al. (ICLR 2025), Guerdan et al. (NeurIPS 2025), Chen et al. (NeurIPS 2025), Wang et al. (Findings EMNLP 2025; confirm from the primary source first), **Li et al. (Findings of EMNLP 2026, [arXiv:2510.00263](https://arxiv.org/abs/2510.00263); pairwise binary, leaves ordinal to future work)**, and Lyu, Do, and Pergola ([arXiv:2610.37868](https://arxiv.org/abs/2610.37868), Oct 2026; representation steering, trains with RPS for ordinal labels and CE for nominal labels, primary metric Jensen–Shannon distance, still reports Spearman gains).
   - Do not cite the G-theory paper (OpenReview YawwpIWqTi) as published: it is a withdrawn ICLR 2027 submission (correction c). The `Gtheory4LLM` CRAN package may be mentioned. Alpha Parity (OpenReview 67Z2nGMzNI) is an ACL ARR May 2026 submission whose details are unverified (correction d); cite it only as such, or not at all.
   - Confirm **zero bibliographic citations of withdrawn `arXiv:2606.19544`** and the correct venue for Li et al. (**Findings of EMNLP 2026**).

3. **Theory, Experiments, Tables and Figures:**
   - Unify notation (`c`, `X`, `Y`, `X_tilde`, `z_X`, `D_B(p* || q_hat)`, `RPS`, `Skill_S` for CE, Brier, and RPS) everywhere in `.tex` and figure labels.
   - Cite exact Lean 4 theorem names verified in Stage 1 (and never overclaim Lean coverage for unformalized statements).
   - Wire every table and figure to the updated committed artifacts (Exp 23, 28, 29, 35, 36, cleaned 39 if included in Discussion/Appendix, 41, 43, and new 44).
   - Compile `paper_latex/` cleanly (e.g., `latexmk -pdf` or `pdflatex` + `bibtex`) with **zero LaTeX errors, zero undefined citations `[?]`, and zero undefined cross-references `??`**.

## STAGE 3: VERIFICATION, COMMIT AND HANDOFF REPORT

1. Check `git diff --stat`, confirm `pytest`, `lake build`, `python3 tally.py`, and `paper_latex/` PDF compilation all succeed cleanly.
2. **Commit logically grouped changes directly on `main`** (correction a; the original said "on the working branch"). Fetch and rebase onto the latest `origin/main` right before each push; do NOT force-push; never overwrite another session's changes. Leave `main` clean and building.
3. Print a concise final report covering:
   - Exact Lean 4 theorems verified and `lake build` exit status.
   - Exp 43 tally counts (49 papers; 29/30; 13/13; 43/49; Lyu, Li and Choi already coded).
   - Cleaned Exp 39 results versus the old flawed headline (plus the exact correction note for the Exp 39 Google Doc).
   - New Exp 44 results (base-rate skill for CE, Brier and RPS, human-agreement stratification, eval-set power/MDD curves).
   - Summary of `paper_latex/` edits, compiled PDF path and page count, and verification of zero `2606.19544` / zero `[?]` / zero overclaims.
   - Reminder of any manual Google Doc or blog appendix cleanup items (e.g., stripping the Gemini prompt and share link from the live blog appendix if it is not hosted in this repository).

**TL;DR (main prompt).** Audit this repository against the merged literature check and Sanmi's Nov 26 guidance; then on CPU ($0 API spend): install the three-pillar repo rules, verify Exp 43 (49 papers), fix and re-run Exp 39, compute Exp 44, verify zero-`sorry` Lean 4 builds; re-scope and compile `paper_latex/`; commit on `main` with a full verification report.

---

# ADDENDUM: FRAMING UPDATE (GENERALITY BEYOND RPS), applies across the repo, Exp 44 and `paper_latex/`

Our core Paper 1 thesis is MORE GENERAL than ordinal RPS alone. Do NOT frame the paper narrowly as "just use RPS." Frame it around the general decision-theoretic principle with two cleanly separated layers:

1. **General Principle (Main Thesis, "Proper Evaluation Matched to the Judgment Space Y"):**
   - Replace correlation, majority-label agreement, and symmetric psychometrics (`alpha`, `kappa`, IRT, G-theory) with a **strictly proper scoring rule `S(q_hat(. | c), y)` matched to the encoding/geometry of the output space `Y`**, paired with its **base-rate skill score** `Skill_S = 1 - S(judge) / S(base-rate)` (answering chance-corrected agreement), human-agreement stratification, and downstream system-ranking checks.
   - Include a crisp **Output-Space -> Encoding -> Strictly Proper Rule** table (echoing Sanmi's Nov 26 "frame around encodings" suggestion):
     * **Binary (`K = 2`):** Bernoulli `y in {0,1}` -> **CE (log-loss), Brier** (noting scalar L2 = 0.5 * Brier, yet Pearson still fails affine calibration, and hard accuracy against the majority label is proper but not strictly proper: it collapses to the mode).
     * **Unordered Categorical (`K >= 3`, e.g., error categories, nominal taxonomies, NLI):** One-hot `e_y in {0,1}^K` -> **CE (log-loss), Multiclass Brier** (`D_B(p* || q_hat)`). Explicitly note that **RPS does NOT apply to unordered categories**: proper evaluation works seamlessly for categorical judging via CE/Brier.
     * **Finite Ordinal (`K >= 3`, e.g., 0–5 / 1–5 Likert rubrics):** Cumulative threshold indicators `1[y <= k]` -> **RPS** (squared error on cumulative encoding, adding within-item distance sensitivity / "how wrong you are") alongside **CE and Brier**.
     * **1D Continuous (`[0,1]` / `0–100`):** Continuous CDF -> **CRPS** (continuous limit of RPS) / log-density.

2. **Flagship Empirical Instantiation (Why Ordinal Scales + RPS Are Spotlighted Without Limiting the Thesis):**
   - Position the general framework as covering **all discrete/1D judging spaces** (binary, unordered categorical, ordinal), while highlighting **ordinal Likert scales (via RPS)** as our primary empirical spotlight alongside **categorical benchmarks (via CE and Brier)** because:
     (a) direct-score ordinal judging is where correlation is near-universal (**29 of the 30 scored/benchmark papers in Exp 43 report correlation**; the old "22/22" count is superseded), and
     (b) Li et al. (**Findings of EMNLP 2026**) already applied Brier/log proper scoring to pairwise binary preferences while explicitly leaving ordinal Likert scales to future work.
   - Check dataset-to-metric matching carefully: on unordered/nominal multi-class tasks (or where class order is non-metric, e.g., note this transparently on ChaosNLI), report **CE and Brier** (plus their base-rate skill scores) prominently alongside any RPS numbers; on genuinely ordinal scales (SummEval, DICES, CertJudge `R_v1`, POPQUORN), report **RPS** (primary distance-sensitive proper score) together with **CE and Brier**.

3. **Concrete Edits Required:**
   - **Title / Abstract / Intro (`paper_latex/`):** Use a general title/framing such as *"Evaluating AI Judges Requires Proper Scoring Rules (Not Correlation or Psychometrics)"* or *"The Judge-Validation Mirage: Why Correlation and Psychometrics Mis-Rank AI Judges, and a Proper Scoring Protocol Across Categorical and Ordinal Scales"* (keeping the Mirage framing Sanmi liked without restricting the principle to RPS).
   - **Exp 44 (`experiments/44_...`):** Compute unbiased per-sample proper risk, **base-rate skill scores** (`1 - S(judge)/S(base-rate)`), human-agreement stratification, and eval-set power/MDD curves for **all three proper rules: CE, Brier, AND RPS**, clearly labeling which proper rule matches each dataset's label-space geometry.
   - **Reference docs:** Keep alignment with our merged strategy doc (`https://docs.google.com/document/d/1Pt4NFmrYn3aclbQqx6_YGIUXS9bQappZsY08IEEBlNo/edit`, with the corrections above) and the Sanmi alignment doc (`https://docs.google.com/document/d/1sH1WHv4qcskd6rPml9HZb8SxBjt1xMLZz_IUhN_q4Uw/edit`).

**TL;DR (addendum).** Broaden Paper 1's framing from "ordinal RPS only" to the general decision-theoretic principle: **evaluate AI judges with a strictly proper scoring rule matched to the label-space encoding `Y` (CE/Brier for binary and unordered categorical; RPS for finite ordinal; CRPS for 1D continuous) plus base-rate skill**, treating ordinal judging (RPS) as the flagship empirical instantiation alongside categorical judging (CE/Brier), and compute Exp 44 skill/stratification/power diagnostics across CE, Brier and RPS.
