# Corrected Claude Code prompt for Paper 1 (27 Nov 2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/prompts/claude_code_paper1_prompt_corrected_27Nov2026.md>

**TLDR:** Archive copy of the Gemini-written Claude Code prompt (main prompt, Stages 0 to 3, plus the "generality beyond RPS" addendum) with eight factual and process corrections applied inline, so it can be pasted into Claude Code as written. As of 27 Nov 2026, Brando is already running a corrected version of this prompt in Claude Code; this file is the record, not a new task.

## Status and provenance

- **Running copy.** Brando is running a corrected version in Claude Code as of 27 Nov 2026. That session's own text is authoritative for what it was actually told; this file is the archived, corrected reference.
- **Source.** The main prompt and the addendum were written by Gemini and pasted by Brando in chat on 27 Nov 2026 (messages t105u and t107u).
- **Archiver's limitation (read this).** The archiving agent received the main prompt with a middle segment cut off (about 12,000 characters: the rest of Section 1, any Stage 0 and Stage 1 text, and the start of Stage 2). The cut parts are marked `[ORIGINAL TEXT NOT AVAILABLE]` below, each followed by a short stand-in written by the archiver from the prompt's own rationale and TL;DR and from this repository's rules. Everything not marked is the original text with the corrections below applied. If the verbatim text of the cut segment matters, copy it from Brando's Claude Code session or the original chat.

## Corrections applied

| # | Correction | Where applied |
| --- | --- | --- |
| a | Commit directly to `main`, not a feature branch (Brando's rule for this repository: `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/merge-to-main.mdc`, `.cursor/rules/experiment-folders-on-main.mdc`). Fetch latest `main` right before each push; never force-push; never rewrite others' changes. | Section 1, Stage 3 |
| b | Li et al., "Judging with Confidence" ([arXiv:2510.00263](https://arxiv.org/abs/2510.00263)), is **Findings of EMNLP 2026**, not ICLR 2026 (the ICLR 2026 OpenReview forum still reads "Submitted to ICLR 2026"). It applies Brier and log rewards to pairwise preference distributions and leaves ordinal/Likert scales to future work; that point stands. | Section 0, Stage 2, addendum |
| c | The G-theory paper "From LLM Judge Scores to Evaluation Design" (OpenReview [YawwpIWqTi](https://openreview.net/forum?id=YawwpIWqTi)) is a **withdrawn ICLR 2027 submission**. Do not cite it as published. The `Gtheory4LLM` CRAN package may be mentioned. | Stage 2 |
| d | Alpha Parity, "Correlation Is Not Agreement" (OpenReview [67Z2nGMzNI](https://openreview.net/forum?id=67Z2nGMzNI)), is an **ACL ARR May 2026 submission**. Only the title and submission status are confirmed; its details (such as 3.06M judgments) are unverified. | Stage 2 |
| e | Exp 43 is now **49 papers**: 29 of 30 scored/benchmark papers report correlation; 13 of 13 pairwise papers report agreement, accuracy or kappa; 43 of 49 have no item-level proper score. Lyu et al. ([2610.37868](https://arxiv.org/abs/2610.37868)), Li et al. and Choi et al. are already coded, so Stage 1.1 is mostly verification. The old "22/22" count is replaced. | Stage 1, Stage 2, Stage 3, addendum |
| f | The "psychometric rebound" is an **emerging thread**: 1 of 49 coded papers reports IRT or G-theory reliability. Correlation and majority-label agreement remain dominant. | Stage 2 |
| g | The thesis is **general**: a strictly proper score matched to the label space, plus a base-rate skill score. RPS is the ordinal flagship; CE and Brier are primary for unordered categories such as ChaosNLI. Accuracy against the majority label is proper but **not strictly proper** (it rewards only picking the mode). Alpha, kappa and IRT are agreement or reliability measures, not scoring rules at all. | Section 1, Stage 2, addendum |
| h | The withdrawn **arXiv:2606.19544** (Norman et al., "Reliability without Validity", withdrawn by arXiv admin on 3 Nov 2026) must not be cited anywhere. | Stage 1, Stage 2, Stage 3 |

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

- **$0 API spend / CPU-only execution first:** Do NOT make paid API calls or launch expensive GPU sweeps without asking. All code fixes, Exp 39 re-runs, Exp 43 updates, RPS base-rate skill / agreement-stra

`[ORIGINAL TEXT NOT AVAILABLE: the rest of Section 1 was cut off in the copy the archiver received. Stand-in, written by the archiver from this repository's rules and the prompt's rationale:]`

- (original first bullet continues: base-rate skill, agreement-stratification and eval-set power diagnostics are computed on saved artifacts on CPU.)
- **Commit directly to `main`** (correction a): fetch and rebase onto the latest `origin/main` right before each push, never force-push, never rewrite another session's commits, keep each experiment in its numbered folder `experiments/NN_<setup>_start_<DDMonYYYY>/`.
- **Proofs:** every formal claim is checked in Lean 4 (zero `sorry`, standard axioms only, `#print axioms` saved) and in Python, per `AGENTS.md`. Do not call anything a proof that has not been compiled.
- **Real data:** every paper-facing empirical claim rests on real multi-annotator human ratings, the unbiased per-sample proper risk at test time, a prompt-level train/validation/test split, and item-clustered bootstrap 95% confidence intervals, per `AGENTS.md`.
- **No invented numbers.** Every number, CI and table in `paper_latex/` comes from a committed artifact.
- **Citations:** never cite withdrawn arXiv:2606.19544 (correction h). Use the corrected venues and statuses in the corrections table above (b, c, d).

## STAGE 0 to STAGE 1 (audit, repo rules, Exp 43, Exp 39, Exp 44, Lean)

`[ORIGINAL TEXT NOT AVAILABLE: Stage 0 and Stage 1 were cut off. Stand-in reconstructed from the prompt's rationale and TL;DR; the original wording and any extra detail are in Brando's Claude Code session.]`

0. **Audit.** Read the repo and the reference docs. Install the three-pillar repo rules (corr / majority-label agreement / symmetric psychometrics are not proper scores; strictly proper score matched to the label space plus base-rate skill; real multi-annotator data plus Lean-checked proofs) in `AGENTS.md`, `CLAUDE.md` (identical files) and `.cursor/rules/`.
1. **1.1 Exp 43.** Lyu et al., Li et al. and Choi et al. are already coded (correction e), so this is mostly verification: confirm `python3 tally.py` reports 49 papers, 29/30 scored/benchmark papers with correlation, 13/13 pairwise papers with agreement/accuracy/kappa, and 43/49 with no item-level proper score. Add any missing verified 2026 paper only from a primary source. Purge any occurrence of arXiv:2606.19544 (a note that it was withdrawn and must not be cited is fine).
2. **1.2 Exp 39.** Fix the review's findings on Experiment 39 (see the 11-27-2026 QA entry for experiment 39 in `experiments/README.md` and the audit doc, reference 3) and re-run it cleanly on CPU. Report the cleaned results next to the old headline and write an exact correction note for the Exp 39 Google Doc (reference 6).
3. **1.3 Exp 44.** New experiment folder: unbiased per-sample proper risk, base-rate skill scores `1 - S(judge)/S(base-rate)`, human-agreement stratification, and eval-set power / minimum-detectable-difference curves, computed on saved artifacts, for CE, Brier and RPS (see the addendum), plus the real-data judge rank-reversal table (correlation-ranked versus proper-score-ranked judges).
4. **1.4 Lean.** Run `lake build`; verify the cited theorem names and zero `sorry`. Optionally add a zero-`sorry` lemma that accuracy against the majority label is proper but not strictly proper (it elicits only the mode). Never claim Lean coverage for an unformalized statement.

## STAGE 2: `paper_latex/` RE-SCOPE

`[ORIGINAL TEXT NOT AVAILABLE: the start of Stage 2, item 1, was cut off. Stand-in:]`

1. **Framing and structure.** Rewrite the title, abstract and introduction around the three rulers that cannot certify a judge as a forecaster of human ratings: correlation (scored judges), majority-label agreement, accuracy or kappa (pairwise judges), and psychometric reliability (alpha, kappa, IRT, G-theory), versus a strictly proper score matched to the label space with a base-rate skill score. Describe the psychometric thread as **emerging** (1 of 49 coded papers reports IRT or G-theory reliability; correction f), not as a rebound. Say alpha, kappa and IRT measure agreement or reliability and are not scoring rules at all; say accuracy is proper but not strictly proper. Lead with the unbiased single-rating proper score and base-rate skill; scope raw-versus-calibrated training results honestly (Exp 41: the CE-over-Pearson gap is up to 0.1282 raw and at most 0.0024 once each head has its own calibration map).

2. **Literature and differentiation** (verbatim fragment follows; the beginning of this item was cut off)
   - `...d Literature Check doc (including Exp 43's updated reproducible tally)` (use the 49-paper tally: 29/30, 13/13, 43/49).
   - Differentiate crisply from Elangovan et al. (ICLR 2025), Guerdan et al. (NeurIPS 2025), Chen et al. (NeurIPS 2025), Wang et al. (Findings EMNLP 2025; confirm from the primary source first), **Li et al. (Findings of EMNLP 2026, [arXiv:2510.00263](https://arxiv.org/abs/2510.00263); pairwise binary, leaves ordinal to future work)**, and Lyu, Do, and Pergola ([arXiv:2610.37868](https://arxiv.org/abs/2610.37868), Oct 2026; representation steering, trains with RPS for ordinal labels and CE for nominal labels, primary metric Jensen–Shannon distance, still reports Spearman gains).
   - Do not cite the G-theory paper (OpenReview YawwpIWqTi) as published: it is a withdrawn ICLR 2027 submission (correction c). The `Gtheory4LLM` CRAN package may be mentioned. Alpha Parity (OpenReview 67Z2nGMzNI) is an ACL ARR May 2026 submission whose details are unverified (correction d); cite it only as such, or not at all.
   - Confirm **zero occurrences of withdrawn `arXiv:2606.19544`** and the correct venue for Li et al. (**Findings of EMNLP 2026**).

3. **Theory, Experiments, Tables and Figures:**
   - Unify notation (`c`, `X`, `Y`, `X_tilde`, `z_X`, `D_B(p* || q_hat)`, `RPS`, `RPS skill`) everywhere in `.tex` and figure labels.
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
