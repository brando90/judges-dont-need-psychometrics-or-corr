# Codex "vals ultra" review of the paper against the advisor's two warnings, 28 Nov 2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/notes/qa_codex_vals_ultra_28Nov2026b.md>

**TLDR:** A second Codex review (reasoning effort ultra, read-only) of the repository at 04ab604, run on Brando's Mac under his "vals" Codex subscription profile. All ten findings were triaged on 11-28-2026: nine were adopted (one in different words) in commits 809bb44 and d6d057e, and finding 8 (a replacement abstract) was not adopted as a whole. The earlier Codex review of an earlier state is in `qa_codex_vals_28Nov2026.md` (F01 to F20).

## Provenance

- Run: 2026-11-28 17:04:16Z to 17:09:53Z (09:04 to 09:09 PST, about 5 min 37 s), exit code 0, no files changed by the reviewer.
- Command (from `run.sh` on the Mac): `CODEX_HOME=$HOME/.codex-vals codex exec -m gpt-6-astra -c 'model_reasoning_effort="ultra"' -s read-only -C <clone> --skip-git-repo-check -o report.md - < prompt.md`, with a 90-minute watchdog. API-key variables were not used; the run drew on the subscription login, so no API dollars (see `docs/STATE_OF_EVIDENCE.md`).
- Target: a shallow clone of origin/main at 04ab604 (README sync, 08:57:27 PST), that is, before Experiment 61 (28f6148, 3c447d1) and before the paper edit pass (809bb44). The prompt (sha256 `a945efc47dacbc05...`) asked for a hostile area-chair check of whether the advisor's two warnings are fixed, with every number and citation verified against committed READMEs and the two papers; it carried the advisor's message verbatim and is not tracked.
- Report file on the Mac: `~/_codex_review_28Nov2026/report.md`, 12,464 bytes, sha256 `6a5ec1b6134179a323519edf9ecbb6bfd42b066546ba0165b39557378142309e`.
- Not checked by the reviewer: Lean compilation, PDF rendering, experiment reruns, external papers, bibliography, model identities, raw-data provenance. Claude did not rerun Codex in the sandbox (Codex is not installed there).

## Disposition (state after d6d057e)

| # | Codex severity and finding | Disposition | Where |
|---|---|---|---|
| 1 | Medium. S3 (conditioning on the item), S4 (hypotheses) and S7 (reliable ranking) only partly met; 218 words | Adopted, S7 in different words | `1-abstract.tex`: "each item's rating distribution for a stated rater pool and rubric"; the sampled-score sentence states independence given the item and variation between and within items; "which ranks judges up to sampling error". The abstract is now 248 words, not about 200 (cut list in `abstract_rewrite_28Nov2026.md`). |
| 2 | High. Experiments 41 and 47 READMEs still say the head comparison supports the evaluation claim; 9c:106 needs a caveat | Adopted in substance, in the repository's dated-note form (results untouched) | Dated notes at the top of the Experiment 41, 45, 46, 47, 54, 57, 58, 59 and 60 READMEs; `9c-appendix-calibration.tex` paragraph on the Experiment 41 map rewritten. |
| 3 | Medium. "about 100", "did at least as well", "at most about 0.01" lose qualifications | Adopted | Experiment 56 reported as the smallest tested budget (`12-supplementary-pool-strata-power.tex`); `7-discussion.tex:10` now adds the raw-proper-score sensitivity run and the random-pick comparison; `:12` says 0.0024 normalized RPS (0.0098 in skill over the base rate; largest five-seed means). |
| 4 | Medium. Experiment 59 interval type; mean readout off the square-root curve; rounded versus unrounded mean; "four" analyses | Adopted | `13-supplementary-calibration-level-selection.tex`: percentile intervals over 200 splits; the mean judge "sits just below sqrt(rho)"; rounded-mean, modal, median and two-point forecasts with Experiment 61; "three preregistered analyses ... one oracle check ... one post hoc follow-up". |
| 5 | High. Estimated calibration gains presented as the Murphy decomposition; 3e-17 proves no equivalence | Adopted | "Estimated calibration gain" and "calibrated skill" in 9c, 13, the Experiment 57 README note, `STATE_OF_EVIDENCE.md` and the README; the 3e-17 is stated as a floating-point check of the code by construction. |
| 6 | Medium. Finer-key sentence, "Correlation" versus squared correlation, "never scored", Lean hypotheses, affine invariance of r, "for all" versus almost surely | Adopted | 9c:63 corrected and checked directly (numerics check 10, 900 laws, plus a worked example and a claimed-false case; Lean pending); squared correlation; "ignores distribution shape beyond those means"; "Lean-checked for finite laws with both variances positive"; sign flip for negative slope (`Point2.FinLaw.pearson_affine`, `condE_affine`); "for a.e." in both displays. |
| 7 | Medium. Lean-status scope of the two-ratings claim; one-rating unbiasedness wrongly unformalized | Adopted | `4-proposal.tex:33` (finite identities and a two-category witness are Lean-checked, the statement for every number of categories is not); `8-appendix-rps.tex` and `abstract_rewrite_28Nov2026.md` cite `PlanAudit.plan_estimator_unbiased` (`PlanClaims.lean`, finite law, forecasts fixed in advance); `LEAN_TODO.md` rows. |
| 8 | Medium. Replacement abstract of about 210 words | **Not adopted as a whole** | Its DICES-350 sentence is the stale four-judge version (the paper now reports all five judges, 9 to 24 percent forecast unsafe against 33 percent, at 80 to 82 percent agreement), and it drops the credit to Lyu et al. Individual fixes from it were applied (items 1, 3). One idea stays open for Brando: adding Experiment 59's unfavorable result to the abstract (this review and the Fable literature review both ask for it; about 20 words). |
| 9 | Medium. Audit categories contradict their exceptions (pooled labels, synthetic votes) | Adopted | `2-introduction.tex:21` ("under a broad code that counts pooled labels and LLM-teacher votes"); `11-supplementary-related-work.tex` audit paragraph. |
| 10 | Low. Stale 100 seeds / nine starts; 64/128 break-even; evidence-map line about raw-score uses | Adopted | `9b-appendix-concordance-and-calibration.tex` (30 seeds, five starts, per the stored Experiment 07 records); `10-supplementary.tex` (64 for DICES-350, 128 for DICES-990); `abstract_rewrite_28Nov2026.md` now says the introduction cites the raw-score uses (MT-Bench, AlpaGasus, FineWeb-Edu). |

What the reviewer verified (its list below): the abstract's rounded numbers, 0 of 50 significant Pearson reversals, 78 of 82, and Experiment 59's 20 of 20 and 17 of 20 against the saved results; the numerics (25 of 25 at the time; 28 of 28 now) twice with identical output.

## Report, as printed by the reviewer

Line breaks were restored by hand (the bridge from the Mac flattened them) and the links to the reviewer's clone were rewritten as repository-relative links; the text is otherwise as printed. After replacing every link target by `X`, collapsing whitespace runs to one space and trimming the ends, the original file on the Mac and the section below have the same SHA-256 digest, `f679f6a04343d4a7e61d828728a93efdb229c119647db35ad4c61e2437f7fe19`.

<!-- begin verbatim -->
## PROBLEMS

1. **Medium — Abstract response to S1–S9.** Paper basenames below refer to `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/`.

   - **S1 MET**, `1-abstract.tex:5`: “LLM judges are validated by correlation or agreement with human ratings.”
   - **S2 MET**, `:9`: “Binned Jensen--Shannon divergence (Elangovan et al., 2025) and squared error to vote means on pairwise preferences (Li et al., 2026) compare judges with such distributions.”
   - **S3 PARTLY MET**, `:7`: “Following Pavlick and Kwiatkowski (2019) and Plank (2022), the target is the rating distribution \(p^*\) of a rater pool and rubric.” The standalone definition omits conditioning on the item. Pass-rate thresholds are named at `:15`; actual raw-score uses are cited in `2-introduction.tex:7`. Fix: “the rating distribution **for each item**, for a specified rater pool and rubric.”
   - **S4 PARTLY MET**, `:17`: “For sampled scores (single draws at nonzero temperature), correlation prefers the conditional mean to a draw from \(p^*\).” Missing conditional independence and positive within-item and between-item variation; temperature alone establishes neither. Add those hypotheses.
   - **S5 MET**, `:15`: “Squared correlation is one minus the best affine fit's mean squared error over the rating variance, so it ignores the level and scale behind pass-rate thresholds.”
   - **S6 MET**, `:11`: “We adopt item-level strictly proper scores: Brier for categories and, for ordinal ratings, the ranked probability score (RPS; Epstein, 1969), normalized absolute error on hard predictions.”
   - **S7 PARTLY MET**, `:13`: “One rating per item gives an unbiased expected score, enough to rank judges.” Unbiasedness does not guarantee a reliable finite-sample ranking. Replace with: “One rating per test item estimates expected score without bias; ranking precision depends on the number of items.”
   - **S8 MET by removal**, `:19`: “On five frontier judges, a preregistered test found no significant rank reversal between Pearson correlation and a proper score.” The audit and circular head comparison are absent.
   - **S9 MET**, `:25`: “We recommend reporting a proper score, split into calibration error and resolution, beside correlation.”

   The supplied script returns **218 words**, near 200.

2. **High — Warning 2 survives in cited READMEs.** [Experiment 41 README:5](../../experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/README.md), repeated at `:52`, says the head comparison “supports the evaluation claim.” [Experiment 47 README:59](../../experiments/47_calibration_label_budget_sweep_exp41_heads_start_27Nov2026/README.md) says “It supports the evaluation thesis” and falsely infers that correlation users have not collected calibration labels. Replace both interpretations with:

   > “This measures calibration behavior and label costs for weak linear heads. The comparison is circular as evidence for the evaluation claim and is retained only as a calibration check.”

   `9c-appendix-calibration.tex:106` also needs: “The fitted maps remove nearly all the observed head-risk difference; this does not independently establish the evaluation claim or identify population calibration error.”

   **The current abstract, introduction, and conclusion no longer contradict either original warning.** No surviving categorical “no existing metric” claim was found in the searched paper sources and cited READMEs.

3. **Medium — Numerical claims lose essential qualifications.**

   - `1-abstract.tex:23`, “needs about 100”: Experiment 56 `README.md:17–23` establishes the **smallest tested budget** where all 25 mean calibrated skills exceed the base rate; eight remain worse than their raw counterparts. Replace: “All 25 judge–dataset mean calibrated skills first exceeded the base rate at 100 ratings, the smallest successful tested budget.”
   - `7-discussion.tex:10`, “did at least as well as choosing by a proper score”: Experiment 59 `README.md:17–18` reports raw-proper-score selection beating correlation in all eight DICES comparisons. Replace: “Correlation selection had lower mean regret than selection by three-fold cross-validated calibrated proper risk; raw-proper-score selection performed better on DICES.”
   - `7-discussion.tex:12`, “at most about 0.01”: Experiment 53 `results/skill_tables.md:10` gives mean **0.0098**, interval **[0.0081, 0.0114]**. Replace: “The largest five-seed mean skill difference after calibration was 0.0098.”

4. **Medium — Supplementary uncertainty and oracle descriptions are inaccurate.**

   - `13-supplementary-calibration-level-selection.tex:14`: Experiment 59 `README.md:25` bootstraps **200 random splits**, not item clusters. Replace: “Experiment 59’s intervals bootstrap labeled-pool/test splits conditional on the fixed dataset; they do not measure population item-sampling uncertainty.”
   - `:84`: mean-readout markers do **not** lie on \(r=\sqrt{\rho}\). Experiment 60 `results/sampled_vs_mean.csv:2–4` gives **0.802259, 0.423959, 0.401048**, versus **0.806750, 0.438112, 0.425276**. Replace: “The curves show population identities; finite-half mean estimates fall below the square-root curve.”
   - `:77–78`: correlation uses the unrounded mean, but skill evaluates a point mass at its **nearest label**. Replace: “For proper-score evaluation we separately round each estimated mean to the nearest label and evaluate that hard forecast.”
   - `:9`: replace “four … analyses … and one oracle check” with “three … analyses … and one oracle check.”

5. **High — Estimated calibration gains are presented as the Murphy decomposition.** `13-supplementary-calibration-level-selection.tex:19,26,43` and `9c-appendix-calibration.tex:99,108–111` call fitted-map improvement “calibration error,” calibrated skill “resolution,” and empirical estimates population lower bounds. A restricted fitted map need not equal the conditional rating law. Arithmetic cancellation to \(3\times10^{-17}\) proves no such equivalence.

   Replace throughout:

   > “We report estimated calibration gain and calibrated skill. These approximate the Murphy terms when the fitted map estimates the conditional rating law well; finite-sample estimates are not guaranteed population bounds.”

   Rename the table columns accordingly.

6. **Medium — Mathematical wording contradicts the equations or Lean hypotheses.**

   - `9c-appendix-calibration.tex:63`: replace “moves part of the calibration error into resolution” with “For a proper score, finer conditioning increases calibration error and resolution by the same nonnegative amount.”
   - `:77`: replace “Correlation” with “Squared correlation.”
   - `:78`: replace “spread about the conditional mean is never scored” with “Correlation computed from forecast means ignores distribution shape beyond those means.”
   - `:80`: `Murphy.lean:119` requires positive score and rating variances and formalizes finite laws. Add: “Lean-checked for finite laws with both variances positive.”
   - `:82`: `Murphy.lean:107` proves conditional-expectation invariance for nonzero affine slope, not the stated invariance of \(r\). Replace: “A nonconstant affine relabelling preserves \(r^2\) and the conditional-expectation random variable; calibration error can change. Pearson \(r\) is preserved only for positive slope.”
   - `:19,27`: replace “for all” forecast values with “almost surely,” avoiding conditioning requirements on zero-probability events.

7. **Medium — Additional Lean-status scope problems.** `4-proposal.tex:33` attaches “finite statements Lean-checked” to a general necessity claim about two ratings. `UncertaintyTerm.lean:57–64` supplies a particular counterexample, not a universal necessity theorem. Replace: “Without additional structure, paired independent ratings permit unbiased estimation of uncertainty; finite identities and a counterexample are Lean-checked.”

   Conversely, `8-appendix-rps.tex:176` and `docs/notes/abstract_rewrite_28Nov2026.md:15` wrongly leave one-rating unbiasedness unformalized. Cite `PlanClaims.lean:130–168`, with its finite, fixed-sample scope.

8. **Medium — Replacement abstract addressing the remaining objections, approximately 210 words:**

   > Large language model judges are commonly validated by correlation or agreement with human ratings. Following Pavlick and Kwiatkowski (2019) and Plank (2022), we target each item's rating distribution for a specified rater pool and rubric. Elangovan et al. (2025) compare binned rating distributions; Li et al. (2026) compare pairwise probabilities with vote means. We adopt established strictly proper scores, whose expected loss uniquely favors the target distribution: Brier for categories and ranked probability score (RPS; Epstein, 1969) for ordinal ratings. Normalized RPS reduces to normalized absolute error on hard predictions. One rating per test item estimates expected score without bias. Squared Pearson correlation equals one minus affine-refitted mean squared error divided by rating variance, ignoring level and scale. With conditional independence and positive within- and between-item variation, Pearson favors conditional-mean readouts over truthful sampled ratings. Five frontier judges showed no significant preregistered Pearson-versus-proper-score ranking reversal. On DICES-350, four achieved 80–82% majority agreement but forecast 9–14% unsafe against 33% human. All 25 judge–dataset mean calibrated scores first beat the base rate at 100 ratings, the smallest tested successful budget; 78/82 transferred maps significantly underperformed in-domain maps. Correlation selection followed by recalibration outperformed selection by cross-validated calibrated proper risk. We recommend proper-score evaluation alongside correlation, distinguishing calibration error from resolution.

9. **Medium — Audit categories contradict their exceptions.** `2-introduction.tex:21` and `11-supplementary-related-work.tex:54–60,67` call six papers item-level human-distribution evaluations, although Elangovan is pooled and Li uses synthetic votes. Replace: “Six papers report a proper score or distributional comparison under our broad code, which includes pooled labels and synthetic votes.” Apply that definition to the table and complementary counts.

10. **Low — Known corrections remain uncorrected in active prose.** `9b-appendix-concordance-and-calibration.tex:35` establishes **30 seeds/five starts**, but `:41,48,52` still say **100/nine**; replace those numbers. `10-supplementary.tex:275` should say “64 ratings for DICES-350 and 128 for DICES-990,” matching `6-case-study.tex:15`. The abstract evidence map `:31` incorrectly says raw-score-use citations are missing; update it to reference `2-introduction.tex:7`.

## VERIFIED

- Abstract rounding is supported: agreement **80.2857–82.2857%**; forecast unsafe rates **8.6086–13.7966%**; human rate **32.6690%**.
- Saved results support five judges, **0/50 significant Pearson reversals**, **78/82** inferior transfers, and Experiment 59’s **20/20** negative mean differences with **17/20** intervals below zero.
- Other requested numerical values matched saved results; Experiment 46/57 skill differences arise from different base-rate denominators.
- The any-score decomposition is algebraically correct when expectations exist; propriety supplies nonnegativity. The population map-improvement bound is correct.
- Numerical checks ran twice: **25/25 PASS**, identical output. The examples give resolution **2/9** and squared correlations **4/7 versus 25/73**.
- Read Lean statements support finite scalar/multiclass decomposition and both sampled-score forms; hard-prediction checks correctly restrict to \(K=2,\ldots,8\).
- Credit is present for Pavlick–Kwiatkowski, Plank, Epstein, Murphy, Bröcker, Elangovan, Li, and concurrent ordinal-scoring work. Classical identities are identified as classical.

## NOT CHECKED

- Lean compilation, PDF rendering, experiment reruns, external papers, bibliographic accuracy, model identities, and raw-data provenance. This was an offline, read-only source/result review; no files were changed.
<!-- end verbatim -->
