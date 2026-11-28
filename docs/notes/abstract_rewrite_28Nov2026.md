# Abstract rewrite, 11-28-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/notes/abstract_rewrite_28Nov2026.md>

**TLDR:** The abstract in `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/1-abstract.tex` was rewritten on 11-28-2026 to 218 words (rule in `abstract_wordcount_28Nov2026.py`: whitespace tokens, each math span one token). Each sentence below points to the place in the paper or the experiment that supports it. The abstract claims no novelty for the distribution target or for proper scores, and it makes no claim that correlation reverses a ranking.

## Sentence by sentence

| # | Abstract sentence (short form) | Where it is supported |
|---|---|---|
| 1 | Judges are validated by correlation or agreement. | Introduction; `11-supplementary-related-work.tex` (label `sec:supp-practice`; Experiments 43 and 54). |
| 2 | Target is the rating distribution of a rater pool and rubric, following Pavlick and Kwiatkowski (2019) and Plank (2022). | Introduction, paragraph "Target versus ruler"; bib keys `pavlick2019inherent`, `plank2022problem`. |
| 3 | Elangovan et al. (2025) and Li et al. (2026) already compare judges with such distributions. | Introduction, same paragraph; `11-supplementary-related-work.tex`, paragraphs on each paper; `docs/notes/elangovan_li_summary_27Nov2026.md`. |
| 4 | We adopt item-level strictly proper scores (Brier; RPS, Epstein 1969); RPS on a hard prediction is normalized absolute error. | `4-proposal.tex` line 11; `8-appendix-rps.tex` line 173. Lean-checked special case only (scales K = 2 to 8, `checks/HalfBrierRps.lean`); the induction on K for general K is open (`docs/LEAN_TODO.md`). |
| 5 | One rating per item gives an unbiased expected score, enough to rank judges. | `4-proposal.tex` line 30; Experiment 52 (exact fractions and simulation); listed in `docs/LEAN_TODO.md` as numerically checked, Lean pending. |
| 6 | Squared correlation is one minus the best affine fit's mean squared error over the rating variance. | Appendix, "Why Correlation Misses Calibration"; Lean `r_sq_resolution`, `condE_affine` (`docs/notes/point2_support_27Nov2026/lean/Point2/Murphy.lean`). |
| 7 | For sampled scores, correlation prefers the conditional mean to a draw from the rating distribution. | Introduction, contribution 2; `12-supplementary-pool-strata-power.tex`; `13-supplementary-calibration-level-selection.tex`. Lean `PlanAudit.Model.anti_proper` (Experiment 35) and `Sampled.Model.anti_proper` (`Point2/SampledReadout.lean`), finite laws; Experiment 60 on three real rating sets (an oracle check, no language model). |
| 8 | Preregistered test: no significant rank reversal between Pearson correlation and a proper score on five frontier judges. | Experiment 46 (0 of 50 Pearson reversals significant). |
| 9 | On DICES-350 four judges with 80 to 82 percent majority agreement forecast unsafe rates of 9 to 14 percent where the human rate is 33 percent. | Experiment 58 (8.6 to 13.8 percent against 32.7 percent); Experiment 46 for the agreement range. |
| 10 | Recalibration needs about 100 single held-out ratings and transfers poorly (78 of 82 worse than in-domain). | Experiment 56 (first budget at which every judge beats the base rate is 100; budgets tested 10, 25, 50, 100, 200); Experiment 48 (82 transfers: 80 frontier plus 2 G-Eval). |
| 11 | Recommend a proper score, split into calibration error and resolution, beside correlation. | Conclusion checklist; calibration appendix (identity for every score; Lean for squared error and Brier). |

## What the abstract does not say

- It does not say that correlation reverses a ranking on real judges. The preregistered Pearson test is null (Experiment 46). The Spearman reversal on two NLI sets is in the body.
- It does not say that selecting by a proper score beats selecting by correlation. Experiment 59 found the opposite in all 20 cells (the correlation-then-calibrate path had lower mean regret), and the conclusion says so.
- It does not mention cross-entropy, Lean, the 49-paper audit or the cross-entropy-versus-correlation head comparison (circular as a test of the evaluation claim; Experiments 41, 47, 53).

## Not yet addressed

- Sentence 6 names "pass-rate thresholds" as a use where level and scale matter, but the paper has no citation showing that practitioners apply thresholds to raw judge scores. That needs sources, not a guess.
