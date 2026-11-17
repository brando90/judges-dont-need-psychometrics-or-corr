# Experiments

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/README.md>

**TLDR:** This project studies which judge objectives identify or learn human-rating distributions. Read the synthetic studies separately from the historical human-rating comparisons; their follow-ups share data and are not independent replications. The applied human-rating collection remains in trustworthy-theorem-judges.

## Folder dates

Folders directly under `experiments/` are active and end in `_start_<DDMonYYYY>` only. An experiment is done only when it moves to `archive/`, and only then does its name gain `_end_<DDMonYYYY>` ([agents-config Rule 88](https://github.com/brando90/agents-config/blob/main/rules/research.md#trigger-rule-88)). Active experiments are numbered from 00. The dates are day-level labels read from git history in America/Los_Angeles time. For exact times, or if a label looks wrong, git history is the authority: run `git log --follow -- <folder>` here, and for experiments 11–14 also in trustworthy-theorem-judges, where they originally ran.

| Experiment | Started | Ended | Evidence |
| --- | --- | --- | --- |
| 00 | 11-07-2026 | Active | Code and first results added together in `f0f1ed6`. |
| 01 | 11-08-2026 | Active | Procedure frozen in `9139470`; results in `8137f52`. Its `results.md` still marks the multi-provider review and main publication `PENDING`. |
| 02 | 11-09-2026 | Active | Handoff `be28db0` (10:26 PST); comparison results `8a44e6d` (11:01 PST); integrated in AAAI-27 `6-case-study.tex` (`11-17-2026`). It is a new comparison on cached experiment 06 data, so it uses its own start date, not experiment 06's 11-06-2026 start. |
| 04 | 11-13-2026 | Active | Plan `cd66258`; protocol frozen before measurement `579915a`; Phases A–C ran the same day (`a78a626`). AAAI-27 reliability appendix added 11-14-2026 (`03ada1e`); main-body cross-reference added in `3-problem.tex` (`11-17-2026`). User-requested identifier 04 is retained; the gap at 03 is intentional. |
| 05 | 11-13-2026 | Active | Derivation and numerical check of the cross-entropy versus correlation information gap (Impromptu Call action item 1). The VeriBench plan (real VeriBench comparison, action item 2) is written but not run. |
| 06 | 11-13-2026 | Active | Protocol and objectives committed before measurement (`d46b984`); sweep and results the same day. Kept active for the one-pager and paper integration. |
| 07 | 11-13-2026 | Active | Protocol committed before measurement (`d9d1c1d`); sweep interrupted at 532 of 540 rows when the session ended, resumed and completed 11-14-2026. Kept active for the one-pager and paper. |
| 08 | 11-14-2026 | Active | Protocol and hypotheses committed before measurement (`4e2a8aa`); amendment for undefined validation correlations (`094edde`). |
| 09 | 11-16-2026 | Active | Open-weight real judges (`mercury2`, `$0.00`): protocol frozen before measurement (`2997834`); evaluates prompt optimization (`ii.a`), harness optimization (`ii.b`), and end-to-end PyTorch low-rank adaptation (LoRA) transformer weight updates (`ii.c`) across 19 task families. |
| 10 | 11-16-2026 | Active | Frontier artificial intelligence (AI) judges (`koyejolab-openai-bm` + `mistral`, `$17.85` / `$18.00` cap): protocol frozen before measurement (`FREEZE.json`, pre-run worktree `aa09730` integrated in `a346d5c`); evaluates native 6-category probabilities, `iii.a` blind prompt optimization, and `iii.b` frontier harness optimization across 19 task families. |
| 11–14 | 11-10-2026 | 11-10-2026 | Archived. Each was run and committed on 11-10-2026 in trustworthy-theorem-judges (`652f075`, `6d11895`, `597f245`, `ae2585b`). Its 11-11-2026 archive move (`bc15da9`) and the import here were moves, not new work. Archived experiments keep their original numbers. Same-day start and end were rechecked on 11-13-2026: these were short runs that built on each other in one evening. 11 was committed at 15:08 PST; 12 at 17:50 (full run 160 s); 13 at 18:46 (sweep 435 s); 14 ran 18:03–20:43 (GPU scoring about 1 h 50 min). No folder changed after its commit (the moves are byte-identical renames). The formulations they test were written down on 11-09-2026 (`930da26`), but no experiment work appears before 11-10. |

## Synthetic illustrations and identification

| Folder | Distinct question | Status |
| --- | --- | --- |
| [00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026](00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026/README.md) | Original scalar example comparing smoothed marginal histograms | Active. Historical code and output preserved; reproduced by experiment 01 |
| [01_conditional_binomial_distribution_identification_start_08Nov2026](01_conditional_binomial_distribution_identification_start_08Nov2026/README.md) | Identify a known conditional population among analytic probability candidates, with full tie accounting | Active. Simulation finished; parent review and paper integration remain open |
| [06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026](06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026/README.md) | Train the same toy judge with Spearman, Kendall or ½ Pearson + ½ Spearman on experiment 13's exact tasks | Active. Complete; none matches the CE-trained judge on held-out distribution metrics |
| [07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026](07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026/README.md) | Sensitivity to L-BFGS-B budgets (400/600 to 2,000/4,000 iterations/evaluations) and best of 5 starts | Active. Complete; held-out crossovers unchanged under these settings; no global optimum certificate |
| [08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026](08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/README.md) | Does correlation training win with little data, and do the ratings needed grow with target complexity? Mean-only targets, validation-tuned ridge (formulations ① and ③) | Active. Complete; among four tested targets, correlation training wins at low data only in scenario A (little mean signal) |
| [13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026](archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026/README.md) | Learn flexible judges from nonlinear synthetic ratings across training sizes | Completed illustration, imported unchanged; not human-data evidence |

## Reliability and distribution-accuracy study

| Folder | Distinct question | Status |
| --- | --- | --- |
| [04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026](04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/README.md) | Are moment matching and generalizability-theory reliability sufficient for human-rating distribution accuracy, and when does variance decomposition improve evaluation design? | Active. Complete 11-13-2026: not sufficient (exact counterexamples; most repeatable historical judge had the worst test ranked probability score (RPS)); design allocation works under its assumptions. [Results](04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/RESULTS_reliability_vs_accuracy.md). AAAI-27 appendix and `3-problem.tex` main-body cross-reference integrated |

## Historical human-rating objective comparisons

| Folder | Distinct question | Status |
| --- | --- | --- |
| [02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026](02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/RESULTS_comparison.md) | Certified cross-entropy-versus-Pearson mixture fits with matched budgets; also owns the earlier experiment 06 history | Active. Comparison finished and verified; paper integration remains open |
| [11_formulation4_train_only_ce_mixture_cv_start_10Nov2026_end_10Nov2026](archive/11_formulation4_train_only_ce_mixture_cv_start_10Nov2026_end_10Nov2026/README.md) | Train-only cross-entropy baseline with repeated five-fold evaluation | Completed, same historical ratings |
| [12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026](archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026/README.md) | Correlation-versus-cross-entropy training on repeated 80/20 task splits | Completed follow-up to the baseline, same ratings and cached judge pool |
| [14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026](archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/README.md) | Extend the comparison to an expanded open-weight judge pool | Completed follow-up to experiment 12; additional model outputs, same human ratings |
| [09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026](09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/RESULTS_open_weight_ce_vs_corr.md) | Open-weight real judges (`ii`): native 6-category probabilities (`q_verbal`, `q_logprob`), `ii.a` prompt optimization, `ii.b` harness optimization, and `ii.c` PyTorch low-rank adaptation (LoRA) transformer weight updates across 19 task families | Active. Complete and verified; integrated in AAAI-27 `6-case-study.tex` and Version 25 one-pager; same retrospective 75-artifact human ratings |
| [10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026](10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/RESULTS_frontier_ce_vs_corr.md) | Frontier artificial intelligence (AI) judges (`iii`): 15-judge multi-family native-probability pool, `iii.a` blind prompt optimization, and `iii.b` frontier harness optimization across 19 task families | Active. Complete and verified (`$17.85`); integrated in AAAI-27 `6-case-study.tex` and Version 25 one-pager; same retrospective 75-artifact human ratings |

The [archive guide and migration receipt](archive/README.md) explain what changed between these studies, which files were duplicates, current paths and reproducibility limits. Original experiment numbers are preserved after migration; no new studies were created by moving folders.

[trustworthy-theorem-judges experiment 09](https://github.com/brando90/trustworthy-theorem-judges/blob/main/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/README.md) owns the current participant task and collection. Its original-rating provenance, practical judge comparison and task/solver/rater size planning stay in that project. This repository owns the general objective-comparison research and links to the authorized shared data rather than creating a competing collection.

**TLDR-end:** [judge objectives: experiments] Separate synthetic distribution evidence, historical human-rating comparisons and the reliability study. The reliability study (04) finds reliability insufficient for distribution accuracy while design allocation works under its assumptions; existing evidence and shared-data limitations remain unchanged.
