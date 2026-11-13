# Experiments

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/README.md>

**TLDR:** This project studies which judge objectives identify or learn human-rating distributions. Read the synthetic studies separately from the historical human-rating comparisons; their follow-ups share data and are not independent replications. The applied human-rating collection remains in trustworthy-theorem-judges.

## Folder dates

Folder names end in `_start_<DDMonYYYY>` once work began and `_end_<DDMonYYYY>` once the study reached its final verdict ([agents-config Rule 88](https://github.com/brando90/agents-config/blob/main/rules/research.md#trigger-rule-88)). The dates are day-level labels read from git history in America/Los_Angeles time. For exact times, or if a label looks wrong, git history is the authority: run `git log --follow -- <folder>` here, and for experiments 11–14 also in trustworthy-theorem-judges, where they originally ran.

| Experiment | Started | Ended | Evidence |
| --- | --- | --- | --- |
| 01 | 11-07-2026 | 11-07-2026 | Code and results added together in `f0f1ed6`. |
| 02 | 11-08-2026 | Not yet | Procedure frozen in `9139470`; results in `8137f52`. Its `results.md` still marks the multi-provider review and main publication `PENDING`, so the folder has no end date. |
| 03 | 11-09-2026 | 11-09-2026 | Handoff `be28db0` (10:26 PST) through verified results `8a44e6d` (11:01 PST). It is a new comparison on cached experiment 06 data, so it uses its own dates, not experiment 06's 11-06-2026 start. |
| 11–14 | 11-10-2026 | 11-10-2026 | Each was run and committed on 11-10-2026 in trustworthy-theorem-judges (`652f075`, `6d11895`, `597f245`, `ae2585b`). Its 11-11-2026 archive move (`bc15da9`) and the import here were moves, not new work. |

## Synthetic illustrations and identification

| Folder | Distinct question | Status |
| --- | --- | --- |
| [01_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026_end_07Nov2026](01_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026_end_07Nov2026/README.md) | Original scalar example comparing smoothed marginal histograms | Historical code and output preserved; reproduced by experiment 02 |
| [02_conditional_binomial_distribution_identification_start_08Nov2026](02_conditional_binomial_distribution_identification_start_08Nov2026/README.md) | Identify a known conditional population among analytic probability candidates, with full tie accounting | Completed simulation; parent review and paper integration remain active |
| [13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026](archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026/README.md) | Learn flexible judges from nonlinear synthetic ratings across training sizes | Completed illustration, imported unchanged; not human-data evidence |

## Historical human-rating objective comparisons

| Folder | Distinct question | Status |
| --- | --- | --- |
| [03_exp06_ce_vs_pearson_matched_budget_start_09Nov2026_end_09Nov2026](03_exp06_ce_vs_pearson_matched_budget_start_09Nov2026_end_09Nov2026/RESULTS_comparison.md) | Certified cross-entropy-versus-Pearson mixture fits with matched budgets; also owns the earlier experiment 06 history | Completed and verified; paper integration remains active |
| [11_formulation4_train_only_ce_mixture_cv_start_10Nov2026_end_10Nov2026](archive/11_formulation4_train_only_ce_mixture_cv_start_10Nov2026_end_10Nov2026/README.md) | Train-only cross-entropy baseline with repeated five-fold evaluation | Completed, same historical ratings |
| [12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026](archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026/README.md) | Correlation-versus-cross-entropy training on repeated 80/20 task splits | Completed follow-up to the baseline, same ratings and cached judge pool |
| [14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026](archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/README.md) | Extend the comparison to an expanded open-weight judge pool | Completed follow-up to experiment 12; additional model outputs, same human ratings |

The [archive guide and migration receipt](archive/README.md) explain what changed between these studies, which files were duplicates, current paths and reproducibility limits. Original experiment numbers are preserved after migration; no new studies were created by moving folders.

[trustworthy-theorem-judges experiment 09](https://github.com/brando90/trustworthy-theorem-judges/blob/main/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/README.md) owns the current participant task and collection. Its original-rating provenance, practical judge comparison and task/solver/rater size planning stay in that project. This repository owns the general objective-comparison research and links to the authorized shared data rather than creating a competing collection.

**TLDR-end:** [judge objectives: experiments] Two evidence groups replace a flat list of seemingly repeated tasks: synthetic distribution studies and historical human-rating objective comparisons. Their original conditions and shared-data limitations remain visible.
