# Experiments

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/README.md>

**TLDR:** This project studies which judge objectives identify or learn human-rating distributions. Read the synthetic studies separately from the historical human-rating comparisons; their follow-ups share data and are not independent replications. The applied human-rating collection remains in trustworthy-theorem-judges.

## Synthetic illustrations and identification

| Folder | Distinct question | Status |
| --- | --- | --- |
| [01_synthetic_pearson_vs_kl_squashed_judge_0to5](01_synthetic_pearson_vs_kl_squashed_judge_0to5/README.md) | Original scalar example comparing smoothed marginal histograms | Historical code and output preserved; reproduced by experiment 02 |
| [02_conditional_binomial_distribution_identification](02_conditional_binomial_distribution_identification/README.md) | Identify a known conditional population among analytic probability candidates, with full tie accounting | Completed simulation; parent review and paper integration remain active |
| [13_toy_pearson_vs_ce_nonlinear_ratings](archive/13_toy_pearson_vs_ce_nonlinear_ratings/README.md) | Learn flexible judges from nonlinear synthetic ratings across training sizes | Completed illustration, imported unchanged; not human-data evidence |

## Historical human-rating objective comparisons

| Folder | Distinct question | Status |
| --- | --- | --- |
| [03_exp06_ce_vs_pearson_matched_budget](03_exp06_ce_vs_pearson_matched_budget/RESULTS_comparison.md) | Certified cross-entropy-versus-Pearson mixture fits with matched budgets; also owns the earlier experiment 06 history | Completed and verified; paper integration remains active |
| [11_formulation4_train_only_ce_mixture_cv](archive/11_formulation4_train_only_ce_mixture_cv/README.md) | Train-only cross-entropy baseline with repeated five-fold evaluation | Completed, same historical ratings |
| [12_formulation2_vs_4_corr_vs_ce_80_20_splits](archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits/README.md) | Correlation-versus-cross-entropy training on repeated 80/20 task splits | Completed follow-up to the baseline, same ratings and cached judge pool |
| [14_open_weight_judge_pool_corr_vs_ce_80_20](archive/14_open_weight_judge_pool_corr_vs_ce_80_20/README.md) | Extend the comparison to an expanded open-weight judge pool | Completed follow-up to experiment 12; additional model outputs, same human ratings |

The [archive guide and migration receipt](archive/README.md) explain what changed between these studies, which files were duplicates, current paths and reproducibility limits. Original experiment numbers are preserved after migration; no new studies were created by moving folders.

[trustworthy-theorem-judges experiment 09](https://github.com/brando90/trustworthy-theorem-judges/blob/main/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/README.md) owns the current participant task and collection. Its original-rating provenance, practical judge comparison and task/solver/rater size planning stay in that project. This repository owns the general objective-comparison research and links to the authorized shared data rather than creating a competing collection.

**TLDR-end:** [judge objectives: experiments] Two evidence groups replace a flat list of seemingly repeated tasks: synthetic distribution studies and historical human-rating objective comparisons. Their original conditions and shared-data limitations remain visible.
