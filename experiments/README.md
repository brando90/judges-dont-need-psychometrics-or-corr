# Experiments

Each experiment has one canonical home. Historical evidence is preserved; completed measurements are distinguished from review and publication.

| Folder | Setup | Status |
| --- | --- | --- |
| [01_synthetic_pearson_vs_kl_squashed_judge_0to5](01_synthetic_pearson_vs_kl_squashed_judge_0to5/README.md) | Historical scalar-score example comparing smoothed marginal histograms | Original code/output preserved; reproduced by experiment02 |
| [02_conditional_binomial_distribution_identification](02_conditional_binomial_distribution_identification/README.md) | Known binomial conditional population; nine candidates, held-out parameter families, one/two/three synthetic ratings per item, full tie accounting | All 768 simulation conditions and 5,376 selections complete; parent review and paper integration remain active |
| [03_exp06_ce_vs_pearson_matched_budget](03_exp06_ce_vs_pearson_matched_budget/ckpt.md) | Continuation of sister-repository experiment 06: diagnose Pearson nonconvergence and compare cross-entropy against correlation with matched budgets | Handoff accepted 11-09-2026; diagnostic and draft comparison prepared, synthetic controls pass; empirical execution awaiting cluster access or local-fallback decision |

The former placeholder `02_kl_vs_corr_judge_fit_theorem_proving_human_ratings` was never created. Historical theorem-judge fitting and the completed weighted-probability follow-up belong to the sister repository’s [experiment06](https://github.com/brando90/trustworthy-theorem-judges/tree/main/experiments/06_information_loss_vs_correlation_judge_selection); experiment02 here is a distinct controlled identification study.

Reproduction commands, every common metric, uncertainty definitions and exact public sister-repository revisions are in the [conditional-identification report](02_conditional_binomial_distribution_identification/RESULTS_conditional_identification.md). No human or private source data are stored here.

**TLDR-end:** [judge objectives: experiments] The old marginal toy and the new conditional study answer different questions. Both remain reproducible, with the new study awaiting parent review.
