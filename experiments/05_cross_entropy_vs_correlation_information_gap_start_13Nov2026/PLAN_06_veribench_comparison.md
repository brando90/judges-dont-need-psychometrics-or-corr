# Plan 06: distribution-matching versus correlation on VeriBench (not yet run)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/05_cross_entropy_vs_correlation_information_gap_start_13Nov2026/PLAN_06_veribench_comparison.md>

**TLDR:** Action item 2 of the 11-13-2026 Impromptu Call. It needs the private rating dataset (75 artifacts, 25 tasks, 195 expert ratings), which sits on Brando's Mac and in no public repository, so it is a separate run there. Free only: $0 hardware, no paid API.

**Question:** on real VeriBench ratings, does a judge trained for distribution matching (CE against the human rating) forecast held-out human ratings better than a correlation-trained judge?

**Already done, do not repeat:** experiment 02 (CE-mixture versus Pearson-mixture on cached judge scores, held-out log loss difference −0.26 [−1.46, 0.20]); experiment 04 phase C (reliability versus RPS on the same data).

**New in 06:**
1. Per-artifact empirical information gap G from the expert ratings (with a shrinkage or Dirichlet-smoothed P because 2–3 ratings are sparse), then the correlation between G and the held-out RPS (ranked probability score) difference of the CE-trained and the Pearson-trained judge: if the theory is right, tasks with larger G favor CE.
2. Same split protocol as experiment 02 (task-level 80/20, repeated splits, task-family bootstrap), primary metric RPS, secondary CE, TVD, Spearman, Pearson; Pearson-trained judge and CE-trained judge matched in data, family, and budget.
3. Label DT or no DT and name harness and model for every run; write the protocol and predictions before measuring (freeze commit), results straight to main.

**Dispatch:** a coding agent on Brando's Mac, with the private root `trustworthy-theorem-judges/private/06_information_loss_vs_correlation_judge_selection/` (cached_v1/dataset.json, SHA-256 b3a05699…). Keep raw ratings out of the public repository.
