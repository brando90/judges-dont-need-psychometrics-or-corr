# How to read Experiment 53's results

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/53_exp41_heads_as_skill_scores_with_item_bootstrap_cis_start_27Nov2026/HOW_TO_READ_RESULTS.md>

- **Data:** held-out test items of Experiment 41's five group-level split seeds on 16 views of 7 public multi-annotator rating sets. Every test rating of each item is scored. These test items were also test items in Experiment 41, so they are not fresh.
- **Metric:** normalized ranked probability score (nRPS = RPS/(K-1), 0 to 1, lower is better). Skill = 1 - nRPS(head)/nRPS(base rate). The base rate is the training-split marginal. Higher is better: 1 is perfect, 0 matches the base rate, and below 0 is worse than the base rate.
- **Heads:** linear heads on frozen text features, trained by cross-entropy, by Pearson correlation or by MSE (ridge 0.01, at most 35 L-BFGS iterations). "Raw" is the head's own forecast. "Cal" is the head's scalar mean passed through its own multinomial-logit map, fitted on the validation items.
- **Differences:** skill of the CE-trained head minus skill of the Pearson-trained head, on the same items. A positive value means the CE-trained head is better.
- **Uncertainty:** item-clustered percentile bootstrap 95% intervals (2,000 draws; test items resampled within each seed; paired across heads; pooled as the mean over seeds). Holm-Bonferroni across 16 views.
- **Not shown here:** label scarcity (Experiment 47), shift (Experiment 48) and real-judge rank reversals (Experiment 46).
