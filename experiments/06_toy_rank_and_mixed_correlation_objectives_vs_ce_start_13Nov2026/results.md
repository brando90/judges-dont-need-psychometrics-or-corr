# Experiment 06: results

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026/results.md>

**Status, 11-13-2026:** sweep complete; kept active because the Sanmi one-pager and the paper's synthetic section depend on it.

**TLDR:** Training the judge on a rank correlation (Spearman or Kendall) or on ½ Pearson + ½ Spearman does not fix what Pearson training misses. On held-out test RPS and cross-entropy, the cross-entropy (CE)-trained judge beats all four correlation-trained judges from 4 training tasks (12 ratings) in scenario B, and from 64 tasks (192 ratings; 8 for Kendall) in scenario A. Even on held-out test correlation, CE training is as good or better until data are plentiful. Synthetic toy; not evidence about real ratings.

| Phase | Status |
| --- | --- |
| Protocol and objectives committed before measurement (`d46b984`) | DONE |
| Gradient checks; data reproduction of experiment 13 | DONE: CE refit matches experiment 13's saved test RPS excess to 7.6e-15 in all 22 (scenario, n) cells |
| Sweep: 2 scenarios × 11 training sizes × 100 seeds × 3 objectives | DONE: 2,200 runs, 3,355 s on 9 local CPU processes, $0 |
| Figures and tables | DONE |

## Main results (held-out test tasks, against the true distribution)

Smallest number of training tasks from which the CE-trained judge is significantly better at every larger size (paired per-seed difference, 95% bootstrap over 100 seeds); each task has 3 ratings:

| Correlation-trained judge | Scenario A, test RPS | Scenario A, test CE | Scenario B, test RPS | Scenario B, test CE |
| --- | --- | --- | --- | --- |
| Pearson | 64 (192 ratings) | 64 (192) | 4 (12) | 4 (12) |
| Spearman | 64 (192) | 32 (96) | 4 (12) | 4 (12) |
| Kendall | 8 (24) | 16 (48) | 4 (12) | 4 (12) |
| ½ Pearson + ½ Spearman | 64 (192) | 64 (192) | 4 (12) | 4 (12) |

At 2,048 training tasks, held-out test RPS excess is 0.0007 for the CE-trained judge against 0.025–0.031 for the correlation-trained judges in scenario A, and 0.0037 against 0.033–0.095 in scenario B.

Held-out test correlation with the true mean rating ([fig_rank_objectives_correlations_vs_n.png](fig_rank_objectives_correlations_vs_n.png)): in scenario A the CE-trained judge has the highest test Pearson and Spearman at every size (Pearson 0.891 vs 0.746–0.846 at 2,048 tasks). In scenario B it leads at small sizes (test Pearson 0.688 vs 0.090–0.123 for the Pearson-, Spearman- and mixed-trained judges at 16 tasks); from about 256 tasks the correlation-trained judges catch up and slightly exceed it on their own metric (Spearman-trained test Spearman 0.982 vs 0.948 at 2,048 tasks), while their test RPS excess stays about 17× larger (0.062 vs 0.0037).

Full tables: [results/tables.md](results/tables.md). Figures: [fig_rank_objectives_vs_n.png](fig_rank_objectives_vs_n.png), [fig_rank_objectives_correlations_vs_n.png](fig_rank_objectives_correlations_vs_n.png). Reading guide: [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md). Regenerate with `python3 experiments/06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026/scripts/make_report.py`.

The prediction in the README holds: no correlation objective matches the CE-trained judge at large n.

**TLDR-end:** [judge objectives: experiment 06] Rank and mixed correlation training lose to CE training on held-out distribution metrics from 12–192 ratings, and match or trail it on held-out correlation until data are plentiful.
