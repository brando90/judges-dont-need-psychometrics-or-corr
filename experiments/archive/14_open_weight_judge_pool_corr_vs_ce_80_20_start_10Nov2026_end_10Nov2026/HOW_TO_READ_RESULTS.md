# How to read the results of experiment 14

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Experiment 14 repeats [experiment 12's](../12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026/HOW_TO_READ_RESULTS.md) held-out protocol on three judge pools. Every comparison table in [results.md](results.md) is a held-out test score; the per-judge table at the end is **in-sample**. (This guide was added on 11-13-2026; `results.md` is a frozen copy and is unchanged.)

## What is compared

The same ② (correlation-trained) and ④ (loss-trained) choosers as experiment 12, on three pools of judges:

- **Old pool:** the 5 cached frontier judges (reproduces experiment 12 exactly).
- **Open pool:** 39 open-weight judges (15 models × 3 prompts, minus 6 unusable).
- **Combined pool:** all 44.

The open-weight judges are new model outputs on the same 75 artifacts. There are no new human ratings.

## Which data each number uses

| Table | Data |
| --- | --- |
| "② minus ④" and "Mean test score of each chooser" | Held-out test tasks: 1,000 random splits of 20 training tasks (used only to choose or fit) and 5 test tasks (scored). Per-artifact test metric averaged over the splits where it was held out. |
| "Each judge on its own (all 75 artifacts, in-sample, no fitting)" | All 75 artifacts, no split. Nothing is fitted, but choosing a judge from this table would be an in-sample choice. |

Splits group whole tasks, not task families. **Prior exposure:** the 75 artifacts and their ratings were used in earlier studies.

## Metrics and differences

Metrics are defined exactly as in experiment 12: ranked probability score (RPS; main; 0 to 1, lower is better), CE (nats, lower is better), TVD (0 to 1, lower is better), Spearman and Pearson (−1 to 1, higher is better).

"② minus ④" is **the column's test metric for the ② judge minus the same test metric for the ④ judge.** For RPS, CE and TVD, positive means the ④ (loss-trained) judge did better; for correlations, positive means the ② judge did better.

Test RPS of each chooser, for scale:

| Pool | ② mixture | ④ mixture | ② best single | ④ best single | Constant |
| --- | --- | --- | --- | --- | --- |
| Old 5 | 0.1419 | 0.1362 | 0.1797 | 0.2051 | 0.1836 |
| Open 39 | 0.3274 | 0.2773 | 0.3680 | 0.3151 | 0.1836 |
| Combined 44 | 0.2063 | 0.1658 | 0.1797 | 0.2114 | 0.1836 |

Example: on the combined pool, test RPS of the ② mixture minus test RPS of the ④ mixture = +0.0405 [+0.0237, +0.0575], so the loss-trained mixture had lower (better) test RPS.

## Uncertainty

95% percentile intervals from 10,000 bootstrap draws over the 25 tasks. No p-values were computed.

**TLDR-end:** [judge objectives: experiment 14] Comparison tables are held-out test scores with experiment 12's sign convention; the per-judge table is in-sample.
