# How to read the results of experiment 00

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Nothing is trained and there is no train/test split. Every number in the [README](README.md) table describes the same 2,000 synthetic items, comparing two hand-written judges with the synthetic human scores.

## What is compared

- **Judge A (noisy faithful):** human score plus noise in {−1, 0, +1}, clipped to 0–5.
- **Judge B (squashed):** 5 if the human score is at least 3, otherwise 4.

## Which data each number uses

All 2,000 synthetic human scores (uniform on 0–5, seed 0) and the two judges' scores on the same items. There is no fitting, so train versus test does not apply.

## Metrics

| Metric | Definition | Range | Better |
| --- | --- | --- | --- |
| Pearson / Spearman | Correlation between human score and judge score across the 2,000 items | −1 to 1 | Higher |
| KL | KL(human ‖ judge) between the two score histograms, Laplace-smoothed (alpha = 1), natural log. Ignores which item each score came from. | 0 upward, nats | Lower |
| nMSE | Mean of ((human − judge)/5)² over items | 0 to 1 | Lower |

There are no differences between methods, no intervals and a single seed.

**TLDR-end:** [judge objectives: experiment 00] No training and no split; all numbers describe the same 2,000 synthetic items.
