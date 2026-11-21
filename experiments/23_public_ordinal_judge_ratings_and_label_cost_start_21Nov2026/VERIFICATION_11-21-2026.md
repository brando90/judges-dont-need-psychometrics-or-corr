# Independent replay of the public ordinal-rating experiment | 11-21-2026

As of 11-21-2026, the replay reproduced all 17,024 per-draw, per-method held-out ranked probability scores (RPS) and all 152 prespecified paired contrasts across the two primary datasets, both training depths and every tested budget. RPS is the mean squared error over the four cumulative grade thresholds for an individual human rating. It ranges from 0 to 1, with lower values better. The largest absolute error was below 2.2 × 10⁻¹⁵ for a draw score and 1.7 × 10⁻¹⁶ for a contrast or simultaneous interval endpoint. [Verification output](results/primary_verification.csv)

| Held-out data | Grades per training item | First sustained budget beating Pearson and uniform | First sustained budget beating all four comparators |
|---|---:|---:|---|
| SummEval factual consistency, 640 summaries from 40 articles | 1 | None through 768 grades | None through 768 |
| Same SummEval summaries | 3 | None through 768 grades | None through 768 |
| KonIQ image quality, 2,015 distinct test images | 1 | 12 grades on 12 images | None through 6,144 |
| Same KonIQ images | 3 | 48 grades on 16 images | None through 6,144 |

“Sustained” requires a positive lower band for the held-out RPS of each comparator minus the cross-entropy (CE)-trained judge at that budget and every larger tested budget. The four comparators were the Pearson-trained judge, uniform, a constant training histogram, and the Pearson-trained judge calibrated to a distribution. For example, on KonIQ with one grade on each of 12 training images, test RPS was 0.12991 for the CE-trained judge, 0.26776 for the Pearson-trained judge, 0.13325 for uniform, 0.11310 for the training histogram, and 0.11012 for calibrated Pearson. On SummEval at the same training budget those values were 0.25710, 0.19316, 0.28367, 0.08179, and 0.07260, respectively. [KonIQ scores](results/image_quality/scores.csv) · [SummEval scores](results/gpt4_consistency/scores.csv)

The replay verified the published image split, the frozen article-hash split, training/test group disjointness, and all 64 nested training schedules. It resampled individual grades from raw counts without replacement, checked that the one-grade and three-grade schedules shared their first observed grade, constructed the candidate grids with direct powers or Gaussian kernels, and obtained the feature principal components by eigendecomposition. It exhaustively reselected CE, Pearson and RPS fits from the matched candidates using individual grade losses and centered Pearson. It then re-scored each method against each possible individual test grade before weighting by the actual counts. These are public, previously studied benchmarks, with new train/test separation for this fitting experiment. [Frozen design](PROTOCOL_11-21-2026.md) · [Independent replay code](verify_primary.py)

For the bands, the replay averaged over the fixed 64 training schedules, drew 4,000 multinomial resamples of complete held-out articles or images, and computed the 97.5th percentile of the maximum absolute standardized deviation over all four comparators and budgets within each benchmark and depth. At one grade per item the two benchmark families give approximate joint 95% coverage by Bonferroni. Training schedules, available raters and systems are fixed in this calculation. [SummEval contrasts](results/gpt4_consistency/contrasts.csv) · [KonIQ contrasts](results/image_quality/contrasts.csv)

## Two-item Pearson correction

With two training items every defined Pearson correlation equals +1 or −1. Computing it from differences of uncentered squares introduced cancellation for candidates with nearly equal predicted scores. Some mathematically tied candidates then missed the first-candidate tie rule. The corrected calculation takes the sign of the product of the two observed and predicted score differences, alongside the existing variance eligibility rule. The discrepancy affected only the secondary six-grade budget with three grades on each of two items. The table gives the old and corrected held-out RPS of the Pearson-trained judge minus the CE-trained judge, with the same simultaneous bands defined above. A negative value favors the Pearson-trained judge. [Original candidate and analysis source](https://github.com/brando90/judges-dont-need-psychometrics-or-corr/tree/8d238e6/experiments/23_public_ordinal_judge_ratings_and_label_cost_start_21Nov2026) · [Exact-pair replay](results/two_item_pearson_check.csv)

| Track | Changed selections among 64 draws | Original difference [band] | Corrected difference [band] |
|---|---:|---:|---:|
| KonIQ image features | 12 | +0.01491 [+0.01170, +0.01812] | −0.00909 [−0.01346, −0.00472] |
| SummEval consistency, four metric features | 4 | +0.21233 [+0.17883, +0.24582] | +0.22766 [+0.19307, +0.26226] |
| SummEval relevance, post-hoc smoothed GPT-4 votes | 1 | +0.04830 [+0.02921, +0.06739] | +0.04669 [+0.02765, +0.06573] |

The fixed native GPT-4 consistency, native relevance, and post-hoc smoothed consistency tracks had zero changed selections. The KonIQ sign reversal is the sole change in a band’s directional conclusion. Simultaneous intervals at other budgets in the affected depth can vary because the critical value uses the entire grid. Primary one-grade selections, scores and conclusions were identical before and after the correction. [Corrected KonIQ](results/image_quality/contrasts.csv) · [Corrected features](results/summary_metrics/contrasts.csv) · [Corrected smoothed relevance](results/jeffreys_relevance/contrasts.csv)

## Numerical check for score-bin boundaries

Discrete GPT-4 score distributions can put many items exactly at a quantile cut. For an exact replay of calibrated Pearson we used the frozen floating-point candidate means to decide bin membership after comparing every candidate probability with an independently implemented grid. Repeating bin assignment with the algebraically equivalent direct-power implementation changed the calibrated judge’s average held-out RPS by at most 0.000032 at one grade per item and 0.000043 at three. The largest difference for any single training draw was 0.00290. The tested sustained thresholds stayed the same. [Sensitivity columns](results/primary_verification.csv)

To replay from this experiment directory with the prepared public inputs, run `python3 verify_primary.py /tmp/corr-ce-public23`. It checks the canonical CSVs and writes two small verification tables. [SummEval data preparation and source hashes](summeval/DATA_11-21-2026.md) · [KonIQ data preparation](koniq/DATA_11-21-2026.md)
