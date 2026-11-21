# Reading and reproducing the public human-rating results | 11-21-2026

As of 11-21-2026. The primary unit of annotation cost is a single observed human grade on one criterion. Each test item is weighted equally. Each budget fits judges to 64 matched label acquisitions using the same unique items and votes for all objectives. The result tests an acquisition procedure on new items within a public archive. All primary data splits, candidate search spaces, metrics and stopping criteria are in the [prespecified protocol](PROTOCOL_11-21-2026.md).

## Where to read each result

- [Human-readable results](RESULTS_11-21-2026.md) and [all primary budgets with simultaneous bands](BUDGET_TABLES_11-21-2026.md) give both the first successful budget and budgets where CE lost. A practical win requires CE to improve on raw Pearson, uniform, training grade frequencies, and a Pearson score calibrated from the same labels.
- Each subfolder of [results](results) has a scores table with the absolute held-out, lower-is-better losses. Ranked probability score (RPS) is the average squared error on the four cumulative probabilities (0–1). Cross-entropy (CE) is negative log probability assigned to a human grade (nats). Total variation distance (TVD) is half the L1 distance from the item's empirical histogram (0–1). The multiclass Brier score sums the squared error across five class probabilities (0–2).
- The two primary directories are [GPT-4 factual consistency](results/gpt4_consistency) and [technical image quality](results/image_quality). SummEval has 640 test summaries from 40 articles, 16 systems repeated across splits, three expert ratings each. KonIQ has 2,015 held-out images with 215,415 votes. The image judge uses seven measured image properties. Neither test establishes performance for new raters. For secondary analyses see [GPT-4 relevance](results/gpt4_relevance) and [summary metric features](results/summary_metrics). The four original judge libraries contain 28, 217, 28 and 145 candidate distributions respectively, counting uniform. Only the chosen output distribution is fitted in this comparison.
- Each contrasts table subtracts test RPS of the CE fit from the specified comparator’s test RPS on the same cases. Positive favors CE. Intervals use 4,000 paired test-group resamples, with 97.5% simultaneous bands within each primary benchmark over all studied budgets and four comparisons. This gives approximately 95% joint coverage across both primary benchmarks for the primary depth. The second rating depth and the other tracks are supplementary. The p-value column is a within-benchmark max-statistic adjusted estimate. The crossed-sample analysis applies independent multinomial bootstrap weights to the held-out groups and the 64 matched acquisition draws, then recomputes the paired mean and the simultaneous bands.
- The diagnostics tables give the number of training draws with a defined test correlation. Pearson, Spearman, Kendall and Lin concordance range from −1 to 1 (higher agreement is better); constant predictions can have undefined correlations. Mean bias and mean absolute error are in 1–5 grade points. Expected calibration error averages absolute cumulative probability discrepancies across fixed bins. Modal and per-vote exact agreement are proportions and assign uniform credit among tied predicted modes. System-level outcomes on SummEval aggregate the same 16 systems before measuring rank agreement or absolute mean error. The human reference forecasts each held-out vote from other votes on that same test item with half-count smoothing.
- Exploratory results have separate directories for [model-frequency smoothing on consistency](results/jeffreys_consistency), [smoothing on relevance](results/jeffreys_relevance), and [identical score calibration](matched_calibration/RESULTS_11-21-2026.md). The [prespecified continuous model and longer-optimizer sensitivity](continuous/RESULTS_11-21-2026.md) have separate draws, validation choices and uncertainty. Their numbers describe the recorded optimizer, two starts and budget.

## Reproduction

Run from this experiment directory in a Python environment with NumPy, SciPy and Matplotlib. The first two commands download the authors' releases. Data archives and raw documents stay outside Git. Existing output folders can be reproduced in a separate copy of this experiment.

```sh
cache=/tmp/corr-ce-public-ordinal
inputs=/tmp/corr-ce-public23
mkdir -p "$inputs"
python3 summeval/prepare.py --cache "$cache/summeval" --output "$inputs"
python3 koniq/prepare.py "$cache/koniq" "$inputs/koniq.npz"
python3 diagnostics.py
python3 analyze.py "$inputs/summeval_consistency.npz" native results/gpt4_consistency
python3 analyze.py "$inputs/koniq.npz" features results/image_quality
python3 analyze.py "$inputs/summeval_relevance.npz" native results/gpt4_relevance
python3 analyze.py "$inputs/summeval_consistency.npz" features results/summary_metrics
python3 sensitivity.py smooth "$inputs/summeval_consistency.npz" native results/jeffreys_consistency
python3 sensitivity.py smooth "$inputs/summeval_relevance.npz" native results/jeffreys_relevance
python3 sensitivity.py crossed "$inputs/summeval_consistency.npz" native results/gpt4_consistency
python3 sensitivity.py crossed "$inputs/koniq.npz" features results/image_quality
python3 verify_primary.py "$inputs"
python3 report.py
```

[Source-specific provenance](summeval/DATA_11-21-2026.md) · [Independent replay and two-item numerical tie correction](VERIFICATION_11-21-2026.md) · [Continuous-model reproduction](continuous/RESULTS_11-21-2026.md) · [Matched-calibration reproduction](matched_calibration/RESULTS_11-21-2026.md)
