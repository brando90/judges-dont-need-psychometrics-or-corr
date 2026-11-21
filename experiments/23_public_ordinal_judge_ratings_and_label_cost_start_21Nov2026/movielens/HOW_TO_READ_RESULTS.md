# Reading and reproducing the MovieLens results | 11-21-2026

As of 11-21-2026. The exploratory MovieLens study evaluates movie-level rating distributions from static genre and release-year metadata. Its [protocol](PROTOCOL_11-21-2026.md) preceded the fits. See the [results](RESULTS_11-21-2026.md) for the threshold, intervals, coverage and conclusions.

## Reproduction

From this directory, obtain the original research archive from the publisher and run the scripts with Python, NumPy and SciPy. Both scripts read the ZIP directly. The adapter checks SHA-256 `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`. Runtime is a few minutes or less on a modern laptop. The repository contains aggregate scores and selection frequencies. A new local directory holds the downloaded archive and replay outputs. [GroupLens data use terms](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt) · [Dataset citation](https://doi.org/10.1145/2827872)

```sh
mkdir -p /tmp/movielens-replay
curl --fail --location https://files.grouplens.org/datasets/movielens/ml-100k.zip --output /tmp/movielens-replay/ml-100k.zip
python3 run.py /tmp/movielens-replay/ml-100k.zip /tmp/movielens-replay/results
python3 verify.py /tmp/movielens-replay/ml-100k.zip /tmp/movielens-replay/results
diff -r results /tmp/movielens-replay/results
```

## Interpreting the files

- Each panel has a `coverage.json` with the movie and rating denominators, overlap, missing outcomes, and any undefined Pearson draws. The primary panel includes all participating users, the sensitivity partitions them before fitting. The model uses no user attributes.
- In `scores.csv`, budget counts distinct training movies with one observed grade each. The test set is identical across all methods in each panel. All nine forecasts report normalized individual-rating ranked probability score (RPS, 0–1), individual-rating cross-entropy (CE, nats, nonnegative), and total variation distance (TVD, 0–1) to each movie's empirical histogram. Lower is better on every metric. For each training draw, average votes within movie and weight all held-out movies equally. Next average across 64 draws. Uniform predicts .2 for each grade. The training histogram smooths with a half-count per grade.
- In `contrasts.csv`, the reported test RPS difference subtracts direct CE selection from the named reference. Positive favors direct CE. Simultaneous 95% intervals resample 4,000 sets of held-out movies and use the maximum studentized difference across eight budgets and four comparators. The primary decision requires the lower endpoints versus direct Pearson and uniform to be positive at a given budget and every larger budget. Pointwise intervals for the disjoint-user sensitivity come from the same bootstrap draws. Full-400 identification bounds allow every uncovered movie to have any signed difference in [−1,1]. These bounds express missingness rather than sampling variability.
- In `matched.csv`, each reported difference subtracts the named calibrated alternative from calibrated Pearson on the same held-out metric. Positive favors the alternative. Each score predictor undergoes identical quartile-bin calibration using only training outcomes, plus eight smoothed-prior pseudo-observations per bin. The separate leave-one-out CE procedure selects a score by deleting each training vote from its bin and global prior, then refitting the calibrator to the full training sample. Pointwise intervals use 10,000 held-out movie resamples, conditional on the training draws. `head_counts.csv` aggregates the number of times each discrete forecast was selected. There are 64 selections per objective and budget.
- The two direct fitting rules minimize training log loss or maximize training Pearson in the same 529-prespecified-candidate library (22 single scores × 24 ordered-distribution parameter settings plus uniform). The two principal-component scores and score-bin boundaries use unlabeled training features only. Ties within 10⁻¹⁰ go to the first candidate in the fixed order. The default score uses release year. Exhaustive search guarantees the selected training optimum to this tolerance within this library. Wide variation in outcome probabilities across the same expected-score ordering can affect direct-Pearson performance. The calibration comparison retains the selected scalar score and refits its grade probabilities. [Fixed candidate family](../candidates.py) · [Vote sampling](../design.py) · [Exact calibration selector](../matched_calibration/run.py)
