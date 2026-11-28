# judge_scores

Score an LLM judge against a human rating distribution. The scores are the ones this repository's experiments use (Experiments 44, 46,
55, 56): item-level strictly proper scores, skill over the human base rate, a split of the proper risk into calibration error and
resolution, and item-clustered bootstrap intervals. It is small on purpose: numpy and scipy only, no model calls.

```python
import numpy as np
from judge_scores import agreement_vs_skill_report, ratings_to_counts

# q: [items, K] forecast probabilities of the judge; ratings: per item, the list of human ratings (integers 0..K-1)
counts = ratings_to_counts(ratings, K=3)
rep = agreement_vs_skill_report(q, counts, ordered=True, calibrate=True)   # ordered=False for unordered labels
print(rep["top1_agreement"], rep["always_majority_agreement"], rep["pearson"])    # what is usually reported
print(rep["skill"])                       # matched proper score, skill over the human base rate, 95% item-clustered interval
print(rep["calibration_split"])           # calibration error and resolution (cross-fitted)
```

Command line: `judge-scores report FORECASTS.csv RATINGS.csv --labels 3 --ordered --calibrate` (forecast file `item_id,p0,...`; rating
file `item_id,rating`, one row per rating).

## Definitions

| name | definition (loss; lower is better) | use for |
|---|---|---|
| `brier` | sum over labels of (q_k - 1{y=k})^2, range 0 to 2 | unordered labels, binary labels |
| `ce` | -log q_y after flooring q at 1e-6 | unordered or ordered labels (ignores order) |
| `nrps` | normalized ranked probability score: mean over the K-1 thresholds of (Q_m - 1{y<=m})^2, range 0 to 1 | ordered labels with K >= 3 |

An item's score is its expected score over its own ratings, so the mean over items is the unbiased per-sample proper risk, and
one rating per item is enough for an unbiased risk (the skill ratio is consistent, not unbiased).
`skill = 1 - risk(judge) / risk(base-rate forecast)`; the base rate is the human rating histogram of the other cross-fitting folds.
0 means no better than the human base rate. Match the rule to the label encoding: `matched_rule(K, ordered)`. The package refuses the
ordinal RPS on unordered labels (the ordinal score depends on a label order that does not exist there).
`calibration_split` reports `calibration_error = risk_raw - risk_cal` and `resolution = unc - risk_cal`, so that
`risk_raw = unc - resolution + calibration_error`; the calibration map is the one of Experiment 46 (multinomial logit on [log q, 1],
ridge 1e-3 toward the identity). Report these beside correlation, not instead of it.

## What it does not do

- It calls no model and fits nothing except that one calibration map.
- It does not make significance claims beyond the item-clustered percentile bootstrap it computes (default 4,000 draws).
- Correlation, agreement and kappa are reported for comparison; the package does not decide that any of them is wrong.
- The tests check the code against the definitions, strict propriety on random laws, the decomposition identity on finite laws, exact
  one-rating unbiasedness by enumeration, and 75 published rows of Experiment 55. They do not validate the package for use outside
  this repository's data.

Tests: `cd src/judge_scores && python -m pytest -q` (22 tests; two skip with a printed reason if the Experiment 39 bundle or the
Experiment 46/55 files are absent). Install: `pip install -e src/judge_scores`, or put `src/judge_scores` on `PYTHONPATH`.
