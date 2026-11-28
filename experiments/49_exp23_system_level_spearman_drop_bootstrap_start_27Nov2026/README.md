# Experiment 49: Experiment 23's system-level Spearman drop (0.877 raw vs 0.835 calibrated) is not significant, and it comes from the four-bin calibrator

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/49_exp23_system_level_spearman_drop_bootstrap_start_27Nov2026/README.md>

**TLDR:** The two numbers reproduce exactly from Experiment 23's code and inputs (0.8765 and 0.8349). Their difference, +0.042, is stable across training draws but not across test data: the 95% interval is [-0.049, +0.114] when the 40 test documents are resampled (p = 0.52) and [-0.095, +0.241] when the 16 systems are resampled (p = 0.63). By the preregistered rule the drop is **not significant**. The cause is the calibrator, not calibration: Experiment 23's calibrated forecast is a four-bin step function, so every test item gets one of only 4 expected grades (the raw judge gives 308 distinct values). A continuous calibrator fitted on the same 768 training votes keeps the system ranking (0.8754, difference +0.001) and has a better proper score than either (test normalized RPS 0.0492, against 0.0543 for the four-bin calibrator and 0.1436 for the raw judge). This removes the reviewer reading "calibration hurts system ranking".

**As of:** 11-27-2026 (Friday, Pacific time). CPU, \$0. Protocol frozen first: [PROTOCOL.md](PROTOCOL.md), commit 83b9782.

## Data and setup

SummEval consistency, G-Eval GPT-4 (20 archived sampled completions per summary) as the judge, three expert grades per summary. Experiment 23's fixed split: 960 training summaries (60 source documents), 640 test summaries (40 documents x 16 systems). Depth 1, budget 768 training votes, 64 training draws, exactly as in Experiment 23's `analyze.py`, whose modules are imported unchanged. "pearson" = the 27-transform head selected by Pearson on the training votes, used raw. "calibrated_pearson" = that head followed by Experiment 23's distribution calibration (four quantile bins of the selected expected grade, grade frequencies per bin). "continuous" (new, diagnostic) = the same selected head followed by Experiment 41's multinomial logit on its standardized expected grade, fitted by cross-entropy on the same 768 training votes. System-level Spearman: Spearman correlation between the 16 systems' mean predicted expected grade and their mean human grade over the 40 test documents (Experiment 23's `system_scores`).

## Results

| quantity | raw Pearson-selected | four-bin calibrated | continuous calibrated |
|---|---|---|---|
| system-level Spearman, mean over 64 draws (higher is better) | **0.8765** | **0.8349** | 0.8754 |
| 95% interval, documents resampled | [0.721, 0.937] | [0.685, 0.915] | [0.631, 0.931] |
| 95% interval, systems resampled | [0.605, 0.977] | [0.483, 0.986] | [0.574, 0.979] |
| distinct expected grades per test item (median over draws) | 308 | 4 | 308 |
| system pair accuracy | 0.867 | 0.854 | 0.866 |
| test normalized RPS (lower is better) | 0.1436 | 0.0543 | **0.0492** |
| test CE, nats (lower is better) | 1.933 | 0.421 | **0.396** |

Difference Delta = system Spearman of the raw Pearson-selected head minus that of the calibrated forecast (positive = raw ranks systems better), mean over the 64 draws, with intervals from 2,000 paired resamples (seed 20261127):

| comparison | Delta | training draws, 2.5 to 97.5 percentile | documents resampled, 95% | systems resampled, 95% | both resampled, 95% | significant? |
|---|---|---|---|---|---|---|
| raw minus four-bin | +0.0417 | [+0.028, +0.058] | [-0.049, +0.114], p = 0.52 | [-0.095, +0.241], p = 0.63 | [-0.121, +0.227], p = 0.75 | **no** |
| raw minus continuous | +0.0011 | | [-0.072, +0.164], p = 0.58 | [-0.118, +0.109], p = 0.92 | [-0.156, +0.266], p = 0.76 | no |

Leave-one-system-out jackknife of the raw minus four-bin Delta: range +0.014 to +0.077, standard error 0.071.

## Diagnosis

1. **Few systems, many near-ties.** Eight of the 16 systems have mean human consistency between 4.94 and 5.00 on the 1 to 5 scale, and 4 system pairs are exactly tied in the human means. Their order is set by differences of 0.01 to 0.03 grade points over 40 documents and 3 experts. The bootstrap SD of the system Spearman itself is 0.10 (raw) to 0.14 (four-bin) when systems are resampled, so a 0.04 difference cannot be resolved with 16 systems.
2. **Binning, not calibration.** The four-bin calibrator maps the 640 test items to four expected grades, which discards the within-bin order that system means need. The continuous calibrator keeps the order (Spearman 0.8754) and is better on both proper scores.
3. The draw-to-draw spread is small (all 64 draws favor raw, [+0.028, +0.058]) because the test set is fixed. That interval conditions on the 40 test documents and 16 systems, so it does not support a population claim.

## Does this support or undercut the thesis?

It removes an apparent counterexample. The one data point where correlation-style validation seemed to keep an advantage (system ranking) is within noise, and a calibrator that keeps the score order has both a better proper score and the same system ranking. A real ranking test needs more than 16 systems (for example, many judge configurations or benchmark submissions).

**Proposed paper text (not inserted):** "Post-hoc calibration need not cost system-level ranking. On SummEval (16 systems, 40 held-out documents), Experiment 23's four-bin calibrator lowered the mean system-level Spearman from 0.877 to 0.835, but the difference is not significant (document bootstrap 95% CI [-0.05, 0.11]; system bootstrap [-0.10, 0.24]). It is explained by the step-function calibrator: a continuous map fitted on the same 768 ratings keeps the ranking (0.875) and has the lowest held-out RPS (0.049 vs 0.144 raw)."

## Reproduce

```
python ../23_public_ordinal_judge_ratings_and_label_cost_start_21Nov2026/summeval/prepare.py --cache CACHE --output INPUTS   # SHA-256 bbd81648... checked
python scripts/system_level_drop.py INPUTS/summeval_consistency.npz results     # about 3 minutes on one core
```

Outputs: [results/system_level_drop.json](results/system_level_drop.json) (every number above) and `results/per_draw_system_spearman.npz` (on the box only; binary).
