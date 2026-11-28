# Experiment 58: pass-rate gates where level and scale matter

> **Note 11-28-2026 (QA and Experiment 61; results unchanged).** The hard-label rate is compared with the rater-share human rate, which a hard-label judge that matched the human majority on every item would itself miss: by -9.8 points on DICES-350 and -9.5 on DICES-990 (Experiment 61). Read the hard-label errors (-20.4 to -25.2 points on DICES-350) with that gap in mind; against the human majority rate they are -15.4 to -10.6 (DICES-350) and -6.8 to -1.8 (DICES-990). The soft-rate comparison does not have this issue.


Status: run 11-28-2026 on the frozen bundle, CPU only, no model calls, API spend $0. Pre-registered: [`PROTOCOL.md`](PROTOCOL.md) was committed in `ae58f5c` before the first run
of `scripts/run_exp58.py`; the one addition after that (a post hoc comparator) is labeled below and does not touch any pre-registered quantity. Reading guide: [`HOW_TO_READ_RESULTS.md`](HOW_TO_READ_RESULTS.md).
Run it again with `python experiments/58_*/scripts/run_exp58.py` (two seconds; reruns are byte-identical) and the tests with `python -m pytest experiments/58_*/tests -q` (7 pass). Data prerequisite: the same frozen bundle as Experiment 57 (not tracked by git; rebuild steps and checksum in Experiment 57's README).

## Question
Correlation ranks and ignores level and scale, but judge scores are also used where level matters: the share of outputs flagged, a pass rate against a threshold. On the same five frozen frontier judges and five rating sets as Experiment 57, how far is the event rate
that a judge implies from the human rate, with and without calibration, and how often does a gate "rate at most tau" give a different verdict from the humans? The frozen sets contain no separate systems, so system win rates are not available and are not simulated.
Events (fixed in the protocol): "Yes (unsafe)" on the DICES sets (secondary: "Unsure or Yes"), entailment on SNLI and MNLI-m, label 0 on alphaNLI.

## What was found (answer first)
1. **Raw verdict rates are far from the human rate on four of the five sets.** The raw soft rate (mean forecast mass on the event) is off by more than 5 points with a 95% interval entirely above 5 in 13 of the 25 cells (H1 supported, threshold 10): all five DICES-350 cells, four DICES-990 cells and four MNLI cells.
   On DICES-350 the human "unsafe" rate is 32.7% and the judges' mean forecast mass on unsafe is 13.8% (Astra), 24.0% (Opus), 12.0% (Kimi), 13.6% (GLM) and 8.6% (DeepSeek); the largest error is DeepSeek at -24.1 points (95% interval -25.9 to -22.2). Counting hard verdicts is no better: -20.4 to -25.2 points on DICES-350.
   The direction is the dangerous one for a launch gate: the judges under-call unsafe, so 30 of the 39 flipped primary gates are false passes. On MNLI every judge under-calls entailment (-6.4 to -12.4 points); on SNLI (entailment) and alphaNLI (label 0) they over-call the event (+1.3 to +5.7).
2. **Gate flips (descriptive, no test).** Over the 125 primary gates (five thresholds per cell: the human rate rounded to the nearest 5 points, and plus or minus 5 and 10 points), the raw soft rate gives a different verdict from the humans on 39 (20 in at least 97.5% of bootstrap draws; 30 false passes, 9 false fails)
   and the hard-label rate on 38 (22 significant; 31 false passes, 7 false fails). The calibrated soft rate flips none. On the 50 secondary gates the raw soft rate flips 17 and the hard-label rate 20, all false passes. These counts depend on the offsets chosen for the grid and are an illustration of exposure, not the probability of a real decision going wrong.
3. **H2 and H3 are supported, and they are mechanical.** The cross-fitted calibrated soft rate has a smaller absolute error than the raw soft rate in 25 of 25 cells, and the hard-label rate has a larger one than the calibrated rate in 25 of 25 (thresholds 18). Calibration never made an error larger (the least improved cell, SNLI with Kimi, goes from 1.26 to 0.03 points).
   But the calibration map has an unpenalized intercept per label, so by construction it reproduces the human label shares of the training folds, and the calibrated rate is therefore within about 0.1 points of the human rate in every cell. A test pins this down (`test_calibration_matches_the_training_label_shares_by_construction`).
   **Post hoc, not pre-registered:** the cross-fitted human base rate, which uses no judge at all, has an error of -0.02 to +0.00 points on the primary events (`results/posthoc_base_rate_rates.csv`). So H2 and H3 say the level is easy to restore with the labels of the other folds; they do not say the calibrated judge discriminates. Discrimination is what resolution measures in Experiment 57.
4. What this means for the paper's thesis: a judge can be reported with a respectable correlation (Experiment 57) while the share of outputs it flags is wrong by tens of points, and a label-based recalibration repairs the level. The honest claim is about LEVEL AND SCALE, which correlation cannot see; it is not that raw judges are uninformative.

## Method in one paragraph
Human rate = mean over items of the fraction of the item's ratings in the event (each item counts once). Estimators per judge and set: (a) raw soft, mean forecast mass on the event; (b) hard label, share of items whose argmax label is in the event (argmax ties, for example the uniform forecast used for an invalid output, share the item equally);
(c) calibrated soft, the same as (a) after the cross-fitted per-judge calibration of Experiment 46 (five group folds, seed `SeedSequence([20261128, 58, 0, set_index])`). Group-clustered bootstrap, 4,000 draws, one shared weight matrix per set, seed `SeedSequence([20261128, 58, 1, set_index])`;
intervals do not include the variability of the fitted calibration map. A gate flips when the estimator and the humans fall on opposite sides of "rate at most tau"; it is significant when that happens in at least 97.5% of the draws.

## Scope and limits
Five judges at temperature 0, five sets with the selection biases stated in Experiment 28 (ChaosNLI SNLI and MNLI-m contain only examples whose original five labels were not unanimous). Rates are item-averaged. The events are fixed in the protocol; other events (for example "No") are the complements and carry no extra information.
The hard-label rates are a convenience for the comparison and would differ if the judges were asked for verdicts instead of probabilities. No Lean claim is made in this experiment (it is empirical); the one mechanical property above is pinned by a test, not a theorem.

## Files
`PROTOCOL.md`, `scripts/run_exp58.py`, `tests/test_exp58.py`, `results/human_rates.csv`, `results/rates.csv` (every cell, event and estimator with intervals), `results/flips.csv`, `results/hypotheses.json`, `results/decisions_summary.md`,
`results/rates_figure.png` with `results/rates_figure_values.csv`, `results/posthoc_base_rate_rates.csv` (post hoc), `HOW_TO_READ_RESULTS.md`.
