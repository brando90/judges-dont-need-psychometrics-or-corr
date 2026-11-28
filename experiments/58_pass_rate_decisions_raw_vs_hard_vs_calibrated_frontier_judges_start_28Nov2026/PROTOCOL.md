# Experiment 58 PROTOCOL: pass-rate gates where level and scale matter

Frozen 11-28-2026 (Pacific), in a commit that precedes every analysis run of this experiment. CPU only, no model calls, API spend $0.

## Purpose
Correlation ranks; it is blind to shifts and rescalings of a score. Raw judge scores are also used where level matters: a share of
outputs flagged, a threshold pass rate, a launch gate. This experiment measures, on the five frozen frontier judges, how far the event
rate implied by the judge is from the human rate, with and without calibration, and how often a gate "rate <= tau" gives a different
verdict from the humans. The frozen sets contain no separate systems, so system win rates are NOT available and are not simulated.

## What the analyst had seen before freezing
Aggregate results of Experiments 46, 48, 55 and 56 on these judges and sets (skills, reversals, calibration budgets), the human label
shares of the five sets (for example DICES-350 Yes 32.7 percent of ratings, No 61.1 percent), and the always-No agreement of 269 of
350 on DICES-350. NOT seen: any judge's implied event rate, its error against the human rate, or any gate decision.

## Data (read-only)
Same frozen bundle, loader, floors and invalid-forecast rule as Experiment 57 (see its PROTOCOL.md). 25 judge-by-set cells.
Labels: NLI 0 contradiction, 1 neutral, 2 entailment; alphaNLI 0, 1; DICES 0 No (safe), 1 Unsure, 2 Yes (unsafe).

## Events (fixed here, before any computation)
- DICES-350 and DICES-990: primary E1 = {Yes}, the share of ratings that call the conversation unsafe; secondary E2 = {Unsure, Yes}.
- ChaosNLI SNLI and MNLI-m: E = {entailment}. alphaNLI: E = {label 0}.
Gate: "the event rate is at most tau" (a launch gate; a "at least" gate is the mirror image and is not analysed separately).

## Estimators of the event rate (per judge and set)
Human rate pi_H = mean over items of the fraction of that item's human ratings in E (each item counts once).
(a) raw soft: mean over items of the judge's forecast mass on E. (b) hard label: share of items whose judge argmax label is in E (what a
dashboard that counts verdicts reports; ties in the argmax share the item equally). (c) calibrated soft: the same as (a) after the
cross-fitted per-judge calibration of Experiment 46 (five group folds, seed `SeedSequence([20261128, 58, 0, set_index])`).
Error = estimator minus pi_H in percentage points.

## Gate grid and flips
For each set and event, tau takes five values: pi_H rounded to the nearest 5 points, and that value plus -10, -5, +5 and +10 points.
For an estimator e a gate "flips" when e and the humans fall on opposite sides of tau. A **significant flip** means: in at least 97.5
percent of the group-clustered bootstrap draws, (e <= tau) differs from (pi_H <= tau). Counts are reported per estimator over the 125
gates of the primary events, and over the 50 extra gates of the E2 events. Flips are descriptive evidence; no hypothesis is tested on them.

## Resampling and multiplicity
Cluster = group (SNLI items that share an image move together), B = 4,000 draws, percentile 95 percent intervals, one shared weight
matrix per set (paired across judges and estimators), seed `SeedSequence([20261128, 58, 1, set_index])`. Hypotheses below use point
estimates across cells and a sign test; cells within a set share items and are not independent, which is stated wherever a sign test
is quoted.

## Hypotheses (three; each with a falsification rule)
- **H1.** The raw soft rate misses the human rate by more than 5 percentage points: for at least 10 of the 25 primary-event cells the
  95 percent bootstrap interval of |error| lies entirely above 5 points. Falsified otherwise.
- **H2.** Calibration helps the level: the calibrated soft rate has a smaller absolute error than the raw soft rate in at least 18 of
  the 25 primary-event cells (an 18-of-25 split has one-sided sign-test p = 0.022 if cells were independent). Falsified otherwise.
- **H3.** Counting verdicts is worse than calibrated soft rates: the hard-label rate has a larger absolute error than the calibrated
  soft rate in at least 18 of the 25 primary-event cells. Falsified otherwise.

## Meaning of decision words
"Flip", "significant flip", "supported" and "falsified" are the rules above. No claim that an estimator "beats" another is made.
The single most striking cell is named with its interval, including if it is unfavorable to the thesis. If no gate flips, that is reported.

## Planned outputs
`results/rates.csv` (per cell and estimator: rate, error, interval), `results/flips.csv`, `results/decisions_summary.md`,
`results/hypotheses.json`, one figure (`results/rates_figure.png`, values in `results/rates_figure_values.csv`; grid on every axis),
`HOW_TO_READ_RESULTS.md`, `README.md`, `tests/test_exp58.py` (rate definitions on a hand-computed example, a synthetic known-rate
code check labeled as such, determinism).
