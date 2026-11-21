# Experiment 16 results: RPS-selected arm and probability-floor sweep

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/16_rps_selected_arm_and_probability_floor_on_scalar_pools_start_20Nov2026/results.md>

**TLDR:** On the scalar-vote pools, picking one judge by train RPS does not fix single-judge selection: it is no better than picking by CE (clearly worse on the open-weight pool) and no better than picking by correlation, and no single-judge picker clearly beats the constant at the original floor (correlation 0.1797 against 0.1836, within noise; CE and RPS picks are worse). Mixing judges by train RPS is the best fitted arm on every pool, and beats the constant by 0.05 to 0.06 RPS on the five cached judges; on the open-weight pool no arm, mixture included, beats the constant (0.2601 against 0.1836). The floor matters a lot for CE (8.9 down to 2.2 at 5e-2) but not enough to put CE-picked single judges below the constant. Retrospective, same 75 artifacts and 195 expert ratings, 1,000 task-grouped 20/5 splits, CPU only, $0.

Run: 11-20-2026 22:14 to 22:16 on skampere1 (12 cells, 21 s to 100 s in total), plain Python, no language model. Files: `results/<pool>_eps<floor>/results.json`. Smaller RPS is better. Pools: `old5` = 5 cached frontier scalar judges, `open` = 39 open-weight scalar judges, `all` = 44.

## Test RPS, floor 1e-6 (the floor of experiments 12 and 14)

| Pool | Constant | Equal mixture | corr single | corr mixture | CE single | CE mixture | RPS single | RPS mixture |
|---|---|---|---|---|---|---|---|---|
| old5 | 0.1836 | 0.1496 | 0.1797 | 0.1419 | 0.2051 | 0.1362 | 0.2068 | **0.1313** |
| open | 0.1836 | 0.3107 | 0.3680 | 0.3274 | 0.3151 | 0.2773 | 0.3402 | **0.2601** |
| all | 0.1836 | 0.2733 | 0.1797 | 0.2063 | 0.2114 | 0.1658 | 0.2068 | **0.1289** |

Secondary columns (mean over splits of the test-set correlation of the picked judge or mixture with the human mean; Spearman / Pearson), floor-free:

| Pool | corr single | corr mixture | CE single | RPS single | RPS mixture |
|---|---|---|---|---|---|
| old5 | 0.719 / 0.750 | 0.669 / 0.730 | 0.609 / 0.644 | 0.575 / 0.603 | 0.648 / 0.726 |
| open | 0.209 / 0.190 | 0.220 / 0.196 | 0.112 / 0.024 | 0.039 / -0.040 | 0.029 / 0.022 |
| all | 0.719 / 0.750 | 0.600 / 0.654 | 0.595 / 0.627 | 0.575 / 0.603 | 0.650 / 0.725 |

## Primary endpoints (paired difference, test RPS, floor 1e-6; 95% task-bootstrap interval)

| Pool | corr single minus RPS single | corr mixture minus RPS mixture |
|---|---|---|
| old5 | -0.0271 [-0.068, +0.001] | +0.0105 [-0.001, +0.023] |
| open | +0.0278 [-0.007, +0.063] | +0.0673 [+0.025, +0.109] |
| all | -0.0271 [-0.068, +0.001] | +0.0775 [+0.042, +0.112] |

Negative on the left means the correlation-picked judge has the lower loss; positive on the right means the RPS-mixture has the lower loss.

## Floor sweep (test RPS, `old5`; the same pattern holds on `all`)

| Floor | Constant | corr single | CE single | RPS single | RPS mixture |
|---|---|---|---|---|---|
| 1e-6 | 0.1836 | 0.1797 | 0.2051 | 0.2068 | 0.1313 |
| 1e-3 | 0.1836 | 0.1785 | 0.2034 | 0.2055 | 0.1309 |
| 1e-2 | 0.1836 | 0.1690 | 0.1924 | 0.1928 | 0.1273 |
| 5e-2 | 0.1835 | 0.1434 | 0.1624 | 0.1583 | 0.1199 |

Test CE of the CE-picked single judge falls from 8.93 (floor 1e-6) to 2.18 (floor 5e-2) on `old5`, against the constant's 1.85: still above it (+0.325, interval [-0.026, +0.650]); on `open` it stays above (+0.627, [0.289, 0.919]). At 5e-2 every arm improves on RPS because the floor smooths the three-sample histograms, so the floor helps all selectors, not only CE.

## Pre-registered predictions, scored

1. `rps_single` no better than `ce_single`, interval includes 0, on every pool: **holds on `old5` and `all`, fails on `open`** (CE single minus RPS single is -0.0251 [-0.047, -0.002], so RPS-picking is clearly worse there).
2. Floor 5e-2 puts CE-picked single CE below the constant on every pool: **fails** (still above the constant on all three pools; the interval excludes 0 on `open`). The second half (neither single beats the constant on RPS on `all` and `open`) holds.
3. `corr_single` stays better than `rps_single` on `all`: **direction holds** (-0.0271), interval includes 0 by 0.001.
4. `rps_mixture` better than `corr_mixture_pearson` on every pool, interval excluding 0 on `open` and `all`: **holds**.

## Reading

- Sanmi's floor explanation is only partly right for these judges: the floor does most of the work on CE, and raising it helps every selector on RPS, but the single-judge selection loss at the original floor is the same for RPS (which has no floor problem in its objective) as for CE. With 25 tasks and 20 training tasks, picking one of five judges is noisy under any objective.
- Mixing is the robust part: the RPS mixture beats the constant on `old5` at every floor by 0.052 to 0.064 (all intervals exclude 0), and is the best fitted arm on every pool (on `open` the constant, 0.1836, still beats it at 0.2601). This is the retrospective version of "fit a calibrated, regularized head, not a flexible selector".
- The 39 open-weight scalar judges are worse than the constant for every selector at every floor tried (even the best, the RPS mixture at floor 5e-2, is 0.2256 against 0.1835), so open-weight scalar votes carry little human-rating signal here. Experiment 09 (native probabilities, open weights) is the better test of open-weight judges.
- RPS-mixture optimizer certificate failures (Frank-Wolfe gap above 1e-8): 4 of 1,000 splits on `old5`, 52 on `open`, 78 on `all` at floor 1e-6, kept in all numbers above as pre-registered.
- Retrospective: the ratings were seen by the people who built the judges. A prospective test needs fresh ratings.

**TLDR-end:** [judge objectives: experiment 16] RPS-picked single judges do not beat correlation-picked or the constant; RPS-picked mixtures are the best fitted arm on every pool (the constant wins on the open-weight pool); floor helps all selectors but does not rescue single-judge CE.

*Correction 11-20-2026 (after a cross-check of this file): the first version said the RPS mixture was "the best arm on every pool" and that RPS-picking was "as bad as" CE or correlation; on the open-weight pool the constant beats every arm and RPS-picking is clearly worse than CE-picking (see scored prediction 1). The text above is corrected; no number changed.*
