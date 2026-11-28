# Experiment 47 protocol: how many held-out labels does post-hoc calibration need to close the CE-vs-Pearson gap? (frozen 11-27-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/47_calibration_label_budget_sweep_exp41_heads_start_27Nov2026/PROTOCOL.md>

**TLDR:** Experiment 41 found that a validation-fitted calibration map closes the Pearson-trained head's raw gap to at most 0.0024 normalized RPS, but its map used every validation item with all of its ratings. Here the map is fitted on n calibration labels, n in {25, 50, 100, 200, 500, 1000, all}, and we measure the residual gap and its spread. Frozen before the script exists.

**As of:** 11-27-2026 (Friday, Pacific time). CPU, $0.

## Design (reuses Experiment 41 unchanged)

- Data, views (16), group split (0.6/0.2/0.2, seeds 0-4), heads (CE-, Pearson- and MSE-trained linear heads on 24 train-only principal components, L2 0.01, at most 35 L-BFGS iterations = the preregistered Exp 41 run) and the calibration map family (`e40_common.calibrate`: standardize the scalar mean score with the calibration items' moments, multinomial logit on [z, 1] fitted by CE) are imported from Experiments 36/41.
- **Calibration labels.** Drawn from the validation split only. Design R1 (primary): n distinct validation items, one rating each drawn uniformly from the item's ratings (n labels = n ratings). Design RA (secondary): n validation items with all their ratings (Exp 41's design when n = all). n is capped at the number of validation items; "all" = every validation item. 20 calibration draws per (view, split seed, n), draw seed SeedSequence([20261127, 47, seed, n, draw]). Heads are trained once per split seed and do not change with n.
- **Test.** The full test split, every rating; normalized RPS (primary), CE (secondary).

## Quantities (all test, lower is better for scores)

- Residual matched gap M(n) = nRPS(Pearson-trained head + map fitted on n labels) - nRPS(CE-trained head + map fitted on the same n labels). Positive = the CE-trained head is still better.
- Closure gap G(n) = nRPS(Pearson-trained head + map on n labels) - nRPS(CE-trained head, raw, no map). Positive = the correlation-trained judge plus n calibration labels is still worse than the proper-score-trained judge used as is.
- Excess over constant E(n) = nRPS(Pearson-trained head + map on n labels) - nRPS(train-marginal constant forecast).
- Reported per view and n: mean over 5 seeds x 20 draws, SD over the 100 (seed, draw) cells, and, per split seed, a paired item-clustered bootstrap 95% interval (2,000 draws, seed 1) of the draw-averaged difference.

## Predictions and decision rule

- **Q1.** On views with signal (Exp 41's "carries signal" flag), G(n) exceeds 0.0024 (Exp 41's largest matched gap) at n = 25 on most views and falls to at most 0.0024 by n = all.
- **Q2.** The across-draw SD of G(n) falls with n roughly as n^(-1/2).
- **Decision rule.** n\*(view) = the smallest n with seed-mean G(n) <= 0.0024 under design R1. Reported for every view; "not reached" is a result.
- Failures and views where calibration never closes the gap are reported.
