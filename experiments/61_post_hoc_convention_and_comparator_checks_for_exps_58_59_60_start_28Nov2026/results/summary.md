# Experiment 61 summary (POST HOC, descriptive; protocol `PROTOCOL.md`)

## C1: skill of the truthful and of four collapsed forecasts (Experiment 60's splits; skill = 1 - risk / base-rate risk on half B; higher is better; 0 is the base rate)

### ChaosNLI alphaNLI (rule brier, 400 items, 200 splits, 4000 item-clustered draws)

forecast | skill [95% interval] | truthful minus this forecast, same score, same half B [95% interval] | share of items whose forecast label is the middle label
---|---|---|---
truthful | +0.644 [+0.611, +0.676] |  | 
base | +0.000 [+0.000, +0.000] |  | 
rounded_mean | +0.529 [+0.476, +0.582] | +0.114 [+0.093, +0.137] | 
modal_label | +0.528 [+0.476, +0.581] | +0.115 [+0.094, +0.138] | 
median_label | +0.528 [+0.476, +0.581] | +0.115 [+0.094, +0.138] | 
mean_preserving_two_point | +0.644 [+0.611, +0.676] | +0.000 [-0.000, +0.000] | 

### DICES-350 (rule nrps, 350 items, 200 splits, 4000 item-clustered draws)

forecast | skill [95% interval] | truthful minus this forecast, same score, same half B [95% interval] | share of items whose forecast label is the middle label
---|---|---|---
truthful | +0.167 [+0.142, +0.192] |  | 
base | +0.000 [+0.000, +0.000] |  | 
rounded_mean | -0.475 [-0.535, -0.416] | +0.642 [+0.606, +0.680] | 56%
modal_label | -0.245 [-0.298, -0.190] | +0.411 [+0.378, +0.445] | 0%
median_label | -0.237 [-0.290, -0.185] | +0.404 [+0.373, +0.435] | 
mean_preserving_two_point | -0.152 [-0.197, -0.107] | +0.319 [+0.296, +0.342] | 

### DICES-990 (rule nrps, 400 items, 200 splits, 4000 item-clustered draws)

forecast | skill [95% interval] | truthful minus this forecast, same score, same half B [95% interval] | share of items whose forecast label is the middle label
---|---|---|---
truthful | +0.148 [+0.128, +0.168] |  | 
base | +0.000 [+0.000, +0.000] |  | 
rounded_mean | -0.492 [-0.539, -0.444] | +0.640 [+0.607, +0.672] | 53%
modal_label | -0.275 [-0.318, -0.231] | +0.423 [+0.393, +0.454] | 0%
median_label | -0.273 [-0.316, -0.230] | +0.421 [+0.392, +0.450] | 
mean_preserving_two_point | -0.190 [-0.225, -0.153] | +0.338 [+0.318, +0.358] | 

## C2: hard-label error of a perfect judge (human rater-share rate versus human majority rate, percentage points; majority minus rater-share)

set | event | rater-share rate | majority rate | perfect-judge hard-label error [95% interval] | items with a tied top label
---|---|---|---|---|---
ChaosNLI SNLI | primary (entailment) | 29.4% | 27.8% | -1.6 [-3.2, +0.0] | 1.2%
ChaosNLI MNLI-m | primary (entailment) | 41.2% | 46.4% | +5.2 [+3.0, +7.5] | 1.2%
ChaosNLI alphaNLI | primary (label 0) | 47.8% | 47.4% | -0.4 [-2.1, +1.3] | 0.2%
DICES-350 | primary (Yes (unsafe)) | 32.7% | 22.9% | -9.8 [-12.7, -7.1] | 0.6%
DICES-350 | secondary (Unsure or Yes) | 38.9% | 22.9% | -16.1 [-18.9, -13.3] | 0.6%
DICES-990 | primary (Yes (unsafe)) | 28.2% | 18.8% | -9.5 [-12.1, -6.9] | 1.0%
DICES-990 | secondary (Unsure or Yes) | 33.3% | 18.8% | -14.5 [-17.2, -11.9] | 1.0%

Hard-label judges, primary events, 25 (set, judge) cells: error against the rater-share rate ranges -25.2 to +5.5 points; against the human majority rate -21.9 to +5.9 points; the error against the majority rate is smaller in absolute value than against the rater-share rate in 12 of 25 cells.

- ChaosNLI SNLI: error against rater-share -2.9 to +3.2; against majority -1.3 to +4.8.
- ChaosNLI MNLI-m: error against rater-share -16.7 to -4.1; against majority -21.9 to -9.4.
- ChaosNLI alphaNLI: error against rater-share +1.0 to +5.5; against majority +1.4 to +5.9.
- DICES-350: error against rater-share -25.2 to -20.4; against majority -15.4 to -10.6.
- DICES-990: error against rater-share -16.2 to -11.2; against majority -6.8 to -1.8.

## C3: Experiment 59 selection paths against a uniformly random pick (regret in percentage points of skill; lower is better; path minus random: positive means the path is worse than picking at random)

set | n | regret A (Pearson) | regret B (3-fold CV calibrated proper risk) | regret C (raw proper risk) | random pick | A minus random [95%] | B minus random [95%] | C minus random [95%]
---|---|---|---|---|---|---|---|---
ChaosNLI SNLI | 25 | 6.89 | 10.43 | 7.31 | 8.35 | -1.46 [-2.48, -0.42] | +2.08 [+0.88, +3.30] | -1.04 [-2.06, -0.01]
ChaosNLI SNLI | 50 | 3.69 | 4.59 | 3.60 | 4.27 | -0.59 [-1.13, -0.02] | +0.32 [-0.23, +0.89] | -0.68 [-1.20, -0.10]
ChaosNLI SNLI | 100 | 1.85 | 2.22 | 2.07 | 3.01 | -1.16 [-1.46, -0.83] | -0.80 [-1.13, -0.43] | -0.94 [-1.27, -0.58]
ChaosNLI SNLI | 200 | 0.95 | 1.05 | 1.18 | 2.58 | -1.63 [-1.84, -1.41] | -1.53 [-1.74, -1.32] | -1.39 [-1.62, -1.16]
ChaosNLI MNLI-m | 25 | 7.82 | 10.18 | 7.03 | 7.64 | +0.18 [-0.79, +1.14] | +2.54 [+1.52, +3.57] | -0.61 [-1.37, +0.18]
ChaosNLI MNLI-m | 50 | 3.49 | 5.59 | 2.93 | 4.29 | -0.79 [-1.28, -0.25] | +1.31 [+0.71, +1.92] | -1.36 [-1.82, -0.87]
ChaosNLI MNLI-m | 100 | 1.66 | 2.40 | 1.68 | 2.74 | -1.08 [-1.36, -0.79] | -0.35 [-0.67, -0.02] | -1.06 [-1.32, -0.80]
ChaosNLI MNLI-m | 200 | 1.00 | 1.12 | 0.95 | 2.22 | -1.23 [-1.39, -1.06] | -1.10 [-1.29, -0.92] | -1.28 [-1.44, -1.11]
ChaosNLI alphaNLI | 25 | 4.96 | 6.08 | 4.84 | 6.31 | -1.35 [-2.15, -0.53] | -0.22 [-1.18, +0.77] | -1.46 [-2.29, -0.56]
ChaosNLI alphaNLI | 50 | 2.95 | 3.70 | 2.99 | 4.78 | -1.84 [-2.37, -1.28] | -1.08 [-1.63, -0.51] | -1.79 [-2.28, -1.25]
ChaosNLI alphaNLI | 100 | 1.65 | 1.98 | 1.87 | 4.50 | -2.85 [-3.13, -2.56] | -2.52 [-2.84, -2.16] | -2.63 [-2.91, -2.35]
ChaosNLI alphaNLI | 200 | 1.12 | 1.22 | 1.64 | 4.27 | -3.15 [-3.39, -2.89] | -3.06 [-3.30, -2.80] | -2.63 [-2.89, -2.37]
DICES-350 | 25 | 6.53 | 12.72 | 4.98 | 8.01 | -1.48 [-2.43, -0.56] | +4.71 [+3.48, +5.98] | -3.03 [-3.89, -2.11]
DICES-350 | 50 | 3.70 | 6.43 | 2.49 | 4.18 | -0.48 [-1.06, +0.11] | +2.25 [+1.62, +2.93] | -1.70 [-2.16, -1.25]
DICES-350 | 100 | 2.06 | 3.28 | 1.17 | 2.30 | -0.24 [-0.51, +0.02] | +0.98 [+0.67, +1.29] | -1.13 [-1.36, -0.91]
DICES-350 | 175 | 1.54 | 1.76 | 0.79 | 1.74 | -0.21 [-0.40, -0.02] | +0.02 [-0.19, +0.23] | -0.96 [-1.10, -0.80]
DICES-990 | 25 | 5.69 | 11.15 | 3.95 | 7.36 | -1.67 [-2.62, -0.69] | +3.79 [+2.73, +4.94] | -3.41 [-4.26, -2.54]
DICES-990 | 50 | 2.95 | 5.44 | 1.87 | 4.00 | -1.05 [-1.52, -0.56] | +1.44 [+0.87, +2.04] | -2.13 [-2.56, -1.70]
DICES-990 | 100 | 1.12 | 2.83 | 0.45 | 2.48 | -1.36 [-1.66, -1.06] | +0.35 [+0.02, +0.70] | -2.03 [-2.24, -1.82]
DICES-990 | 200 | 0.27 | 1.29 | 0.06 | 1.86 | -1.59 [-1.71, -1.46] | -0.57 [-0.78, -0.37] | -1.80 [-1.90, -1.70]

- Path A (Pearson correlation): mean regret above the random pick's in 1 of 20 cells (0 with the whole interval above 0); whole interval below 0 (better than random) in 17.
- Path B (three-fold cross-validated recalibrated proper risk): mean regret above the random pick's in 11 of 20 cells (9 with the whole interval above 0); whole interval below 0 (better than random) in 8.
- Path C (raw proper risk): mean regret above the random pick's in 0 of 20 cells (0 with the whole interval above 0); whole interval below 0 (better than random) in 19.

Integrity: C1 reproduces Experiment 60's rounded-mean and truthful skills and intervals; C3 reproduces the picks and regrets of paths A, B and C of all 4000 draws in Experiment 59's draws.csv (both asserted by this script).
