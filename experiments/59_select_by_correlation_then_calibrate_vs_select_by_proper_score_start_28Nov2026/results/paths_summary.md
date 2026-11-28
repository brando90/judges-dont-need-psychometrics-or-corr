# Experiment 59 summary (regret in percentage points of skill = 100 x regret, lower is better; A minus B positive = proper-score selection (B) better, negative = correlation selection (A) better)

## ChaosNLI SNLI (brier)

n used | mean regret A | mean regret B | mean regret C | A minus B [95% interval] | share picking the oracle judge A / B / C | sensitivity: A minus C [95% interval]
---|---|---|---|---|---|---
25 | 6.89 | 10.43 | 7.31 | -3.55 [-5.12, -2.01] | 0.25 / 0.14 / 0.22 | -0.42 [-1.45, +0.60]
50 | 3.69 | 4.59 | 3.60 | -0.90 [-1.58, -0.28] | 0.28 / 0.17 / 0.28 | +0.09 [-0.40, +0.59]
100 | 1.85 | 2.22 | 2.07 | -0.36 [-0.69, -0.02] | 0.31 / 0.30 / 0.28 | -0.22 [-0.53, +0.10]
200 | 0.95 | 1.05 | 1.18 | -0.10 [-0.38, +0.18] | 0.46 / 0.39 / 0.40 | -0.23 [-0.47, +0.02]

## ChaosNLI MNLI-m (brier)

n used | mean regret A | mean regret B | mean regret C | A minus B [95% interval] | share picking the oracle judge A / B / C | sensitivity: A minus C [95% interval]
---|---|---|---|---|---|---
25 | 7.82 | 10.18 | 7.03 | -2.37 [-3.73, -1.00] | 0.21 / 0.14 / 0.20 | +0.78 [-0.29, +1.91]
50 | 3.49 | 5.59 | 2.93 | -2.10 [-2.83, -1.40] | 0.25 / 0.10 / 0.28 | +0.57 [+0.01, +1.14]
100 | 1.66 | 2.40 | 1.68 | -0.74 [-1.11, -0.39] | 0.32 / 0.28 / 0.33 | -0.02 [-0.32, +0.28]
200 | 1.00 | 1.12 | 0.95 | -0.12 [-0.34, +0.09] | 0.30 / 0.23 / 0.39 | +0.05 [-0.13, +0.23]

## ChaosNLI alphaNLI (brier)

n used | mean regret A | mean regret B | mean regret C | A minus B [95% interval] | share picking the oracle judge A / B / C | sensitivity: A minus C [95% interval]
---|---|---|---|---|---|---
25 | 4.96 | 6.08 | 4.84 | -1.13 [-2.23, -0.13] | 0.24 / 0.19 / 0.26 | +0.11 [-0.65, +0.71]
50 | 2.95 | 3.70 | 2.99 | -0.76 [-1.27, -0.29] | 0.32 / 0.20 / 0.27 | -0.05 [-0.24, +0.17]
100 | 1.65 | 1.98 | 1.87 | -0.33 [-0.66, -0.01] | 0.47 / 0.39 / 0.41 | -0.22 [-0.41, -0.04]
200 | 1.12 | 1.22 | 1.64 | -0.09 [-0.28, +0.09] | 0.64 / 0.61 / 0.46 | -0.52 [-0.70, -0.33]

## DICES-350 (nrps)

n used | mean regret A | mean regret B | mean regret C | A minus B [95% interval] | share picking the oracle judge A / B / C | sensitivity: A minus C [95% interval]
---|---|---|---|---|---|---
25 | 6.53 | 12.72 | 4.98 | -6.19 [-7.85, -4.66] | 0.21 / 0.08 / 0.29 | +1.55 [+0.35, +2.81]
50 | 3.70 | 6.43 | 2.49 | -2.73 [-3.58, -1.94] | 0.25 / 0.10 / 0.34 | +1.22 [+0.57, +1.86]
100 | 2.06 | 3.28 | 1.17 | -1.22 [-1.60, -0.88] | 0.21 / 0.12 / 0.45 | +0.89 [+0.59, +1.20]
175 | 1.54 | 1.76 | 0.79 | -0.23 [-0.45, -0.01] | 0.23 / 0.21 / 0.36 | +0.75 [+0.55, +0.96]

## DICES-990 (nrps)

n used | mean regret A | mean regret B | mean regret C | A minus B [95% interval] | share picking the oracle judge A / B / C | sensitivity: A minus C [95% interval]
---|---|---|---|---|---|---
25 | 5.69 | 11.15 | 3.95 | -5.46 [-6.85, -4.12] | 0.30 / 0.10 / 0.43 | +1.74 [+0.85, +2.70]
50 | 2.95 | 5.44 | 1.87 | -2.49 [-3.29, -1.69] | 0.30 / 0.20 / 0.46 | +1.08 [+0.66, +1.52]
100 | 1.12 | 2.83 | 0.45 | -1.72 [-2.12, -1.33] | 0.54 / 0.22 / 0.69 | +0.67 [+0.42, +0.94]
200 | 0.27 | 1.29 | 0.06 | -1.02 [-1.25, -0.79] | 0.78 / 0.45 / 0.85 | +0.21 [+0.11, +0.32]

## Decisions
- H1: falsified (sets where B has lower regret at n = 25: []; with the interval excluding 0: []).
- H2: falsified (|A minus B| at n = 200: chaosnli_snli 0.10, chaosnli_mnli 0.12, chaosnli_alphanli 0.09, dices350 0.23, dices990 1.02 percentage points of skill; margin 0.50).
- H3: smallest n at which |A minus B| < 0.005 in every set: none.
