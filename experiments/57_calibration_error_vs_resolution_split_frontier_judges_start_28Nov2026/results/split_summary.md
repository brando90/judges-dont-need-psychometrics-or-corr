# Experiment 57 summary (primary rule per set; skill = 1 - risk/base-rate risk, higher is better)

Bundle sha256 55935c4daacdf2ad...; B = 4000 group-clustered draws; maximum |identity residual| = 2.78e-17.

## ChaosNLI SNLI (brier, n = 600)

judge | raw skill | calibrated skill (resolution) | calibration error (skill pts) | Pearson r
---|---|---|---|---
Astra | +0.303 | +0.371 | +0.068 | +0.926
Opus | +0.363 | +0.370 | +0.007 | +0.942
Kimi | +0.297 | +0.328 | +0.031 | +0.889
GLM | +0.342 | +0.349 | +0.007 | +0.924
DeepSeek | +0.300 | +0.329 | +0.028 | +0.888

Kendall tau-b of the judge ranking by Pearson r against resolution: +0.60; against raw skill: +0.60; difference interval [-0.40, +0.60]. Spearman version: +1.00 against +0.60.

## ChaosNLI MNLI-m (brier, n = 600)

judge | raw skill | calibrated skill (resolution) | calibration error (skill pts) | Pearson r
---|---|---|---|---
Astra | +0.042 | +0.193 | +0.150 | +0.836
Opus | +0.160 | +0.193 | +0.033 | +0.862
Kimi | +0.041 | +0.150 | +0.109 | +0.762
GLM | +0.140 | +0.172 | +0.032 | +0.837
DeepSeek | +0.009 | +0.162 | +0.153 | +0.751

Kendall tau-b of the judge ranking by Pearson r against resolution: +0.40; against raw skill: +1.00; difference interval [-0.60, +0.40]. Spearman version: +0.80 against +0.60.

## ChaosNLI alphaNLI (brier, n = 400)

judge | raw skill | calibrated skill (resolution) | calibration error (skill pts) | Pearson r
---|---|---|---|---
Astra | +0.560 | +0.582 | +0.022 | +0.935
Opus | +0.545 | +0.558 | +0.012 | +0.917
Kimi | +0.534 | +0.553 | +0.018 | +0.911
GLM | +0.525 | +0.549 | +0.024 | +0.904
DeepSeek | +0.419 | +0.458 | +0.039 | +0.835

Kendall tau-b of the judge ranking by Pearson r against resolution: +1.00; against raw skill: +1.00; difference interval [-0.40, +0.20]. Spearman version: +0.80 against +0.80.

## DICES-350 (nrps, n = 350)

judge | raw skill | calibrated skill (resolution) | calibration error (skill pts) | Pearson r
---|---|---|---|---
Astra | -0.143 | +0.072 | +0.215 | +0.588
Opus | +0.032 | +0.069 | +0.037 | +0.607
Kimi | -0.138 | +0.050 | +0.188 | +0.519
GLM | -0.098 | +0.055 | +0.153 | +0.550
DeepSeek | -0.179 | +0.052 | +0.231 | +0.533

Kendall tau-b of the judge ranking by Pearson r against resolution: +0.80; against raw skill: +0.40; difference interval [-0.20, +1.00]. Spearman version: +0.80 against +0.40.

## DICES-990 (nrps, n = 400)

judge | raw skill | calibrated skill (resolution) | calibration error (skill pts) | Pearson r
---|---|---|---|---
Astra | -0.088 | +0.105 | +0.193 | +0.677
Opus | +0.090 | +0.120 | +0.029 | +0.786
Kimi | -0.086 | +0.090 | +0.177 | +0.634
GLM | -0.025 | +0.108 | +0.133 | +0.683
DeepSeek | -0.112 | +0.095 | +0.207 | +0.618

Kendall tau-b of the judge ranking by Pearson r against resolution: +0.80; against raw skill: +0.80; difference interval [-0.20, +0.40]. Spearman version: +1.00 against +0.60.

## Decisions
- H1: falsified (1 of 5 sets).
- H2: supported (4 of 5 sets).
- H3: supported.
