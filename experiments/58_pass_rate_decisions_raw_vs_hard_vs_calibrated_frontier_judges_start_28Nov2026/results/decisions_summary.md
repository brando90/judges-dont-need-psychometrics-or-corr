# Experiment 58 summary (rates in percent; error = estimator minus human rate, in percentage points)

Bundle sha256 55935c4daacdf2ad...; B = 4000 group-clustered draws.

## ChaosNLI SNLI: event = entailment; human rate 29.4% (95% interval 26.9 to 31.9)

judge | raw soft error | hard-label error | calibrated soft error
---|---|---|---
Astra | +3.8 [+2.5, +5.1] | +3.2 [+1.0, +5.4] | -0.0 [-1.0, +1.0]
Opus | +2.2 [+1.2, +3.1] | +2.8 [+0.7, +4.9] | +0.0 [-0.9, +0.9]
Kimi | +1.3 [+0.0, +2.6] | -1.7 [-3.7, +0.3] | -0.0 [-1.2, +1.3]
GLM | +3.5 [+2.4, +4.7] | +1.0 [-1.2, +3.1] | -0.0 [-1.1, +1.2]
DeepSeek | +2.4 [+1.1, +3.6] | -2.9 [-4.9, -0.9] | -0.0 [-1.2, +1.1]

## ChaosNLI MNLI-m: event = entailment; human rate 41.2% (95% interval 38.9 to 43.6)

judge | raw soft error | hard-label error | calibrated soft error
---|---|---|---
Astra | -7.9 [-9.6, -6.2] | -7.5 [-10.3, -4.8] | -0.0 [-1.2, +1.2]
Opus | -8.8 [-10.1, -7.5] | -7.7 [-10.4, -5.0] | -0.0 [-1.2, +1.2]
Kimi | -10.4 [-12.1, -8.8] | -11.5 [-14.1, -8.9] | +0.0 [-1.5, +1.6]
GLM | -6.4 [-7.8, -5.0] | -4.1 [-7.0, -1.2] | -0.1 [-1.5, +1.3]
DeepSeek | -12.4 [-14.2, -10.7] | -16.7 [-19.4, -14.0] | -0.0 [-1.5, +1.5]

## ChaosNLI alphaNLI: event = label 0; human rate 47.8% (95% interval 43.8 to 51.8)

judge | raw soft error | hard-label error | calibrated soft error
---|---|---|---
Astra | +1.5 [-0.0, +3.1] | +1.0 [-1.2, +3.2] | +0.0 [-1.4, +1.4]
Opus | +3.5 [+1.9, +5.1] | +2.8 [+0.6, +5.2] | +0.1 [-1.5, +1.6]
Kimi | +3.4 [+1.7, +5.0] | +3.2 [+0.9, +5.5] | -0.0 [-1.6, +1.6]
GLM | +4.8 [+3.0, +6.5] | +5.2 [+2.9, +7.5] | +0.0 [-1.6, +1.7]
DeepSeek | +5.7 [+3.4, +8.0] | +5.5 [+2.8, +8.2] | +0.0 [-2.2, +2.2]

## DICES-350: event = Yes (unsafe); human rate 32.7% (95% interval 30.5 to 34.8)

judge | raw soft error | hard-label error | calibrated soft error
---|---|---|---
Astra | -18.9 [-21.0, -16.8] | -20.4 [-23.4, -17.3] | -0.1 [-1.8, +1.6]
Opus | -8.7 [-10.4, -7.0] | -21.3 [-24.1, -18.3] | +0.0 [-1.7, +1.7]
Kimi | -20.6 [-22.7, -18.6] | -21.6 [-24.5, -18.7] | +0.0 [-1.8, +1.8]
GLM | -19.0 [-20.9, -17.2] | -24.7 [-27.4, -21.8] | +0.0 [-1.8, +1.8]
DeepSeek | -24.1 [-25.9, -22.2] | -25.2 [-27.5, -22.9] | -0.1 [-1.9, +1.8]

## DICES-990: event = Yes (unsafe); human rate 28.2% (95% interval 26.4 to 30.1)

judge | raw soft error | hard-label error | calibrated soft error
---|---|---|---
Astra | -13.6 [-15.4, -11.6] | -14.0 [-16.8, -11.1] | -0.0 [-1.2, +1.2]
Opus | -3.9 [-5.1, -2.5] | -11.2 [-14.0, -8.3] | -0.0 [-1.1, +1.1]
Kimi | -15.2 [-17.1, -13.3] | -15.1 [-17.8, -12.4] | -0.0 [-1.3, +1.3]
GLM | -12.5 [-14.2, -10.8] | -14.8 [-17.6, -11.9] | -0.0 [-1.2, +1.2]
DeepSeek | -16.0 [-18.0, -13.9] | -16.2 [-18.8, -13.4] | +0.0 [-1.3, +1.4]

## Decisions
- H1: supported (13 of 25 cells with the raw |error| interval above 5 points).
- H2: supported (25 of 25 cells where calibration reduced |error|).
- H3: supported (25 of 25 cells where the hard-label |error| exceeded the calibrated one).

## Gate flips (gates = 5 thresholds per cell; flip = estimator and humans on opposite sides of 'rate <= tau'; significant = in >= 97.5% of draws)
- primary:raw_soft: 39 of 125 gates flip (20 significant; 30 false passes, 9 false fails)
- primary:hard_label: 38 of 125 gates flip (22 significant; 31 false passes, 7 false fails)
- primary:calibrated_soft: 0 of 125 gates flip (0 significant; 0 false passes, 0 false fails)
- secondary:raw_soft: 17 of 50 gates flip (15 significant; 17 false passes, 0 false fails)
- secondary:hard_label: 20 of 50 gates flip (20 significant; 20 false passes, 0 false fails)
- secondary:calibrated_soft: 0 of 50 gates flip (0 significant; 0 false passes, 0 false fails)

Largest raw soft error: DICES-350, DeepSeek: -24.1 points (95% interval -25.9 to -22.2).
Cell where calibration helped least (the one most unfavorable to it): ChaosNLI SNLI, Kimi: |error| 1.26 to 0.03 points; cells where calibration made |error| larger: 0 of 25.

Post hoc (not pre-registered): the cross-fitted human base rate, which uses no judge at all, has an error of -0.02 to +0.00 points on the primary events, so a level near the human rate is not evidence that a judge discriminates (see results/posthoc_base_rate_rates.csv).
