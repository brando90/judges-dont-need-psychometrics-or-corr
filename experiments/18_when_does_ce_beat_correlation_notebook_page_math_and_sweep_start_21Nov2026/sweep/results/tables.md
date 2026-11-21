## Sweep A, verdicts CE-tuned vs each other arm (cells = scenario x n x k x J; CE wins = both RPS and CE-excess intervals above 0)


### CE-tuned vs pearson_raw: counts by n (CE / tie / other / mixed)

| n | CE | tie | other | mixed |
|---|---|---|---|---|
| 8 | 1 | 5 | 5 | 25 |
| 16 | 13 | 2 | 3 | 18 |
| 32 | 22 | 3 | 1 | 10 |
| 64 | 31 | 2 | 0 | 3 |
| 128 | 33 | 0 | 0 | 3 |
| 256 | 36 | 0 | 0 | 0 |
| 512 | 36 | 0 | 0 | 0 |

### CE-tuned vs pearson_head_tuned: counts by n (CE / tie / other / mixed)

| n | CE | tie | other | mixed |
|---|---|---|---|---|
| 8 | 31 | 1 | 0 | 4 |
| 16 | 26 | 2 | 0 | 8 |
| 32 | 20 | 6 | 0 | 10 |
| 64 | 16 | 4 | 4 | 12 |
| 128 | 15 | 0 | 8 | 13 |
| 256 | 13 | 0 | 11 | 12 |
| 512 | 13 | 0 | 14 | 9 |

### CE-tuned vs mse_tuned: counts by n (CE / tie / other / mixed)

| n | CE | tie | other | mixed |
|---|---|---|---|---|
| 8 | 1 | 23 | 0 | 12 |
| 16 | 12 | 14 | 0 | 10 |
| 32 | 21 | 8 | 2 | 5 |
| 64 | 25 | 2 | 4 | 5 |
| 128 | 31 | 1 | 0 | 4 |
| 256 | 34 | 0 | 0 | 2 |
| 512 | 36 | 0 | 0 | 0 |

### CE-tuned vs rps_tuned: counts by n (CE / tie / other / mixed)

| n | CE | tie | other | mixed |
|---|---|---|---|---|
| 8 | 1 | 12 | 6 | 17 |
| 16 | 1 | 21 | 2 | 12 |
| 32 | 2 | 14 | 4 | 16 |
| 64 | 4 | 5 | 3 | 24 |
| 128 | 6 | 4 | 4 | 22 |
| 256 | 9 | 0 | 3 | 24 |
| 512 | 3 | 2 | 1 | 30 |

### CE-tuned vs constant: counts by n (CE / tie / other / mixed)

| n | CE | tie | other | mixed |
|---|---|---|---|---|
| 8 | 16 | 3 | 0 | 17 |
| 16 | 22 | 3 | 0 | 11 |
| 32 | 24 | 3 | 0 | 9 |
| 64 | 30 | 1 | 0 | 5 |
| 128 | 35 | 0 | 0 | 1 |
| 256 | 36 | 0 | 0 | 0 |
| 512 | 36 | 0 | 0 | 0 |

### Mean test RPS excess over true-distribution RPS floor (all scenarios, k=3, J=3), by n

| n | constant | pearson_raw | pearson_head | mse | ce | rps |
|---|---|---|---|---|---|---|
| 8 | 0.0566 | 0.0572 | 0.0669 | 0.0466 | 0.0441 | 0.0431 |
| 16 | 0.0521 | 0.0570 | 0.0517 | 0.0364 | 0.0294 | 0.0306 |
| 32 | 0.0494 | 0.0566 | 0.0347 | 0.0262 | 0.0183 | 0.0201 |
| 64 | 0.0477 | 0.0571 | 0.0192 | 0.0190 | 0.0115 | 0.0124 |
| 128 | 0.0471 | 0.0577 | 0.0117 | 0.0155 | 0.0084 | 0.0084 |
| 256 | 0.0466 | 0.0587 | 0.0076 | 0.0130 | 0.0061 | 0.0054 |
| 512 | 0.0463 | 0.0577 | 0.0057 | 0.0125 | 0.0053 | 0.0042 |

### Crossover n (smallest n from which CE-tuned beats the Pearson-selected+calibrated arm at every larger n; '-' = never)

| scenario | k | J=1 | J=3 | J=6 |
|---|---|---|---|---|
| A | 1 | 8 | 64 | 128 |
| A | 3 | 8 | 8 | 64 |
| A | 10 | 8 | 8 | 8 |
| B | 1 | - | 128 | 512 |
| B | 3 | - | - | 32 |
| B | 10 | - | - | 8 |
| C | 1 | - | - | - |
| C | 3 | - | - | - |
| C | 10 | - | - | - |
| D | 1 | - | - | - |
| D | 3 | - | - | - |
| D | 10 | - | - | - |

## Sweep B, selection of one of 24 candidate judges: CE-selection vs Pearson-selection, counts (CE / tie / other / mixed) by pool and n; cells = scenario x k

| pool | n | CE-sel better | tie | Pearson better | mixed | mean dRPS (Pearson-sel minus CE-sel) |
|---|---|---|---|---|---|---|
| prob | 4 | 9 | 2 | 0 | 1 | +0.0156 |
| prob | 8 | 12 | 0 | 0 | 0 | +0.0190 |
| prob | 16 | 12 | 0 | 0 | 0 | +0.0196 |
| prob | 32 | 12 | 0 | 0 | 0 | +0.0176 |
| prob | 64 | 12 | 0 | 0 | 0 | +0.0151 |
| prob | 128 | 12 | 0 | 0 | 0 | +0.0120 |
| prob | 256 | 12 | 0 | 0 | 0 | +0.0093 |
| votes3 | 4 | 3 | 8 | 0 | 1 | +0.0040 |
| votes3 | 8 | 4 | 7 | 0 | 1 | +0.0034 |
| votes3 | 16 | 4 | 5 | 0 | 3 | +0.0045 |
| votes3 | 32 | 4 | 6 | 0 | 2 | +0.0040 |
| votes3 | 64 | 3 | 7 | 0 | 2 | +0.0047 |
| votes3 | 128 | 6 | 3 | 0 | 3 | +0.0065 |
| votes3 | 256 | 6 | 3 | 0 | 3 | +0.0059 |
| votes3_f5e-2 | 4 | 6 | 6 | 0 | 0 | +0.0055 |
| votes3_f5e-2 | 8 | 3 | 7 | 0 | 2 | +0.0037 |
| votes3_f5e-2 | 16 | 6 | 6 | 0 | 0 | +0.0034 |
| votes3_f5e-2 | 32 | 5 | 5 | 0 | 2 | +0.0031 |
| votes3_f5e-2 | 64 | 4 | 7 | 0 | 1 | +0.0032 |
| votes3_f5e-2 | 128 | 6 | 5 | 0 | 1 | +0.0032 |
| votes3_f5e-2 | 256 | 9 | 2 | 0 | 1 | +0.0039 |

### Sweep B, same for RPS-selection vs Pearson-selection

| pool | n | RPS-sel better | tie | Pearson better | mixed | mean dRPS |
|---|---|---|---|---|---|---|
| prob | 4 | 11 | 1 | 0 | 0 | +0.0146 |
| prob | 8 | 12 | 0 | 0 | 0 | +0.0177 |
| prob | 16 | 12 | 0 | 0 | 0 | +0.0183 |
| prob | 32 | 12 | 0 | 0 | 0 | +0.0166 |
| prob | 64 | 12 | 0 | 0 | 0 | +0.0150 |
| prob | 128 | 12 | 0 | 0 | 0 | +0.0123 |
| prob | 256 | 11 | 0 | 0 | 1 | +0.0099 |
| votes3 | 4 | 7 | 4 | 0 | 1 | +0.0066 |
| votes3 | 8 | 8 | 1 | 0 | 3 | +0.0073 |
| votes3 | 16 | 8 | 3 | 0 | 1 | +0.0061 |
| votes3 | 32 | 11 | 1 | 0 | 0 | +0.0083 |
| votes3 | 64 | 10 | 0 | 0 | 2 | +0.0080 |
| votes3 | 128 | 11 | 1 | 0 | 0 | +0.0096 |
| votes3 | 256 | 11 | 1 | 0 | 0 | +0.0083 |
| votes3_f5e-2 | 4 | 10 | 1 | 0 | 1 | +0.0071 |
| votes3_f5e-2 | 8 | 10 | 2 | 0 | 0 | +0.0055 |
| votes3_f5e-2 | 16 | 10 | 1 | 0 | 1 | +0.0045 |
| votes3_f5e-2 | 32 | 11 | 0 | 0 | 1 | +0.0054 |
| votes3_f5e-2 | 64 | 12 | 0 | 0 | 0 | +0.0052 |
| votes3_f5e-2 | 128 | 11 | 0 | 0 | 1 | +0.0055 |
| votes3_f5e-2 | 256 | 12 | 0 | 0 | 0 | +0.0054 |
