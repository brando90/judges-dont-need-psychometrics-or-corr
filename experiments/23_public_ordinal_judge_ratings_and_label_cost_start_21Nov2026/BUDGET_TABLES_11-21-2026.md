# Held-out budget tables | 11-21-2026

As of 11-21-2026. Every budget counts individual training grades, one per item unless shown otherwise. Cross-entropy (CE) and Pearson fits use exactly the same candidate library and observed training labels. Positive differences in normalized ranked probability score (RPS) favor the CE fit. Test items have equal weight.

## SummEval factual consistency

Paired held-out RPS for the named comparator minus the CE fit. Brackets are simultaneous 97.5% within-benchmark bands from 4,000 paired held-out article/image bootstrap samples across all budgets and four comparators. ‘Both’ counts matched training draws in which CE had a strictly lower test RPS than raw Pearson and uniform. ‘Crossed’ also resamples the 64 training draws and tests both simultaneous lower bounds against zero. [Design](PROTOCOL_11-21-2026.md)

| Training grades | Pearson − CE fit [band] | Uniform − CE fit [band] | Calibrated Pearson − CE fit [band] | Training histogram − CE fit [band] | Both / 64 | Crossed clears both |
|---:|---:|---:|---:|---:|---:|:---:|
| 6 | -0.0252 [-0.0427, -0.0078] | +0.0545 [+0.0456, +0.0634] | -0.1432 [-0.1612, -0.1252] | -0.1354 [-0.1534, -0.1174] | 19 | no |
| 12 | -0.0639 [-0.0849, -0.0430] | +0.0266 [+0.0223, +0.0308] | -0.1845 [-0.2034, -0.1656] | -0.1753 [-0.1950, -0.1556] | 10 | no |
| 24 | -0.0736 [-0.0938, -0.0535] | +0.0244 [+0.0205, +0.0282] | -0.1943 [-0.2136, -0.1749] | -0.1832 [-0.2047, -0.1617] | 6 | no |
| 48 | -0.0984 [-0.1190, -0.0778] | +0.0155 [+0.0130, +0.0180] | -0.2078 [-0.2268, -0.1887] | -0.1942 [-0.2170, -0.1715] | 4 | no |
| 96 | -0.1142 [-0.1360, -0.0924] | +0.0066 [+0.0056, +0.0077] | -0.2198 [-0.2377, -0.2019] | -0.2038 [-0.2270, -0.1807] | 0 | no |
| 192 | -0.1271 [-0.1495, -0.1046] | +0.0022 [+0.0019, +0.0026] | -0.2257 [-0.2432, -0.2082] | -0.2087 [-0.2323, -0.1850] | 0 | no |
| 384 | -0.1344 [-0.1570, -0.1118] | +0.0000 [+0.0000, +0.0000] | -0.2290 [-0.2460, -0.2119] | -0.2111 [-0.2348, -0.1873] | 0 | no |
| 768 | -0.1401 [-0.1626, -0.1176] | +0.0000 [+0.0000, +0.0000] | -0.2294 [-0.2463, -0.2125] | -0.2112 [-0.2351, -0.1874] | 0 | no |

### Absolute losses at 768 training grades

Held-out RPS (0–1), CE (natural-log units per human grade) and total variation distance (TVD, 0–1) are lower when better. TVD compares the predicted probabilities to each item's empirical histogram. Test populations are 640 summaries from 40 held-out articles, three expert grades per summary, or 2,015 held-out photographs with 215,415 crowd votes.

| Method | Test RPS | Test CE (nats) | Test TVD |
|---|---:|---:|---:|
| CE fit | 0.28367 | 1.60944 | 0.76719 |
| Pearson fit | 0.14361 | 1.93348 | 0.49402 |
| Uniform | 0.28367 | 1.60944 | 0.76719 |
| Training histogram | 0.07243 | 0.54723 | 0.22961 |
| Pearson + calibration | 0.05429 | 0.42135 | 0.17096 |
| Original judge | 0.20952 | 4.65515 | 0.66110 |
| RPS fit | 0.14198 | 1.88002 | 0.48747 |

[Scores at every budget and depth](results/gpt4_consistency/scores.csv) · [Uncertainty for both depths](results/gpt4_consistency/contrasts.csv) · [Separate ranking and calibration metrics](results/gpt4_consistency/diagnostics.csv) · [Crossed bands](results/gpt4_consistency/crossed_sensitivity.csv)

## KonIQ-10k image quality

Paired held-out RPS for the named comparator minus the CE fit. Brackets are simultaneous 97.5% within-benchmark bands from 4,000 paired held-out article/image bootstrap samples across all budgets and four comparators. ‘Both’ counts matched training draws in which CE had a strictly lower test RPS than raw Pearson and uniform. ‘Crossed’ also resamples the 64 training draws and tests both simultaneous lower bounds against zero. [Design](PROTOCOL_11-21-2026.md)

| Training grades | Pearson − CE fit [band] | Uniform − CE fit [band] | Calibrated Pearson − CE fit [band] | Training histogram − CE fit [band] | Both / 64 | Crossed clears both |
|---:|---:|---:|---:|---:|---:|:---:|
| 6 | +0.1422 [+0.1405, +0.1439] | -0.0199 [-0.0221, -0.0178] | -0.0377 [-0.0401, -0.0353] | -0.0373 [-0.0398, -0.0348] | 21 | no |
| 12 | +0.1379 [+0.1343, +0.1414] | +0.0033 [+0.0012, +0.0055] | -0.0198 [-0.0222, -0.0174] | -0.0168 [-0.0196, -0.0140] | 38 | no |
| 24 | +0.1467 [+0.1435, +0.1500] | +0.0089 [+0.0066, +0.0112] | -0.0192 [-0.0217, -0.0167] | -0.0149 [-0.0180, -0.0118] | 44 | yes |
| 48 | +0.1386 [+0.1346, +0.1426] | +0.0190 [+0.0164, +0.0216] | -0.0145 [-0.0170, -0.0120] | -0.0067 [-0.0102, -0.0033] | 56 | yes |
| 96 | +0.1414 [+0.1356, +0.1472] | +0.0233 [+0.0206, +0.0260] | -0.0138 [-0.0162, -0.0114] | -0.0033 [-0.0070, +0.0004] | 61 | yes |
| 192 | +0.1379 [+0.1320, +0.1438] | +0.0245 [+0.0219, +0.0272] | -0.0145 [-0.0167, -0.0123] | -0.0027 [-0.0064, +0.0009] | 63 | yes |
| 384 | +0.1555 [+0.1489, +0.1621] | +0.0250 [+0.0224, +0.0276] | -0.0152 [-0.0173, -0.0131] | -0.0026 [-0.0063, +0.0011] | 64 | yes |
| 768 | +0.1667 [+0.1599, +0.1734] | +0.0261 [+0.0236, +0.0285] | -0.0146 [-0.0166, -0.0127] | -0.0017 [-0.0052, +0.0019] | 64 | yes |
| 1,536 | +0.1745 [+0.1678, +0.1812] | +0.0263 [+0.0239, +0.0286] | -0.0147 [-0.0165, -0.0128] | -0.0015 [-0.0050, +0.0020] | 64 | yes |
| 3,072 | +0.1712 [+0.1646, +0.1778] | +0.0264 [+0.0241, +0.0287] | -0.0146 [-0.0164, -0.0128] | -0.0014 [-0.0049, +0.0020] | 64 | yes |
| 6,144 | +0.2005 [+0.1935, +0.2074] | +0.0265 [+0.0242, +0.0288] | -0.0146 [-0.0164, -0.0128] | -0.0014 [-0.0048, +0.0021] | 64 | yes |

### Absolute losses at 6,144 training grades

Held-out RPS (0–1), CE (natural-log units per human grade) and total variation distance (TVD, 0–1) are lower when better. TVD compares the predicted probabilities to each item's empirical histogram. Test populations are 640 summaries from 40 held-out articles, three expert grades per summary, or 2,015 held-out photographs with 215,415 crowd votes.

| Method | Test RPS | Test CE (nats) | Test TVD |
|---|---:|---:|---:|
| CE fit | 0.10678 | 1.27885 | 0.37369 |
| Pearson fit | 0.30723 | 4.34565 | 0.78809 |
| Uniform | 0.13325 | 1.60944 | 0.53019 |
| Training histogram | 0.10541 | 1.17923 | 0.29698 |
| Pearson + calibration | 0.09215 | 1.07173 | 0.23373 |
| Original judge | 0.12225 | 1.35883 | 0.40839 |
| RPS fit | 0.10664 | 1.27831 | 0.37595 |

[Scores at every budget and depth](results/image_quality/scores.csv) · [Uncertainty for both depths](results/image_quality/contrasts.csv) · [Separate ranking and calibration metrics](results/image_quality/diagnostics.csv) · [Crossed bands](results/image_quality/crossed_sensitivity.csv)

## Training cost scenarios

Assumed US dollars per single retained grade. Acquisition excludes recruitment, overhead and test-set grades. The original per-grade prices on these two tasks are unknown. Reuse required no additional paid annotations.

| Training grades | At $0.10/grade | At $1/grade | At $5/grade |
|---:|---:|---:|---:|
| 6 | $0.60 | $6.00 | $30.00 |
| 12 | $1.20 | $12.00 | $60.00 |
| 24 | $2.40 | $24.00 | $120.00 |
| 48 | $4.80 | $48.00 | $240.00 |
| 96 | $9.60 | $96.00 | $480.00 |
| 192 | $19.20 | $192.00 | $960.00 |
| 384 | $38.40 | $384.00 | $1,920.00 |
| 768 | $76.80 | $768.00 | $3,840.00 |
| 1,536 | $153.60 | $1,536.00 | $7,680.00 |
| 3,072 | $307.20 | $3,072.00 | $15,360.00 |
| 6,144 | $614.40 | $6,144.00 | $30,720.00 |
