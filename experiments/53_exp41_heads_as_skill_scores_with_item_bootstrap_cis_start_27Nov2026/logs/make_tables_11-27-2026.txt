# Experiment 53: Experiment 41's heads as skill scores over the human base rate

Held-out test items of Experiment 41's preregistered K-sweep (ridge 0.01, at most 35 L-BFGS iterations), five group-level split seeds. Skill = 1 - nRPS(head)/nRPS(base rate); nRPS = ranked probability score divided by K - 1 (0 to 1, lower is better); base rate = training-split marginal. Skill: 1 = perfect, 0 = base rate, negative = worse than the base rate. Values are means over the five seeds; brackets are item-clustered percentile bootstrap 95% intervals (2,000 draws, test items resampled within each seed, paired across heads). Differences are CE-trained head minus Pearson-trained head (positive = CE-trained better). 'raw' = the head's own forecast; 'cal' = the head's scalar mean through its own validation-fitted multinomial-logit map. Holm p across 16 views (bootstrap p; '<0.001' means no bootstrap draw crossed zero).

| view | base-rate nRPS | skill CE-trained raw | skill Pearson-trained raw | skill difference raw (CE minus Pearson) | Holm p | skill CE-trained cal | skill Pearson-trained cal | skill difference cal (CE minus Pearson) | Holm p | raw nRPS gap (Pearson minus CE) |
|---|---|---|---|---|---|---|---|---|---|---|
| popquorn_offensiveness|K2 | 0.1156 | +0.0031 [+0.0022, +0.0040] | -1.1141 [-1.2554, -1.0082] | +1.1173 [+1.0116, +1.2586] | <0.001 | +0.0023 [-0.0022, +0.0070] | +0.0017 [-0.0027, +0.0062] | +0.0007 [+0.0003, +0.0011] | <0.001 | +0.1277 |
| popquorn_offensiveness|K3 | 0.1529 | +0.0037 [+0.0025, +0.0049] | -0.6239 [-0.6923, -0.5657] | +0.6276 [+0.5702, +0.6962] | <0.001 | +0.0054 [+0.0007, +0.0104] | +0.0045 [-0.0001, +0.0094] | +0.0010 [+0.0003, +0.0016] | 0.036 | +0.0952 |
| popquorn_offensiveness|K5 | 0.1501 | +0.0048 [+0.0033, +0.0064] | -0.6938 [-0.7477, -0.6438] | +0.6986 [+0.6487, +0.7529] | <0.001 | +0.0074 [+0.0020, +0.0130] | +0.0076 [+0.0023, +0.0131] | -0.0002 [-0.0011, +0.0006] | 1 | +0.1044 |
| popquorn_politeness|K2 | 0.2489 | +0.0712 [+0.0655, +0.0767] | +0.0862 [+0.0768, +0.0952] | -0.0149 [-0.0189, -0.0111] | <0.001 | +0.0988 [+0.0878, +0.1095] | +0.0890 [+0.0783, +0.0995] | +0.0098 [+0.0081, +0.0114] | <0.001 | -0.0037 |
| popquorn_politeness|K3 | 0.2184 | +0.0529 [+0.0489, +0.0567] | -0.0853 [-0.1009, -0.0707] | +0.1381 [+0.1257, +0.1512] | <0.001 | +0.0796 [+0.0703, +0.0886] | +0.0721 [+0.0632, +0.0808] | +0.0075 [+0.0063, +0.0088] | <0.001 | +0.0301 |
| popquorn_politeness|K5 | 0.1758 | +0.0340 [+0.0316, +0.0364] | -0.3911 [-0.4122, -0.3693] | +0.4251 [+0.4046, +0.4450] | <0.001 | +0.0720 [+0.0637, +0.0800] | +0.0667 [+0.0587, +0.0744] | +0.0053 [+0.0044, +0.0062] | <0.001 | +0.0747 |
| summeval_expert|K2 | 0.2001 | +0.0130 [+0.0112, +0.0148] | -0.2231 [-0.2496, -0.1996] | +0.2361 [+0.2126, +0.2627] | <0.001 | +0.0206 [+0.0127, +0.0284] | +0.0204 [+0.0129, +0.0279] | +0.0001 [-0.0008, +0.0011] | 1 | +0.0470 |
| summeval_expert|K3 | 0.1517 | +0.0094 [+0.0081, +0.0108] | -0.5902 [-0.6369, -0.5486] | +0.5996 [+0.5582, +0.6459] | <0.001 | +0.0150 [+0.0084, +0.0221] | +0.0139 [+0.0076, +0.0205] | +0.0011 [+0.0000, +0.0022] | 0.4 | +0.0904 |
| summeval_expert|K5 | 0.1449 | +0.0120 [+0.0109, +0.0132] | -0.6720 [-0.7067, -0.6394] | +0.6840 [+0.6517, +0.7185] | <0.001 | +0.0287 [+0.0210, +0.0372] | +0.0240 [+0.0178, +0.0308] | +0.0046 [+0.0017, +0.0077] | 0.02 | +0.0988 |
| summeval_crowd|K2 | 0.2233 | -0.0018 [-0.0031, -0.0006] | -0.1502 [-0.1692, -0.1331] | +0.1484 [+0.1314, +0.1670] | <0.001 | +0.0002 [-0.0027, +0.0031] | +0.0004 [-0.0025, +0.0032] | -0.0002 [-0.0005, +0.0000] | 0.665 | +0.0329 |
| summeval_crowd|K3 | 0.1516 | -0.0010 [-0.0019, -0.0002] | -0.6622 [-0.6981, -0.6286] | +0.6611 [+0.6281, +0.6975] | <0.001 | +0.0003 [-0.0023, +0.0027] | +0.0005 [-0.0020, +0.0029] | -0.0003 [-0.0006, +0.0001] | 0.69 | +0.1000 |
| summeval_crowd|K5 | 0.1277 | -0.0009 [-0.0014, -0.0004] | -1.0058 [-1.0363, -0.9779] | +1.0049 [+0.9769, +1.0353] | <0.001 | -0.0020 [-0.0040, +0.0001] | -0.0017 [-0.0037, +0.0004] | -0.0003 [-0.0006, +0.0001] | 0.75 | +0.1282 |
| dices990|K2 | 0.1867 | +0.0131 [+0.0097, +0.0162] | -0.3937 [-0.4450, -0.3477] | +0.4067 [+0.3615, +0.4581] | <0.001 | +0.0187 [+0.0105, +0.0258] | +0.0155 [+0.0080, +0.0222] | +0.0031 [+0.0016, +0.0047] | 0.011 | +0.0756 |
| dices990|K3 | 0.1980 | +0.0138 [+0.0104, +0.0169] | -0.3103 [-0.3560, -0.2692] | +0.3240 [+0.2839, +0.3694] | <0.001 | +0.0195 [+0.0113, +0.0268] | +0.0163 [+0.0085, +0.0232] | +0.0032 [+0.0016, +0.0047] | <0.001 | +0.0639 |
| chaosnli_snli_mnli|K3 | 0.1950 | +0.0066 [+0.0048, +0.0086] | -0.2883 [-0.3058, -0.2714] | +0.2950 [+0.2784, +0.3119] | <0.001 | +0.0069 [+0.0033, +0.0107] | +0.0070 [+0.0035, +0.0106] | -0.0001 [-0.0009, +0.0006] | 1 | +0.0575 |
| chaosnli_alphanli|K2 | 0.2501 | -0.0054 [-0.0089, -0.0020] | -0.0133 [-0.0205, -0.0065] | +0.0079 [+0.0036, +0.0122] | <0.001 | -0.0050 [-0.0105, +0.0009] | -0.0048 [-0.0104, +0.0009] | -0.0002 [-0.0010, +0.0007] | 1 | +0.0020 |

## Summary

- Raw skill difference (CE-trained minus Pearson-trained): -0.0149 to +1.1173; positive on 15/16 views.
- Pearson-trained raw skill: -1.1141 to +0.0862; below zero (worse than the base rate) on 15/16 views.
- CE-trained raw skill: -0.0054 to +0.0712; CE-trained calibrated skill: -0.0050 to +0.0988.
- Calibrated skill difference (CE minus Pearson): -0.0003 to +0.0098; |value| <= 0.01 on 16/16.
- Calibrated skill difference (CE minus MSE): -0.0105 to +0.0033.
- Sanity check: passed: every (view, seed, head) mean matches Exp 41 maxiter35 JSON within 1e-3 (max deviation per head below); max |deviation| by head: constant_raw 0.0e+00, ce_raw 3.1e-07, ce_cal 7.4e-08, pearson_raw 1.3e-06, pearson_cal 1.2e-07, mse_raw 8.3e-17, mse_cal 4.4e-15.

## Preregistered predictions

- P1 (raw difference > 0, Holm p < 0.05 where Exp 41 raw gap > 0.01 nRPS): holds on 14/14 such views.
- P2 (|calibrated difference| <= 0.025): holds on 16/16.
- P3 (descriptive; CE-trained raw skill < 0.10): 16/16.
