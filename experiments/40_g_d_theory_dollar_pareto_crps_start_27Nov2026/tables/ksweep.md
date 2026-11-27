# Output-granularity sweep

Real rows rebin a native ordinal scale only to K=2, K=3, and the native K. Synthetic rows are the continuous mixture on {0,...,100} and are not human ratings.

## Real rebin, held-out test cross-entropy

| Dataset | Mode | K | k ratings | CE-trained CE | MSE-trained CE | MSE minus CE |
|---|---|---:|---:|---|---|---|
| summeval_expert | K2 | 2 | 1 | 0.5746 [0.5563, 0.5925] | 0.5675 [0.5497, 0.5862] | -0.0070 [-0.0107, -0.0033] |
| summeval_expert | K2 | 2 | 2 | 0.5744 [0.5565, 0.5920] | 0.5673 [0.5495, 0.5855] | -0.0071 [-0.0109, -0.0033] |
| summeval_expert | K3 | 3 | 1 | 0.7565 [0.7233, 0.7886] | inf [nan, nan] | inf [nan, nan] |
| summeval_expert | K3 | 3 | 2 | 0.7565 [0.7237, 0.7883] | inf [nan, nan] | inf [nan, nan] |
| summeval_expert | native | 5 | 1 | 1.2097 [1.1704, 1.2473] | inf [nan, nan] | inf [nan, nan] |
| summeval_expert | native | 5 | 2 | 1.2098 [1.1711, 1.2471] | inf [nan, nan] | inf [nan, nan] |
| summeval_crowd | K2 | 2 | 1 | 0.6466 [0.6324, 0.6608] | 0.6502 [0.6354, 0.6648] | 0.0036 [0.0013, 0.0059] |
| summeval_crowd | K2 | 2 | 2 | 0.6473 [0.6328, 0.6621] | 0.6523 [0.6373, 0.6678] | 0.0050 [0.0024, 0.0077] |
| summeval_crowd | K2 | 2 | 4 | 0.6472 [0.6326, 0.6621] | 0.6520 [0.6370, 0.6675] | 0.0048 [0.0021, 0.0075] |
| summeval_crowd | K3 | 3 | 1 | 0.8495 [0.8201, 0.8793] | inf [nan, nan] | inf [nan, nan] |
| summeval_crowd | K3 | 3 | 2 | 0.8501 [0.8204, 0.8805] | inf [nan, nan] | inf [nan, nan] |
| summeval_crowd | K3 | 3 | 4 | 0.8498 [0.8199, 0.8802] | inf [nan, nan] | inf [nan, nan] |
| summeval_crowd | native | 5 | 1 | 1.3265 [1.3013, 1.3527] | inf [nan, nan] | inf [nan, nan] |
| summeval_crowd | native | 5 | 2 | 1.3264 [1.3008, 1.3525] | inf [nan, nan] | inf [nan, nan] |
| summeval_crowd | native | 5 | 4 | 1.3265 [1.3006, 1.3530] | inf [nan, nan] | inf [nan, nan] |
| popquorn_offensiveness | K2 | 2 | 1 | 0.4380 [0.3953, 0.4810] | inf [0.4087, 0.4962] | inf [0.0030, 0.0217] |
| popquorn_offensiveness | K2 | 2 | 2 | 0.4345 [0.3935, 0.4755] | 0.4391 [0.3970, 0.4806] | 0.0046 [-0.0025, 0.0121] |
| popquorn_offensiveness | K2 | 2 | 4 | 0.4325 [0.3923, 0.4725] | 0.4315 [0.3923, 0.4710] | -0.0010 [-0.0071, 0.0052] |
| popquorn_offensiveness | K2 | 2 | 8 | 0.4323 [0.3921, 0.4716] | 0.4305 [0.3919, 0.4690] | -0.0017 [-0.0074, 0.0038] |
| popquorn_offensiveness | K3 | 3 | 1 | 0.7921 [0.7430, 0.8443] | inf [nan, nan] | inf [nan, nan] |
| popquorn_offensiveness | K3 | 3 | 2 | 0.7891 [0.7405, 0.8418] | inf [nan, nan] | inf [nan, nan] |
| popquorn_offensiveness | K3 | 3 | 4 | 0.7877 [0.7392, 0.8414] | inf [nan, nan] | inf [nan, nan] |
| popquorn_offensiveness | K3 | 3 | 8 | 0.7876 [0.7393, 0.8406] | inf [nan, nan] | inf [nan, nan] |
| popquorn_offensiveness | native | 5 | 1 | 1.3011 [1.2439, 1.3638] | inf [nan, nan] | inf [nan, nan] |
| popquorn_offensiveness | native | 5 | 2 | 1.2993 [1.2428, 1.3627] | inf [nan, nan] | inf [nan, nan] |
| popquorn_offensiveness | native | 5 | 4 | 1.2990 [1.2423, 1.3625] | inf [nan, nan] | inf [nan, nan] |
| popquorn_offensiveness | native | 5 | 8 | 1.2990 [1.2417, 1.3627] | inf [nan, nan] | inf [nan, nan] |
| popquorn_politeness | K2 | 2 | 1 | 0.6530 [0.6463, 0.6596] | inf [0.6250, 0.6495] | inf [-0.0235, -0.0092] |
| popquorn_politeness | K2 | 2 | 2 | 0.6515 [0.6447, 0.6585] | inf [0.6202, 0.6471] | inf [-0.0262, -0.0097] |
| popquorn_politeness | K2 | 2 | 4 | 0.6524 [0.6459, 0.6592] | inf [0.6200, 0.6457] | inf [-0.0269, -0.0120] |
| popquorn_politeness | K2 | 2 | 8 | 0.6578 [0.6497, 0.6663] | 0.6463 [0.6331, 0.6603] | -0.0115 [-0.0179, -0.0042] |
| popquorn_politeness | K3 | 3 | 1 | 1.0304 [1.0136, 1.0461] | inf [nan, nan] | inf [nan, nan] |
| popquorn_politeness | K3 | 3 | 2 | 1.0293 [1.0134, 1.0446] | inf [nan, nan] | inf [nan, nan] |
| popquorn_politeness | K3 | 3 | 4 | 1.0303 [1.0148, 1.0449] | inf [nan, nan] | inf [nan, nan] |
| popquorn_politeness | K3 | 3 | 8 | 1.0342 [1.0211, 1.0471] | inf [nan, nan] | inf [nan, nan] |
| popquorn_politeness | native | 5 | 1 | 1.5297 [1.5136, 1.5457] | inf [nan, nan] | inf [nan, nan] |
| popquorn_politeness | native | 5 | 2 | 1.5283 [1.5129, 1.5437] | inf [nan, nan] | inf [nan, nan] |
| popquorn_politeness | native | 5 | 4 | 1.5282 [1.5133, 1.5428] | inf [nan, nan] | inf [nan, nan] |
| popquorn_politeness | native | 5 | 8 | 1.5322 [1.5200, 1.5447] | inf [nan, nan] | inf [nan, nan] |
| dices990 | K2 | 2 | 1 | 0.5663 [0.5306, 0.6001] | inf [0.5538, 0.6287] | inf [0.0076, 0.0380] |
| dices990 | K2 | 2 | 2 | 0.5618 [0.5300, 0.5930] | 0.5693 [0.5367, 0.6010] | 0.0074 [-0.0009, 0.0153] |
| dices990 | K2 | 2 | 4 | 0.5593 [0.5282, 0.5903] | 0.5577 [0.5257, 0.5884] | -0.0016 [-0.0076, 0.0040] |
| dices990 | K2 | 2 | 8 | 0.5581 [0.5276, 0.5875] | 0.5548 [0.5236, 0.5857] | -0.0033 [-0.0090, 0.0019] |
| dices990 | K3 | 3 | 1 | 0.7533 [0.7140, 0.7927] | inf [nan, nan] | inf [nan, nan] |
| dices990 | K3 | 3 | 2 | 0.7485 [0.7130, 0.7845] | inf [nan, nan] | inf [nan, nan] |
| dices990 | K3 | 3 | 4 | 0.7453 [0.7095, 0.7808] | inf [nan, nan] | inf [nan, nan] |
| dices990 | K3 | 3 | 8 | 0.7442 [0.7094, 0.7790] | inf [nan, nan] | inf [nan, nan] |
| dices990 | native | 3 | 1 | 0.7533 [0.7140, 0.7927] | inf [nan, nan] | inf [nan, nan] |
| dices990 | native | 3 | 2 | 0.7485 [0.7130, 0.7845] | inf [nan, nan] | inf [nan, nan] |
| dices990 | native | 3 | 4 | 0.7453 [0.7095, 0.7808] | inf [nan, nan] | inf [nan, nan] |
| dices990 | native | 3 | 8 | 0.7442 [0.7094, 0.7790] | inf [nan, nan] | inf [nan, nan] |
| chaosnli_snli_mnli | K2 | 2 | 1 | 0.5127 [0.4848, 0.5442] | 0.5203 [0.4912, 0.5525] | 0.0076 [0.0032, 0.0123] |
| chaosnli_snli_mnli | K2 | 2 | 2 | 0.5116 [0.4850, 0.5414] | 0.5143 [0.4868, 0.5446] | 0.0028 [-0.0003, 0.0058] |
| chaosnli_snli_mnli | K2 | 2 | 4 | 0.5112 [0.4845, 0.5406] | 0.5133 [0.4855, 0.5428] | 0.0021 [-0.0006, 0.0048] |
| chaosnli_snli_mnli | K2 | 2 | 8 | 0.5112 [0.4851, 0.5403] | 0.5131 [0.4862, 0.5428] | 0.0019 [-0.0005, 0.0044] |
| chaosnli_snli_mnli | K3 | 3 | 1 | 1.0542 [1.0393, 1.0699] | inf [nan, nan] | inf [nan, nan] |
| chaosnli_snli_mnli | K3 | 3 | 2 | 1.0536 [1.0399, 1.0684] | inf [nan, nan] | inf [nan, nan] |
| chaosnli_snli_mnli | K3 | 3 | 4 | 1.0533 [1.0397, 1.0681] | inf [nan, nan] | inf [nan, nan] |
| chaosnli_snli_mnli | K3 | 3 | 8 | 1.0529 [1.0397, 1.0672] | inf [nan, nan] | inf [nan, nan] |
| chaosnli_snli_mnli | native | 3 | 1 | 1.0542 [1.0393, 1.0699] | inf [nan, nan] | inf [nan, nan] |
| chaosnli_snli_mnli | native | 3 | 2 | 1.0536 [1.0399, 1.0684] | inf [nan, nan] | inf [nan, nan] |
| chaosnli_snli_mnli | native | 3 | 4 | 1.0533 [1.0397, 1.0681] | inf [nan, nan] | inf [nan, nan] |
| chaosnli_snli_mnli | native | 3 | 8 | 1.0529 [1.0397, 1.0672] | inf [nan, nan] | inf [nan, nan] |
| veribench_fsc_native_main | K2 | 2 | 1 | 0.2488 [0.1474, 0.3918] | 0.3526 [0.2592, 0.4518] | 0.1038 [-0.0009, 0.2012] |
| veribench_fsc_native_main | K2 | 2 | 2 | 0.2595 [0.1636, 0.3944] | inf [0.2136, 0.4992] | inf [-0.0028, 0.2321] |
| veribench_fsc_native_main | K3 | 3 | 1 | 0.2994 [0.1567, 0.5127] | inf [0.2518, 0.5325] | inf [0.1045, 0.2735] |
| veribench_fsc_native_main | K3 | 3 | 2 | 0.3070 [0.1737, 0.5048] | inf [0.2104, 0.3228] | inf [0.0434, 0.1829] |
| veribench_fsc_native_main | native | 6 | 1 | 0.7144 [0.5502, 0.9359] | inf [nan, nan] | inf [nan, nan] |
| veribench_fsc_native_main | native | 6 | 2 | 0.6780 [0.5131, 0.8980] | inf [nan, nan] | inf [nan, nan] |

## Synthetic CRPS crossover (not human data)

| K | smallest k whose 95% CI says the CE head has lower CRPS | smallest k on the point estimate |
|---:|---|---|
| 2 | 2 | 1 |
| 3 | 1 | 1 |
| 5 | 1 | 1 |
| 10 | 1 | 1 |
| 20 | 1 | 1 |
| 101 | 1 | 1 |
