# Generated results: experiment 11

Cells with all 100 seeds in every scenario: n = [4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048] (3 ratings/task, 500 held-out test tasks/seed).

True distribution test RPS baselines (empirical truth on test tasks): checked per scenario below.

All pairwise comparisons use 10,000 paired seed bootstrap draws (95% percentile CIs). CE uses exact unclipped log-sum-exp loss/gradient; Stage-2 heads tune slope ridge c_b in {0.01, 0.1, 1, 10, 100, 1e4, 1e8} by 4-fold CV RPS.


## Scenario A: mean nearly constant, disagreement varies (true test RPS ≈ 0.1870)

### Held-out test RPS excess over truth (mean over 100 seeds; 0 optimal, lower is better)

| n (ratings) | true RPS | Constant (train histogram) | CE-trained, ridge 1 (④) | CE-trained, val RPS ridge (③) | Pearson-trained, ridge 1 (②) | Pearson + naive head (1/n ridge) | Pearson + tuned in-sample head | Pearson + cross-fitted tuned head | MSE-trained, ridge 1 | MSE-trained, val RPS ridge | MSE + cross-fitted tuned head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 (12) | 0.1869 | 0.0267 | 0.0106 | 0.0140 | 0.0096 | 0.1272 | 0.0428 | 0.0432 | 0.0163 | 0.0131 | 0.0445 |
| 8 (24) | 0.1871 | 0.0185 | 0.0116 | 0.0141 | 0.0096 | 0.1362 | 0.0254 | 0.0245 | 0.0228 | 0.0172 | 0.0283 |
| 16 (48) | 0.1869 | 0.0147 | 0.0123 | 0.0117 | 0.0096 | 0.1617 | 0.0153 | 0.0153 | 0.0301 | 0.0135 | 0.0178 |
| 20 (60) | 0.1870 | 0.0125 | 0.0113 | 0.0104 | 0.0097 | 0.1309 | 0.0134 | 0.0134 | 0.0284 | 0.0116 | 0.0150 |
| 32 (96) | 0.1868 | 0.0119 | 0.0111 | 0.0096 | 0.0098 | 0.0904 | 0.0124 | 0.0125 | 0.0302 | 0.0106 | 0.0131 |
| 64 (192) | 0.1870 | 0.0107 | 0.0085 | 0.0078 | 0.0102 | 0.0375 | 0.0110 | 0.0109 | 0.0258 | 0.0098 | 0.0111 |
| 128 (384) | 0.1870 | 0.0101 | 0.0055 | 0.0055 | 0.0116 | 0.0203 | 0.0105 | 0.0102 | 0.0217 | 0.0094 | 0.0101 |
| 256 (768) | 0.1871 | 0.0098 | 0.0032 | 0.0035 | 0.0130 | 0.0136 | 0.0097 | 0.0096 | 0.0176 | 0.0089 | 0.0093 |
| 512 (1536) | 0.1872 | 0.0097 | 0.0018 | 0.0019 | 0.0155 | 0.0104 | 0.0090 | 0.0090 | 0.0180 | 0.0084 | 0.0087 |
| 1024 (3072) | 0.1871 | 0.0096 | 0.0011 | 0.0011 | 0.0194 | 0.0087 | 0.0084 | 0.0083 | 0.0181 | 0.0079 | 0.0081 |
| 2048 (6144) | 0.1869 | 0.0096 | 0.0007 | 0.0007 | 0.0253 | 0.0080 | 0.0079 | 0.0079 | 0.0185 | 0.0076 | 0.0078 |

### Held-out test RPS relative to Constant and Val-CE (n = 8, 32, 256, 2048)

| n | Judge | Raw RPS | RPS excess | Judge − Constant [95% CI] | Judge − ③ Val-CE [95% CI] | CE excess (nats) |
|---|---|---|---|---|---|---|
| 8 | Constant (train histogram) | 0.2056 | 0.0185 | +0.0000 [+0.0000, +0.0000] | +0.0045 [+0.0016, +0.0071] | 0.4102 |
| 8 | CE-trained, ridge 1 (④) | 0.1987 | 0.0116 | -0.0069 [-0.0085, -0.0055] | -0.0025 [-0.0049, -0.0005] | 0.1448 |
| 8 | CE-trained, val RPS ridge (③) | 0.2012 | 0.0141 | -0.0045 [-0.0071, -0.0017] | +0.0000 [+0.0000, +0.0000] | 0.1906 |
| 8 | Pearson-trained, ridge 1 (②) | 0.1967 | 0.0096 | -0.0089 [-0.0107, -0.0073] | -0.0044 [-0.0069, -0.0024] | 0.1291 |
| 8 | Pearson + naive head (1/n ridge) | 0.3233 | 0.1362 | +0.1177 [+0.0984, +0.1371] | +0.1222 [+0.1025, +0.1415] | 3.8029 |
| 8 | Pearson + tuned in-sample head | 0.2125 | 0.0254 | +0.0069 [+0.0024, +0.0125] | +0.0113 [+0.0059, +0.0178] | 0.4921 |
| 8 | Pearson + cross-fitted tuned head | 0.2116 | 0.0245 | +0.0060 [+0.0021, +0.0116] | +0.0105 [+0.0058, +0.0162] | 0.5793 |
| 8 | MSE-trained, ridge 1 | 0.2099 | 0.0228 | +0.0043 [+0.0025, +0.0060] | +0.0087 [+0.0062, +0.0111] | 0.1832 |
| 8 | MSE-trained, val RPS ridge | 0.2043 | 0.0172 | -0.0013 [-0.0043, +0.0019] | +0.0032 [+0.0001, +0.0063] | 0.1708 |
| 8 | MSE + cross-fitted tuned head | 0.2154 | 0.0283 | +0.0097 [+0.0065, +0.0132] | +0.0142 [+0.0102, +0.0184] | 0.5831 |
| 32 | Constant (train histogram) | 0.1987 | 0.0119 | +0.0000 [+0.0000, +0.0000] | +0.0023 [+0.0017, +0.0029] | 0.1553 |
| 32 | CE-trained, ridge 1 (④) | 0.1979 | 0.0111 | -0.0008 [-0.0014, -0.0002] | +0.0015 [+0.0011, +0.0019] | 0.1341 |
| 32 | CE-trained, val RPS ridge (③) | 0.1963 | 0.0096 | -0.0023 [-0.0029, -0.0017] | +0.0000 [+0.0000, +0.0000] | 0.1199 |
| 32 | Pearson-trained, ridge 1 (②) | 0.1966 | 0.0098 | -0.0021 [-0.0025, -0.0017] | +0.0003 [-0.0003, +0.0008] | 0.1358 |
| 32 | Pearson + naive head (1/n ridge) | 0.2772 | 0.0904 | +0.0785 [+0.0730, +0.0843] | +0.0809 [+0.0754, +0.0866] | 1.4658 |
| 32 | Pearson + tuned in-sample head | 0.1992 | 0.0124 | +0.0005 [+0.0001, +0.0011] | +0.0028 [+0.0020, +0.0037] | 0.1590 |
| 32 | Pearson + cross-fitted tuned head | 0.1993 | 0.0125 | +0.0006 [+0.0002, +0.0013] | +0.0030 [+0.0022, +0.0038] | 0.1666 |
| 32 | MSE-trained, ridge 1 | 0.2170 | 0.0302 | +0.0183 [+0.0168, +0.0199] | +0.0206 [+0.0193, +0.0221] | 0.2969 |
| 32 | MSE-trained, val RPS ridge | 0.1974 | 0.0106 | -0.0013 [-0.0019, -0.0004] | +0.0011 [+0.0005, +0.0018] | 0.1322 |
| 32 | MSE + cross-fitted tuned head | 0.1999 | 0.0131 | +0.0012 [+0.0006, +0.0020] | +0.0036 [+0.0028, +0.0044] | 0.1674 |
| 256 | Constant (train histogram) | 0.1969 | 0.0098 | +0.0000 [+0.0000, +0.0000] | +0.0064 [+0.0062, +0.0065] | 0.1284 |
| 256 | CE-trained, ridge 1 (④) | 0.1903 | 0.0032 | -0.0066 [-0.0068, -0.0064] | -0.0002 [-0.0003, -0.0001] | 0.0391 |
| 256 | CE-trained, val RPS ridge (③) | 0.1905 | 0.0035 | -0.0064 [-0.0065, -0.0062] | +0.0000 [+0.0000, +0.0000] | 0.0431 |
| 256 | Pearson-trained, ridge 1 (②) | 0.2000 | 0.0130 | +0.0031 [+0.0024, +0.0040] | +0.0095 [+0.0088, +0.0104] | 0.1916 |
| 256 | Pearson + naive head (1/n ridge) | 0.2006 | 0.0136 | +0.0037 [+0.0033, +0.0041] | +0.0101 [+0.0097, +0.0105] | 0.1484 |
| 256 | Pearson + tuned in-sample head | 0.1968 | 0.0097 | -0.0001 [-0.0003, +0.0001] | +0.0063 [+0.0061, +0.0065] | 0.1281 |
| 256 | Pearson + cross-fitted tuned head | 0.1967 | 0.0096 | -0.0002 [-0.0003, -0.0001] | +0.0062 [+0.0060, +0.0064] | 0.1276 |
| 256 | MSE-trained, ridge 1 | 0.2047 | 0.0176 | +0.0078 [+0.0071, +0.0085] | +0.0142 [+0.0135, +0.0149] | 0.3267 |
| 256 | MSE-trained, val RPS ridge | 0.1959 | 0.0089 | -0.0010 [-0.0011, -0.0008] | +0.0054 [+0.0053, +0.0055] | 0.1252 |
| 256 | MSE + cross-fitted tuned head | 0.1964 | 0.0093 | -0.0005 [-0.0006, -0.0004] | +0.0059 [+0.0057, +0.0061] | 0.1264 |
| 2048 | Constant (train histogram) | 0.1965 | 0.0096 | +0.0000 [+0.0000, +0.0000] | +0.0089 [+0.0088, +0.0089] | 0.1250 |
| 2048 | CE-trained, ridge 1 (④) | 0.1876 | 0.0007 | -0.0089 [-0.0089, -0.0088] | -0.0000 [-0.0000, -0.0000] | 0.0065 |
| 2048 | CE-trained, val RPS ridge (③) | 0.1876 | 0.0007 | -0.0089 [-0.0089, -0.0088] | +0.0000 [+0.0000, +0.0000] | 0.0066 |
| 2048 | Pearson-trained, ridge 1 (②) | 0.2122 | 0.0253 | +0.0157 [+0.0130, +0.0189] | +0.0246 [+0.0218, +0.0278] | 0.3961 |
| 2048 | Pearson + naive head (1/n ridge) | 0.1949 | 0.0080 | -0.0016 [-0.0016, -0.0015] | +0.0073 [+0.0073, +0.0074] | 0.1190 |
| 2048 | Pearson + tuned in-sample head | 0.1948 | 0.0079 | -0.0016 [-0.0016, -0.0016] | +0.0073 [+0.0072, +0.0073] | 0.1184 |
| 2048 | Pearson + cross-fitted tuned head | 0.1948 | 0.0079 | -0.0017 [-0.0017, -0.0016] | +0.0072 [+0.0071, +0.0073] | 0.1182 |
| 2048 | MSE-trained, ridge 1 | 0.2054 | 0.0185 | +0.0090 [+0.0082, +0.0098] | +0.0179 [+0.0170, +0.0187] | 0.4305 |
| 2048 | MSE-trained, val RPS ridge | 0.1945 | 0.0076 | -0.0020 [-0.0020, -0.0019] | +0.0069 [+0.0069, +0.0070] | 0.1184 |
| 2048 | MSE + cross-fitted tuned head | 0.1947 | 0.0078 | -0.0018 [-0.0018, -0.0017] | +0.0071 [+0.0070, +0.0071] | 0.1177 |

### Crossovers & small-n stability (n <= 32)

- ③ Val-CE significantly beats Pearson-trained, ridge 1 (②) at every larger n from: 64 tasks (192 ratings)
- ③ Val-CE significantly beats Pearson + naive head (1/n ridge) at every larger n from: 4 tasks (12 ratings)
- ③ Val-CE significantly beats Pearson + tuned in-sample head at every larger n from: 4 tasks (12 ratings)
- ③ Val-CE significantly beats Pearson + cross-fitted tuned head at every larger n from: 4 tasks (12 ratings)
- ③ Val-CE significantly beats MSE-trained, ridge 1 at every larger n from: 4 tasks (12 ratings)
- ③ Val-CE significantly beats MSE-trained, val RPS ridge at every larger n from: 8 tasks (24 ratings)
- ③ Val-CE significantly beats MSE + cross-fitted tuned head at every larger n from: 4 tasks (12 ratings)
- ③ Val-CE significantly beats Constant (train histogram) at every larger n from: 4 tasks (12 ratings)
- Max small-n (n <= 32) test RPS minus constant for Pearson + naive head (1/n ridge): +0.1470
- Max small-n (n <= 32) test RPS minus constant for Pearson + tuned in-sample head: +0.0161
- Max small-n (n <= 32) test RPS minus constant for Pearson + cross-fitted tuned head: +0.0165
- Max small-n (n <= 32) test RPS minus constant for MSE-trained, ridge 1: +0.0183
- Max small-n (n <= 32) test RPS minus constant for MSE-trained, val RPS ridge: -0.0009
- Max small-n (n <= 32) test RPS minus constant for MSE + cross-fitted tuned head: +0.0178

## Scenario B: nonlinear mean, disagreement varies (true test RPS ≈ 0.1362)

### Held-out test RPS excess over truth (mean over 100 seeds; 0 optimal, lower is better)

| n (ratings) | true RPS | Constant (train histogram) | CE-trained, ridge 1 (④) | CE-trained, val RPS ridge (③) | Pearson-trained, ridge 1 (②) | Pearson + naive head (1/n ridge) | Pearson + tuned in-sample head | Pearson + cross-fitted tuned head | MSE-trained, ridge 1 | MSE-trained, val RPS ridge | MSE + cross-fitted tuned head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 (12) | 0.1361 | 0.0923 | 0.0740 | 0.0829 | 0.0901 | 0.2199 | 0.1262 | 0.1310 | 0.0707 | 0.0803 | 0.1063 |
| 8 (24) | 0.1362 | 0.0803 | 0.0616 | 0.0706 | 0.0896 | 0.1844 | 0.1011 | 0.0967 | 0.0585 | 0.0706 | 0.0770 |
| 16 (48) | 0.1361 | 0.0733 | 0.0454 | 0.0519 | 0.0899 | 0.1926 | 0.0768 | 0.0761 | 0.0451 | 0.0518 | 0.0572 |
| 20 (60) | 0.1361 | 0.0727 | 0.0399 | 0.0455 | 0.0897 | 0.1585 | 0.0749 | 0.0747 | 0.0406 | 0.0467 | 0.0499 |
| 32 (96) | 0.1362 | 0.0705 | 0.0292 | 0.0304 | 0.0897 | 0.0986 | 0.0707 | 0.0702 | 0.0318 | 0.0345 | 0.0364 |
| 64 (192) | 0.1363 | 0.0682 | 0.0156 | 0.0160 | 0.0903 | 0.0319 | 0.0384 | 0.0446 | 0.0235 | 0.0237 | 0.0205 |
| 128 (384) | 0.1360 | 0.0678 | 0.0102 | 0.0099 | 0.0915 | 0.0171 | 0.0176 | 0.0190 | 0.0200 | 0.0176 | 0.0145 |
| 256 (768) | 0.1362 | 0.0671 | 0.0068 | 0.0064 | 0.0917 | 0.0119 | 0.0121 | 0.0125 | 0.0184 | 0.0136 | 0.0112 |
| 512 (1536) | 0.1363 | 0.0671 | 0.0052 | 0.0049 | 0.0918 | 0.0096 | 0.0096 | 0.0098 | 0.0164 | 0.0124 | 0.0090 |
| 1024 (3072) | 0.1359 | 0.0671 | 0.0043 | 0.0041 | 0.0927 | 0.0082 | 0.0081 | 0.0083 | 0.0161 | 0.0123 | 0.0078 |
| 2048 (6144) | 0.1363 | 0.0670 | 0.0037 | 0.0036 | 0.0951 | 0.0073 | 0.0073 | 0.0074 | 0.0172 | 0.0118 | 0.0071 |

### Held-out test RPS relative to Constant and Val-CE (n = 8, 32, 256, 2048)

| n | Judge | Raw RPS | RPS excess | Judge − Constant [95% CI] | Judge − ③ Val-CE [95% CI] | CE excess (nats) |
|---|---|---|---|---|---|---|
| 8 | Constant (train histogram) | 0.2165 | 0.0803 | +0.0000 [+0.0000, +0.0000] | +0.0097 [+0.0059, +0.0135] | 0.8829 |
| 8 | CE-trained, ridge 1 (④) | 0.1978 | 0.0616 | -0.0187 [-0.0216, -0.0160] | -0.0090 [-0.0119, -0.0064] | 0.3542 |
| 8 | CE-trained, val RPS ridge (③) | 0.2068 | 0.0706 | -0.0097 [-0.0135, -0.0059] | +0.0000 [+0.0000, +0.0000] | 0.5407 |
| 8 | Pearson-trained, ridge 1 (②) | 0.2259 | 0.0896 | +0.0093 [+0.0066, +0.0118] | +0.0191 [+0.0155, +0.0225] | 0.4819 |
| 8 | Pearson + naive head (1/n ridge) | 0.3206 | 0.1844 | +0.1041 [+0.0829, +0.1259] | +0.1138 [+0.0917, +0.1365] | 3.5021 |
| 8 | Pearson + tuned in-sample head | 0.2373 | 0.1011 | +0.0208 [+0.0104, +0.0333] | +0.0305 [+0.0189, +0.0435] | 1.2626 |
| 8 | Pearson + cross-fitted tuned head | 0.2329 | 0.0967 | +0.0164 [+0.0088, +0.0252] | +0.0261 [+0.0171, +0.0363] | 1.2518 |
| 8 | MSE-trained, ridge 1 | 0.1948 | 0.0585 | -0.0218 [-0.0250, -0.0186] | -0.0121 [-0.0150, -0.0092] | 0.3403 |
| 8 | MSE-trained, val RPS ridge | 0.2069 | 0.0706 | -0.0097 [-0.0135, -0.0057] | +0.0000 [-0.0024, +0.0024] | 0.4184 |
| 8 | MSE + cross-fitted tuned head | 0.2132 | 0.0770 | -0.0033 [-0.0065, +0.0001] | +0.0064 [+0.0024, +0.0103] | 0.9898 |
| 32 | Constant (train histogram) | 0.2067 | 0.0705 | +0.0000 [+0.0000, +0.0000] | +0.0401 [+0.0383, +0.0418] | 0.3854 |
| 32 | CE-trained, ridge 1 (④) | 0.1653 | 0.0292 | -0.0413 [-0.0429, -0.0398] | -0.0012 [-0.0022, -0.0003] | 0.1965 |
| 32 | CE-trained, val RPS ridge (③) | 0.1666 | 0.0304 | -0.0401 [-0.0418, -0.0383] | +0.0000 [+0.0000, +0.0000] | 0.2659 |
| 32 | Pearson-trained, ridge 1 (②) | 0.2259 | 0.0897 | +0.0192 [+0.0182, +0.0202] | +0.0593 [+0.0575, +0.0610] | 0.4834 |
| 32 | Pearson + naive head (1/n ridge) | 0.2348 | 0.0986 | +0.0281 [+0.0194, +0.0372] | +0.0682 [+0.0595, +0.0769] | 1.1537 |
| 32 | Pearson + tuned in-sample head | 0.2069 | 0.0707 | +0.0002 [-0.0020, +0.0027] | +0.0403 [+0.0376, +0.0431] | 0.4293 |
| 32 | Pearson + cross-fitted tuned head | 0.2063 | 0.0702 | -0.0004 [-0.0014, +0.0008] | +0.0398 [+0.0377, +0.0417] | 0.3993 |
| 32 | MSE-trained, ridge 1 | 0.1680 | 0.0318 | -0.0387 [-0.0407, -0.0367] | +0.0014 [+0.0001, +0.0027] | 0.2946 |
| 32 | MSE-trained, val RPS ridge | 0.1707 | 0.0345 | -0.0360 [-0.0379, -0.0340] | +0.0041 [+0.0026, +0.0057] | 0.2824 |
| 32 | MSE + cross-fitted tuned head | 0.1725 | 0.0364 | -0.0342 [-0.0364, -0.0318] | +0.0059 [+0.0038, +0.0081] | 0.2744 |
| 256 | Constant (train histogram) | 0.2033 | 0.0671 | +0.0000 [+0.0000, +0.0000] | +0.0606 [+0.0601, +0.0612] | 0.3496 |
| 256 | CE-trained, ridge 1 (④) | 0.1431 | 0.0068 | -0.0603 [-0.0608, -0.0597] | +0.0004 [+0.0003, +0.0005] | 0.0539 |
| 256 | CE-trained, val RPS ridge (③) | 0.1427 | 0.0064 | -0.0606 [-0.0612, -0.0601] | +0.0000 [+0.0000, +0.0000] | 0.0664 |
| 256 | Pearson-trained, ridge 1 (②) | 0.2280 | 0.0917 | +0.0247 [+0.0238, +0.0255] | +0.0853 [+0.0844, +0.0862] | 0.5119 |
| 256 | Pearson + naive head (1/n ridge) | 0.1482 | 0.0119 | -0.0551 [-0.0557, -0.0546] | +0.0055 [+0.0052, +0.0058] | 0.1348 |
| 256 | Pearson + tuned in-sample head | 0.1483 | 0.0121 | -0.0550 [-0.0556, -0.0544] | +0.0057 [+0.0054, +0.0060] | 0.1362 |
| 256 | Pearson + cross-fitted tuned head | 0.1487 | 0.0125 | -0.0546 [-0.0552, -0.0540] | +0.0061 [+0.0058, +0.0064] | 0.1357 |
| 256 | MSE-trained, ridge 1 | 0.1546 | 0.0184 | -0.0487 [-0.0495, -0.0480] | +0.0119 [+0.0113, +0.0125] | 0.3435 |
| 256 | MSE-trained, val RPS ridge | 0.1499 | 0.0136 | -0.0534 [-0.0540, -0.0529] | +0.0072 [+0.0070, +0.0074] | 0.1725 |
| 256 | MSE + cross-fitted tuned head | 0.1474 | 0.0112 | -0.0559 [-0.0564, -0.0553] | +0.0048 [+0.0045, +0.0050] | 0.1298 |
| 2048 | Constant (train histogram) | 0.2033 | 0.0670 | +0.0000 [+0.0000, +0.0000] | +0.0633 [+0.0629, +0.0638] | 0.3461 |
| 2048 | CE-trained, ridge 1 (④) | 0.1401 | 0.0037 | -0.0632 [-0.0637, -0.0628] | +0.0001 [+0.0001, +0.0001] | 0.0222 |
| 2048 | CE-trained, val RPS ridge (③) | 0.1400 | 0.0036 | -0.0633 [-0.0638, -0.0629] | +0.0000 [+0.0000, +0.0000] | 0.0224 |
| 2048 | Pearson-trained, ridge 1 (②) | 0.2314 | 0.0951 | +0.0281 [+0.0262, +0.0302] | +0.0915 [+0.0895, +0.0936] | 0.5957 |
| 2048 | Pearson + naive head (1/n ridge) | 0.1436 | 0.0073 | -0.0597 [-0.0602, -0.0592] | +0.0037 [+0.0036, +0.0038] | 0.1091 |
| 2048 | Pearson + tuned in-sample head | 0.1436 | 0.0073 | -0.0597 [-0.0602, -0.0592] | +0.0037 [+0.0036, +0.0038] | 0.1092 |
| 2048 | Pearson + cross-fitted tuned head | 0.1437 | 0.0074 | -0.0596 [-0.0601, -0.0591] | +0.0037 [+0.0036, +0.0038] | 0.1091 |
| 2048 | MSE-trained, ridge 1 | 0.1536 | 0.0172 | -0.0497 [-0.0505, -0.0490] | +0.0136 [+0.0130, +0.0142] | 0.5400 |
| 2048 | MSE-trained, val RPS ridge | 0.1481 | 0.0118 | -0.0552 [-0.0557, -0.0547] | +0.0082 [+0.0080, +0.0083] | 0.1972 |
| 2048 | MSE + cross-fitted tuned head | 0.1434 | 0.0071 | -0.0599 [-0.0604, -0.0594] | +0.0034 [+0.0033, +0.0035] | 0.1077 |

### Crossovers & small-n stability (n <= 32)

- ③ Val-CE significantly beats Pearson-trained, ridge 1 (②) at every larger n from: 4 tasks (12 ratings)
- ③ Val-CE significantly beats Pearson + naive head (1/n ridge) at every larger n from: 4 tasks (12 ratings)
- ③ Val-CE significantly beats Pearson + tuned in-sample head at every larger n from: 4 tasks (12 ratings)
- ③ Val-CE significantly beats Pearson + cross-fitted tuned head at every larger n from: 4 tasks (12 ratings)
- ③ Val-CE significantly beats MSE-trained, ridge 1 at every larger n from: 32 tasks (96 ratings)
- ③ Val-CE significantly beats MSE-trained, val RPS ridge at every larger n from: 32 tasks (96 ratings)
- ③ Val-CE significantly beats MSE + cross-fitted tuned head at every larger n from: 4 tasks (12 ratings)
- ③ Val-CE significantly beats Constant (train histogram) at every larger n from: 4 tasks (12 ratings)
- Max small-n (n <= 32) test RPS minus constant for Pearson + naive head (1/n ridge): +0.1276
- Max small-n (n <= 32) test RPS minus constant for Pearson + tuned in-sample head: +0.0339
- Max small-n (n <= 32) test RPS minus constant for Pearson + cross-fitted tuned head: +0.0387
- Max small-n (n <= 32) test RPS minus constant for MSE-trained, ridge 1: -0.0216
- Max small-n (n <= 32) test RPS minus constant for MSE-trained, val RPS ridge: -0.0097
- Max small-n (n <= 32) test RPS minus constant for MSE + cross-fitted tuned head: +0.0141

## Scenario C: nonlinear mean, constant concentration κ (true test RPS ≈ 0.1273)

### Held-out test RPS excess over truth (mean over 100 seeds; 0 optimal, lower is better)

| n (ratings) | true RPS | Constant (train histogram) | CE-trained, ridge 1 (④) | CE-trained, val RPS ridge (③) | Pearson-trained, ridge 1 (②) | Pearson + naive head (1/n ridge) | Pearson + tuned in-sample head | Pearson + cross-fitted tuned head | MSE-trained, ridge 1 | MSE-trained, val RPS ridge | MSE + cross-fitted tuned head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 (12) | 0.1272 | 0.0877 | 0.0682 | 0.0776 | 0.0830 | 0.1822 | 0.1232 | 0.1102 | 0.0637 | 0.0729 | 0.1093 |
| 8 (24) | 0.1274 | 0.0788 | 0.0572 | 0.0669 | 0.0824 | 0.1717 | 0.0969 | 0.0914 | 0.0531 | 0.0642 | 0.0776 |
| 16 (48) | 0.1274 | 0.0689 | 0.0434 | 0.0482 | 0.0824 | 0.1729 | 0.0722 | 0.0735 | 0.0401 | 0.0460 | 0.0482 |
| 20 (60) | 0.1275 | 0.0685 | 0.0394 | 0.0434 | 0.0824 | 0.1382 | 0.0720 | 0.0714 | 0.0389 | 0.0419 | 0.0409 |
| 32 (96) | 0.1273 | 0.0674 | 0.0290 | 0.0322 | 0.0827 | 0.0843 | 0.0661 | 0.0670 | 0.0297 | 0.0317 | 0.0284 |
| 64 (192) | 0.1274 | 0.0644 | 0.0163 | 0.0156 | 0.0825 | 0.0236 | 0.0298 | 0.0350 | 0.0213 | 0.0210 | 0.0134 |
| 128 (384) | 0.1274 | 0.0643 | 0.0097 | 0.0085 | 0.0838 | 0.0101 | 0.0101 | 0.0113 | 0.0182 | 0.0150 | 0.0078 |
| 256 (768) | 0.1272 | 0.0640 | 0.0065 | 0.0054 | 0.0842 | 0.0062 | 0.0061 | 0.0063 | 0.0165 | 0.0126 | 0.0053 |
| 512 (1536) | 0.1276 | 0.0639 | 0.0044 | 0.0037 | 0.0823 | 0.0041 | 0.0041 | 0.0041 | 0.0154 | 0.0118 | 0.0034 |
| 1024 (3072) | 0.1272 | 0.0631 | 0.0034 | 0.0030 | 0.0843 | 0.0028 | 0.0028 | 0.0028 | 0.0145 | 0.0117 | 0.0024 |
| 2048 (6144) | 0.1270 | 0.0640 | 0.0032 | 0.0029 | 0.0893 | 0.0022 | 0.0021 | 0.0022 | 0.0154 | 0.0116 | 0.0019 |

### Held-out test RPS relative to Constant and Val-CE (n = 8, 32, 256, 2048)

| n | Judge | Raw RPS | RPS excess | Judge − Constant [95% CI] | Judge − ③ Val-CE [95% CI] | CE excess (nats) |
|---|---|---|---|---|---|---|
| 8 | Constant (train histogram) | 0.2062 | 0.0788 | +0.0000 [+0.0000, +0.0000] | +0.0119 [+0.0075, +0.0167] | 0.9209 |
| 8 | CE-trained, ridge 1 (④) | 0.1846 | 0.0572 | -0.0216 [-0.0265, -0.0176] | -0.0097 [-0.0130, -0.0066] | 0.3087 |
| 8 | CE-trained, val RPS ridge (③) | 0.1943 | 0.0669 | -0.0119 [-0.0168, -0.0074] | +0.0000 [+0.0000, +0.0000] | 0.4968 |
| 8 | Pearson-trained, ridge 1 (②) | 0.2099 | 0.0824 | +0.0036 [-0.0016, +0.0078] | +0.0156 [+0.0118, +0.0191] | 0.3855 |
| 8 | Pearson + naive head (1/n ridge) | 0.2992 | 0.1717 | +0.0929 [+0.0740, +0.1117] | +0.1049 [+0.0861, +0.1236] | 3.6782 |
| 8 | Pearson + tuned in-sample head | 0.2243 | 0.0969 | +0.0181 [+0.0082, +0.0296] | +0.0300 [+0.0192, +0.0422] | 1.4402 |
| 8 | Pearson + cross-fitted tuned head | 0.2189 | 0.0914 | +0.0126 [+0.0061, +0.0210] | +0.0246 [+0.0167, +0.0336] | 1.2282 |
| 8 | MSE-trained, ridge 1 | 0.1806 | 0.0531 | -0.0257 [-0.0303, -0.0218] | -0.0137 [-0.0169, -0.0108] | 0.2941 |
| 8 | MSE-trained, val RPS ridge | 0.1917 | 0.0642 | -0.0146 [-0.0197, -0.0099] | -0.0026 [-0.0054, +0.0003] | 0.3705 |
| 8 | MSE + cross-fitted tuned head | 0.2050 | 0.0776 | -0.0012 [-0.0060, +0.0038] | +0.0107 [+0.0047, +0.0169] | 1.0537 |
| 32 | Constant (train histogram) | 0.1947 | 0.0674 | +0.0000 [+0.0000, +0.0000] | +0.0351 [+0.0333, +0.0369] | 0.3380 |
| 32 | CE-trained, ridge 1 (④) | 0.1563 | 0.0290 | -0.0384 [-0.0398, -0.0369] | -0.0032 [-0.0044, -0.0021] | 0.2082 |
| 32 | CE-trained, val RPS ridge (③) | 0.1595 | 0.0322 | -0.0351 [-0.0369, -0.0333] | +0.0000 [+0.0000, +0.0000] | 0.3158 |
| 32 | Pearson-trained, ridge 1 (②) | 0.2100 | 0.0827 | +0.0153 [+0.0143, +0.0163] | +0.0505 [+0.0485, +0.0524] | 0.3887 |
| 32 | Pearson + naive head (1/n ridge) | 0.2116 | 0.0843 | +0.0169 [+0.0081, +0.0261] | +0.0520 [+0.0439, +0.0609] | 0.9792 |
| 32 | Pearson + tuned in-sample head | 0.1934 | 0.0661 | -0.0013 [-0.0038, +0.0013] | +0.0339 [+0.0312, +0.0366] | 0.3587 |
| 32 | Pearson + cross-fitted tuned head | 0.1943 | 0.0670 | -0.0004 [-0.0025, +0.0019] | +0.0347 [+0.0322, +0.0373] | 0.3512 |
| 32 | MSE-trained, ridge 1 | 0.1570 | 0.0297 | -0.0377 [-0.0397, -0.0356] | -0.0025 [-0.0037, -0.0013] | 0.2818 |
| 32 | MSE-trained, val RPS ridge | 0.1590 | 0.0317 | -0.0357 [-0.0375, -0.0339] | -0.0005 [-0.0017, +0.0006] | 0.2622 |
| 32 | MSE + cross-fitted tuned head | 0.1557 | 0.0284 | -0.0390 [-0.0412, -0.0366] | -0.0039 [-0.0057, -0.0019] | 0.1918 |
| 256 | Constant (train histogram) | 0.1912 | 0.0640 | +0.0000 [+0.0000, +0.0000] | +0.0586 [+0.0580, +0.0591] | 0.3060 |
| 256 | CE-trained, ridge 1 (④) | 0.1336 | 0.0065 | -0.0575 [-0.0581, -0.0570] | +0.0010 [+0.0009, +0.0012] | 0.0562 |
| 256 | CE-trained, val RPS ridge (③) | 0.1326 | 0.0054 | -0.0586 [-0.0591, -0.0580] | +0.0000 [+0.0000, +0.0000] | 0.0691 |
| 256 | Pearson-trained, ridge 1 (②) | 0.2114 | 0.0842 | +0.0203 [+0.0195, +0.0210] | +0.0788 [+0.0781, +0.0795] | 0.4074 |
| 256 | Pearson + naive head (1/n ridge) | 0.1334 | 0.0062 | -0.0578 [-0.0584, -0.0572] | +0.0008 [+0.0005, +0.0010] | 0.0408 |
| 256 | Pearson + tuned in-sample head | 0.1333 | 0.0061 | -0.0578 [-0.0585, -0.0572] | +0.0007 [+0.0005, +0.0010] | 0.0405 |
| 256 | Pearson + cross-fitted tuned head | 0.1334 | 0.0063 | -0.0577 [-0.0584, -0.0571] | +0.0008 [+0.0006, +0.0011] | 0.0410 |
| 256 | MSE-trained, ridge 1 | 0.1437 | 0.0165 | -0.0474 [-0.0482, -0.0467] | +0.0111 [+0.0107, +0.0115] | 0.3293 |
| 256 | MSE-trained, val RPS ridge | 0.1398 | 0.0126 | -0.0514 [-0.0519, -0.0508] | +0.0072 [+0.0071, +0.0074] | 0.1772 |
| 256 | MSE + cross-fitted tuned head | 0.1325 | 0.0053 | -0.0587 [-0.0593, -0.0581] | -0.0001 [-0.0003, +0.0000] | 0.0353 |
| 2048 | Constant (train histogram) | 0.1910 | 0.0640 | +0.0000 [+0.0000, +0.0000] | +0.0612 [+0.0606, +0.0617] | 0.3040 |
| 2048 | CE-trained, ridge 1 (④) | 0.1302 | 0.0032 | -0.0609 [-0.0614, -0.0604] | +0.0003 [+0.0003, +0.0003] | 0.0200 |
| 2048 | CE-trained, val RPS ridge (③) | 0.1299 | 0.0029 | -0.0612 [-0.0617, -0.0607] | +0.0000 [+0.0000, +0.0000] | 0.0202 |
| 2048 | Pearson-trained, ridge 1 (②) | 0.2163 | 0.0893 | +0.0252 [+0.0237, +0.0268] | +0.0864 [+0.0848, +0.0880] | 0.5733 |
| 2048 | Pearson + naive head (1/n ridge) | 0.1292 | 0.0022 | -0.0619 [-0.0624, -0.0613] | -0.0007 [-0.0008, -0.0007] | 0.0141 |
| 2048 | Pearson + tuned in-sample head | 0.1291 | 0.0021 | -0.0619 [-0.0624, -0.0614] | -0.0007 [-0.0008, -0.0007] | 0.0141 |
| 2048 | Pearson + cross-fitted tuned head | 0.1292 | 0.0022 | -0.0619 [-0.0624, -0.0613] | -0.0007 [-0.0008, -0.0006] | 0.0141 |
| 2048 | MSE-trained, ridge 1 | 0.1424 | 0.0154 | -0.0487 [-0.0493, -0.0480] | +0.0125 [+0.0121, +0.0129] | 0.4382 |
| 2048 | MSE-trained, val RPS ridge | 0.1386 | 0.0116 | -0.0525 [-0.0530, -0.0520] | +0.0087 [+0.0086, +0.0088] | 0.1868 |
| 2048 | MSE + cross-fitted tuned head | 0.1289 | 0.0019 | -0.0621 [-0.0627, -0.0616] | -0.0010 [-0.0011, -0.0009] | 0.0126 |

### Crossovers & small-n stability (n <= 32)

- ③ Val-CE significantly beats Pearson-trained, ridge 1 (②) at every larger n from: 4 tasks (12 ratings)
- ③ Val-CE significantly beats Pearson + naive head (1/n ridge) at every larger n from: not at every larger n
- ③ Val-CE significantly beats Pearson + tuned in-sample head at every larger n from: not at every larger n
- ③ Val-CE significantly beats Pearson + cross-fitted tuned head at every larger n from: not at every larger n
- ③ Val-CE significantly beats MSE-trained, ridge 1 at every larger n from: 64 tasks (192 ratings)
- ③ Val-CE significantly beats MSE-trained, val RPS ridge at every larger n from: 64 tasks (192 ratings)
- ③ Val-CE significantly beats MSE + cross-fitted tuned head at every larger n from: not at every larger n
- ③ Val-CE significantly beats Constant (train histogram) at every larger n from: 4 tasks (12 ratings)
- Max small-n (n <= 32) test RPS minus constant for Pearson + naive head (1/n ridge): +0.1040
- Max small-n (n <= 32) test RPS minus constant for Pearson + tuned in-sample head: +0.0355
- Max small-n (n <= 32) test RPS minus constant for Pearson + cross-fitted tuned head: +0.0225
- Max small-n (n <= 32) test RPS minus constant for MSE-trained, ridge 1: -0.0240
- Max small-n (n <= 32) test RPS minus constant for MSE-trained, val RPS ridge: -0.0146
- Max small-n (n <= 32) test RPS minus constant for MSE + cross-fitted tuned head: +0.0216

## Scenario D: linear mean, constant concentration κ (true test RPS ≈ 0.1464)

### Held-out test RPS excess over truth (mean over 100 seeds; 0 optimal, lower is better)

| n (ratings) | true RPS | Constant (train histogram) | CE-trained, ridge 1 (④) | CE-trained, val RPS ridge (③) | Pearson-trained, ridge 1 (②) | Pearson + naive head (1/n ridge) | Pearson + tuned in-sample head | Pearson + cross-fitted tuned head | MSE-trained, ridge 1 | MSE-trained, val RPS ridge | MSE + cross-fitted tuned head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 (12) | 0.1464 | 0.0654 | 0.0410 | 0.0481 | 0.0452 | 0.1389 | 0.0824 | 0.0799 | 0.0423 | 0.0444 | 0.0732 |
| 8 (24) | 0.1463 | 0.0563 | 0.0372 | 0.0458 | 0.0453 | 0.1661 | 0.0703 | 0.0701 | 0.0414 | 0.0451 | 0.0571 |
| 16 (48) | 0.1465 | 0.0517 | 0.0307 | 0.0367 | 0.0450 | 0.1528 | 0.0515 | 0.0534 | 0.0366 | 0.0379 | 0.0421 |
| 20 (60) | 0.1465 | 0.0499 | 0.0278 | 0.0327 | 0.0450 | 0.1328 | 0.0513 | 0.0504 | 0.0352 | 0.0327 | 0.0346 |
| 32 (96) | 0.1465 | 0.0480 | 0.0216 | 0.0238 | 0.0451 | 0.0781 | 0.0488 | 0.0478 | 0.0291 | 0.0256 | 0.0236 |
| 64 (192) | 0.1464 | 0.0466 | 0.0144 | 0.0152 | 0.0454 | 0.0258 | 0.0310 | 0.0351 | 0.0243 | 0.0179 | 0.0135 |
| 128 (384) | 0.1464 | 0.0457 | 0.0091 | 0.0092 | 0.0457 | 0.0109 | 0.0114 | 0.0117 | 0.0212 | 0.0136 | 0.0081 |
| 256 (768) | 0.1464 | 0.0455 | 0.0056 | 0.0052 | 0.0462 | 0.0057 | 0.0057 | 0.0059 | 0.0180 | 0.0119 | 0.0047 |
| 512 (1536) | 0.1463 | 0.0455 | 0.0041 | 0.0038 | 0.0476 | 0.0037 | 0.0037 | 0.0037 | 0.0165 | 0.0115 | 0.0030 |
| 1024 (3072) | 0.1463 | 0.0454 | 0.0034 | 0.0032 | 0.0493 | 0.0027 | 0.0027 | 0.0027 | 0.0167 | 0.0116 | 0.0022 |
| 2048 (6144) | 0.1465 | 0.0449 | 0.0028 | 0.0027 | 0.0523 | 0.0017 | 0.0017 | 0.0018 | 0.0183 | 0.0104 | 0.0015 |

### Held-out test RPS relative to Constant and Val-CE (n = 8, 32, 256, 2048)

| n | Judge | Raw RPS | RPS excess | Judge − Constant [95% CI] | Judge − ③ Val-CE [95% CI] | CE excess (nats) |
|---|---|---|---|---|---|---|
| 8 | Constant (train histogram) | 0.2027 | 0.0563 | +0.0000 [+0.0000, +0.0000] | +0.0105 [+0.0080, +0.0131] | 0.5714 |
| 8 | CE-trained, ridge 1 (④) | 0.1835 | 0.0372 | -0.0192 [-0.0214, -0.0171] | -0.0086 [-0.0108, -0.0066] | 0.2223 |
| 8 | CE-trained, val RPS ridge (③) | 0.1921 | 0.0458 | -0.0105 [-0.0132, -0.0081] | +0.0000 [+0.0000, +0.0000] | 0.3672 |
| 8 | Pearson-trained, ridge 1 (②) | 0.1917 | 0.0453 | -0.0110 [-0.0134, -0.0089] | -0.0005 [-0.0029, +0.0016] | 0.2182 |
| 8 | Pearson + naive head (1/n ridge) | 0.3124 | 0.1661 | +0.1098 [+0.0893, +0.1297] | +0.1203 [+0.0998, +0.1408] | 3.5355 |
| 8 | Pearson + tuned in-sample head | 0.2167 | 0.0703 | +0.0140 [+0.0067, +0.0229] | +0.0246 [+0.0173, +0.0332] | 0.8893 |
| 8 | Pearson + cross-fitted tuned head | 0.2164 | 0.0701 | +0.0137 [+0.0064, +0.0224] | +0.0243 [+0.0169, +0.0332] | 0.9264 |
| 8 | MSE-trained, ridge 1 | 0.1877 | 0.0414 | -0.0150 [-0.0173, -0.0127] | -0.0044 [-0.0069, -0.0021] | 0.2415 |
| 8 | MSE-trained, val RPS ridge | 0.1915 | 0.0451 | -0.0112 [-0.0140, -0.0085] | -0.0007 [-0.0032, +0.0023] | 0.2530 |
| 8 | MSE + cross-fitted tuned head | 0.2035 | 0.0571 | +0.0008 [-0.0016, +0.0034] | +0.0113 [+0.0080, +0.0149] | 0.6942 |
| 32 | Constant (train histogram) | 0.1945 | 0.0480 | +0.0000 [+0.0000, +0.0000] | +0.0241 [+0.0228, +0.0255] | 0.2490 |
| 32 | CE-trained, ridge 1 (④) | 0.1681 | 0.0216 | -0.0264 [-0.0274, -0.0254] | -0.0023 [-0.0034, -0.0012] | 0.1759 |
| 32 | CE-trained, val RPS ridge (③) | 0.1703 | 0.0238 | -0.0241 [-0.0255, -0.0228] | +0.0000 [+0.0000, +0.0000] | 0.2113 |
| 32 | Pearson-trained, ridge 1 (②) | 0.1916 | 0.0451 | -0.0029 [-0.0035, -0.0023] | +0.0213 [+0.0199, +0.0226] | 0.2193 |
| 32 | Pearson + naive head (1/n ridge) | 0.2246 | 0.0781 | +0.0302 [+0.0232, +0.0374] | +0.0543 [+0.0475, +0.0614] | 0.9709 |
| 32 | Pearson + tuned in-sample head | 0.1953 | 0.0488 | +0.0009 [-0.0014, +0.0035] | +0.0250 [+0.0223, +0.0280] | 0.2832 |
| 32 | Pearson + cross-fitted tuned head | 0.1943 | 0.0478 | -0.0002 [-0.0012, +0.0009] | +0.0240 [+0.0222, +0.0257] | 0.2645 |
| 32 | MSE-trained, ridge 1 | 0.1756 | 0.0291 | -0.0188 [-0.0203, -0.0172] | +0.0053 [+0.0037, +0.0069] | 0.2851 |
| 32 | MSE-trained, val RPS ridge | 0.1721 | 0.0256 | -0.0224 [-0.0236, -0.0210] | +0.0017 [+0.0005, +0.0031] | 0.1874 |
| 32 | MSE + cross-fitted tuned head | 0.1701 | 0.0236 | -0.0244 [-0.0260, -0.0227] | -0.0002 [-0.0017, +0.0012] | 0.1568 |
| 256 | Constant (train histogram) | 0.1919 | 0.0455 | +0.0000 [+0.0000, +0.0000] | +0.0402 [+0.0399, +0.0406] | 0.2205 |
| 256 | CE-trained, ridge 1 (④) | 0.1520 | 0.0056 | -0.0399 [-0.0403, -0.0395] | +0.0004 [+0.0003, +0.0004] | 0.0526 |
| 256 | CE-trained, val RPS ridge (③) | 0.1517 | 0.0052 | -0.0402 [-0.0406, -0.0399] | +0.0000 [+0.0000, +0.0000] | 0.0630 |
| 256 | Pearson-trained, ridge 1 (②) | 0.1926 | 0.0462 | +0.0007 [+0.0006, +0.0009] | +0.0410 [+0.0406, +0.0414] | 0.2331 |
| 256 | Pearson + naive head (1/n ridge) | 0.1522 | 0.0057 | -0.0397 [-0.0402, -0.0393] | +0.0005 [+0.0003, +0.0008] | 0.0352 |
| 256 | Pearson + tuned in-sample head | 0.1522 | 0.0057 | -0.0397 [-0.0402, -0.0393] | +0.0005 [+0.0003, +0.0008] | 0.0356 |
| 256 | Pearson + cross-fitted tuned head | 0.1523 | 0.0059 | -0.0396 [-0.0400, -0.0391] | +0.0007 [+0.0004, +0.0009] | 0.0367 |
| 256 | MSE-trained, ridge 1 | 0.1645 | 0.0180 | -0.0275 [-0.0280, -0.0268] | +0.0128 [+0.0123, +0.0133] | 0.3461 |
| 256 | MSE-trained, val RPS ridge | 0.1583 | 0.0119 | -0.0336 [-0.0339, -0.0332] | +0.0067 [+0.0065, +0.0068] | 0.1585 |
| 256 | MSE + cross-fitted tuned head | 0.1511 | 0.0047 | -0.0408 [-0.0412, -0.0404] | -0.0005 [-0.0007, -0.0004] | 0.0303 |
| 2048 | Constant (train histogram) | 0.1914 | 0.0449 | +0.0000 [+0.0000, +0.0000] | +0.0422 [+0.0419, +0.0426] | 0.2162 |
| 2048 | CE-trained, ridge 1 (④) | 0.1493 | 0.0028 | -0.0421 [-0.0425, -0.0418] | +0.0001 [+0.0001, +0.0001] | 0.0180 |
| 2048 | CE-trained, val RPS ridge (③) | 0.1492 | 0.0027 | -0.0422 [-0.0426, -0.0419] | +0.0000 [+0.0000, +0.0000] | 0.0183 |
| 2048 | Pearson-trained, ridge 1 (②) | 0.1989 | 0.0523 | +0.0074 [+0.0063, +0.0087] | +0.0497 [+0.0485, +0.0510] | 0.3687 |
| 2048 | Pearson + naive head (1/n ridge) | 0.1483 | 0.0017 | -0.0432 [-0.0435, -0.0428] | -0.0009 [-0.0010, -0.0009] | 0.0106 |
| 2048 | Pearson + tuned in-sample head | 0.1483 | 0.0017 | -0.0432 [-0.0435, -0.0428] | -0.0009 [-0.0010, -0.0009] | 0.0107 |
| 2048 | Pearson + cross-fitted tuned head | 0.1483 | 0.0018 | -0.0432 [-0.0435, -0.0428] | -0.0009 [-0.0010, -0.0008] | 0.0107 |
| 2048 | MSE-trained, ridge 1 | 0.1648 | 0.0183 | -0.0266 [-0.0273, -0.0260] | +0.0156 [+0.0151, +0.0162] | 0.5286 |
| 2048 | MSE-trained, val RPS ridge | 0.1569 | 0.0104 | -0.0345 [-0.0349, -0.0342] | +0.0077 [+0.0077, +0.0077] | 0.1434 |
| 2048 | MSE + cross-fitted tuned head | 0.1480 | 0.0015 | -0.0434 [-0.0438, -0.0431] | -0.0012 [-0.0013, -0.0011] | 0.0090 |

### Crossovers & small-n stability (n <= 32)

- ③ Val-CE significantly beats Pearson-trained, ridge 1 (②) at every larger n from: 16 tasks (48 ratings)
- ③ Val-CE significantly beats Pearson + naive head (1/n ridge) at every larger n from: not at every larger n
- ③ Val-CE significantly beats Pearson + tuned in-sample head at every larger n from: not at every larger n
- ③ Val-CE significantly beats Pearson + cross-fitted tuned head at every larger n from: not at every larger n
- ③ Val-CE significantly beats MSE-trained, ridge 1 at every larger n from: 20 tasks (60 ratings)
- ③ Val-CE significantly beats MSE-trained, val RPS ridge at every larger n from: 32 tasks (96 ratings)
- ③ Val-CE significantly beats MSE + cross-fitted tuned head at every larger n from: not at every larger n
- ③ Val-CE significantly beats Constant (train histogram) at every larger n from: 4 tasks (12 ratings)
- Max small-n (n <= 32) test RPS minus constant for Pearson + naive head (1/n ridge): +0.1098
- Max small-n (n <= 32) test RPS minus constant for Pearson + tuned in-sample head: +0.0171
- Max small-n (n <= 32) test RPS minus constant for Pearson + cross-fitted tuned head: +0.0145
- Max small-n (n <= 32) test RPS minus constant for MSE-trained, ridge 1: -0.0148
- Max small-n (n <= 32) test RPS minus constant for MSE-trained, val RPS ridge: -0.0112
- Max small-n (n <= 32) test RPS minus constant for MSE + cross-fitted tuned head: +0.0079
