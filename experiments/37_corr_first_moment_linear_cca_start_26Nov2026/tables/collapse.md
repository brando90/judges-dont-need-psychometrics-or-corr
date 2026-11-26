# Within-prompt collapse

Positive Pearson difference: collapsed conditional mean correlates more with the held-out ratings than the raw ratings do. Positive Brier, CE, or RPS difference: the floored point mass is worse than the add-one probability mass function. Half-split needs at least 4 ratings. Leave-one-rater-out needs at least 3.

| Dataset | Design | Split | Items | Pearson collapsed minus spread | Brier point minus add-one | CE point minus add-one | RPS point minus add-one |
| --- | --- | --- | --- | --- | --- | --- | --- |
| summeval_expert | half-split | held-out test | 0 | unavailable | unavailable | unavailable | unavailable |
| summeval_expert | leave-one-rater-out | held-out test | 1280 | 0.0578 [0.0520, 0.0631] p-val=< 0.0010 | 0.2081 [0.1626, 0.2532] p-val=< 0.0010 | 4.4677 [4.1100, 4.8012] p-val=< 0.0010 | -0.1093 [-0.1426, -0.0761] p-val=< 0.0010 |
| summeval_expert | half-split | full collection | 0 | unavailable | unavailable | unavailable | unavailable |
| summeval_expert | leave-one-rater-out | full collection | 6400 | 0.0572 [0.0547, 0.0598] p-val=< 0.0010 | 0.2155 [0.1975, 0.2360] p-val=< 0.0010 | 4.5164 [4.3624, 4.6791] p-val=< 0.0010 | -0.1137 [-0.1271, -0.1004] p-val=< 0.0010 |
| summeval_crowd | half-split | held-out test | 1280 | 0.0805 [0.0592, 0.1013] p-val=< 0.0010 | 0.1030 [0.0700, 0.1370] p-val=< 0.0010 | 3.6146 [3.3243, 3.9063] p-val=< 0.0010 | -0.0221 [-0.0457, 0.0016] p-val=0.0700 |
| summeval_crowd | leave-one-rater-out | held-out test | 1280 | 0.1295 [0.1191, 0.1396] p-val=< 0.0010 | 0.1371 [0.0985, 0.1789] p-val=< 0.0010 | 3.3915 [3.0656, 3.7068] p-val=< 0.0010 | 0.0032 [-0.0186, 0.0260] p-val=0.8000 |
| summeval_crowd | half-split | full collection | 6400 | 0.0694 [0.0600, 0.0784] p-val=< 0.0010 | 0.0794 [0.0648, 0.0940] p-val=< 0.0010 | 3.4303 [3.2973, 3.5575] p-val=< 0.0010 | -0.0441 [-0.0553, -0.0340] p-val=< 0.0010 |
| summeval_crowd | leave-one-rater-out | full collection | 6400 | 0.1349 [0.1299, 0.1398] p-val=< 0.0010 | 0.1291 [0.1101, 0.1462] p-val=< 0.0010 | 3.3140 [3.1682, 3.4518] p-val=< 0.0010 | -0.0023 [-0.0119, 0.0074] p-val=0.6360 |
| chaosnli_snli_mnli | half-split | held-out test | 624 | 0.2013 [0.1911, 0.2118] p-val=< 0.0010 | 0.2522 [0.2317, 0.2749] p-val=< 0.0010 | 3.9026 [3.7022, 4.1139] p-val=< 0.0010 | 0.1132 [0.1038, 0.1221] p-val=< 0.0010 |
| chaosnli_snli_mnli | leave-one-rater-out | held-out test | 624 | 0.1978 [0.1837, 0.2126] p-val=< 0.0010 | 0.2741 [0.2213, 0.3303] p-val=< 0.0010 | 4.1964 [3.7108, 4.7215] p-val=< 0.0010 | 0.1241 [0.0978, 0.1500] p-val=< 0.0010 |
| chaosnli_snli_mnli | half-split | full collection | 3113 | 0.2071 [0.2026, 0.2121] p-val=< 0.0010 | 0.2491 [0.2391, 0.2591] p-val=< 0.0010 | 3.8741 [3.7778, 3.9625] p-val=< 0.0010 | 0.1121 [0.1077, 0.1162] p-val=< 0.0010 |
| chaosnli_snli_mnli | leave-one-rater-out | full collection | 3113 | 0.2102 [0.2042, 0.2166] p-val=< 0.0010 | 0.2475 [0.2232, 0.2714] p-val=< 0.0010 | 3.7976 [3.5893, 4.0198] p-val=< 0.0010 | 0.1105 [0.0998, 0.1212] p-val=< 0.0010 |
| chaosnli_alpha | half-split | held-out test | 307 | 0.1598 [0.1439, 0.1779] p-val=< 0.0010 | 0.0623 [0.0485, 0.0753] p-val=< 0.0010 | 1.3941 [1.1989, 1.5814] p-val=< 0.0010 | 0.0311 [0.0243, 0.0386] p-val=< 0.0010 |
| chaosnli_alpha | leave-one-rater-out | held-out test | 307 | 0.1648 [0.1466, 0.1838] p-val=< 0.0010 | 0.0616 [0.0238, 0.1015] p-val=< 0.0010 | 1.2822 [0.8393, 1.6978] p-val=< 0.0010 | 0.0308 [0.0126, 0.0519] p-val=< 0.0010 |
| chaosnli_alpha | half-split | full collection | 1532 | 0.1527 [0.1457, 0.1599] p-val=< 0.0010 | 0.0617 [0.0555, 0.0682] p-val=< 0.0010 | 1.3631 [1.2826, 1.4531] p-val=< 0.0010 | 0.0308 [0.0276, 0.0340] p-val=< 0.0010 |
| chaosnli_alpha | leave-one-rater-out | full collection | 1532 | 0.1545 [0.1474, 0.1623] p-val=< 0.0010 | 0.0688 [0.0526, 0.0891] p-val=< 0.0010 | 1.4150 [1.2102, 1.6236] p-val=< 0.0010 | 0.0344 [0.0260, 0.0439] p-val=< 0.0010 |
| dices990 | half-split | held-out test | 198 | 0.2229 [0.1952, 0.2486] p-val=< 0.0010 | 0.5719 [0.4787, 0.6597] p-val=< 0.0010 | 5.8821 [5.2150, 6.5738] p-val=< 0.0010 | 0.2173 [0.1874, 0.2499] p-val=< 0.0010 |
| dices990 | leave-one-rater-out | held-out test | 198 | 0.2285 [0.1669, 0.2862] p-val=< 0.0010 | 0.5115 [0.3891, 0.6226] p-val=< 0.0010 | 5.5218 [4.6508, 6.4942] p-val=< 0.0010 | 0.1946 [0.1442, 0.2487] p-val=< 0.0010 |
| dices990 | half-split | full collection | 990 | 0.2324 [0.2214, 0.2436] p-val=< 0.0010 | 0.6067 [0.5654, 0.6510] p-val=< 0.0010 | 6.1273 [5.8065, 6.4522] p-val=< 0.0010 | 0.2291 [0.2161, 0.2428] p-val=< 0.0010 |
| dices990 | leave-one-rater-out | full collection | 990 | 0.2278 [0.1977, 0.2565] p-val=< 0.0010 | 0.6154 [0.5658, 0.6728] p-val=< 0.0010 | 6.2459 [5.8405, 6.6493] p-val=< 0.0010 | 0.2250 [0.2033, 0.2461] p-val=< 0.0010 |
| veribench_fsc_exp09 | half-split | held-out test | 1 | unavailable | unavailable | unavailable | unavailable |
| veribench_fsc_exp09 | leave-one-rater-out | held-out test | 13 | 0.1283 [0.0718, 0.3564] p-val=< 0.0010 | -0.2208 [-0.5192, 0.1995] p-val=0.3080 | 1.0329 [-1.0744, 4.1148] p-val=0.4540 | -0.6364 [-0.8091, -0.4158] p-val=< 0.0010 |
| veribench_fsc_exp09 | half-split | full collection | 2 | unavailable | unavailable | unavailable | unavailable |
| veribench_fsc_exp09 | leave-one-rater-out | full collection | 48 | 0.0251 [0.0133, 0.0601] p-val=< 0.0010 | 0.0340 [-0.1711, 0.2632] p-val=0.7560 | 3.0820 [1.4431, 4.7946] p-val=< 0.0010 | -0.5107 [-0.6378, -0.3622] p-val=< 0.0010 |
