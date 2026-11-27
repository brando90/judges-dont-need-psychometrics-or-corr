### summeval_expert

SummEval expert annotations on four aspects (coherence, consistency, fluency, relevance), native grades 1–5 stored as class values [1,2,3,4,5]. Every summary has [3] expert annotation records. The jsonl does not publish a stable rater id, so rater_id is the within-item vote index, not a person. Split group is the source article (filepath), so every system and aspect of one article stays in one split. Binary views use the pre-registered cut grade ≥ 4.

Split by prompt group, seed 0: train 3840, validation 1280, test 1280. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 1.210 [1.171, 1.247] | 0.749 [0.731, 0.764] | 0.615 [0.595, 0.634] | 0.1402 | 0.253 | 0.331 | 0.238 |
| brier-trained | 1.213 [1.174, 1.251] | 0.750 [0.733, 0.766] | 0.617 [0.597, 0.636] | 0.1408 | 0.256 | 0.345 | 0.247 |
| rps-trained | 1.213 [1.174, 1.251] | 0.750 [0.732, 0.765] | 0.617 [0.597, 0.636] | 0.1405 | 0.245 | 0.309 | 0.223 |
| pearson-trained | 4.731 [4.509, 4.961] | 0.998 [0.994, 1.002] | 0.905 [0.883, 0.929] | 0.2490 | 0.225 | 0.272 | 0.196 |
| mse-trained | +inf [undefined, undefined] | 0.902 [0.887, 0.918] | 1.291 [1.266, 1.316] | 0.2035 | 0.220 | 0.274 | 0.197 |

Uniform forecast held-out test CE 1.6094. Train-marginal forecast held-out test CE 1.2210. CE head refit after shuffling train features: held-out test CE 1.2217.

### summeval_expert_binary

Binary view of summeval_expert: class 1 iff the original value ≥ 4. This is a threshold of the same ratings, not a separately collected binary task. SummEval expert annotations on four aspects (coherence, consistency, fluency, relevance), native grades 1–5 stored as class values [1,2,3,4,5]. Every summary has [3] expert annotation records. The jsonl does not publish a stable rater id, so rater_id is the within-item vote index, not a person. Split group is the source article (filepath), so every system and aspect of one article stays in one split. Binary views use the pre-registered cut grade ≥ 4.

Split by prompt group, seed 0: train 3840, validation 1280, test 1280. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.575 [0.556, 0.593] | 0.440 [0.430, 0.449] | 0.387 [0.370, 0.404] | 0.1934 | 0.200 | 0.254 | 0.195 |
| brier-trained | 0.576 [0.558, 0.593] | 0.440 [0.431, 0.450] | 0.388 [0.371, 0.404] | 0.1939 | 0.200 | 0.253 | 0.194 |
| rps-trained | 0.578 [0.560, 0.596] | 0.441 [0.432, 0.451] | 0.390 [0.373, 0.407] | 0.1948 | 0.201 | 0.254 | 0.195 |
| pearson-trained | 0.689 [0.686, 0.693] | 0.498 [0.496, 0.500] | 0.496 [0.492, 0.500] | 0.2481 | 0.202 | 0.257 | 0.197 |
| mse-trained | 0.567 [0.549, 0.585] | 0.436 [0.427, 0.446] | 0.381 [0.364, 0.398] | 0.1904 | 0.193 | 0.247 | 0.189 |

Uniform forecast held-out test CE 0.6931. Train-marginal forecast held-out test CE 0.5819. CE head refit after shuffling train features: held-out test CE 0.5814.

### chaosnli_snli_mnli

Pooled ChaosNLI-SNLI and ChaosNLI-MNLI-matched. ChaosNLI SNLI or MNLI-matched, 3 classes, 100 votes per item in the author files. label_count order is entailment, neutral, contradiction. Ranked probability score uses that order as an imposed order (entailment < neutral < contradiction); it is not a native Likert scale. rater_id is a within-item vote index expanded from the counts, not a person. Each uid is its own split group. Item counts by file are kept distinct by the group prefix.

Split by prompt group, seed 0: train 1867, validation 622, test 624. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 1.053 [1.040, 1.068] | 0.627 [0.619, 0.634] | 0.637 [0.629, 0.647] | 0.1963 | 0.078 | 0.111 | 0.074 |
| brier-trained | 1.053 [1.040, 1.068] | 0.627 [0.619, 0.634] | 0.638 [0.629, 0.647] | 0.1964 | 0.081 | 0.114 | 0.076 |
| rps-trained | 1.054 [1.041, 1.068] | 0.627 [0.619, 0.634] | 0.638 [0.630, 0.647] | 0.1965 | 0.082 | 0.114 | 0.076 |
| pearson-trained | 4.022 [3.867, 4.176] | 0.711 [0.707, 0.715] | 0.940 [0.919, 0.962] | 0.2530 | 0.086 | 0.118 | 0.078 |
| mse-trained | +inf [undefined, undefined] | 0.694 [0.682, 0.706] | 0.903 [0.872, 0.934] | 0.2409 | 0.063 | 0.079 | 0.052 |

Uniform forecast held-out test CE 1.0986. Train-marginal forecast held-out test CE 1.0569. CE head refit after shuffling train features: held-out test CE 1.0554.

### chaosnli_alpha

ChaosNLI abductive NLI, native two-option labels. The release stores counts, not person ids; rater_id is the within-item vote index obtained by expanding label_count. Order is option 1 then option 2. Each item is its own split group.

Split by prompt group, seed 0: train 919, validation 306, test 307. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.696 [0.692, 0.699] | 0.501 [0.500, 0.503] | 0.502 [0.499, 0.506] | 0.2512 | -0.057 | -0.064 | -0.045 |
| brier-trained | 0.696 [0.692, 0.699] | 0.501 [0.500, 0.503] | 0.502 [0.499, 0.506] | 0.2512 | -0.058 | -0.064 | -0.045 |
| rps-trained | 0.694 [0.692, 0.696] | 0.501 [0.500, 0.502] | 0.501 [0.499, 0.503] | 0.2505 | -0.049 | -0.057 | -0.040 |
| pearson-trained | 0.696 [0.689, 0.703] | 0.501 [0.498, 0.505] | 0.503 [0.496, 0.510] | 0.2514 | -0.034 | -0.049 | -0.036 |
| mse-trained | 0.708 [0.699, 0.718] | 0.507 [0.503, 0.512] | 0.515 [0.506, 0.524] | 0.2573 | -0.083 | -0.087 | -0.060 |

Uniform forecast held-out test CE 0.6931. Train-marginal forecast held-out test CE 0.6930. CE head refit after shuffling train features: held-out test CE 0.6916.

### dices990

DICES-990 rater-level Q_overall. Yes means unsafe (DICES-990 README; DICES-350 uses the same question family). Order for ranked probability score: No < Unsure < Yes, values 0,1,2. Dropped 7217 rows whose rater_id is in the README quality-filter list (12 raters). rater_id is the released person id. Split group is item_id. Raters are crossed, so the same person can appear in train and test on different conversations; no conversation's raters are split.

Split by prompt group, seed 0: train 594, validation 198, test 198. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.743 [0.708, 0.778] | 0.628 [0.608, 0.647] | 0.442 [0.415, 0.469] | 0.1969 | 0.127 | 0.280 | 0.189 |
| brier-trained | 0.744 [0.710, 0.778] | 0.628 [0.608, 0.647] | 0.443 [0.416, 0.470] | 0.1972 | 0.124 | 0.278 | 0.187 |
| rps-trained | 0.744 [0.710, 0.779] | 0.628 [0.608, 0.647] | 0.443 [0.416, 0.470] | 0.1972 | 0.124 | 0.276 | 0.186 |
| pearson-trained | 1.164 [1.118, 1.210] | 0.717 [0.709, 0.726] | 0.565 [0.552, 0.578] | 0.2572 | 0.110 | 0.255 | 0.172 |
| mse-trained | +inf [undefined, undefined] | 0.720 [0.699, 0.739] | 0.825 [0.781, 0.866] | 0.2593 | 0.150 | 0.312 | 0.210 |

Uniform forecast held-out test CE 1.0986. Train-marginal forecast held-out test CE 0.7488. CE head refit after shuffling train features: held-out test CE 0.7492.

### dices990_binary_yes

Binary view of dices990: class 1 iff the original value ≥ 2. This is a threshold of the same ratings, not a separately collected binary task. DICES-990 rater-level Q_overall. Yes means unsafe (DICES-990 README; DICES-350 uses the same question family). Order for ranked probability score: No < Unsure < Yes, values 0,1,2. Dropped 7217 rows whose rater_id is in the README quality-filter list (12 raters). rater_id is the released person id. Split group is item_id. Raters are crossed, so the same person can appear in train and test on different conversations; no conversation's raters are split.

Split by prompt group, seed 0: train 594, validation 198, test 198. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.557 [0.527, 0.586] | 0.431 [0.415, 0.446] | 0.371 [0.344, 0.397] | 0.1856 | 0.127 | 0.293 | 0.196 |
| brier-trained | 0.558 [0.528, 0.587] | 0.431 [0.415, 0.446] | 0.372 [0.345, 0.398] | 0.1858 | 0.125 | 0.291 | 0.196 |
| rps-trained | 0.560 [0.530, 0.589] | 0.432 [0.416, 0.447] | 0.373 [0.346, 0.400] | 0.1865 | 0.118 | 0.278 | 0.185 |
| pearson-trained | 0.710 [0.698, 0.722] | 0.508 [0.502, 0.514] | 0.516 [0.505, 0.529] | 0.2582 | 0.110 | 0.270 | 0.180 |
| mse-trained | 0.552 [0.521, 0.583] | 0.428 [0.412, 0.443] | 0.367 [0.340, 0.393] | 0.1835 | 0.149 | 0.323 | 0.216 |

Uniform forecast held-out test CE 0.6931. Train-marginal forecast held-out test CE 0.5627. CE head refit after shuffling train features: held-out test CE 0.5631.

### popquorn_offensiveness

POPQUORN offensiveness (Jiaxin-Pei/Potato-Prolific-Dataset@dbb118c2), native grades 1–5 stored as class values [1,2,3,4,5]. Dropped 0 invalid rows. Split group is instance_id; raters are crossed across instances. Binary view uses cut grade >= 4.

Split by prompt group, seed 0: train 900, validation 300, test 300. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 1.298 [1.242, 1.361] | 0.799 [0.773, 0.825] | 0.650 [0.620, 0.683] | 0.1595 | 0.139 | 0.247 | 0.172 |
| brier-trained | 1.299 [1.243, 1.363] | 0.799 [0.773, 0.826] | 0.651 [0.620, 0.684] | 0.1597 | 0.126 | 0.236 | 0.164 |
| rps-trained | 1.299 [1.242, 1.362] | 0.799 [0.773, 0.825] | 0.651 [0.620, 0.684] | 0.1596 | 0.143 | 0.248 | 0.171 |
| pearson-trained | 4.996 [4.695, 5.312] | 1.004 [0.999, 1.010] | 0.907 [0.878, 0.937] | 0.2523 | 0.144 | 0.244 | 0.167 |
| mse-trained | +inf [undefined, undefined] | 0.939 [0.916, 0.963] | 1.257 [1.218, 1.295] | 0.2204 | 0.141 | 0.209 | 0.143 |

Uniform forecast held-out test CE 1.6094. Train-marginal forecast held-out test CE 1.3025. CE head refit after shuffling train features: held-out test CE 1.3030.

### popquorn_offensiveness_binary

Binary view of popquorn_offensiveness: class 1 iff the original value ≥ 4. This is a threshold of the same ratings, not a separately collected binary task. POPQUORN offensiveness (Jiaxin-Pei/Potato-Prolific-Dataset@dbb118c2), native grades 1–5 stored as class values [1,2,3,4,5]. Dropped 0 invalid rows. Split group is instance_id; raters are crossed across instances. Binary view uses cut grade >= 4.

Split by prompt group, seed 0: train 900, validation 300, test 300. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.432 [0.392, 0.471] | 0.362 [0.340, 0.382] | 0.262 [0.231, 0.292] | 0.1310 | 0.093 | 0.144 | 0.103 |
| brier-trained | 0.433 [0.393, 0.472] | 0.362 [0.340, 0.382] | 0.262 [0.231, 0.293] | 0.1311 | 0.088 | 0.131 | 0.095 |
| rps-trained | 0.433 [0.393, 0.472] | 0.362 [0.340, 0.383] | 0.262 [0.231, 0.293] | 0.1312 | 0.086 | 0.129 | 0.093 |
| pearson-trained | 0.669 [0.664, 0.674] | 0.488 [0.485, 0.491] | 0.476 [0.471, 0.481] | 0.2382 | 0.084 | 0.126 | 0.091 |
| mse-trained | 0.429 [0.390, 0.467] | 0.361 [0.339, 0.381] | 0.260 [0.230, 0.290] | 0.1300 | 0.102 | 0.164 | 0.117 |

Uniform forecast held-out test CE 0.6931. Train-marginal forecast held-out test CE 0.4337. CE head refit after shuffling train features: held-out test CE 0.4340.

### popquorn_politeness

POPQUORN politeness (Jiaxin-Pei/Potato-Prolific-Dataset@dbb118c2), native grades 1–5 stored as class values [1,2,3,4,5]. Dropped 0 invalid rows. Split group is instance_id; raters are crossed across instances. Binary view uses cut grade >= 4.

Split by prompt group, seed 0: train 2230, validation 743, test 745. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 1.528 [1.513, 1.543] | 0.828 [0.818, 0.838] | 0.768 [0.763, 0.774] | 0.1714 | 0.338 | 0.473 | 0.330 |
| brier-trained | 1.544 [1.529, 1.558] | 0.835 [0.826, 0.846] | 0.774 [0.768, 0.779] | 0.1744 | 0.333 | 0.465 | 0.323 |
| rps-trained | 1.533 [1.520, 1.547] | 0.829 [0.820, 0.839] | 0.770 [0.765, 0.775] | 0.1719 | 0.337 | 0.471 | 0.328 |
| pearson-trained | 9.547 [9.316, 9.764] | 0.983 [0.975, 0.991] | 1.167 [1.145, 1.187] | 0.2417 | 0.328 | 0.459 | 0.319 |
| mse-trained | +inf [undefined, undefined] | 0.900 [0.885, 0.916] | 1.082 [1.059, 1.107] | 0.2025 | 0.362 | 0.509 | 0.357 |

Uniform forecast held-out test CE 1.6094. Train-marginal forecast held-out test CE 1.5598. CE head refit after shuffling train features: held-out test CE 1.5603.

### popquorn_politeness_binary

Binary view of popquorn_politeness: class 1 iff the original value ≥ 4. This is a threshold of the same ratings, not a separately collected binary task. POPQUORN politeness (Jiaxin-Pei/Potato-Prolific-Dataset@dbb118c2), native grades 1–5 stored as class values [1,2,3,4,5]. Dropped 0 invalid rows. Split group is instance_id; raters are crossed across instances. Binary view uses cut grade >= 4.

Split by prompt group, seed 0: train 2230, validation 743, test 745. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.652 [0.645, 0.659] | 0.479 [0.476, 0.483] | 0.460 [0.453, 0.466] | 0.2298 | 0.314 | 0.468 | 0.332 |
| brier-trained | 0.653 [0.646, 0.660] | 0.480 [0.476, 0.483] | 0.460 [0.454, 0.467] | 0.2302 | 0.314 | 0.468 | 0.332 |
| rps-trained | 0.663 [0.658, 0.668] | 0.485 [0.482, 0.487] | 0.470 [0.465, 0.475] | 0.2350 | 0.308 | 0.460 | 0.326 |
| pearson-trained | 0.646 [0.636, 0.657] | 0.477 [0.472, 0.482] | 0.455 [0.445, 0.465] | 0.2273 | 0.298 | 0.447 | 0.317 |
| mse-trained | +inf [0.620, 0.645] | 0.471 [0.464, 0.477] | 0.443 [0.431, 0.455] | 0.2215 | 0.326 | 0.483 | 0.343 |

Uniform forecast held-out test CE 0.6931. Train-marginal forecast held-out test CE 0.6889. CE head refit after shuffling train features: held-out test CE 0.6907.

### veribench_fsc_rv1

VeriBench functional semantic correctness (FSC) R_v1 (all5) from brando90/veribench@d1c47355 Exp 100 (75 artifacts, 25 tasks, 0–5 scale). Split group is task so all 3 samples per task stay together. Binary view uses cut score >= 4.

Split by prompt group, seed 0: train 45, validation 15, test 15. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 1.737 [1.685, 1.786] | 0.952 [0.866, 1.034] | 0.819 [0.800, 0.837] | 0.1813 | 0.594 | 0.774 | 0.585 |
| brier-trained | 1.724 [1.685, 1.763] | 0.944 [0.861, 1.023] | 0.815 [0.801, 0.830] | 0.1781 | 0.633 | 0.874 | 0.723 |
| rps-trained | 1.736 [1.659, 1.813] | 0.942 [0.874, 1.008] | 0.812 [0.784, 0.839] | 0.1775 | 0.672 | 0.841 | 0.644 |
| pearson-trained | 11.011 [8.179, 13.693] | 1.072 [1.007, 1.136] | 1.137 [0.923, 1.352] | 0.2297 | 0.328 | 0.410 | 0.248 |
| mse-trained | +inf [undefined, undefined] | 1.133 [1.024, 1.242] | 1.425 [1.248, 1.594] | 0.2568 | 0.445 | 0.446 | 0.327 |

Uniform forecast held-out test CE 1.7918. Train-marginal forecast held-out test CE 1.8229. CE head refit after shuffling train features: held-out test CE 1.7678.

### veribench_fsc_rv1_expert3

VeriBench functional semantic correctness (FSC) R_v1 (expert3) from brando90/veribench@d1c47355 Exp 100 (75 artifacts, 25 tasks, 0–5 scale). Split group is task so all 3 samples per task stay together. Binary view uses cut score >= 4.

Split by prompt group, seed 0: train 45, validation 15, test 15. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 1.928 [1.775, 2.087] | 1.076 [0.938, 1.205] | 0.887 [0.836, 0.939] | 0.2314 | 0.386 | 0.694 | 0.478 |
| brier-trained | 1.822 [1.719, 1.914] | 1.002 [0.889, 1.112] | 0.848 [0.812, 0.881] | 0.2008 | 0.582 | 0.640 | 0.418 |
| rps-trained | 4.163 [2.803, 5.766] | 1.015 [0.914, 1.113] | 0.871 [0.801, 0.948] | 0.2061 | 0.525 | 0.620 | 0.378 |
| pearson-trained | 12.280 [9.463, 14.673] | 1.587 [1.501, 1.668] | 1.645 [1.513, 1.768] | 0.5038 | 0.561 | 0.676 | 0.458 |
| mse-trained | +inf [undefined, undefined] | 1.091 [0.949, 1.224] | 1.241 [1.079, 1.411] | 0.2381 | 0.361 | 0.426 | 0.359 |

Uniform forecast held-out test CE 1.7918. Train-marginal forecast held-out test CE 2.1268. CE head refit after shuffling train features: held-out test CE 2.0858.

### veribench_fsc_rv1_binary

Binary view of veribench_fsc_rv1: class 1 iff the original value ≥ 4. This is a threshold of the same ratings, not a separately collected binary task. VeriBench functional semantic correctness (FSC) R_v1 (all5) from brando90/veribench@d1c47355 Exp 100 (75 artifacts, 25 tasks, 0–5 scale). Split group is task so all 3 samples per task stay together. Binary view uses cut score >= 4.

Split by prompt group, seed 0: train 45, validation 15, test 15. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.766 [0.666, 0.873] | 0.534 [0.487, 0.581] | 0.571 [0.475, 0.675] | 0.2854 | 0.631 | 0.780 | 0.612 |
| brier-trained | 0.750 [0.662, 0.845] | 0.527 [0.485, 0.570] | 0.556 [0.471, 0.649] | 0.2780 | 0.614 | 0.727 | 0.545 |
| rps-trained | 0.771 [0.675, 0.876] | 0.537 [0.491, 0.582] | 0.577 [0.483, 0.678] | 0.2883 | 0.619 | 0.750 | 0.567 |
| pearson-trained | 1.071 [0.861, 1.291] | 0.630 [0.552, 0.706] | 0.794 [0.608, 0.996] | 0.3972 | 0.531 | 0.742 | 0.567 |
| mse-trained | 0.701 [0.618, 0.809] | 0.503 [0.463, 0.548] | 0.505 [0.429, 0.601] | 0.2525 | 0.462 | 0.765 | 0.589 |

Uniform forecast held-out test CE 0.6931. Train-marginal forecast held-out test CE 0.8799. CE head refit after shuffling train features: held-out test CE 0.8556.

### veribench_fsc_native_main

VeriBench FSC native_main (native-heavy-11-11-2026) synced 11-26-2026 from trustworthy-theorem-judges@460466dd Exp 09 (100 artifacts, 25 tasks, 0–5 scale, 250 non-coordinator numeric ratings across 6 active raters; 2950 blank/N/A rows excluded and never treated as 0; T30 HTTP 401 unverified). Split group is task_folder.

Split by prompt group, seed 0: train 60, validation 20, test 20. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 1.267 [0.810, 1.787] | 0.740 [0.530, 0.953] | 0.667 [0.436, 0.923] | 0.1096 | 0.362 | 0.382 | 0.282 |
| brier-trained | 1.629 [0.995, 2.389] | 0.732 [0.527, 0.940] | 0.651 [0.431, 0.896] | 0.1072 | 0.396 | 0.505 | 0.362 |
| rps-trained | 3.084 [1.546, 4.870] | 0.737 [0.544, 0.929] | 0.677 [0.454, 0.932] | 0.1087 | 0.271 | 0.178 | 0.075 |
| pearson-trained | 8.238 [5.261, 11.498] | 0.864 [0.626, 1.090] | 0.938 [0.594, 1.304] | 0.1494 | -0.062 | -0.019 | -0.017 |
| mse-trained | +inf [undefined, undefined] | 0.847 [0.670, 1.030] | 0.992 [0.728, 1.255] | 0.1435 | -0.095 | -0.134 | -0.098 |

Uniform forecast held-out test CE 1.7918. Train-marginal forecast held-out test CE 1.3927. CE head refit after shuffling train features: held-out test CE 1.4319.

### veribench_fsc_pooled

Pooled VeriBench FSC R_v1 (75 items, 25 tasks, 345 ratings) and live-synced native_main (100 items, 25 disjoint tasks, 250 ratings) = 175 items, 50 tasks, 595 ratings on 0–5 scale. Caveat: native_main has 74.0% ratings at 5 (mean 4.59) vs R_v1 mean 2.97, so pooled results mix two rating regimes.

Split by prompt group, seed 0: train 107, validation 34, test 34. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 1.276 [1.043, 1.511] | 0.810 [0.707, 0.920] | 0.597 [0.487, 0.701] | 0.1314 | 0.315 | 0.462 | 0.338 |
| brier-trained | 1.296 [1.064, 1.531] | 0.815 [0.715, 0.920] | 0.606 [0.499, 0.708] | 0.1329 | 0.334 | 0.515 | 0.372 |
| rps-trained | 2.216 [1.570, 2.972] | 0.824 [0.722, 0.930] | 0.642 [0.515, 0.764] | 0.1357 | 0.305 | 0.437 | 0.323 |
| pearson-trained | 6.869 [5.010, 8.554] | 1.131 [1.010, 1.254] | 0.910 [0.715, 1.100] | 0.2560 | 0.284 | 0.486 | 0.349 |
| mse-trained | +inf [undefined, undefined] | 0.991 [0.888, 1.096] | 1.210 [1.067, 1.352] | 0.1966 | 0.305 | 0.381 | 0.274 |

Uniform forecast held-out test CE 1.7918. Train-marginal forecast held-out test CE 1.3635. CE head refit after shuffling train features: held-out test CE 1.3810.

### veribench_fsc_pooled_binary

Binary view of veribench_fsc_pooled: class 1 iff the original value ≥ 4. This is a threshold of the same ratings, not a separately collected binary task. Pooled VeriBench FSC R_v1 (75 items, 25 tasks, 345 ratings) and live-synced native_main (100 items, 25 disjoint tasks, 250 ratings) = 175 items, 50 tasks, 595 ratings on 0–5 scale. Caveat: native_main has 74.0% ratings at 5 (mean 4.59) vs R_v1 mean 2.97, so pooled results mix two rating regimes.

Split by prompt group, seed 0: train 107, validation 34, test 34. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.554 [0.438, 0.678] | 0.433 [0.369, 0.492] | 0.375 [0.273, 0.484] | 0.1874 | 0.319 | 0.420 | 0.335 |
| brier-trained | 0.558 [0.461, 0.662] | 0.434 [0.380, 0.485] | 0.376 [0.288, 0.471] | 0.1882 | 0.329 | 0.404 | 0.322 |
| rps-trained | 0.564 [0.472, 0.663] | 0.436 [0.384, 0.485] | 0.380 [0.295, 0.470] | 0.1900 | 0.327 | 0.417 | 0.335 |
| pearson-trained | 0.641 [0.531, 0.756] | 0.477 [0.422, 0.529] | 0.455 [0.357, 0.559] | 0.2274 | 0.317 | 0.423 | 0.339 |
| mse-trained | 0.551 [0.441, 0.663] | 0.437 [0.382, 0.491] | 0.382 [0.292, 0.482] | 0.1912 | 0.334 | 0.406 | 0.314 |

Uniform forecast held-out test CE 0.6931. Train-marginal forecast held-out test CE 0.6042. CE head refit after shuffling train features: held-out test CE 0.5773.
