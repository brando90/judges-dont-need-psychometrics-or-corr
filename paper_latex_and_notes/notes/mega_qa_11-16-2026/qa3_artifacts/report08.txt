# Generated results: experiment 08

Synthetic data: n training tasks with three sampled ratings each; 500 held-out test tasks per seed, scored against the true distribution. 100 seeds per scenario and size. Earlier A/B tasks are reused. Undefined test correlations count as zero: C, n=4 has 99 defined seeds and 1 undefined per Pearson variant; raw missingness is preserved.

Ranked probability score (RPS) excess is the mean squared cumulative-probability gap to truth (0 to 1; zero optimal). Cross-entropy (CE) excess is Kullback–Leibler divergence KL(true || judge), in nats (0 upward; zero optimal). Both are lower-is-better. Pearson and Spearman compare expected ratings with the true mean on test tasks (−1 to 1; higher is better).

Same 78-weight softmax family; train objectives and (c/n)||W||^2 ridge. L-BFGS-B: 400 iterations / 600 evaluations; log-histogram start plus Gaussian weight noise (sd 0.01). Exact CE is convex; refits preserve printed means and crossovers, but one ridge share changes (HOW_TO_READ_RESULTS.md); the other objectives have no global-optimum certificate. Ridge c is 1 or selected from {0.01, 0.1, 1, 10, 100} by pooled four-fold training-task validation; test tasks do not enter selection.

Uncertainty: paired 95% percentile bootstrap over seeds, 10,000 draws; no p-values. Intervals are pointwise. Crossovers refer to every larger tested size, not an unobserved sample size or a simultaneous-coverage guarantee.


Cells with all 100 seeds in every scenario: n = [4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048].


## Scenario A: mean nearly constant, disagreement varies

Held-out test RPS excess (mean over 100 seeds):

| n (ratings) | constant (training histogram) | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|---|
| 4 (12) | 0.0267 | 0.0106 | 0.0140 | 0.0121 | 0.0096 | 0.0097 | 0.0098 |
| 8 (24) | 0.0185 | 0.0116 | 0.0141 | 0.0140 | 0.0096 | 0.0111 | 0.0117 |
| 16 (48) | 0.0147 | 0.0123 | 0.0117 | 0.0140 | 0.0096 | 0.0113 | 0.0110 |
| 20 (60) | 0.0125 | 0.0113 | 0.0104 | 0.0118 | 0.0097 | 0.0116 | 0.0120 |
| 32 (96) | 0.0119 | 0.0111 | 0.0096 | 0.0109 | 0.0098 | 0.0208 | 0.0108 |
| 64 (192) | 0.0107 | 0.0085 | 0.0078 | 0.0089 | 0.0102 | 0.0255 | 0.0096 |
| 128 (384) | 0.0101 | 0.0055 | 0.0055 | 0.0070 | 0.0116 | 0.0418 | 0.0096 |
| 256 (768) | 0.0098 | 0.0032 | 0.0035 | 0.0041 | 0.0130 | 0.0553 | 0.0095 |
| 512 (1536) | 0.0097 | 0.0018 | 0.0019 | 0.0021 | 0.0155 | 0.0613 | 0.0095 |
| 1024 (3072) | 0.0096 | 0.0011 | 0.0011 | 0.0012 | 0.0194 | 0.0790 | 0.0095 |
| 2048 (6144) | 0.0096 | 0.0007 | 0.0007 | 0.0007 | 0.0253 | 0.0807 | 0.0096 |

Crossovers on held-out test RPS (smallest n from which the first judge is significantly better at every larger n):
- CE-trained, ridge 1 (④) vs Pearson-trained, ridge 1 (②): 64 tasks (192 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge by validation Pearson (①): 32 tasks (96 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge 1 (②): 64 tasks (192 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge by validation RPS: 20 tasks (60 ratings)
- CE-trained, ridge by validation RPS (③) vs constant (training histogram): 4 tasks (12 ratings)
- Pearson-trained, ridge 1 (②) vs constant (training histogram): not at every larger n

H2 check, sizes where the validation-tuned CE judge (③) is significantly WORSE on test RPS than: Pearson-trained, ridge 1 (②): [4, 8, 16]; Pearson-trained, ridge by validation Pearson (①): [4]; Pearson-trained, ridge by validation RPS: [4]
H1 check, sizes where the Pearson-trained judge (②) is significantly BETTER on test RPS than the CE-trained judge (④): [4, 8, 16, 20, 32]
H2 check, sizes where the CE-trained judge (④) has significantly LOWER test Pearson correlation with the true mean than: Pearson-trained, ridge 1 (②): none; Pearson-trained, ridge by validation Pearson (①): none

Test Pearson correlation with the true mean (mean over 100 seeds; undefined counted as 0):

| n | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|
| 4 | 0.148 | 0.144 | 0.145 | -0.013 | 0.056 | 0.012 |
| 8 | 0.170 | 0.166 | 0.167 | 0.057 | 0.070 | 0.034 |
| 16 | 0.264 | 0.267 | 0.260 | 0.022 | -0.014 | 0.005 |
| 20 | 0.259 | 0.252 | 0.238 | 0.035 | 0.054 | 0.055 |
| 32 | 0.320 | 0.328 | 0.320 | 0.095 | 0.108 | 0.133 |
| 64 | 0.398 | 0.413 | 0.402 | 0.173 | 0.168 | 0.230 |
| 128 | 0.529 | 0.536 | 0.530 | 0.309 | 0.318 | 0.376 |
| 256 | 0.677 | 0.683 | 0.670 | 0.489 | 0.482 | 0.563 |
| 512 | 0.771 | 0.773 | 0.762 | 0.619 | 0.597 | 0.683 |
| 1024 | 0.847 | 0.848 | 0.846 | 0.750 | 0.707 | 0.802 |
| 2048 | 0.891 | 0.892 | 0.897 | 0.845 | 0.802 | 0.882 |

Distance from the constant forecast (mean test total variation distance), small n:

| n | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|
| 4 | 0.232 | 0.243 | 0.237 | 0.248 | 0.250 | 0.253 |
| 8 | 0.187 | 0.199 | 0.191 | 0.187 | 0.190 | 0.192 |
| 16 | 0.187 | 0.159 | 0.159 | 0.143 | 0.150 | 0.148 |
| 20 | 0.188 | 0.137 | 0.134 | 0.123 | 0.142 | 0.141 |
| 32 | 0.206 | 0.129 | 0.118 | 0.116 | 0.191 | 0.122 |
| 64 | 0.212 | 0.115 | 0.096 | 0.118 | 0.231 | 0.089 |

Chosen ridge c (most common value; share of seeds), and Pearson (①) fallbacks to c = 1:

| n | CE ③ | RPS (val RPS) | Pearson ① | Pearson (val RPS) | ① fallbacks |
|---|---|---|---|---|---|
| 4 | 100 (56%) | 100 (52%) | 1 (26%) | 0.01 (41%) | 0 |
| 8 | 100 (41%) | 100 (47%) | 0.1 (25%) | 0.01 (33%) | 0 |
| 16 | 1 (43%) | 100 (37%) | 0.1 (24%) | 0.1 (34%) | 0 |
| 20 | 10 (36%) | 100 (40%) | 100 (28%) | 0.1 (31%) | 0 |
| 32 | 10 (52%) | 1 (39%) | 10 (26%) | 100 (31%) | 0 |
| 64 | 10 (63%) | 1 (53%) | 10 (30%) | 100 (38%) | 0 |
| 128 | 10 (59%) | 1 (57%) | 0.01 (29%) | 10 (46%) | 0 |
| 256 | 1 (52%) | 0.1 (63%) | 0.01 (46%) | 100 (51%) | 0 |
| 512 | 1 (75%) | 0.1 (91%) | 0.01 (61%) | 100 (57%) | 0 |
| 1024 | 1 (86%) | 0.1 (100%) | 0.01 (85%) | 100 (58%) | 0 |
| 2048 | 1 (95%) | 0.1 (99%) | 0.01 (94%) | 100 (71%) | 0 |

## Scenario B: nonlinear mean, disagreement varies

Held-out test RPS excess (mean over 100 seeds):

| n (ratings) | constant (training histogram) | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|---|
| 4 (12) | 0.0923 | 0.0740 | 0.0829 | 0.0787 | 0.0901 | 0.0898 | 0.0893 |
| 8 (24) | 0.0803 | 0.0616 | 0.0706 | 0.0646 | 0.0896 | 0.0896 | 0.0887 |
| 16 (48) | 0.0733 | 0.0454 | 0.0519 | 0.0466 | 0.0899 | 0.0892 | 0.0886 |
| 20 (60) | 0.0727 | 0.0399 | 0.0455 | 0.0420 | 0.0897 | 0.0899 | 0.0887 |
| 32 (96) | 0.0705 | 0.0292 | 0.0304 | 0.0269 | 0.0897 | 0.0908 | 0.0920 |
| 64 (192) | 0.0682 | 0.0156 | 0.0160 | 0.0136 | 0.0903 | 0.0939 | 0.0899 |
| 128 (384) | 0.0678 | 0.0102 | 0.0099 | 0.0091 | 0.0915 | 0.0833 | 0.0834 |
| 256 (768) | 0.0671 | 0.0068 | 0.0064 | 0.0062 | 0.0917 | 0.0624 | 0.0655 |
| 512 (1536) | 0.0671 | 0.0052 | 0.0049 | 0.0042 | 0.0918 | 0.0459 | 0.0456 |
| 1024 (3072) | 0.0671 | 0.0043 | 0.0041 | 0.0029 | 0.0927 | 0.0344 | 0.0343 |
| 2048 (6144) | 0.0670 | 0.0037 | 0.0036 | 0.0023 | 0.0951 | 0.0326 | 0.0321 |

Crossovers on held-out test RPS (smallest n from which the first judge is significantly better at every larger n):
- CE-trained, ridge 1 (④) vs Pearson-trained, ridge 1 (②): 4 tasks (12 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge by validation Pearson (①): 4 tasks (12 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge 1 (②): 4 tasks (12 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge by validation RPS: 4 tasks (12 ratings)
- CE-trained, ridge by validation RPS (③) vs constant (training histogram): 4 tasks (12 ratings)
- Pearson-trained, ridge 1 (②) vs constant (training histogram): not at every larger n

H2 check, sizes where the validation-tuned CE judge (③) is significantly WORSE on test RPS than: Pearson-trained, ridge 1 (②): none; Pearson-trained, ridge by validation Pearson (①): none; Pearson-trained, ridge by validation RPS: none
H1 check, sizes where the Pearson-trained judge (②) is significantly BETTER on test RPS than the CE-trained judge (④): none
H2 check, sizes where the CE-trained judge (④) has significantly LOWER test Pearson correlation with the true mean than: Pearson-trained, ridge 1 (②): [512, 1024, 2048]; Pearson-trained, ridge by validation Pearson (①): [512, 1024, 2048]

Test Pearson correlation with the true mean (mean over 100 seeds; undefined counted as 0):

| n | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|
| 4 | 0.378 | 0.367 | 0.373 | 0.019 | 0.093 | 0.067 |
| 8 | 0.532 | 0.521 | 0.524 | 0.129 | 0.098 | 0.115 |
| 16 | 0.688 | 0.679 | 0.682 | 0.101 | 0.135 | 0.137 |
| 20 | 0.733 | 0.717 | 0.721 | 0.101 | 0.146 | 0.116 |
| 32 | 0.817 | 0.818 | 0.823 | 0.270 | 0.348 | 0.293 |
| 64 | 0.905 | 0.905 | 0.912 | 0.765 | 0.737 | 0.724 |
| 128 | 0.934 | 0.938 | 0.943 | 0.903 | 0.862 | 0.875 |
| 256 | 0.952 | 0.955 | 0.961 | 0.948 | 0.938 | 0.940 |
| 512 | 0.961 | 0.964 | 0.975 | 0.967 | 0.971 | 0.971 |
| 1024 | 0.966 | 0.967 | 0.983 | 0.978 | 0.985 | 0.985 |
| 2048 | 0.970 | 0.970 | 0.988 | 0.985 | 0.992 | 0.991 |

Distance from the constant forecast (mean test total variation distance), small n:

| n | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|
| 4 | 0.307 | 0.326 | 0.317 | 0.345 | 0.339 | 0.344 |
| 8 | 0.258 | 0.318 | 0.286 | 0.298 | 0.299 | 0.294 |
| 16 | 0.240 | 0.320 | 0.279 | 0.252 | 0.257 | 0.254 |
| 20 | 0.245 | 0.298 | 0.269 | 0.250 | 0.254 | 0.252 |
| 32 | 0.258 | 0.313 | 0.284 | 0.241 | 0.261 | 0.289 |
| 64 | 0.295 | 0.333 | 0.307 | 0.228 | 0.302 | 0.264 |

Chosen ridge c (most common value; share of seeds), and Pearson (①) fallbacks to c = 1:

| n | CE ③ | RPS (val RPS) | Pearson ① | Pearson (val RPS) | ① fallbacks |
|---|---|---|---|---|---|
| 4 | 100 (35%) | 0.01 (47%) | 0.01 (24%) | 0.01 (44%) | 0 |
| 8 | 0.1 (39%) | 0.01 (50%) | 10 (25%) | 0.01 (58%) | 0 |
| 16 | 1 (47%) | 0.1 (61%) | 1 (24%) | 0.01 (59%) | 0 |
| 20 | 1 (65%) | 0.1 (71%) | 0.1 (26%) | 0.01 (49%) | 0 |
| 32 | 1 (65%) | 0.1 (83%) | 10 (31%) | 0.01 (50%) | 0 |
| 64 | 1 (62%) | 0.1 (95%) | 1 (27%) | 1 (29%) | 0 |
| 128 | 0.1 (60%) | 0.1 (92%) | 0.01 (76%) | 0.01 (40%) | 0 |
| 256 | 0.1 (59%) | 0.1 (80%) | 0.01 (99%) | 0.01 (83%) | 0 |
| 512 | 0.01 (70%) | 0.01 (59%) | 0.01 (99%) | 0.01 (98%) | 0 |
| 1024 | 0.01 (86%) | 0.01 (94%) | 0.01 (100%) | 0.01 (98%) | 0 |
| 2048 | 0.01 (100%) | 0.01 (100%) | 0.01 (95%) | 0.01 (78%) | 0 |

## Scenario C: nonlinear mean, constant concentration κ

Held-out test RPS excess (mean over 100 seeds):

| n (ratings) | constant (training histogram) | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|---|
| 4 (12) | 0.0877 | 0.0682 | 0.0776 | 0.0724 | 0.0830 | 0.0836 | 0.0833 |
| 8 (24) | 0.0788 | 0.0572 | 0.0669 | 0.0609 | 0.0824 | 0.0831 | 0.0830 |
| 16 (48) | 0.0689 | 0.0434 | 0.0482 | 0.0420 | 0.0824 | 0.0824 | 0.0809 |
| 20 (60) | 0.0685 | 0.0394 | 0.0434 | 0.0375 | 0.0824 | 0.0827 | 0.0829 |
| 32 (96) | 0.0674 | 0.0290 | 0.0322 | 0.0267 | 0.0827 | 0.0843 | 0.0843 |
| 64 (192) | 0.0644 | 0.0163 | 0.0156 | 0.0137 | 0.0825 | 0.0908 | 0.0826 |
| 128 (384) | 0.0643 | 0.0097 | 0.0085 | 0.0086 | 0.0838 | 0.0782 | 0.0772 |
| 256 (768) | 0.0640 | 0.0065 | 0.0054 | 0.0060 | 0.0842 | 0.0577 | 0.0597 |
| 512 (1536) | 0.0639 | 0.0044 | 0.0037 | 0.0036 | 0.0823 | 0.0448 | 0.0430 |
| 1024 (3072) | 0.0631 | 0.0034 | 0.0030 | 0.0024 | 0.0843 | 0.0356 | 0.0351 |
| 2048 (6144) | 0.0640 | 0.0032 | 0.0029 | 0.0020 | 0.0893 | 0.0333 | 0.0320 |

Crossovers on held-out test RPS (smallest n from which the first judge is significantly better at every larger n):
- CE-trained, ridge 1 (④) vs Pearson-trained, ridge 1 (②): 4 tasks (12 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge by validation Pearson (①): 4 tasks (12 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge 1 (②): 4 tasks (12 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge by validation RPS: 4 tasks (12 ratings)
- CE-trained, ridge by validation RPS (③) vs constant (training histogram): 4 tasks (12 ratings)
- Pearson-trained, ridge 1 (②) vs constant (training histogram): not at every larger n

H2 check, sizes where the validation-tuned CE judge (③) is significantly WORSE on test RPS than: Pearson-trained, ridge 1 (②): none; Pearson-trained, ridge by validation Pearson (①): none; Pearson-trained, ridge by validation RPS: none
H1 check, sizes where the Pearson-trained judge (②) is significantly BETTER on test RPS than the CE-trained judge (④): none
H2 check, sizes where the CE-trained judge (④) has significantly LOWER test Pearson correlation with the true mean than: Pearson-trained, ridge 1 (②): [512, 1024, 2048]; Pearson-trained, ridge by validation Pearson (①): [512, 1024, 2048]

Test Pearson correlation with the true mean (mean over 100 seeds; undefined counted as 0):

| n | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|
| 4 | 0.412 | 0.389 | 0.404 | 0.098 | 0.148 | 0.071 |
| 8 | 0.554 | 0.546 | 0.550 | 0.148 | 0.123 | 0.098 |
| 16 | 0.700 | 0.699 | 0.699 | 0.089 | 0.097 | 0.132 |
| 20 | 0.730 | 0.729 | 0.732 | 0.123 | 0.196 | 0.173 |
| 32 | 0.829 | 0.822 | 0.834 | 0.333 | 0.414 | 0.333 |
| 64 | 0.912 | 0.918 | 0.920 | 0.798 | 0.737 | 0.732 |
| 128 | 0.947 | 0.954 | 0.953 | 0.928 | 0.889 | 0.908 |
| 256 | 0.960 | 0.967 | 0.967 | 0.957 | 0.947 | 0.949 |
| 512 | 0.969 | 0.975 | 0.980 | 0.972 | 0.976 | 0.976 |
| 1024 | 0.974 | 0.978 | 0.986 | 0.981 | 0.987 | 0.987 |
| 2048 | 0.976 | 0.978 | 0.988 | 0.986 | 0.993 | 0.993 |

Distance from the constant forecast (mean test total variation distance), small n:

| n | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|
| 4 | 0.287 | 0.300 | 0.292 | 0.323 | 0.320 | 0.318 |
| 8 | 0.235 | 0.293 | 0.264 | 0.267 | 0.270 | 0.269 |
| 16 | 0.215 | 0.298 | 0.252 | 0.214 | 0.216 | 0.217 |
| 20 | 0.223 | 0.306 | 0.265 | 0.208 | 0.224 | 0.231 |
| 32 | 0.237 | 0.303 | 0.279 | 0.198 | 0.216 | 0.233 |
| 64 | 0.255 | 0.313 | 0.283 | 0.193 | 0.296 | 0.233 |

Chosen ridge c (most common value; share of seeds), and Pearson (①) fallbacks to c = 1:

| n | CE ③ | RPS (val RPS) | Pearson ① | Pearson (val RPS) | ① fallbacks |
|---|---|---|---|---|---|
| 4 | 100 (43%) | 0.01 (44%) | 100 (27%) | 0.01 (35%) | 1 |
| 8 | 0.1 (34%) | 0.01 (47%) | 1 (24%) | 0.01 (57%) | 0 |
| 16 | 1 (50%) | 0.1 (67%) | 0.1 (23%) | 0.01 (53%) | 0 |
| 20 | 1 (54%) | 0.1 (72%) | 1 (26%) | 0.01 (51%) | 0 |
| 32 | 1 (60%) | 0.1 (78%) | 10 (25%) | 0.01 (35%) | 0 |
| 64 | 0.1 (54%) | 0.1 (93%) | 0.01 (29%) | 0.1 (36%) | 0 |
| 128 | 0.1 (63%) | 0.1 (87%) | 0.01 (77%) | 0.01 (35%) | 0 |
| 256 | 0.01 (69%) | 0.1 (54%) | 0.01 (99%) | 0.01 (84%) | 0 |
| 512 | 0.01 (91%) | 0.01 (86%) | 0.01 (96%) | 0.01 (100%) | 0 |
| 1024 | 0.01 (100%) | 0.01 (100%) | 0.01 (98%) | 0.01 (99%) | 0 |
| 2048 | 0.01 (100%) | 0.01 (100%) | 0.01 (95%) | 0.01 (90%) | 0 |

## Scenario D: linear mean, constant concentration κ

Held-out test RPS excess (mean over 100 seeds):

| n (ratings) | constant (training histogram) | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|---|
| 4 (12) | 0.0654 | 0.0410 | 0.0481 | 0.0447 | 0.0452 | 0.0474 | 0.0456 |
| 8 (24) | 0.0563 | 0.0372 | 0.0458 | 0.0424 | 0.0453 | 0.0454 | 0.0456 |
| 16 (48) | 0.0517 | 0.0307 | 0.0367 | 0.0354 | 0.0450 | 0.0468 | 0.0467 |
| 20 (60) | 0.0499 | 0.0278 | 0.0327 | 0.0310 | 0.0450 | 0.0460 | 0.0467 |
| 32 (96) | 0.0480 | 0.0216 | 0.0238 | 0.0224 | 0.0451 | 0.0498 | 0.0470 |
| 64 (192) | 0.0466 | 0.0144 | 0.0152 | 0.0138 | 0.0454 | 0.0541 | 0.0455 |
| 128 (384) | 0.0457 | 0.0091 | 0.0092 | 0.0085 | 0.0457 | 0.0612 | 0.0453 |
| 256 (768) | 0.0455 | 0.0056 | 0.0052 | 0.0054 | 0.0462 | 0.0548 | 0.0452 |
| 512 (1536) | 0.0455 | 0.0041 | 0.0038 | 0.0038 | 0.0476 | 0.0455 | 0.0446 |
| 1024 (3072) | 0.0454 | 0.0034 | 0.0032 | 0.0028 | 0.0493 | 0.0417 | 0.0417 |
| 2048 (6144) | 0.0449 | 0.0028 | 0.0027 | 0.0021 | 0.0523 | 0.0359 | 0.0341 |

Crossovers on held-out test RPS (smallest n from which the first judge is significantly better at every larger n):
- CE-trained, ridge 1 (④) vs Pearson-trained, ridge 1 (②): 4 tasks (12 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge by validation Pearson (①): 16 tasks (48 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge 1 (②): 16 tasks (48 ratings)
- CE-trained, ridge by validation RPS (③) vs Pearson-trained, ridge by validation RPS: 16 tasks (48 ratings)
- CE-trained, ridge by validation RPS (③) vs constant (training histogram): 4 tasks (12 ratings)
- Pearson-trained, ridge 1 (②) vs constant (training histogram): not at every larger n

H2 check, sizes where the validation-tuned CE judge (③) is significantly WORSE on test RPS than: Pearson-trained, ridge 1 (②): [4]; Pearson-trained, ridge by validation Pearson (①): none; Pearson-trained, ridge by validation RPS: [4]
H1 check, sizes where the Pearson-trained judge (②) is significantly BETTER on test RPS than the CE-trained judge (④): none
H2 check, sizes where the CE-trained judge (④) has significantly LOWER test Pearson correlation with the true mean than: Pearson-trained, ridge 1 (②): [1024, 2048]; Pearson-trained, ridge by validation Pearson (①): [1024, 2048]

Test Pearson correlation with the true mean (mean over 100 seeds; undefined counted as 0):

| n | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|
| 4 | 0.364 | 0.351 | 0.358 | 0.125 | 0.083 | 0.076 |
| 8 | 0.516 | 0.504 | 0.504 | 0.094 | 0.126 | 0.106 |
| 16 | 0.674 | 0.664 | 0.658 | 0.115 | 0.126 | 0.120 |
| 20 | 0.708 | 0.698 | 0.700 | 0.138 | 0.213 | 0.173 |
| 32 | 0.808 | 0.803 | 0.806 | 0.268 | 0.328 | 0.374 |
| 64 | 0.885 | 0.882 | 0.887 | 0.690 | 0.665 | 0.739 |
| 128 | 0.925 | 0.928 | 0.930 | 0.883 | 0.838 | 0.897 |
| 256 | 0.952 | 0.957 | 0.957 | 0.941 | 0.924 | 0.943 |
| 512 | 0.961 | 0.965 | 0.970 | 0.963 | 0.962 | 0.962 |
| 1024 | 0.965 | 0.968 | 0.978 | 0.973 | 0.979 | 0.975 |
| 2048 | 0.970 | 0.971 | 0.983 | 0.983 | 0.989 | 0.988 |

Distance from the constant forecast (mean test total variation distance), small n:

| n | CE-trained, ridge 1 (④) | CE-trained, ridge by validation RPS (③) | RPS-trained, ridge by validation RPS | Pearson-trained, ridge 1 (②) | Pearson-trained, ridge by validation Pearson (①) | Pearson-trained, ridge by validation RPS |
|---|---|---|---|---|---|---|
| 4 | 0.240 | 0.261 | 0.254 | 0.256 | 0.250 | 0.258 |
| 8 | 0.197 | 0.240 | 0.218 | 0.200 | 0.201 | 0.201 |
| 16 | 0.188 | 0.205 | 0.194 | 0.141 | 0.147 | 0.150 |
| 20 | 0.199 | 0.225 | 0.211 | 0.127 | 0.135 | 0.141 |
| 32 | 0.214 | 0.228 | 0.207 | 0.104 | 0.148 | 0.128 |
| 64 | 0.234 | 0.246 | 0.240 | 0.080 | 0.188 | 0.079 |

Chosen ridge c (most common value; share of seeds), and Pearson (①) fallbacks to c = 1:

| n | CE ③ | RPS (val RPS) | Pearson ① | Pearson (val RPS) | ① fallbacks |
|---|---|---|---|---|---|
| 4 | 100 (45%) | 100 (48%) | 100 (25%) | 0.01 (36%) | 0 |
| 8 | 1 (31%) | 0.1 (33%) | 100 (25%) | 0.1 (31%) | 0 |
| 16 | 1 (57%) | 0.1 (50%) | 0.01 (22%) | 0.1 (28%) | 0 |
| 20 | 1 (67%) | 0.1 (61%) | 100 (28%) | 0.1 (28%) | 0 |
| 32 | 1 (83%) | 0.1 (76%) | 10 (28%) | 100 (27%) | 0 |
| 64 | 1 (86%) | 0.1 (93%) | 10 (33%) | 10 (34%) | 0 |
| 128 | 1 (55%) | 0.1 (99%) | 0.01 (63%) | 10 (47%) | 0 |
| 256 | 0.1 (44%) | 0.1 (87%) | 0.01 (89%) | 100 (56%) | 0 |
| 512 | 0.01 (82%) | 0.1 (52%) | 0.01 (96%) | 100 (38%) | 0 |
| 1024 | 0.01 (100%) | 0.01 (84%) | 0.01 (98%) | 0.01 (65%) | 0 |
| 2048 | 0.01 (100%) | 0.01 (100%) | 0.01 (92%) | 0.01 (54%) | 0 |

## The Pearson-trained judge (②) versus the constant forecast (added 11-16-2026)

Held-out test RPS excess of the Pearson-trained judge (②, ridge 1) minus that of the constant training-histogram forecast, per seed (positive means ② is worse); paired 95% percentile bootstrap over 100 seeds, 10,000 draws, separate random stream.

| Scenario | smallest n from which ② is significantly worse at every larger n | mean difference at n = 2048 [95% CI] |
|---|---|---|
| A: mean nearly constant, disagreement varies | 128 tasks (384 ratings) | +0.0157 [+0.0128, +0.0189] |
| B: nonlinear mean, disagreement varies | 8 tasks (24 ratings) | +0.0281 [+0.0262, +0.0303] |
| C: nonlinear mean, constant concentration κ | 16 tasks (48 ratings) | +0.0252 [+0.0237, +0.0268] |
| D: linear mean, constant concentration κ | 256 tasks (768 ratings) | +0.0074 [+0.0063, +0.0087] |
