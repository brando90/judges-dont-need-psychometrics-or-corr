# Conditional-spread collapse on held-out test ratings

Rows are individual test ratings. The uncollapsed scalar is the judge's expected rating plus a within-prompt residual that is exactly orthogonal to the centered human ratings and has the judge's predictive variance. Collapsing sets that residual to zero. The point-mass forecast puts mass 1 on the class nearest the expected rating.

| Dataset | CKA uncollapsed | CKA collapsed | OPD uncollapsed | OPD collapsed | Pearson uncollapsed | Pearson collapsed | CE distribution | CE point mass | Brier distribution | Brier point mass | sqrt(RPS) distribution | sqrt(RPS) point mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| summeval_expert_coherence | 0.0000 | 0.0062 | 1.4124 | 1.3576 | 0.0026 | 0.0785 | 1.5393 [1.5034, 1.5740] | inf | 0.7750 [0.7636, 0.7857] | 1.4208 [1.3666, 1.4729] | 0.8348 [0.8178, 0.8531] | 1.0098 [0.9826, 1.0348] |
| summeval_expert_consistency | 0.0000 | 0.0012 | 1.4138 | 1.3892 | 0.0006 | 0.0351 | 0.5461 [0.4472, 0.6480] | inf | 0.2302 [0.1784, 0.2835] | 0.2500 [0.1875, 0.3146] | 0.5282 [0.4559, 0.5930] | 0.5562 [0.4688, 0.6325] |
| summeval_expert_fluency | 0.0000 | 0.0113 | 1.4125 | 1.3371 | -0.0025 | -0.1061 | 0.6265 [0.5391, 0.7194] | inf | 0.2826 [0.2370, 0.3326] | 0.3146 [0.2562, 0.3771] | 0.5235 [0.4652, 0.5751] | 0.5572 [0.4873, 0.6208] |
| summeval_expert_relevance | 0.0000 | 0.0242 | 1.4104 | 1.2996 | 0.0054 | 0.1555 | 1.3315 [1.2954, 1.3676] | inf | 0.7117 [0.6988, 0.7248] | 1.2333 [1.1687, 1.2917] | 0.7236 [0.7058, 0.7418] | 0.8521 [0.8235, 0.8809] |
| chaosNLI_snli | 0.0000 | 0.0083 | 1.4115 | 1.3481 | 0.0039 | 0.0913 | 1.0002 [0.9773, 1.0239] | inf | 0.6006 [0.5847, 0.6173] | 0.9405 [0.8733, 1.0127] | 0.5924 [0.5789, 0.6061] | 0.6857 [0.6608, 0.7116] |
| chaosNLI_mnli_m | 0.0000 | 0.0001 | 1.4139 | 1.4061 | 0.0005 | 0.0115 | 1.0553 [1.0392, 1.0730] | inf | 0.6401 [0.6307, 0.6506] | 1.2206 [1.1708, 1.2673] | 0.6343 [0.6255, 0.6434] | 0.7812 [0.7651, 0.7960] |
| chaosnli_alpha | 0.0000 | 0.0033 | 1.4123 | 1.3730 | -0.0027 | -0.0575 | 0.6966 [0.6925, 0.7009] | inf | 0.5034 [0.4993, 0.5077] | 1.0781 [0.9910, 1.1676] | 0.5017 [0.4997, 0.5039] | 0.7342 [0.7039, 0.7641] |
| dices990 | 0.0001 | 0.0161 | 1.4088 | 1.3215 | 0.0076 | 0.1268 | 0.7430 [0.7099, 0.7769] | inf | 0.4421 [0.4167, 0.4680] | 1.5697 [1.4779, 1.6601] | 0.6276 [0.6086, 0.6464] | 0.9063 [0.8839, 0.9275] |
| veribench_fsc_exp09 | 0.0103 | 0.1825 | 1.3404 | 1.0703 | 0.1017 | 0.4272 | 0.4626 [0.2924, 0.6507] | inf | 0.2273 [0.1134, 0.3510] | 0.4667 [0.1667, 0.7833] | 0.3723 [0.2707, 0.4596] | 0.4830 [0.2887, 0.6258] |
