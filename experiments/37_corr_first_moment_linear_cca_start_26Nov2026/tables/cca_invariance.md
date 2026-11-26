# Adversarial maps of the cross-entropy judge

CCA is between the judge probability vector and the item's empirical human distribution, on held-out test items. A difference of 0 at 1e-8 means the canonical correlations matched to that tolerance. Reversal and the label roll stay on the simplex. Sign flip and the orthogonal mix do not, so their cross-entropy is infinite.

| Dataset | Map | Max abs CCA difference | Identical at 1e-8 | CE | Brier | sqrt(RPS) |
| --- | --- | --- | --- | --- | --- | --- |
| summeval_expert | original | 0 | yes | 1.2442 [1.2055, 1.2801] | 0.6351 [0.6145, 0.6544] | 0.7592 [0.7437, 0.7752] |
| summeval_expert | reversal | 1.388e-16 | True | 2.7195 [2.6849, 2.7583] | 1.1868 [1.1777, 1.1959] | 1.3532 [1.3391, 1.3692] |
| summeval_expert | roll_permutation | 1.665e-16 | True | 2.0304 [2.0095, 2.0504] | 1.0747 [1.0682, 1.0804] | 1.1792 [1.1693, 1.1905] |
| summeval_expert | sign_flip_raw | 0.000e+00 | True | infinite | 2.1068 [2.0874, 2.1285] algebraic | 1.3445 algebraic |
| summeval_expert | orthogonal_mix | 1.527e-15 | True | infinite | 7.9976 [7.8624, 8.1250] algebraic | 5.0006 algebraic |
| summeval_crowd | original | 0 | yes | 1.3098 [1.2826, 1.3373] | 0.6993 [0.6903, 0.7095] | 0.7066 [0.6943, 0.7191] |
| summeval_crowd | reversal | 1.110e-16 | True | 2.5711 [2.5203, 2.6205] | 1.0392 [1.0283, 1.0498] | 1.1199 [1.1060, 1.1332] |
| summeval_crowd | roll_permutation | 1.388e-16 | True | 1.8928 [1.8494, 1.9389] | 0.8764 [0.8646, 0.8880] | 0.7952 [0.7888, 0.8023] |
| summeval_crowd | sign_flip_raw | 0.000e+00 | True | infinite | 1.9117 [1.9014, 1.9208] algebraic | 1.8203 algebraic |
| summeval_crowd | orthogonal_mix | 6.037e-15 | True | infinite | 8.3896 [8.3000, 8.4825] algebraic | 5.1072 algebraic |
| chaosnli_snli_mnli | original | 0 | yes | 1.0516 [1.0379, 1.0653] | 0.6359 [0.6274, 0.6446] | 0.6256 [0.6178, 0.6338] |
| chaosnli_snli_mnli | reversal | 5.551e-17 | True | 1.1294 [1.1123, 1.1455] | 0.6791 [0.6686, 0.6889] | 0.6593 [0.6499, 0.6682] |
| chaosnli_snli_mnli | roll_permutation | 6.245e-17 | True | 1.1885 [1.1725, 1.2048] | 0.7255 [0.7157, 0.7354] | 0.6838 [0.6742, 0.6928] |
| chaosnli_snli_mnli | sign_flip_raw | 0.000e+00 | True | infinite | 2.0907 [2.0818, 2.0993] algebraic | 1.8406 algebraic |
| chaosnli_snli_mnli | orthogonal_mix | 4.996e-16 | True | infinite | 8.7423 [8.6029, 8.8855] algebraic | 1.7285 algebraic |
| chaosnli_alpha | original | 0 | yes | 0.6961 [0.6924, 0.6996] | 0.5029 [0.4993, 0.5065] | 0.5015 [0.4996, 0.5032] |
| chaosnli_alpha | reversal | 0.000e+00 | True | 0.6919 [0.6884, 0.6956] | 0.4988 [0.4952, 0.5024] | 0.4994 [0.4976, 0.5012] |
| chaosnli_alpha | roll_permutation | 1.388e-17 | True | 0.6919 [0.6884, 0.6956] | 0.4988 [0.4952, 0.5024] | 0.4994 [0.4976, 0.5012] |
| chaosnli_alpha | sign_flip_raw | 0.000e+00 | True | infinite | 2.4988 [2.4952, 2.5024] algebraic | 1.1309 algebraic |
| chaosnli_alpha | orthogonal_mix | 6.245e-17 | True | infinite | 1.7899 [1.6783, 1.8979] algebraic | 1.2377 algebraic |
| dices990 | original | 0 | yes | 0.7426 [0.7069, 0.7777] | 0.4423 [0.4145, 0.4692] | 0.6283 [0.6076, 0.6479] |
| dices990 | reversal | 5.603e-16 | True | 1.2301 [1.2023, 1.2585] | 0.8641 [0.8392, 0.8901] | 0.9036 [0.8885, 0.9187] |
| dices990 | roll_permutation | 3.600e-16 | True | 1.7588 [1.7181, 1.8009] | 1.1234 [1.1099, 1.1367] | 0.8035 [0.7969, 0.8100] |
| dices990 | sign_flip_raw | 0.000e+00 | True | infinite | 2.6813 [2.6525, 2.7107] algebraic | 2.1551 algebraic |
| dices990 | orthogonal_mix | 4.635e-15 | True | infinite | 5.4773 [5.4475, 5.5047] algebraic | 1.2613 algebraic |
| veribench_fsc_exp09 | original | 0 | yes | 1.1245 [0.7249, 1.5686] | 0.5634 [0.3515, 0.7862] | 0.6378 [0.4903, 0.7706] |
| veribench_fsc_exp09 | reversal | 6.550e-15 | True | 4.3581 [4.1534, 4.5388] | 1.5284 [1.4975, 1.5613] | 1.9032 [1.8364, 1.9627] |
| veribench_fsc_exp09 | roll_permutation | 6.550e-15 | True | 2.3467 [2.0062, 2.7219] | 1.2825 [1.1967, 1.3734] | 1.6063 [1.5501, 1.6603] |
| veribench_fsc_exp09 | sign_flip_raw | 0.000e+00 | True | infinite | 2.5542 [2.3444, 2.7542] algebraic | 0.8909 algebraic |
| veribench_fsc_exp09 | orthogonal_mix | 4.219e-14 | True | infinite | 9.9016 [9.3706, 10.4065] algebraic | 2.3424 algebraic |
