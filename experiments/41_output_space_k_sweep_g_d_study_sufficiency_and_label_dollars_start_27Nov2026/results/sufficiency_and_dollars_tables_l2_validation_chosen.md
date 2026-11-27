Judge: CE-trained 24-component linear head on hashed text. How its ridge penalty was chosen: L2 penalty chosen per point on validation prompts (grid 1e-5 to 1; not preregistered). Source file: gd_study_l2_validation_chosen.json.

Prices: the $0.05 crowd price per rating is DERIVED from POPQUORN's $12 per hour and 13 minutes per 50 offensiveness annotations (about $0.052), not a published per-rating price; every other dollar figure is an assumption. See the header of scripts/sufficiency_and_dollars.py.

### A. Real-data G-study and D-study (items random, ratings nested in items; one-way unbalanced analysis of variance; independent of the ridge penalty)

| dataset | K | single-rating reliability, intraclass correlation ICC(1) [95% bootstrap] | ratings per item for reliability 0.80 (k*) [95%] | median ratings per item available |
|---|---|---|---|---|
| popquorn_offensiveness | 5 | 0.287 [0.263, 0.310] | 10 [9, 12] | 9 |
| popquorn_politeness | 5 | 0.439 [0.424, 0.453] | 6 [5, 6] | 7 |
| summeval_expert | 5 | 0.719 [0.708, 0.729] | 2 [2, 2] | 3 |
| summeval_crowd | 5 | 0.492 [0.473, 0.509] | 5 [4, 5] | 5 |
| dices990 | 3 | 0.201 [0.185, 0.215] | 16 [15, 18] | 66 |
| chaosnli_snli_mnli | 3 | 0.485 [0.472, 0.497] | 5 [5, 5] | 100 |
| chaosnli_alpha | 2 | 0.648 [0.632, 0.663] | 3 [3, 3] | 100 |

### B. Training items: when does the CE-trained head beat the constant forecast, and when does its curve flatten?

| dataset | carries signal | train items available | gain of the full-training head over the constant forecast [95% paired, item-clustered] | first N from which the gain stays significant | saturation N (k=1, margin 0.001) | saturation N (margin 10% of the full-training gain) | last doubling: drop in test normalized RPS [95%] |
|---|---|---|---|---|---|---|---|
| popquorn_offensiveness | yes | 900 | +0.0019 [+0.0006, +0.0031] | 900 | 256 | 900 | 512 to 900: +0.0012 [+0.0004, +0.0020] |
| popquorn_politeness | yes | 2230 | +0.0153 [+0.0120, +0.0186] | 128 | 2048 | 2048 | 2048 to 2230: -0.0005 [-0.0012, +0.0001] |
| summeval_expert | yes | 3840 | +0.0058 [+0.0043, +0.0073] | 128 | 3840 | 3840 | 2048 to 3840: +0.0021 [+0.0016, +0.0026] |
| summeval_crowd | no | 3840 | -0.0000 [-0.0000, +0.0000] | n/a (no signal) | n/a (no signal) | n/a (no signal) | 2048 to 3840: -0.0000 [-0.0001, +0.0000] |
| dices990 | yes | 594 | +0.0041 [+0.0012, +0.0071] | 594 | 256 | 594 | 512 to 594: +0.0002 [-0.0008, +0.0011] |
| chaosnli_snli_mnli | no | 1867 | +0.0007 [-0.0009, +0.0021] | n/a (no signal) | n/a (no signal) | n/a (no signal) | 1024 to 1867: -0.0009 [-0.0021, +0.0002] |
| chaosnli_alpha | no | 919 | -0.0013 [-0.0028, +0.0002] | n/a (no signal) | n/a (no signal) | n/a (no signal) | 512 to 919: -0.0007 [-0.0037, +0.0020] |

gain = test normalized ranked probability score (RPS; RPS divided by K - 1; 0 to 1; lower is better) of the constant forecast minus that of the CE-trained head at its penalty (positive = the head is better). Carries signal = that gain has a paired 95% item-clustered interval above 0 at the full training set. Saturation N = the smallest N on the grid such that no larger N improves the head's mean test curve by more than the margin (a descriptive summary of one noisy curve: mean of 3 training-subsample draws, one prompt-level split; the margins were fixed in the analysis script, not preregistered). The 10% margin is 10% of the gain of the full-training head over the constant, read at the largest N, not at the minimum of the test curve. An N equal to the training items available means the curve had not flattened within the data.

### C. Test items needed (paired comparison at 80% power, two-sided 5%)

| dataset | test items available | test items to detect the head's actual gain over the constant | test items for a gain of 0.005 (k = 1) | for a gain of 0.0016 (k = 1) |
|---|---|---|---|---|
| popquorn_offensiveness | 300 | 288 | 123 | 1,200 |
| popquorn_politeness | 745 | 75 | 2,126 | 20,764 |
| summeval_expert | 1280 | 153 | 259 | 2,525 |
| summeval_crowd | 1280 | n/a (no gain) | 0 | 0 |
| dices990 | 198 | 202 | 674 | 6,587 |
| chaosnli_snli_mnli | 624 | 6,786 | 388 | 3,791 |
| chaosnli_alpha | 307 | n/a (no gain) | 102 | 996 |

Items to detect the actual gain use each test item's own ratings; the 0.005 and 0.0016 columns rescale the paired standard deviation to one rating per item (approximation, see the header). Where the head's gain over the constant is estimated near zero the first column is not meaningful.

### D. Cost-minimizing ratings per item for the proper-score RISK ESTIMATE (a test-side precision statement, not a training recommendation)

| dataset | share of loss variance between items | k_opt if an item costs 0 extra | 1x a rating | 10x | 100x | D-study k* for reliability 0.80 |
|---|---|---|---|---|---|---|
| popquorn_offensiveness | 0.25 | 1.0 | 1.7 | 5.4 | 17.2 | 10 |
| popquorn_politeness | 0.22 | 1.0 | 1.9 | 5.9 | 18.7 | 6 |
| summeval_expert | 0.68 | 1.0 | 1.0 | 2.2 | 6.9 | 2 |
| summeval_crowd | 0.50 | 1.0 | 1.0 | 3.2 | 10.0 | 5 |
| dices990 | 0.19 | 1.0 | 2.1 | 6.5 | 20.6 | 16 |
| chaosnli_snli_mnli | 0.31 | 1.0 | 1.5 | 4.8 | 15.0 | 5 |
| chaosnli_alpha | 0.64 | 1.0 | 1.0 | 2.4 | 7.5 | 3 |

### E. Label dollars against researcher-months (assumption grid), three definitions of enough training items

| dataset | definition of enough | training items (k = 1) | crowd, $0.05 derived | expert, $5 assumed | expert, in researcher-months at $30k assumed |
|---|---|---|---|---|---|
| popquorn_offensiveness | first N with a significant gain | 900 (curve not flat within the data) | $45.00 | $4,500 | 0.150 |
| popquorn_offensiveness | saturation, margin 0.001 | 256 | $12.80 | $1,280 | 0.043 |
| popquorn_offensiveness | saturation, margin 10% of the gain | 900 (curve not flat within the data) | $45.00 | $4,500 | 0.150 |
| popquorn_politeness | first N with a significant gain | 128 | $6.40 | $640 | 0.021 |
| popquorn_politeness | saturation, margin 0.001 | 2,048 | $102.40 | $10,240 | 0.341 |
| popquorn_politeness | saturation, margin 10% of the gain | 2,048 | $102.40 | $10,240 | 0.341 |
| summeval_expert | first N with a significant gain | 128 | $6.40 | $640 | 0.021 |
| summeval_expert | saturation, margin 0.001 | 3,840 (curve not flat within the data) | $192.00 | $19,200 | 0.640 |
| summeval_expert | saturation, margin 10% of the gain | 3,840 (curve not flat within the data) | $192.00 | $19,200 | 0.640 |
| dices990 | first N with a significant gain | 594 (curve not flat within the data) | $29.70 | $2,970 | 0.099 |
| dices990 | saturation, margin 0.001 | 256 | $12.80 | $1,280 | 0.043 |
| dices990 | saturation, margin 10% of the gain | 594 (curve not flat within the data) | $29.70 | $2,970 | 0.099 |

Not priced (the CE-trained head does not beat the constant forecast on these, so there is no point to price): summeval_crowd, chaosnli_snli_mnli, chaosnli_alpha.

### F. Test items needed to resolve the small gap that is LEFT after each head gets its own calibration map

Pearson-trained head minus CE-trained head, both followed by their own validation-fitted calibration map (L2-sensitivity K-sweep; five split seeds; NOT preregistered). Difference = test normalized RPS of the Pearson-trained head minus test normalized RPS of the CE-trained head; positive means the CE-trained head is better. The standard deviation of the per-item paired difference is implied by the interval half-width (normal approximation).

| view | K | test items available | calibrated gap, mean of 5 seeds | seeds with the 95% interval above / below 0 | minimum detectable difference at the available items (80% power) | items to resolve 0.0022 | items to resolve 0.001 | crowd dollars for 0.0022 at the dataset's own ratings per item |
|---|---|---|---|---|---|---|---|---|
| chaosnli_alphanli|K2 | 2 | 307 | +0.0004 | 0 / 0 of 5 | 0.0035 | 780 | 3,777 | $3,902 |
| chaosnli_snli_mnli|K3 | 3 | 624 | +0.0003 | 1 / 0 of 5 | 0.0008 | 82 | 396 | $409 |
| dices990|K2 | 2 | 198 | +0.0005 | 0 / 0 of 5 | 0.0018 | 130 | 631 | $430 |
| dices990|K3 | 3 | 198 | +0.0009 | 1 / 0 of 5 | 0.0019 | 152 | 736 | $502 |
| popquorn_offensiveness|K2 | 2 | 300 | +0.0004 | 3 / 0 of 5 | 0.0007 | 29 | 139 | $13 |
| popquorn_offensiveness|K3 | 3 | 300 | +0.0003 | 0 / 0 of 5 | 0.0006 | 21 | 100 | $9 |
| popquorn_offensiveness|K5 | 5 | 300 | -0.0001 | 0 / 0 of 5 | 0.0008 | 44 | 213 | $20 |
| popquorn_politeness|K2 | 2 | 745 | +0.0022 | 3 / 0 of 5 | 0.0024 | 880 | 4,258 | $308 |
| popquorn_politeness|K3 | 3 | 745 | +0.0020 | 3 / 0 of 5 | 0.0019 | 539 | 2,607 | $189 |
| popquorn_politeness|K5 | 5 | 745 | +0.0020 | 4 / 0 of 5 | 0.0015 | 363 | 1,755 | $127 |
| summeval_crowd|K2 | 2 | 1280 | +0.0002 | 1 / 1 of 5 | 0.0004 | 33 | 159 | $8 |
| summeval_crowd|K3 | 3 | 1280 | +0.0000 | 0 / 0 of 5 | 0.0004 | 53 | 254 | $13 |
| summeval_crowd|K5 | 5 | 1280 | +0.0001 | 1 / 0 of 5 | 0.0002 | 8 | 37 | $2 |
| summeval_expert|K2 | 2 | 1280 | -0.0013 | 0 / 1 of 5 | 0.0021 | 1,135 | 5,494 | $170 |
| summeval_expert|K3 | 3 | 1280 | -0.0004 | 0 / 1 of 5 | 0.0009 | 238 | 1,152 | $36 |
| summeval_expert|K5 | 5 | 1280 | -0.0004 | 1 / 1 of 5 | 0.0016 | 696 | 3,367 | $104 |

Dollars = items x the dataset's median ratings per item x $0.05 (derived crowd price); datasets with many ratings per item (ChaosNLI 100, DICES 66) show large dollars here because that is how they were collected, not because the item count requires it. A larger target difference needs quadratically fewer items. The raw gap between the heads (before calibration) is 0.005 to 0.10 and is resolved by a few dozen to a few hundred test items.

### G. Formalization-quality rating packet (sister repository planning numbers; minutes per rating is the uncertain input)

Minutes per rating: measured median 4.0 (103 entries, 2 raters) versus an assumed length-scaled 37.3.

| scenario | ratings | hours at measured minutes | hours at length-scaled minutes | dollars at $50/hour (measured to scaled) | in researcher-months at $30k (measured to scaled) |
|---|---|---|---|---|---|
| finish_current_packet | 200 | 13.3 | 124.5 | $667 to $6,224 | 0.02 to 0.21 |
| add_development_pool | 392 | 26.1 | 244.0 | $1,307 to $12,199 | 0.04 to 0.41 |
| add_development_pool_and_reach_half_width_0.01 | 488 | 32.5 | 303.7 | $1,627 to $15,187 | 0.05 to 0.51 |
| add_development_pool_and_reach_half_width_0.005 | 1352 | 90.1 | 841.5 | $4,507 to $42,075 | 0.15 to 1.40 |
