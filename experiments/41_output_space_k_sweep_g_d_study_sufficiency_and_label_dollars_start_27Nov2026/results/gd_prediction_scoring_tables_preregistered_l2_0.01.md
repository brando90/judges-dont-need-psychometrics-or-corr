Source file: gd_study_and_allocation.json (preregistered fixed-penalty run: True).

### P6 (D-study k* for reliability 0.80 on POPQUORN and SummEval crowd, predicted at least 8)

| dataset | single-rating intraclass correlation ICC(1) | k* [95% bootstrap] | meets the predicted 8 or more |
|---|---|---|---|
| popquorn_offensiveness | 0.287 | 10 [9, 12] | yes |
| popquorn_politeness | 0.439 | 6 [5, 6] | NO |
| summeval_crowd | 0.492 | 5 [4, 5] | NO |

P6: **NOT SUPPORTED** (met on ['popquorn_offensiveness']; missed on ['popquorn_politeness', 'summeval_crowd']).

### P7 (equal total rating budget: the one-rating-per-item head against the head trained at the D-study k*)

diff = test normalized ranked probability score (RPS divided by K - 1; lower is better) of the k*-ratings-per-item head minus that of the one-rating-per-item head, same total budget, each head at its own penalty (fixed at 0.01); positive = the one-rating-per-item head is better. Interval = 95% paired, item-clustered (test prompts resampled, 1000 draws).

| dataset | carries signal | budget (ratings) | k* used (truncated?) | items at k* | diff [95% paired item-clustered] | verdict | k with the lowest test mean |
|---|---|---|---|---|---|---|---|
| popquorn_offensiveness | yes | 128 | 8 (capped from 10) | 16 | +0.0023 [+0.0010, +0.0037] | k=1_better | 2 |
| popquorn_offensiveness | yes | 256 | 8 (capped from 10) | 32 | +0.0053 [+0.0032, +0.0077] | k=1_better | 1 |
| popquorn_offensiveness | yes | 512 | 8 (capped from 10) | 64 | +0.0025 [+0.0014, +0.0036] | k=1_better | 1 |
| popquorn_politeness | yes | 128 | 6 | 21 | +0.0069 [+0.0049, +0.0088] | k=1_better | 1 |
| popquorn_politeness | yes | 256 | 6 | 42 | +0.0012 [+0.0006, +0.0019] | k=1_better | 1 |
| popquorn_politeness | yes | 512 | 6 | 85 | +0.0014 [+0.0008, +0.0020] | k=1_better | 1 |
| summeval_expert | yes | 128 | 2 | 64 | -0.0002 [-0.0009, +0.0005] | tie | 2 |
| summeval_expert | yes | 256 | 2 | 128 | +0.0017 [+0.0005, +0.0028] | k=1_better | 1 |
| summeval_expert | yes | 512 | 2 | 256 | -0.0006 [-0.0011, -0.0001] | k*_better (refutes P7) | 2 |
| summeval_crowd | no | 128 | 5 | 25 | +0.0009 [+0.0004, +0.0013] | k=1_better | 1 |
| summeval_crowd | no | 256 | 5 | 51 | +0.0020 [+0.0011, +0.0029] | k=1_better | 1 |
| summeval_crowd | no | 512 | 5 | 102 | +0.0019 [+0.0015, +0.0023] | k=1_better | 1 |
| dices990 | yes | 128 | not formed | n/a | n/a | not evaluated: the k* = 16 design would have 8 training items, fewer than the 16 required; plans formed: k = [1, 2, 4, 8] | n/a |
| dices990 | yes | 256 | 16 | 16 | +0.0039 [+0.0011, +0.0067] | k=1_better | 1 |
| dices990 | yes | 512 | 16 | 32 | +0.0006 [-0.0014, +0.0024] | tie | 1 |
| chaosnli_snli_mnli | yes | 128 | 5 | 25 | +0.0077 [+0.0059, +0.0095] | k=1_better | 1 |
| chaosnli_snli_mnli | yes | 256 | 5 | 51 | +0.0043 [+0.0026, +0.0061] | k=1_better | 1 |
| chaosnli_snli_mnli | yes | 512 | 5 | 102 | +0.0014 [+0.0001, +0.0027] | k=1_better | 1 |
| chaosnli_alpha | no | 128 | 3 | 42 | +0.0174 [+0.0103, +0.0241] | k=1_better | 1 |
| chaosnli_alpha | no | 256 | 3 | 85 | +0.0034 [-0.0005, +0.0073] | tie | 2 |
| chaosnli_alpha | no | 512 | 3 | 170 | +0.0040 [+0.0018, +0.0065] | k=1_better | 1 |

P7 over all 21 cells: the k* design could be formed in 20; in those, the one-rating-per-item head is strictly better in 16, tied in 3, worse in 1; 1 cell(s) not evaluated. On the 5 set(s) where the CE-trained head beats the constant forecast: 11 better, 2 tied, 1 worse of 14 evaluable cells, plus 1 not formed. On sets with no signal over the constant: 5 better, 1 tied, 0 worse of 6 (both heads carry no signal there, so those cells say little about label allocation). **REFUTED in 1 cell(s): summeval_expert B=512**.

### P7, every design in every cell (diff of each k against k = 1 at the same budget)

| dataset | budget | k | training items | test nRPS | penalty chosen | diff vs k = 1 [95%] | verdict |
|---|---|---|---|---|---|---|---|
| popquorn_offensiveness | 128 | 1 | 128 | 0.1638 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_offensiveness | 128 | 2 | 64 | 0.1635 | 0.01 (fixed) | -0.0003 [-0.0015, +0.0009] | tie |
| popquorn_offensiveness | 128 | 4 | 32 | 0.1679 | 0.01 (fixed) | +0.0041 [+0.0024, +0.0059] | k=1_better |
| popquorn_offensiveness | 128 | 8 (k*) | 16 | 0.1662 | 0.01 (fixed) | +0.0023 [+0.0010, +0.0037] | k=1_better |
| popquorn_offensiveness | 256 | 1 | 256 | 0.1616 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_offensiveness | 256 | 2 | 128 | 0.1635 | 0.01 (fixed) | +0.0019 [+0.0006, +0.0033] | k=1_better |
| popquorn_offensiveness | 256 | 4 | 64 | 0.1633 | 0.01 (fixed) | +0.0018 [+0.0006, +0.0031] | k=1_better |
| popquorn_offensiveness | 256 | 8 (k*) | 32 | 0.1668 | 0.01 (fixed) | +0.0053 [+0.0032, +0.0077] | k=1_better |
| popquorn_offensiveness | 512 | 1 | 512 | 0.1596 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_offensiveness | 512 | 2 | 256 | 0.1601 | 0.01 (fixed) | +0.0005 [+0.0000, +0.0010] | k=1_better |
| popquorn_offensiveness | 512 | 4 | 128 | 0.1625 | 0.01 (fixed) | +0.0029 [+0.0017, +0.0042] | k=1_better |
| popquorn_offensiveness | 512 | 8 (k*) | 64 | 0.1621 | 0.01 (fixed) | +0.0025 [+0.0014, +0.0036] | k=1_better |
| popquorn_politeness | 128 | 1 | 128 | 0.1736 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_politeness | 128 | 2 | 64 | 0.1740 | 0.01 (fixed) | +0.0003 [-0.0003, +0.0010] | tie |
| popquorn_politeness | 128 | 4 | 32 | 0.1782 | 0.01 (fixed) | +0.0046 [+0.0026, +0.0065] | k=1_better |
| popquorn_politeness | 128 | 6 (k*) | 21 | 0.1805 | 0.01 (fixed) | +0.0069 [+0.0049, +0.0088] | k=1_better |
| popquorn_politeness | 256 | 1 | 256 | 0.1735 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_politeness | 256 | 2 | 128 | 0.1737 | 0.01 (fixed) | +0.0001 [-0.0004, +0.0007] | tie |
| popquorn_politeness | 256 | 4 | 64 | 0.1737 | 0.01 (fixed) | +0.0002 [-0.0004, +0.0008] | tie |
| popquorn_politeness | 256 | 6 (k*) | 42 | 0.1748 | 0.01 (fixed) | +0.0012 [+0.0006, +0.0019] | k=1_better |
| popquorn_politeness | 512 | 1 | 512 | 0.1727 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_politeness | 512 | 2 | 256 | 0.1727 | 0.01 (fixed) | +0.0001 [-0.0004, +0.0006] | tie |
| popquorn_politeness | 512 | 4 | 128 | 0.1732 | 0.01 (fixed) | +0.0005 [+0.0001, +0.0008] | k=1_better |
| popquorn_politeness | 512 | 6 (k*) | 85 | 0.1740 | 0.01 (fixed) | +0.0014 [+0.0008, +0.0020] | k=1_better |
| summeval_expert | 128 | 1 | 128 | 0.1411 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_expert | 128 | 2 (k*) | 64 | 0.1408 | 0.01 (fixed) | -0.0002 [-0.0009, +0.0005] | tie |
| summeval_expert | 256 | 1 | 256 | 0.1391 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_expert | 256 | 2 (k*) | 128 | 0.1408 | 0.01 (fixed) | +0.0017 [+0.0005, +0.0028] | k=1_better |
| summeval_expert | 512 | 1 | 512 | 0.1398 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_expert | 512 | 2 (k*) | 256 | 0.1392 | 0.01 (fixed) | -0.0006 [-0.0011, -0.0001] | k*_better (refutes P7) |
| summeval_crowd | 128 | 1 | 128 | 0.1315 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_crowd | 128 | 2 | 64 | 0.1319 | 0.01 (fixed) | +0.0004 [-0.0000, +0.0008] | tie |
| summeval_crowd | 128 | 4 | 32 | 0.1330 | 0.01 (fixed) | +0.0015 [+0.0010, +0.0020] | k=1_better |
| summeval_crowd | 128 | 5 (k*) | 25 | 0.1324 | 0.01 (fixed) | +0.0009 [+0.0004, +0.0013] | k=1_better |
| summeval_crowd | 256 | 1 | 256 | 0.1300 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_crowd | 256 | 2 | 128 | 0.1306 | 0.01 (fixed) | +0.0006 [+0.0002, +0.0010] | k=1_better |
| summeval_crowd | 256 | 4 | 64 | 0.1312 | 0.01 (fixed) | +0.0012 [+0.0005, +0.0020] | k=1_better |
| summeval_crowd | 256 | 5 (k*) | 51 | 0.1320 | 0.01 (fixed) | +0.0020 [+0.0011, +0.0029] | k=1_better |
| summeval_crowd | 512 | 1 | 512 | 0.1291 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_crowd | 512 | 2 | 256 | 0.1302 | 0.01 (fixed) | +0.0011 [+0.0008, +0.0015] | k=1_better |
| summeval_crowd | 512 | 4 | 128 | 0.1305 | 0.01 (fixed) | +0.0014 [+0.0011, +0.0018] | k=1_better |
| summeval_crowd | 512 | 5 (k*) | 102 | 0.1310 | 0.01 (fixed) | +0.0019 [+0.0015, +0.0023] | k=1_better |
| dices990 | 256 | 1 | 256 | 0.1982 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| dices990 | 256 | 2 | 128 | 0.1990 | 0.01 (fixed) | +0.0007 [-0.0013, +0.0028] | tie |
| dices990 | 256 | 4 | 64 | 0.2012 | 0.01 (fixed) | +0.0030 [-0.0008, +0.0070] | tie |
| dices990 | 256 | 8 | 32 | 0.1992 | 0.01 (fixed) | +0.0009 [-0.0020, +0.0038] | tie |
| dices990 | 256 | 16 (k*) | 16 | 0.2022 | 0.01 (fixed) | +0.0039 [+0.0011, +0.0067] | k=1_better |
| dices990 | 512 | 1 | 512 | 0.1978 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| dices990 | 512 | 2 | 256 | 0.1981 | 0.01 (fixed) | +0.0003 [-0.0014, +0.0020] | tie |
| dices990 | 512 | 4 | 128 | 0.1982 | 0.01 (fixed) | +0.0003 [-0.0013, +0.0022] | tie |
| dices990 | 512 | 8 | 64 | 0.1995 | 0.01 (fixed) | +0.0017 [-0.0016, +0.0051] | tie |
| dices990 | 512 | 16 (k*) | 32 | 0.1984 | 0.01 (fixed) | +0.0006 [-0.0014, +0.0024] | tie |
| chaosnli_snli_mnli | 128 | 1 | 128 | 0.2001 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_snli_mnli | 128 | 2 | 64 | 0.2007 | 0.01 (fixed) | +0.0006 [-0.0006, +0.0019] | tie |
| chaosnli_snli_mnli | 128 | 4 | 32 | 0.2044 | 0.01 (fixed) | +0.0043 [+0.0028, +0.0059] | k=1_better |
| chaosnli_snli_mnli | 128 | 5 (k*) | 25 | 0.2077 | 0.01 (fixed) | +0.0077 [+0.0059, +0.0095] | k=1_better |
| chaosnli_snli_mnli | 128 | 8 | 16 | 0.2044 | 0.01 (fixed) | +0.0044 [+0.0023, +0.0064] | k=1_better |
| chaosnli_snli_mnli | 256 | 1 | 256 | 0.1978 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_snli_mnli | 256 | 2 | 128 | 0.1992 | 0.01 (fixed) | +0.0014 [+0.0003, +0.0026] | k=1_better |
| chaosnli_snli_mnli | 256 | 4 | 64 | 0.2002 | 0.01 (fixed) | +0.0024 [+0.0012, +0.0037] | k=1_better |
| chaosnli_snli_mnli | 256 | 5 (k*) | 51 | 0.2021 | 0.01 (fixed) | +0.0043 [+0.0026, +0.0061] | k=1_better |
| chaosnli_snli_mnli | 256 | 8 | 32 | 0.2025 | 0.01 (fixed) | +0.0046 [+0.0033, +0.0060] | k=1_better |
| chaosnli_snli_mnli | 256 | 16 | 16 | 0.2034 | 0.01 (fixed) | +0.0056 [+0.0043, +0.0069] | k=1_better |
| chaosnli_snli_mnli | 512 | 1 | 512 | 0.1968 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_snli_mnli | 512 | 2 | 256 | 0.1976 | 0.01 (fixed) | +0.0008 [-0.0000, +0.0016] | tie |
| chaosnli_snli_mnli | 512 | 4 | 128 | 0.1981 | 0.01 (fixed) | +0.0013 [-0.0001, +0.0027] | tie |
| chaosnli_snli_mnli | 512 | 5 (k*) | 102 | 0.1981 | 0.01 (fixed) | +0.0014 [+0.0001, +0.0027] | k=1_better |
| chaosnli_snli_mnli | 512 | 8 | 64 | 0.2003 | 0.01 (fixed) | +0.0036 [+0.0021, +0.0051] | k=1_better |
| chaosnli_snli_mnli | 512 | 16 | 32 | 0.2020 | 0.01 (fixed) | +0.0053 [+0.0039, +0.0067] | k=1_better |
| chaosnli_alpha | 128 | 1 | 128 | 0.2553 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_alpha | 128 | 2 | 64 | 0.2631 | 0.01 (fixed) | +0.0078 [+0.0036, +0.0116] | k=1_better |
| chaosnli_alpha | 128 | 3 (k*) | 42 | 0.2727 | 0.01 (fixed) | +0.0174 [+0.0103, +0.0241] | k=1_better |
| chaosnli_alpha | 128 | 4 | 32 | 0.2793 | 0.01 (fixed) | +0.0240 [+0.0162, +0.0310] | k=1_better |
| chaosnli_alpha | 128 | 8 | 16 | 0.2816 | 0.01 (fixed) | +0.0263 [+0.0148, +0.0380] | k=1_better |
| chaosnli_alpha | 256 | 1 | 256 | 0.2545 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_alpha | 256 | 2 | 128 | 0.2536 | 0.01 (fixed) | -0.0009 [-0.0039, +0.0020] | tie |
| chaosnli_alpha | 256 | 3 (k*) | 85 | 0.2579 | 0.01 (fixed) | +0.0034 [-0.0005, +0.0073] | tie |
| chaosnli_alpha | 256 | 4 | 64 | 0.2635 | 0.01 (fixed) | +0.0090 [+0.0044, +0.0133] | k=1_better |
| chaosnli_alpha | 256 | 8 | 32 | 0.2752 | 0.01 (fixed) | +0.0207 [+0.0130, +0.0279] | k=1_better |
| chaosnli_alpha | 256 | 16 | 16 | 0.2757 | 0.01 (fixed) | +0.0212 [+0.0115, +0.0305] | k=1_better |
| chaosnli_alpha | 512 | 1 | 512 | 0.2520 | 0.01 (fixed) | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_alpha | 512 | 2 | 256 | 0.2531 | 0.01 (fixed) | +0.0012 [-0.0004, +0.0027] | tie |
| chaosnli_alpha | 512 | 3 (k*) | 170 | 0.2560 | 0.01 (fixed) | +0.0040 [+0.0018, +0.0065] | k=1_better |
| chaosnli_alpha | 512 | 4 | 128 | 0.2539 | 0.01 (fixed) | +0.0019 [-0.0003, +0.0042] | tie |
| chaosnli_alpha | 512 | 8 | 64 | 0.2602 | 0.01 (fixed) | +0.0083 [+0.0041, +0.0121] | k=1_better |
| chaosnli_alpha | 512 | 16 | 32 | 0.2718 | 0.01 (fixed) | +0.0198 [+0.0129, +0.0262] | k=1_better |

### Ceiling check on real ratings (leave-one-rating-out; pooled over items and ratings)

| dataset | ICC(1) = rho | sampled-judge Corr (theory: rho) | collapsed-mean Corr measured | finite-n formula | sqrt(rho) | lattice-rounded mean Corr | rounded still above sampled |
|---|---|---|---|---|---|---|---|
| popquorn_offensiveness | 0.287 | 0.287 | 0.466 | 0.468 | 0.536 | 0.435 | yes |
| popquorn_politeness | 0.439 | 0.439 | 0.599 | 0.602 | 0.663 | 0.576 | yes |
| summeval_expert | 0.719 | 0.719 | 0.775 | 0.775 | 0.848 | 0.755 | yes |
| summeval_crowd | 0.492 | 0.492 | 0.625 | 0.625 | 0.701 | 0.603 | yes |
| dices990 | 0.201 | 0.201 | 0.435 | 0.435 | 0.448 | 0.389 | yes |
| chaosnli_snli_mnli | 0.485 | 0.485 | 0.693 | 0.693 | 0.696 | 0.617 | yes |
| chaosnli_alpha | 0.648 | 0.648 | 0.803 | 0.803 | 0.805 | 0.757 | yes |
