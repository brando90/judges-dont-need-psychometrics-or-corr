Source file: gd_study_l2_validation_chosen.json (preregistered fixed-penalty run: False).

### P6 (D-study k* for reliability 0.80 on POPQUORN and SummEval crowd, predicted at least 8)

| dataset | single-rating intraclass correlation ICC(1) | k* [95% bootstrap] | meets the predicted 8 or more |
|---|---|---|---|
| popquorn_offensiveness | 0.287 | 10 [9, 12] | yes |
| popquorn_politeness | 0.439 | 6 [5, 6] | NO |
| summeval_crowd | 0.492 | 5 [4, 5] | NO |

P6: **NOT SUPPORTED** (met on ['popquorn_offensiveness']; missed on ['popquorn_politeness', 'summeval_crowd']).

### P7 (equal total rating budget: the one-rating-per-item head against the head trained at the D-study k*)

diff = test normalized ranked probability score (RPS divided by K - 1; lower is better) of the k*-ratings-per-item head minus that of the one-rating-per-item head, same total budget, each head at its own penalty (chosen per cell on validation prompts); positive = the one-rating-per-item head is better. Interval = 95% paired, item-clustered (test prompts resampled, 1000 draws).

| dataset | carries signal | budget (ratings) | k* used (truncated?) | items at k* | diff [95% paired item-clustered] | verdict | k with the lowest test mean |
|---|---|---|---|---|---|---|---|
| popquorn_offensiveness | yes | 128 | 8 (capped from 10) | 16 | +0.0012 [+0.0006, +0.0019] | k=1_better | 2 |
| popquorn_offensiveness | yes | 256 | 8 (capped from 10) | 32 | +0.0042 [+0.0022, +0.0063] | k=1_better | 1 |
| popquorn_offensiveness | yes | 512 | 8 (capped from 10) | 64 | +0.0025 [+0.0014, +0.0036] | k=1_better | 1 |
| popquorn_politeness | yes | 128 | 6 | 21 | +0.0069 [+0.0049, +0.0088] | k=1_better | 1 |
| popquorn_politeness | yes | 256 | 6 | 42 | +0.0046 [+0.0027, +0.0063] | k=1_better | 1 |
| popquorn_politeness | yes | 512 | 6 | 85 | +0.0032 [+0.0017, +0.0046] | k=1_better | 1 |
| summeval_expert | yes | 128 | 2 | 64 | -0.0002 [-0.0009, +0.0005] | tie | 2 |
| summeval_expert | yes | 256 | 2 | 128 | +0.0041 [+0.0028, +0.0053] | k=1_better | 1 |
| summeval_expert | yes | 512 | 2 | 256 | -0.0009 [-0.0019, +0.0001] | tie | 2 |
| summeval_crowd | no | 128 | 5 | 25 | +0.0011 [+0.0009, +0.0013] | k=1_better | 1 |
| summeval_crowd | no | 256 | 5 | 51 | +0.0011 [+0.0004, +0.0017] | k=1_better | 1 |
| summeval_crowd | no | 512 | 5 | 102 | +0.0012 [+0.0009, +0.0015] | k=1_better | 1 |
| dices990 | yes | 128 | not formed | n/a | n/a | not evaluated: the k* = 16 design would have 8 training items, fewer than the 16 required; plans formed: k = [1, 2, 4, 8] | n/a |
| dices990 | yes | 256 | 16 | 16 | +0.0024 [-0.0004, +0.0053] | tie | 1 |
| dices990 | yes | 512 | 16 | 32 | +0.0006 [-0.0014, +0.0024] | tie | 1 |
| chaosnli_snli_mnli | no | 128 | 5 | 25 | +0.0042 [+0.0024, +0.0060] | k=1_better | 2 |
| chaosnli_snli_mnli | no | 256 | 5 | 51 | +0.0025 [+0.0010, +0.0040] | k=1_better | 1 |
| chaosnli_snli_mnli | no | 512 | 5 | 102 | +0.0014 [+0.0001, +0.0027] | k=1_better | 1 |
| chaosnli_alpha | no | 128 | 3 | 42 | +0.0132 [+0.0091, +0.0174] | k=1_better | 1 |
| chaosnli_alpha | no | 256 | 3 | 85 | +0.0022 [+0.0009, +0.0034] | k=1_better | 1 |
| chaosnli_alpha | no | 512 | 3 | 170 | +0.0024 [+0.0020, +0.0027] | k=1_better | 1 |

P7 over all 21 cells: the k* design could be formed in 20; in those, the one-rating-per-item head is strictly better in 16, tied in 4, worse in 0; 1 cell(s) not evaluated. On the 4 set(s) where the CE-trained head beats the constant forecast: 7 better, 4 tied, 0 worse of 11 evaluable cells, plus 1 not formed. On sets with no signal over the constant: 9 better, 0 tied, 0 worse of 9 (both heads carry no signal there, so those cells say little about label allocation). **NOT REFUTED in any evaluable cell**.

### P7, every design in every cell (diff of each k against k = 1 at the same budget)

| dataset | budget | k | training items | test nRPS | penalty chosen | diff vs k = 1 [95%] | verdict |
|---|---|---|---|---|---|---|---|
| popquorn_offensiveness | 128 | 1 | 128 | 0.1636 | 0.1 | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_offensiveness | 128 | 2 | 64 | 0.1631 | 0.1 | -0.0005 [-0.0011, +0.0002] | tie |
| popquorn_offensiveness | 128 | 4 | 32 | 0.1665 | 0.1 | +0.0029 [+0.0021, +0.0039] | k=1_better |
| popquorn_offensiveness | 128 | 8 (k*) | 16 | 0.1648 | 0.1 | +0.0012 [+0.0006, +0.0019] | k=1_better |
| popquorn_offensiveness | 256 | 1 | 256 | 0.1616 | 0.01 | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_offensiveness | 256 | 2 | 128 | 0.1635 | 0.1 | +0.0020 [+0.0006, +0.0034] | k=1_better |
| popquorn_offensiveness | 256 | 4 | 64 | 0.1635 | 0.1 | +0.0019 [+0.0006, +0.0034] | k=1_better |
| popquorn_offensiveness | 256 | 8 (k*) | 32 | 0.1657 | 0.1 | +0.0042 [+0.0022, +0.0063] | k=1_better |
| popquorn_offensiveness | 512 | 1 | 512 | 0.1596 | 0.01 | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_offensiveness | 512 | 2 | 256 | 0.1601 | 0.01 | +0.0005 [+0.0000, +0.0010] | k=1_better |
| popquorn_offensiveness | 512 | 4 | 128 | 0.1625 | 0.01 | +0.0029 [+0.0017, +0.0042] | k=1_better |
| popquorn_offensiveness | 512 | 8 (k*) | 64 | 0.1621 | 0.01 | +0.0025 [+0.0014, +0.0036] | k=1_better |
| popquorn_politeness | 128 | 1 | 128 | 0.1736 | 0.01 | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_politeness | 128 | 2 | 64 | 0.1740 | 0.01 | +0.0003 [-0.0003, +0.0010] | tie |
| popquorn_politeness | 128 | 4 | 32 | 0.1782 | 0.01 | +0.0046 [+0.0026, +0.0065] | k=1_better |
| popquorn_politeness | 128 | 6 (k*) | 21 | 0.1805 | 0.01 | +0.0069 [+0.0049, +0.0088] | k=1_better |
| popquorn_politeness | 256 | 1 | 256 | 0.1703 | 0.001 | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_politeness | 256 | 2 | 128 | 0.1717 | 0.001 | +0.0014 [+0.0001, +0.0026] | k=1_better |
| popquorn_politeness | 256 | 4 | 64 | 0.1722 | 0.001 | +0.0019 [+0.0004, +0.0034] | k=1_better |
| popquorn_politeness | 256 | 6 (k*) | 42 | 0.1749 | 0.001 | +0.0046 [+0.0027, +0.0063] | k=1_better |
| popquorn_politeness | 512 | 1 | 512 | 0.1680 | 0.001 | +0.0000 [+0.0000, +0.0000] | baseline |
| popquorn_politeness | 512 | 2 | 256 | 0.1685 | 0.001 | +0.0005 [-0.0005, +0.0015] | tie |
| popquorn_politeness | 512 | 4 | 128 | 0.1690 | 0.001 | +0.0009 [-0.0002, +0.0020] | tie |
| popquorn_politeness | 512 | 6 (k*) | 85 | 0.1712 | 0.001 | +0.0032 [+0.0017, +0.0046] | k=1_better |
| summeval_expert | 128 | 1 | 128 | 0.1411 | 0.01 | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_expert | 128 | 2 (k*) | 64 | 0.1408 | 0.01 | -0.0002 [-0.0009, +0.0005] | tie |
| summeval_expert | 256 | 1 | 256 | 0.1367 | 0.001 | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_expert | 256 | 2 (k*) | 128 | 0.1408 | 0.01 | +0.0041 [+0.0028, +0.0053] | k=1_better |
| summeval_expert | 512 | 1 | 512 | 0.1372 | 0.001 | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_expert | 512 | 2 (k*) | 256 | 0.1363 | 0.001 | -0.0009 [-0.0019, +0.0001] | tie |
| summeval_crowd | 128 | 1 | 128 | 0.1298 | 1.0 | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_crowd | 128 | 2 | 64 | 0.1306 | 1.0 | +0.0007 [+0.0006, +0.0009] | k=1_better |
| summeval_crowd | 128 | 4 | 32 | 0.1311 | 0.1 | +0.0013 [+0.0010, +0.0015] | k=1_better |
| summeval_crowd | 128 | 5 (k*) | 25 | 0.1309 | 0.1 | +0.0011 [+0.0009, +0.0013] | k=1_better |
| summeval_crowd | 256 | 1 | 256 | 0.1294 | 1.0 | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_crowd | 256 | 2 | 128 | 0.1297 | 1.0 | +0.0003 [-0.0000, +0.0007] | tie |
| summeval_crowd | 256 | 4 | 64 | 0.1301 | 1.0 | +0.0007 [+0.0001, +0.0013] | k=1_better |
| summeval_crowd | 256 | 5 (k*) | 51 | 0.1305 | 1.0 | +0.0011 [+0.0004, +0.0017] | k=1_better |
| summeval_crowd | 512 | 1 | 512 | 0.1292 | 0.1 | +0.0000 [+0.0000, +0.0000] | baseline |
| summeval_crowd | 512 | 2 | 256 | 0.1297 | 0.1 | +0.0005 [+0.0002, +0.0008] | k=1_better |
| summeval_crowd | 512 | 4 | 128 | 0.1299 | 1.0 | +0.0007 [+0.0005, +0.0009] | k=1_better |
| summeval_crowd | 512 | 5 (k*) | 102 | 0.1303 | 1.0 | +0.0012 [+0.0009, +0.0015] | k=1_better |
| dices990 | 256 | 1 | 256 | 0.1982 | 0.01 | +0.0000 [+0.0000, +0.0000] | baseline |
| dices990 | 256 | 2 | 128 | 0.1990 | 0.01 | +0.0007 [-0.0013, +0.0028] | tie |
| dices990 | 256 | 4 | 64 | 0.2012 | 0.01 | +0.0030 [-0.0008, +0.0070] | tie |
| dices990 | 256 | 8 | 32 | 0.1992 | 0.01 | +0.0009 [-0.0020, +0.0038] | tie |
| dices990 | 256 | 16 (k*) | 16 | 0.2007 | 0.1 | +0.0024 [-0.0004, +0.0053] | tie |
| dices990 | 512 | 1 | 512 | 0.1978 | 0.01 | +0.0000 [+0.0000, +0.0000] | baseline |
| dices990 | 512 | 2 | 256 | 0.1981 | 0.01 | +0.0003 [-0.0014, +0.0020] | tie |
| dices990 | 512 | 4 | 128 | 0.1982 | 0.01 | +0.0003 [-0.0013, +0.0022] | tie |
| dices990 | 512 | 8 | 64 | 0.1995 | 0.01 | +0.0017 [-0.0016, +0.0051] | tie |
| dices990 | 512 | 16 (k*) | 32 | 0.1984 | 0.01 | +0.0006 [-0.0014, +0.0024] | tie |
| chaosnli_snli_mnli | 128 | 1 | 128 | 0.2001 | 0.01 | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_snli_mnli | 128 | 2 | 64 | 0.1992 | 0.1 | -0.0008 [-0.0021, +0.0004] | tie |
| chaosnli_snli_mnli | 128 | 4 | 32 | 0.2012 | 0.1 | +0.0011 [-0.0004, +0.0026] | tie |
| chaosnli_snli_mnli | 128 | 5 (k*) | 25 | 0.2043 | 0.1 | +0.0042 [+0.0024, +0.0060] | k=1_better |
| chaosnli_snli_mnli | 128 | 8 | 16 | 0.2013 | 0.1 | +0.0012 [-0.0005, +0.0030] | tie |
| chaosnli_snli_mnli | 256 | 1 | 256 | 0.1978 | 0.01 | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_snli_mnli | 256 | 2 | 128 | 0.1992 | 0.01 | +0.0014 [+0.0003, +0.0026] | k=1_better |
| chaosnli_snli_mnli | 256 | 4 | 64 | 0.1991 | 0.1 | +0.0013 [+0.0002, +0.0024] | k=1_better |
| chaosnli_snli_mnli | 256 | 5 (k*) | 51 | 0.2003 | 0.1 | +0.0025 [+0.0010, +0.0040] | k=1_better |
| chaosnli_snli_mnli | 256 | 8 | 32 | 0.1996 | 0.1 | +0.0018 [+0.0009, +0.0026] | k=1_better |
| chaosnli_snli_mnli | 256 | 16 | 16 | 0.2012 | 0.1 | +0.0034 [+0.0024, +0.0044] | k=1_better |
| chaosnli_snli_mnli | 512 | 1 | 512 | 0.1968 | 0.01 | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_snli_mnli | 512 | 2 | 256 | 0.1976 | 0.01 | +0.0008 [-0.0000, +0.0016] | tie |
| chaosnli_snli_mnli | 512 | 4 | 128 | 0.1981 | 0.01 | +0.0013 [-0.0001, +0.0027] | tie |
| chaosnli_snli_mnli | 512 | 5 (k*) | 102 | 0.1981 | 0.01 | +0.0014 [+0.0001, +0.0027] | k=1_better |
| chaosnli_snli_mnli | 512 | 8 | 64 | 0.1992 | 0.1 | +0.0024 [+0.0009, +0.0038] | k=1_better |
| chaosnli_snli_mnli | 512 | 16 | 32 | 0.1999 | 0.1 | +0.0031 [+0.0019, +0.0043] | k=1_better |
| chaosnli_alpha | 128 | 1 | 128 | 0.2530 | 1.0 | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_alpha | 128 | 2 | 64 | 0.2562 | 1.0 | +0.0031 [+0.0010, +0.0052] | k=1_better |
| chaosnli_alpha | 128 | 3 (k*) | 42 | 0.2662 | 0.1 | +0.0132 [+0.0091, +0.0174] | k=1_better |
| chaosnli_alpha | 128 | 4 | 32 | 0.2633 | 1.0 | +0.0103 [+0.0065, +0.0140] | k=1_better |
| chaosnli_alpha | 128 | 8 | 16 | 0.2635 | 1.0 | +0.0105 [+0.0041, +0.0168] | k=1_better |
| chaosnli_alpha | 256 | 1 | 256 | 0.2509 | 1.0 | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_alpha | 256 | 2 | 128 | 0.2513 | 1.0 | +0.0004 [-0.0002, +0.0010] | tie |
| chaosnli_alpha | 256 | 3 (k*) | 85 | 0.2531 | 1.0 | +0.0022 [+0.0009, +0.0034] | k=1_better |
| chaosnli_alpha | 256 | 4 | 64 | 0.2571 | 1.0 | +0.0062 [+0.0041, +0.0083] | k=1_better |
| chaosnli_alpha | 256 | 8 | 32 | 0.2623 | 1.0 | +0.0114 [+0.0073, +0.0153] | k=1_better |
| chaosnli_alpha | 256 | 16 | 16 | 0.2596 | 1.0 | +0.0087 [+0.0035, +0.0138] | k=1_better |
| chaosnli_alpha | 512 | 1 | 512 | 0.2502 | 1.0 | +0.0000 [+0.0000, +0.0000] | baseline |
| chaosnli_alpha | 512 | 2 | 256 | 0.2507 | 0.1 | +0.0005 [+0.0001, +0.0010] | k=1_better |
| chaosnli_alpha | 512 | 3 (k*) | 170 | 0.2526 | 1.0 | +0.0024 [+0.0020, +0.0027] | k=1_better |
| chaosnli_alpha | 512 | 4 | 128 | 0.2513 | 1.0 | +0.0012 [+0.0010, +0.0013] | k=1_better |
| chaosnli_alpha | 512 | 8 | 64 | 0.2548 | 1.0 | +0.0046 [+0.0023, +0.0069] | k=1_better |
| chaosnli_alpha | 512 | 16 | 32 | 0.2608 | 1.0 | +0.0106 [+0.0073, +0.0138] | k=1_better |

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
