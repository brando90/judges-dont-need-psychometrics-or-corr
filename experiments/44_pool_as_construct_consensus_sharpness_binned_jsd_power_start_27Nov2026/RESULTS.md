# Experiment 44 results: rater pool as construct, consensus items, binned JSD, and power at one rating per item

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/RESULTS.md>

**TLDR:** Five frozen frontier judges, five public multi-annotator rating sets, nothing fitted, CPU only, $0. (P) Switching the rater pool that defines p* between demographic groups did not detectably move the item-level rating distributions, the judges' scores or their ranking in the two DICES sets (0 of 30 score cells and 0 of 60 pair-ordering cells Holm-significant; pools as small as 11 raters, so low power). (S) A proper score penalizes sharpening much less on consensus items than on contested items in 25 of 25 judge-by-set cells, but sharpening improved the consensus-stratum score in only 5 of 25. (J) Binned JSD is exactly unchanged when forecasts are shuffled among items of the same human bin, while the proper-score risk gets worse in 25 of 25 cells; its top judge differs from the proper-score top judge in 2 of 5 sets. (M) At one rating per item and 400 items, 0 to 6 of the 10 judge pairs per set have a difference large enough to resolve; at equal total ratings, more items always beat more ratings per item.

**As of:** 11-27-2026 (Friday, Pacific time). Written by Claude (Sonnet 5.5) in Claude Code for Brando Miranda. Every number below is printed by `src/summarize.py` into `results/key_numbers.md` from `results/*.csv`, and the per-rating numbers are re-derived by `src/verify.py` (185 of 185 checks pass).

## What was run

- Frozen protocol: `PROTOCOL.md` (commit `13d1177`). Amendment: `AMENDMENT_A1.md` (commit `3cba685`, three statistical corrections written before any arm ran: a rater-level permutation null for P1, a two-way items-by-raters bootstrap for P2 and P3, and a comparison of the H-M1 subsampling check with the exact subsampling variance at tolerance 0.9 to 1.1). One further line of the amendment (raters whose demographics differ between their own rows are excluded from the affected contrast) was written when the rater table was built, before any arm produced a result, and is committed together with the code rather than before it.
- Deviations from the frozen text, all disclosed: (1) the output files are nine CSVs (`arm_p_tvd`, `arm_p_judges`, `arm_p_pairs`, `arm_s_curves`, `arm_s_gain`, `arm_j_jsb`, `arm_j_permutation`, `arm_m_pairs`, `arm_m_check`) instead of the four named in the protocol; (2) the amendments above; (3) a development smoke run (200 bootstrap draws, 20 permutations, written to a temporary folder and not kept) was executed before the full run and printed the same hypothesis outcomes; no analysis choice was changed after any numerical output was seen.
- Run: `python3 src/run_all.py` (4,000 bootstrap draws, 500 permutations, seeds `SeedSequence([20261127, 44, arm, ...])`). Two runs give byte-identical CSVs and `summary.json`. `python3 src/verify.py` recomputes primary risks, binned JSD, pool TVD and pool risks, stratum risks on a temperature grid and the variance components with plain per-rating loops and matches all of them to 1e-9. `pytest tests` passes (16 tests).
- Data and judges: five frozen judges of Experiment 28 (gpt-6-astra, claude-opus-5-5, kimi-k3, glm-5.3, deepseek-v4.1-flash), one temperature-0 forecast per item; sets ChaosNLI SNLI (600 items), MNLI-m (600), alphaNLI (400), DICES-350 (350), DICES-990 (400). No fitting happens, so there is no train/validation/test split and every item is scored once. The analyst had seen Experiment 28's all-item rankings of these judges before freezing (listed in `PROTOCOL.md`), so none of the judge orderings here is a blind prospective result; the arms below were not computed before freezing.
- Scores (all lower is better, unbiased per-sample risk: mean over each item's ratings, then mean over items): multiclass Brier score for ChaosNLI, ranked probability score (RPS) normalized by K minus 1 for DICES. Reference values: a uniform forecast scores 0.6667 on three-class Brier and 0.5 on two-class Brier for every rating, and on three-class normalized RPS it scores 0.2778 for a rating of No or Yes and 0.1111 for Unsure (`tests/test_lib44.py`). The judges' all-item risks in this experiment range from 0.1933 to 0.6335.

## Outcome of every pre-registered hypothesis

| ID | Statement (short) | Outcome | Key number |
|---|---|---|---|
| H-P1 | in each DICES set at least one pool contrast has a Holm-significant distribution difference | **not supported** | 0 of 6 cells; smallest raw p 0.098, its Holm p 0.587 |
| H-P2 | in each DICES set at least one judge-by-contrast score difference between pools is Holm-significant | **not supported** | 0 of 30 cells; smallest raw p 0.114 |
| P3 | pool-dependent ordering (descriptive, no direction predicted) | **pool-robust ranking** by the registered rule | Kendall tau-b 0.8 to 1.0 in all six contrasts; 4 of 60 pairs reverse by point estimate, 0 Holm-significant |
| H-S1 | sharpening helps relatively more on consensus than on contested items | **supported** | 25 of 25 cells positive, 25 of 25 intervals exclude 0 |
| H-J1 | binned JSD exactly invariant to within-bin forecast shuffles | **supported** | largest change 2.4e-16 (limit 1e-12) over 200 shuffles in each of 25 cells |
| H-J2 | the same shuffle worsens the proper-score risk | **supported** | 25 of 25 cells, increase +0.0086 to +0.1075 |
| H-M1 | planning variance formula matches subsampling | **supported** at the amended tolerance | 30 of 30 cells, ratio to exact variance 0.941 to 1.053; the originally frozen criterion (0.6 to 1.1 times V(N, k)) is also met, ratio 0.645 to 0.992 |
| H-M2 | at equal total ratings the variance is nondecreasing in k | **identity holds** (checked, not tested) | 50 of 50 pairs |

## Arm P: does the pool that defines p* change the distribution, the score or the ranking?

Data: DICES-350 (350 items, 123 raters per item) and DICES-990 (400 items, 69 to 76 raters per item), all eligible items (at least 5 raters of each pool); nothing fitted. Pool A versus pool B: gender Man versus Woman, age group gen z versus millenial plus gen x+, race White versus Black/African American. Raters whose attribute differs between their own rows are excluded from that contrast (DICES-990: one rater for age, one for race).

| Set, contrast (raters in A / B) | Items | Mean TVD of the two pools' per-item distributions | Rater-label permutation null: mean (95th percentile) | Raw p | Holm p |
|---|---|---|---|---|---|
| DICES-350 gender (61 / 62) | 350 | 0.0917 | 0.0777 (0.0980) | 0.098 | 0.587 |
| DICES-350 age (56 / 67) | 350 | 0.0758 | 0.0786 (0.0970) | 0.521 | 1.000 |
| DICES-350 race (30 / 29) | 350 | 0.1175 | 0.1095 (0.1346) | 0.228 | 1.000 |
| DICES-990 gender (80 / 88) | 400 | 0.0934 | 0.0980 (0.1224) | 0.567 | 1.000 |
| DICES-990 age (30 / 139) | 400 | 0.1262 | 0.1282 (0.1567) | 0.477 | 1.000 |
| DICES-990 race (27 / 11) | 396 | 0.1830 | 0.1858 (0.2334) | 0.449 | 1.000 |

TVD is total variation distance, 0.5 times the sum over the three ratings of the absolute difference of the two pools' rating shares, averaged equally over items; 0 means identical distributions, 1 means disjoint. Even two random rater groups differ (null mean 0.08 to 0.19), which is why the observed values are compared with a rater-level permutation null (500 permutations, group sizes preserved) and not with 0.

Scores. For each judge, risk against pool A minus risk against pool B (normalized RPS; positive means the judge looks worse against pool A), with 95% two-way (items by raters) bootstrap intervals from 4,000 draws. Within every contrast the five judges move in the same direction and by similar amounts: DICES-350 gender −0.0361 to −0.0247, age −0.0102 to −0.0072, race −0.0126 to −0.0034; DICES-990 gender +0.0161 to +0.0229, age +0.0355 to +0.0524, race −0.0912 to −0.0698. The largest shift in absolute value is 0.0912 (DICES-990 race, pool B has only 11 raters) and its interval is wide (gpt-6-astra: −0.0912, interval −0.2904 to +0.0416). No cell is Holm-significant (smallest raw p 0.114).

Ranking. Kendall tau-b between the judges' order under pool A and under pool B is 0.8 or 1.0 in all six contrasts. Four of the 60 judge-pair-by-contrast cells reverse by point estimate, and all four involve gpt-6-astra against kimi-k3 or deepseek-v4.1-flash with a difference below 0.01 under both pools (largest raw p 0.267; none Holm-significant). The best judge (claude-opus-5-5) and the second best (glm-5.3) hold their places in all twelve (set, contrast, pool) orders. By the rule fixed in the protocol (no Holm-significant cell and tau-b at least 0.8 in all six contrasts) the reading is **pool-robust ranking**.

What this does and does not show. It shows that in these two DICES subsets, with these pool definitions and pool sizes, the ordering of five strong judges is not detectably pool-dependent and the pool shifts every judge about equally. It does not show that the pool never matters: the distribution test has low power (null 95th percentiles of 0.10 to 0.23 against observed values of 0.08 to 0.18), the smallest pools have 11 and 27 raters, and the judges were told to forecast a whole crowd, not a sub-pool. The construct statement (p* is defined by pool and rubric, stated before any metric) is a requirement of the method, not a finding of this arm.

## Arm S: are peaked forecasts rewarded on consensus items?

Data: all five sets, all items, frozen judges. Each item's ratings are split at random into two halves; half A defines the item's agreement (share of its most common category) and the strata (lowest third = contested, highest third = consensus, floor(N/3) items each); half B is scored. The temperature transform raises the forecast to the power 1/tau and renormalizes (tau < 1 sharpens, tau > 1 softens; the top-1 label never changes, checked in all cells). Judge-mean risk on half B, Brier for ChaosNLI and normalized RPS for DICES, lower is better:

| Set | Stratum (median half-A agreement) | tau = 0.5 (sharper) | tau = 1 (as forecast) | tau = 2 (softer) |
|---|---|---|---|---|
| ChaosNLI SNLI | consensus (0.92) | 0.2117 | 0.2086 | 0.2757 |
| | contested (0.56) | 0.7471 | 0.6304 | 0.5719 |
| ChaosNLI MNLI-m | consensus (0.82) | 0.4858 | 0.4382 | 0.4540 |
| | contested (0.52) | 0.8500 | 0.7084 | 0.6259 |
| ChaosNLI alphaNLI | consensus (0.98) | 0.0737 | 0.0783 | 0.1195 |
| | contested (0.74) | 0.5653 | 0.4937 | 0.4438 |
| DICES-350 | consensus (0.84) | 0.1941 | 0.1723 | 0.1685 |
| | contested (0.54) | 0.3971 | 0.3357 | 0.2797 |
| DICES-990 | consensus (0.89) | 0.1149 | 0.1080 | 0.1146 |
| | contested (0.54) | 0.3779 | 0.3170 | 0.2674 |

H-S1 is supported. The sharpening gain G (risk at tau = 1 minus risk at tau = 0.5; positive means sharpening helps) is higher on consensus than on contested items in all 25 judge-by-set cells, by 0.0320 to 0.1351, and the 95% item-bootstrap interval of that difference excludes 0 in all 25.

The honest reading is narrower than "judges should be peaky on consensus items". Sharpening to tau = 0.5 lowered the consensus-stratum risk in only 5 of 25 cells (alphaNLI for four of five judges, and glm-5.3 on SNLI) and the contested-stratum risk in none. On contested items the best temperature on the grid is the softest value, tau = 2, in 23 of 25 cells: the judges are overconfident there. On consensus items the best temperature is sharper than on contested items in 24 of 25 cells (equal in one), but it is still above 1 (softer than the judge's own forecast) in 12 of 25 cells, equal to 1 in 5 and below 1 in 8. What the arm supports for the paper is therefore: a proper score does not punish a peaked forecast on consensus items the way it punishes it on contested items, which is the reason to report scores by agreement stratum; it does not show that these five judges are too soft on consensus items. ChaosNLI SNLI and MNLI-m items were selected toward moderate original disagreement, so their consensus strata (median agreement 0.92 and 0.82) are milder than alphaNLI's (0.98).

## Arm J: binned JSD

Binned JSD (Elangovan et al., ICLR 2025; log base 2; lower is better) pools the human ratings and averages the judge's forecasts inside bins defined by the human median (ordinal, DICES) or the most frequent human label (nominal, ChaosNLI), then averages the Jensen-Shannon divergence over bins weighted by bin size. Frozen judges, all items.

H-J1 and H-J2 are both supported. Reassigning each judge's forecast vectors at random among the items of the same human bin (200 shuffles per judge and set) leaves binned JSD unchanged up to floating-point rounding (largest change 2.4e-16 in 25 cells), while the proper-score risk (Brier or normalized RPS, all items) gets worse in all 25 cells: by 0.0943 to 0.1075 on SNLI, 0.0657 to 0.0858 on MNLI-m, 0.0571 to 0.0734 on alphaNLI, 0.0086 to 0.0101 on DICES-350 and 0.0126 to 0.0202 on DICES-990 (largest standard error over the 200 shuffles 0.0004). The shuffle is a diagnostic of the metric, not a judge: it shows that binned JSD cannot see which item inside a bin a forecast belongs to.

Descriptive comparison of the five-judge rankings (no test; five judges give a tau-b resolution of 0.2):

| Set | Bins | Kendall tau-b, binned-JSD order versus proper-score order | Best judge by binned JSD | Best judge by proper score |
|---|---|---|---|---|
| ChaosNLI SNLI | 3 | −0.8 | kimi-k3 | claude-opus-5-5 |
| ChaosNLI MNLI-m | 3 | +0.6 | glm-5.3 | claude-opus-5-5 |
| ChaosNLI alphaNLI | 2 | +0.4 | gpt-6-astra | gpt-6-astra |
| DICES-350 | 3 | +0.6 | claude-opus-5-5 | claude-opus-5-5 |
| DICES-990 | 3 | +0.8 | claude-opus-5-5 | claude-opus-5-5 |

The top judge differs in 2 of 5 sets (SNLI and MNLI-m). No claim about which ranking is right is made from five judges and no pre-registered test; the supported statement is the invariance and the paired shuffle result.

## Arm M: how many held-out items does a paired comparison need at one rating per item?

Data: all 10 judge pairs in each of the five sets, primary score, variance components estimated from all items and all ratings (sigma_b squared between items, sigma_w squared within items), minimum detectable difference (MDD) at two-sided 5% and 80% power, V(N, k) = (sigma_b squared + sigma_w squared divided by k) divided by N.

| Set | Observed difference between judge pairs, absolute value | MDD at N = 400, k = 1 | MDD at N = 400, k = 10 | Pairs resolvable at N = 400, k = 1 (N_80 at most 400) | N_80 at k = 1: median (smallest to largest over the 10 pairs) |
|---|---|---|---|---|---|
| ChaosNLI SNLI (Brier) | 0.0018 to 0.0418 | 0.0337 to 0.0526 | 0.0179 to 0.0270 | 0 of 10 | 1,607 (538 to 302,931) |
| ChaosNLI MNLI-m (Brier) | 0.0008 to 0.0966 | 0.0381 to 0.0646 | 0.0188 to 0.0328 | 6 of 10 | 292 (157 to 1,868,671) |
| ChaosNLI alphaNLI (Brier) | 0.0049 to 0.0711 | 0.0355 to 0.0625 | 0.0207 to 0.0498 | 2 of 10 | 2,440 (259 to 23,791) |
| DICES-350 (normalized RPS) | 0.0012 to 0.0483 | 0.0163 to 0.0247 | 0.0084 to 0.0123 | 5 of 10 | 785 (102 to 89,468) |
| DICES-990 (normalized RPS) | 0.0002 to 0.0430 | 0.0144 to 0.0235 | 0.0072 to 0.0113 | 5 of 10 | 419 (110 to 1,617,440) |

N_80 is the number of items at which the pair's observed full-sample difference is exactly detectable with 80% power. It is a planning scale built from the observed difference, not a post-hoc power claim about a true effect. Ten ratings per item shrink the MDD at N = 400 by a median factor 2.0 (range 1.21 to 2.23); ten times as many items would shrink it by 3.16. At equal total ratings the variance is (sigma_b squared k + sigma_w squared) divided by the budget, which never decreases in k, so items beat ratings per item (H-M2, 50 of 50 pairs). Consequences for the paper: with enough items one rating per held-out item is statistically adequate for comparing judges, the number of items needed depends on how close the two judges are (smallest N_80 of 102 to 538 for the best-separated pairs, 23,791 to 1,868,671 for the nearest ties), and the nearest ties need far more items than the 350 to 600 these sets contain.

H-M1: the empirical variance of the paired mean over 2,000 subsamples (items drawn without replacement, k ratings per item without replacement; N in {50, 100, 200}, k in {1, 5}, the pair with the median observed difference in each set) is 0.941 to 1.053 times the exact subsampling variance in all 30 cells, and 0.645 to 0.992 times the infinite-population planning formula V(N, k). The frozen criterion (0.6 to 1.1 times V(N, k)) was also met.

## Sensitivity: drop every item on which any judge's forecast was invalid

Invalid forecasts are replaced by the uniform forecast in the primary analysis (Experiment 28 and 46 rule). Dropping every item on which any of the five judges was invalid leaves 521, 465, 320, 323 and 368 items in SNLI, MNLI-m, alphaNLI, DICES-350 and DICES-990. Every hypothesis outcome above is unchanged (`results/sensitivity_drop_invalid/`). The share of resolvable pairs at N = 400 and k = 1 changes to 2, 6, 4, 5 and 6 of 10 (SNLI, MNLI-m, alphaNLI, DICES-350, DICES-990); the tau-b values change by at most 0.6 (SNLI binned JSD versus proper score moves from −0.8 to −0.2). `src/verify.py` covers the main run only.

## Limits

- Five frozen judges, one temperature-0 forecast each; rank statistics over five judges have a resolution of 0.2, and none of the ranking comparisons is a test.
- The data had been seen in Experiment 28 and the judges' all-item rankings were known before freezing.
- Arm P uses sub-pools of a whole-crowd forecast on purpose; pools of 11 to 139 raters; ChaosNLI has no rater identities, and rubric shifts were not tested.
- Arms S, J and M use the item-cluster bootstrap and ignore rater sampling variance (amendment A1, point 2); paired differences between judges on the same items largely cancel shared rater effects, but intervals there are not two-way.
- No multiplicity correction across arms; Holm is applied inside the families named in the protocol, and the arms without registered tests (J ranking, M planning) are descriptive.
- Not run: training-budget split between items and raters, drift across judge generations, any API or GPU job.

## Files

`results/*.csv` and `summary.json` (nine CSVs plus the summary), `results/key_numbers.md` (every number above), `results/sensitivity_drop_invalid/`, `figures/fig1_pool_contrast.png`, `fig2_sharpening_by_stratum.png`, `fig3_binned_jsd.png`, `fig4_power.png`, `src/` (`lib44.py`, `run_all.py`, `verify.py`, `summarize.py`, `make_figures.py`), `tests/test_lib44.py`, `HOW_TO_READ_RESULTS.md`.
