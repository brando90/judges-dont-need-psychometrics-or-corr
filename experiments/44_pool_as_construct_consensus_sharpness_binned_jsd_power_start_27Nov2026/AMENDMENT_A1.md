# Amendment A1 to the Experiment 44 protocol (11-27-2026, written before any arm was run)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/AMENDMENT_A1.md>

**TLDR:** Three statistical corrections found while writing the analysis code, before any arm ran. `PROTOCOL.md` is unchanged; where the two differ, this file wins. Nothing in the results can have influenced them.

## Corrections

1. **P1 permutation null is at the rater level.** The protocol shuffled pool labels among each item's raters. The same raters rate many items (DICES-350: all 123 raters rate every item), so rater-level heterogeneity shared across items would make a within-item shuffle too narrow and flag random rater groups as different pools. The null now permutes the pool label across raters (group sizes at the rater level preserved) and recomputes the mean per-item TVD with the same eligibility rule (at least 5 raters of each pool on the item) under each permutation. Items that fail the rule under a permutation are dropped from that permutation's mean; the count of permutations with no eligible item is reported (expected 0). Still 500 permutations, p = (1 + #{null >= observed}) / 501, Holm over the 6 cells.
2. **P2 and P3 use a two-way bootstrap, not an item-only bootstrap.** Raters are shared across items, so item resampling alone understates the uncertainty of a pool-level quantity. Each draw resamples items with replacement and, independently inside each pool, raters with replacement; an item with no resampled rater in a pool is dropped from that draw's mean for both pools. 4,000 draws, same percentile intervals, p-values and Holm families as the protocol. D(a, b) equals Delta_pool(a) - Delta_pool(b), so the 60 interaction cells are contrasts of the same draws. Arms S, J and M keep the item-cluster bootstrap of the protocol; ChaosNLI has no rater ids, and for paired differences between judges shared rater effects largely cancel. This limit is stated in `RESULTS.md`.
3. **H-M1 compares with the exact variance of the subsampling design.** The planning formula V(N, k) assumes items are drawn from an infinite population. The check draws N items without replacement from N_tot items and k ratings without replacement from each item's n_i, whose exact variance is V_sub(N, k) = (1 - N/N_tot) S_D^2 / N + (1/N) mean_i[(s_i^2 / k)(1 - k/n_i)], with S_D^2 the sample variance (ddof 1) of the item means D_i. H-M1 now reads: the empirical variance over 2,000 subsamples is within 0.9 to 1.1 of V_sub(N, k) (the Monte Carlo relative standard error of a variance from 2,000 draws is about 3%). The ratio to V(N, k) is reported, not tested. "Pair median" means the pair whose observed |R(a) - R(b)| is the median of the 10 pairs in that set, so the check has 5 sets x 3 values of N x 2 values of k = 30 cells.

## Clarifications (no change of meaning)

- Arm S: strata order items by (half-A plurality share, item index) ascending; temperature uses the floored forecast, then the transform, then the floor again.
- Arm J: ordinal bins use the lower median, sorted[(n - 1) // 2] (0-based) of an item's ratings; nominal bins use the first index of the largest count.
- Invalid forecasts become the uniform forecast before flooring (primary); the sensitivity run drops every item on which any of the five judges was invalid.
- Eligibility in Arm P is evaluated on the observed data for the reported eligible-item counts and per draw or per permutation as stated above.
