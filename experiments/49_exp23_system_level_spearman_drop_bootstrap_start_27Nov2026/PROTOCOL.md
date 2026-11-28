# Experiment 49 protocol: is Experiment 23's system-level Spearman drop (0.877 raw Pearson fit vs 0.835 calibrated) significant, and why does it happen? (frozen 11-27-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/49_exp23_system_level_spearman_drop_bootstrap_start_27Nov2026/PROTOCOL.md>

**TLDR:** Reproduce the two numbers from Experiment 23's code and inputs, then put intervals on their difference by resampling test documents and, separately, the 16 systems, and diagnose ties, the number of systems and the four-bin calibrator. Frozen before the script exists.

**As of:** 11-27-2026 (Friday, Pacific time). CPU, $0.

## Reproduction

`23_*/summeval/prepare.py` rebuilds `summeval_consistency.npz` (SHA-256 bbd81648...7dbd7d0 must match; checked: it does). Experiment 23's `analyze.py` logic (native mode, depth 1, budget 768, 64 training draws) is re-run with the per-draw test forecasts kept. Success = the mean over draws of system-level Spearman reproduces 0.8765 (pearson) and 0.8349 (calibrated_pearson) to 4 decimals. If it does not, the discrepancy is the first result.

## Tests

- Statistic: Delta = mean over the 64 draws of [system Spearman(pearson) - system Spearman(calibrated_pearson)].
- (i) Training-draw spread: SD and 2.5/97.5 percentiles of the per-draw difference (conditional on the fixed test set).
- (ii) Document bootstrap: resample the 40 test source documents with replacement (all 16 systems' summaries of a document move together), recompute system means, both Spearmans and Delta for every draw; 2,000 resamples, seed 20261127; percentile 95% interval and two-sided p.
- (iii) System bootstrap: resample the 16 systems with replacement (paired across methods); 2,000 resamples; and the leave-one-system-out jackknife of Delta.
- (iv) Both: documents and systems resampled jointly.
- **Decision rule.** The drop is "significant" only if the 95% intervals of Delta under (ii) and (iii) both exclude 0.
- Diagnostics: number of distinct item-level expected scores per method; tied pairs in the human and predicted system means; system pair accuracy; Kendall tau-b; a continuous calibrator in place of the four-bin one (Experiment 41's multinomial logit on the standardized Pearson-selected mean score, fitted on the same training votes) to test whether the drop comes from the binning.
- A power note: the SD of Spearman over 16 systems at the observed level, from (iii).
