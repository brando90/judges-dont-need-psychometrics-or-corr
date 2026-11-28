# Experiment 53 protocol: Experiment 41's heads re-expressed as skill scores over the human base rate (27 Nov 2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/53_exp41_heads_as_skill_scores_with_item_bootstrap_cis_start_27Nov2026/PROTOCOL.md>

**Status:** frozen before `scripts/skill_reanalysis.py` produced any number. Committed to `main` before the run.

## Why

The advisor's note S8 on the abstract: a maximum raw ranked probability score (RPS) gap of 0.128 cannot be judged without a scale, and comparing cross-entropy-trained (CE-trained) with Pearson-trained heads on RPS favors the proper fit by construction. This experiment re-expresses Experiment 41's preregistered K-sweep (16 views, 5 split seeds, ridge penalty 0.01, at most 35 L-BFGS iterations) as skill scores with item-clustered intervals. It trains nothing new: it rebuilds the same heads with Experiment 41's unchanged code (imported read-only) to recover per-item test scores, which Experiment 41's JSON does not store.

## Definitions

- Score S: normalized RPS = RPS/(K-1) of one held-out human rating per (item, rating) pair, averaged as in Experiment 41 (per-item mean over that item's test ratings, then mean over test items). Range 0 to 1, lower is better.
- Base rate: the training-split marginal rating distribution, used as a constant forecast for every test item (Experiment 41's "constant").
- Skill of head h: Skill(h) = 1 - S(h)/S(base rate), computed on the same test items. 0 = no better than the base rate, 1 = perfect, negative = worse than the base rate.
- Heads: CE-trained, Pearson-trained, MSE-trained linear heads; "raw" = the head's own forecast; "calibrated" = the head's scalar mean passed through its own validation-fitted multinomial-logit map (Experiment 41's matched calibration).
- Skill difference: Skill(CE-trained) minus Skill(Pearson-trained), raw and calibrated; positive = the CE-trained head is better. Equivalently (S(Pearson) - S(CE))/S(base rate).

## Uncertainty

Item-clustered percentile bootstrap, 2,000 draws, seed 20261127: within each split seed, test items are resampled with replacement (the same draw for every head, so differences are paired); the reported value is the mean over the 5 split seeds. Also reported: the range of the per-seed values. Two-sided bootstrap p-value = 2 min(share of draws <= 0, share >= 0); Holm-Bonferroni across the 16 views within each family (raw, calibrated).

## Sanity check (must pass before any result is reported)

Per (view, seed), the mean normalized RPS of the base rate and of every head (raw and calibrated) must equal Experiment 41's `results/k_sweep_matched_calibration_maxiter35.json` within 1e-9. If not, stop and report.

## Predictions

- P1. The raw skill difference is positive with a Holm-adjusted p < 0.05 on every view where Experiment 41's raw gap exceeds 0.01 normalized RPS.
- P2. The calibrated skill difference lies within [-0.025, +0.025] skill units on all 16 views (0.025 is about 0.0024 / 0.1156, Experiment 41's largest matched gap over its smallest base-rate score).
- P3 (descriptive, no decision rule). The CE-trained head's raw skill is below 0.10 on every view, so headroom over the base rate is small compared with the raw gap.

## What this cannot show

It cannot break the circularity: the comparison is still scored by a proper rule that the CE-trained head optimizes. It quantifies on a bounded scale how much a correlation-trained head loses before and after calibration with ample labels. Label scarcity (Experiment 47), shift (Experiment 48) and real-judge rank reversals (Experiment 46) are other workers' experiments and are cited, not redone.

## Command

`E40_DATA_CACHE=<copy of the pinned data cache> python3 scripts/skill_reanalysis.py --boot 2000` (CPU, about 5 minutes).
