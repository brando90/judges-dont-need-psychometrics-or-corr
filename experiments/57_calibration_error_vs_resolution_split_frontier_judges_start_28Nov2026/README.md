# Experiment 57: calibration error versus resolution on the five frozen frontier judges

> **Note 11-28-2026 (QA; results unchanged).** Read "calibration error" and "resolution" below as the estimated calibration gain (R_raw − R_cal) and the calibrated skill (UNC − R_cal): finite-sample estimates with a restricted map family, not guaranteed lower bounds on the population terms. The relation raw skill = calibrated skill − estimated gain holds by construction, so the residual of 3e-17 is a floating-point check of the code, not evidence for the decomposition.


Status: run 11-28-2026 on the frozen bundle, CPU only, no model calls, API spend $0. Pre-registered: [`PROTOCOL.md`](PROTOCOL.md) was committed in
`ae58f5c` before the first run of `scripts/run_exp57.py`; nothing in it was changed afterwards. Reading guide: [`HOW_TO_READ_RESULTS.md`](HOW_TO_READ_RESULTS.md).
Run it again with `python experiments/57_*/scripts/run_exp57.py` (about fifteen seconds; two runs give byte-identical result files) and the tests with
`python -m pytest experiments/57_*/tests -q` (9 pass). Data prerequisite: the frozen bundle `experiments/39_*/data/processed/study_a_bundle.npz` is not tracked by git. On a fresh clone, run `python3 src/fetch_raw_data.py` and then `python3 src/prepare_data.py` inside `experiments/39_*/` (sources and checksums: `experiments/39_*/data/README.md`). On 11-28-2026 `prepare_data.py`, run on the raw files in a scratch copy, wrote a bundle byte-identical to the one these results used (sha256 beginning `55935c4d`, full value in `results/hypotheses.json`); the fetch step was not re-run.

## Question
The paper recommends reporting a strictly proper score split into calibration error and resolution next to correlation. This experiment shows that split on
real judges: GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3 and DeepSeek V4.1 Flash (temperature 0, frozen forecasts) on ChaosNLI SNLI (600 items), MNLI-m (600), alphaNLI (400), DICES-350 (350) and DICES-990 (400),
25 judge-by-set cells. The primary rule is matched to the label encoding: Brier score on the three NLI sets, normalized ranked probability score (RPS) on the two DICES sets (ordered labels).
Skill is one minus the risk of the judge divided by the risk of the human base-rate forecast, so 0 means no better than the human label mix and higher is better.

## What was found (answer first)
1. **The calibration error is large on DICES and it is invisible to correlation.** Four of the five judges have raw skill below zero on both DICES sets (eight of the 25 cells), for example Astra on DICES-350 at
   -0.143 (95% interval -0.188 to -0.100), while its Pearson correlation with the human item mean is 0.588 (0.510 to 0.661), the second best of the five. After a per-judge calibration fitted on other folds, all 25 cells have positive skill with the interval above zero.
   Calibration error on DICES-350 (skill points): Astra 0.215, Opus 0.037, Kimi 0.188, GLM 0.153, DeepSeek 0.231; on DICES-990: 0.193, 0.029, 0.177, 0.133, 0.207.
2. **H3 supported (the DICES-350 rescue is not noise).** The four judges with negative raw skill (Astra, Kimi, GLM, DeepSeek) all have calibrated skill above zero, Holm-adjusted one-sided p = 0.001 for each (the smallest value 4,000 draws can give, times four tests).
3. **H2 supported (4 of 5 sets): judges differ more in calibration error than in resolution.** Between-judge standard deviations, calibration share versus resolution share: SNLI 0.025 versus 0.021 (thin), MNLI 0.060 versus 0.019,
   alphaNLI 0.010 versus 0.048 (the set that fails: binary labels, small calibration errors, DeepSeek lower in resolution), DICES-350 0.077 versus 0.010, DICES-990 0.072 versus 0.012.
4. **H1 falsified as pre-registered: correlation does not track resolution better than raw skill.** Kendall tau-b between the judge ranking by Pearson r and the ranking by resolution (calibrated skill), against the same with raw skill:
   SNLI 0.60 versus 0.60, MNLI 0.40 versus 1.00, alphaNLI 1.00 versus 1.00, DICES-350 0.80 versus 0.40, DICES-990 0.80 versus 0.80. The rule needed a strict gain in at least four sets and only DICES-350 shows one.
   All five bootstrap intervals of the difference contain zero (five judges is a coarse statistic: one swap moves tau by 0.2). The Spearman sensitivity points the other way in four of five sets (resolution 1.00, 0.80, 0.80, 0.80, 1.00 against raw 0.60, 0.60, 0.80, 0.40, 0.60),
   but it is a sensitivity run, the decision is the pre-registered Pearson one, and it stays falsified.
5. What the data therefore support: on these judges the disagreement between correlation and a proper score is mostly about LEVEL (a judge can be well correlated and worse than the base rate), not about the rank order of the judges. Rank reversals are limited to DICES-350 (Astra: second by Pearson, fourth of five by raw skill, first by resolution, 0.072 against Opus 0.069, an essentially tied top).

## Method in one paragraph
Per cell, five-fold cross-fitting over groups (SNLI items that share an image share a fold; every other item is its own group; seed `SeedSequence([20261128, 57, 0, set_index])`). UNC is the risk of the cross-fitted human base rate, R_raw the risk of the raw forecast, R_cal the risk of the forecast recalibrated by a map fitted on the other four folds
(Experiment 46's multinomial logit on [log q, 1], ridge 1e-3 toward the identity, individual-rating cross-entropy, L-BFGS-B from the identity). Calibration error = R_raw - R_cal, resolution = UNC - R_cal, and R_raw = UNC - resolution + calibration error by construction (largest residual 3e-17).
Intervals: group-clustered bootstrap, 4,000 draws, percentile 95%, one weight matrix per set shared by every judge and statistic. The bootstrap resamples the item scores of the cross-fitted forecasts; it does not refit the calibration map inside each draw (the Experiment 46 practice), so it does not include the variability of the fitted map.
Every number is the unbiased per-sample proper risk (mean over each item's ratings, then over items). Secondary rules (cross-entropy on all sets; Brier on DICES) are in `results/cells.csv`; no decision uses them.

## Scope and limits
- Five judges and five sets; ChaosNLI SNLI and MNLI-m contain only examples whose original five labels were not unanimous, so they over-represent disagreement (selection bias stated in the Experiment 28 protocol); the NLI correlation uses an ordinal 0, 1, 2 coding that is an assumption (it enters only the correlation side).
- Calibration error here is what this one map family removes: a lower bound on the population miscalibration term within the class, produced by this optimizer, these folds and this ridge; it is not a certified optimum over all maps. No cell had negative calibration error (calibration never hurt out of fold).
- The statements about correlation are descriptive for these 25 cells. They do not say correlation is wrong in general; the theory (a correlation sees a forecast only through its mean) is in the paper and Lean files, not established by this run.
- Theory status of the split: the Murphy-type identity is Lean-checked for scalar squared error and multiclass Brier (`docs/notes/point2_support_27Nov2026/lean/Point2/Murphy.lean`); for cross-entropy and ranked probability score it is numerically checked and Lean is pending (`docs/LEAN_TODO.md`).
  The identity used here is the algebraic definition of the two quantities and holds exactly for every rule.

## Files
`PROTOCOL.md`, `scripts/frozen.py` (bundle loader shared with Experiments 58 to 60), `scripts/run_exp57.py`, `tests/test_exp57.py`, `results/cells.csv` (60 rows: 25 cells, secondary rules included),
`results/correlations.csv`, `results/hypotheses.json`, `results/split_summary.md`, `results/split_figure.png` and `results/split_figure_values.csv`, `HOW_TO_READ_RESULTS.md`.
