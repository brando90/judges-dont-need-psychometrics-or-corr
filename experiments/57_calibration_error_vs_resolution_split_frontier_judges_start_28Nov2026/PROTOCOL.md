# Experiment 57 PROTOCOL: calibration error versus resolution on the five frozen frontier judges

Frozen 11-28-2026 (Pacific), in a commit that precedes every analysis run of this experiment. CPU only, no model calls, API spend $0.

## Purpose
The paper recommends reporting a strictly proper score split into calibration error and resolution, beside correlation. No experiment
shows that split on real judges. This one does, and asks whether correlation tracks resolution rather than calibration error.
Terminology: "calibration error" (never "reliability", which means something else in psychometrics).

## What the analyst had seen before freezing
Aggregate results of Experiments 46 (raw and per-judge calibrated skill, reversals), 48, 55 (skill under three rules) and 56 (label
budget of calibration) on these same judges and sets. In particular: on DICES-350 four of five judges have negative raw skill, and
per-judge calibration (Experiment 46, H3) gave positive skill on both DICES sets. NOT seen: the calibration-error and resolution
shares of any cell, and any rank comparison between correlation and resolution. Before this commit only array shapes and encodings
were inspected, and the `judge_scores` package was shown to reproduce 75 published Experiment 55 rows (24 tests).

## Data (read-only)
Frozen forecasts of five judges (GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3, DeepSeek V4.1 Flash; temperature 0) in
`experiments/39_*/data/processed/study_a_bundle.npz`, loaded through `experiments/44_*/src/lib44.py`; invalid forecasts are replaced
by the uniform forecast (Experiment 28 rule) and every forecast is floored at 1e-6. Sets (items): ChaosNLI SNLI (600), MNLI-m (600),
alphaNLI (400), DICES-350 (350), DICES-990 (400). That is 25 judge-by-set cells. Human ratings are the individual ratings per item
(100, 100, 100, 123 and 69 to 76 per item). Labels: NLI contradiction < neutral < entailment (an ordinal coding that is an
assumption, Experiment 28 protocol), alphaNLI binary, DICES No < Unsure < Yes.

## Rules and encodings
Primary rule matched to the encoding (Experiment 55 table): Brier for the three NLI sets (unordered, or binary), normalized ranked
probability score (RPS) for the two DICES sets (ordered). Secondary: cross-entropy (CE) on all sets, Brier on the DICES sets. Ordinal
RPS is never applied to the unordered NLI sets. Risk = unbiased per-sample proper risk (mean over an item's ratings, then over items).

## Split (per cell; implemented by `judge_scores.calibration_split`)
UNC = risk of the cross-fitted base-rate forecast; R_raw = risk of the raw forecast; R_cal = risk of the cross-fitted calibrated
forecast. Calibration map and fitting: Experiment 46's (multinomial logit on [log q, 1], ridge 1e-3 toward the identity, individual-
rating cross-entropy, L-BFGS-B), fitted on the other four folds. Folds: five folds over GROUPS (SNLI items that share an image move
together; every other set's group is the item), seed `SeedSequence([20261128, 57, 0, set_index])`. Then
calibration error CAL = R_raw - R_cal; resolution RES = UNC - R_cal; identity R_raw = UNC - RES + CAL (by construction, tested);
raw skill = (RES - CAL)/UNC; calibrated skill = RES/UNC. CAL is what this map class removes, a lower bound on the population
miscalibration term within the class; RES is the discrimination left after that recalibration. They reflect this calibrator, these
folds and this optimizer, not a certified optimum over all maps.
Correlation side: Pearson and Spearman correlation across items between the judge's forecast mean (labels coded 0, 1, ..., K-1) and
the human item-mean rating.

## Resampling and multiplicity
Cluster = group (as in Experiment 46); groups are resampled with replacement, B = 4,000 draws, percentile 95 percent intervals, one
weight matrix per set shared by every judge and every statistic (paired), seed `SeedSequence([20261128, 57, 1, set_index])`. Holm
over the declared cells of a hypothesis where a test is used.

## Hypotheses (three; each with a falsification rule, descriptive criteria are labeled as such)
- **H1, correlation tracks resolution, not calibration error.** For each set, let tau_res be Kendall's tau-b between the five judges'
  Pearson-correlation ranking and their calibrated-skill (resolution) ranking, and tau_raw the same against their raw-skill ranking.
  Supported if tau_res > tau_raw (strict, point estimates) in at least 4 of the 5 sets; falsified otherwise. A bootstrap interval of
  tau_res - tau_raw is reported but is not part of the decision (five judges give a coarse statistic). The same is reported for
  Spearman as a sensitivity.
- **H2, judges differ more in calibration error than in resolution.** For each set, the between-judge standard deviation of the
  calibration share CAL/UNC exceeds that of the resolution share RES/UNC. Supported if so in at least 4 of 5 sets; falsified otherwise.
- **H3, the DICES-350 rescue is not noise.** Let J- be the judges whose raw skill (primary rule) on DICES-350 is below 0 by point
  estimate. Supported if every judge in J- has calibrated skill above 0 with Holm-adjusted (over |J-|) one-sided bootstrap p < 0.05
  (p = (1 + #draws <= 0)/(B + 1)); falsified otherwise.

## Meaning of decision words
"Supported" and "falsified" are the rules above. No claim that one method "beats" another is made here, because no paired test of
two methods is run. Failures are reported. Every cell is reported, including those that do not fit a hypothesis.

## Planned outputs
`results/cells.csv` (25 cells, all rules, with intervals), `results/split_summary.md`, `results/hypotheses.json`, one figure
(`results/split_figure.png` and its values in `results/split_figure_values.csv`; grid on every axis), `HOW_TO_READ_RESULTS.md`,
`README.md`, `tests/test_exp57.py` (identity, tie, determinism, a synthetic known-law code check labeled as such, consistency with
Experiment 55 raw skill within 0.01 because the base-rate folds differ).
