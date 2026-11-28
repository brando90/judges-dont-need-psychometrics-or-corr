# Experiment 59 PROTOCOL: select by correlation then calibrate (path A) versus select by proper score (path B)

Frozen 11-28-2026 (Pacific), in a commit that precedes every analysis run of this experiment. CPU only, no model calls, API spend $0.

## Purpose
The advisor-style objection is that recalibration removes the gap, so a practitioner can select the judge by correlation and then
recalibrate. Experiment 56 measured the label budget of calibration; Experiment 46 found no significant reversal after calibration,
while a reviewer noted that no significant reversal is not evidence of negligible selection regret and asked for a predefined regret
or equivalence criterion. This experiment is that analysis, with the same labels spent either way.

## What the analyst had seen before freezing
Aggregate results of Experiments 46 (including that on calibrated MNLI both correlations pick Astra and the proper score picks Opus),
48, 55 and 56. NOT seen: any regret of any selection path, and which judge each path selects on any draw.

## Data (read-only)
Same frozen bundle, loader, floors and invalid-forecast rule as Experiment 57. Five sets, five judges. Labels coded 0..K-1; the NLI
ordinal coding is an assumption used only by the correlation of path A (Experiment 28 protocol).

## Design (one draw)
Split the groups of a set at random into a pool P and a test set T of half the groups each (as in Experiment 56; SNLI items sharing
an image move together). Draw n items from P at random; for each, one human rating drawn at random from that item's ratings. These n
labeled items are L. Candidate judges: the five frozen judges. n in {25, 50, 100, 200}. R = 200 draws per (set, n).
- **Calibration** of a chosen judge: Experiment 46's map fitted on the single ratings of L (the `fit_calibrator` of `judge_scores`).
- **Path A:** choose the judge with the highest Pearson correlation between its forecast mean and the single ratings of L (ties to
  the fixed judge order), calibrate it on L, evaluate on T.
- **Path B:** choose the judge with the lowest 3-fold cross-validated calibrated proper risk on L (matched rule; folds over the items
  of L), calibrate that judge on all of L, evaluate on T.
- **Path C (sensitivity only):** choose the judge with the lowest raw proper risk on L (single ratings), calibrate it on L, evaluate on T.
- **Oracle:** the judge with the lowest calibrated test risk when every judge is calibrated on the same L.
Test risk is the unbiased per-sample proper risk on T using all ratings of each test item, under the matched rule (Brier for the NLI
sets, normalized RPS for DICES). Regret of a path on a draw = (calibrated test risk of the chosen judge - calibrated test risk of the
oracle) / (test risk of the base-rate forecast), the base rate being the histogram of all ratings of P (the same normalizer for every
path), so regret is in skill points (0 means the path picked the oracle's judge or one just as good).
Seed per draw `SeedSequence([20261128, 59, set_index, n_index, draw])`.

## Primary comparison and uncertainty
For each (set, n): mean regret of A minus mean regret of B, paired over the same 200 draws; positive means B is better. Interval:
percentile bootstrap over the draws, B = 4,000, seed `SeedSequence([20261128, 59, 99, set_index, n_index])`. This is variability over
labeled pools conditional on the finite set (it does not include the sampling of the items of the set from a population) and is
stated as such. Secondary: the share of draws in which each path selects the oracle's judge.
Multiplicity: the hypotheses below are decided on point estimates and the stated counts; no Holm correction is used, and the 20 (set, n)
cells are all reported.

## Hypotheses (three; each with a falsification rule)
- **H1.** At n = 25, path B has lower mean regret than path A in at least 3 of the 5 sets, and the paired 95 percent interval excludes 0
  in at least 2 of those sets. Falsified otherwise.
- **H2 (equivalence).** At n = 200 the absolute difference in mean regret between A and B is below 0.005 skill points in all 5 sets
  (the margin is fixed here, before any result). Falsified if any set exceeds it. If H2 holds, "select by correlation, then calibrate"
  is within that margin of proper-score selection at that label budget on these judges, and nothing more is claimed.
- **H3 (descriptive).** The smallest n on the grid at which |A minus B| < 0.005 skill points in every set; "none" if it never is.

## Meaning of decision words
"Supported", "falsified" and the equivalence margin are as above. "Beats" is not used; a path is "better" in a cell only if its mean
regret is lower and the paired interval excludes 0.

## Planned outputs
`results/regret.csv` (all 20 cells and 3 paths, with intervals), `results/paths_summary.md`, `results/hypotheses.json`, one figure
(`results/regret_figure.png`, values in `results/regret_figure_values.csv`; grid on every axis), `HOW_TO_READ_RESULTS.md`,
`README.md`, `tests/test_exp59.py` (a synthetic known-winner code check labeled as such, regret zero when all judges are identical,
determinism).

## Amendment A1 (11-28-2026, committed before the first real-data run of this experiment; no result of it had been seen)
The pool P holds half of the groups, so a label budget n cannot exceed the number of items in P. On DICES-350 the pool has 175 items, so the cell n = 200 uses all 175 pool items
(n_used = |P| = 175); every other (set, n) cell has |P| >= n (on alphaNLI and DICES-990, |P| = 200, so n = 200 labels the whole pool; on SNLI and MNLI-m, |P| is about 300). The label "200" stays in the grid so that H2 and H3 are still read at the largest
value, and the tables report n_used. Reason: infeasibility found while writing the script, not a response to any result. Details the protocol left open, fixed here: (i) a judge whose correlation in path A is undefined (a constant forecast mean or constant single ratings)
gets correlation minus infinity, so undefined values and ties go to the fixed judge order (Astra, Opus, Kimi, GLM, DeepSeek); (ii) the single rating of a labeled item is drawn uniformly from that item's valid ratings; (iii) the three folds of path B are a random permutation
of the items of L taken modulo 3; (iv) the random split puts floor(G/2) of the G groups into P and the rest into T; (v) draws are independent and their seeds do not depend on the number of worker processes, so the output is identical for any worker count.
