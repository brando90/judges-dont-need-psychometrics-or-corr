# Experiment 56: how many held-out human ratings does per-judge calibration of the frontier LLM judges need?

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/56_frontier_judge_calibration_label_budget_sweep_start_27Nov2026/README.md>

**TLDR:** About 100 single held-out ratings per task. Experiment 46 showed that per-judge calibration removes the frontier judges' reversals and negative skill, and cited Experiment 47's "100–200 labels" for the cost, but Experiment 47 measured linear heads, not these judges. Here, on the five frozen frontier judges and five public sets, calibrating on n held-out items with **one rating each** leaves the judge's mean skill below its uncalibrated skill in 25 of 25 judge-set cases at n = 10 and 21 of 25 at n = 25. At n = 100 every judge's mean calibrated skill is above the base rate on all five sets (8 of 25 cases are still below their uncalibrated skill); at n = 200, 2 of 25 are. Even 200 single ratings (every pool item on alphaNLI and DICES-990; 175, the whole pool, on DICES-350) stay 0.003 to 0.022 skill below a map fitted on all ratings of the pool, so the preregistered n\* ≤ 200 holds only on alphaNLI (H1 fails); H2 holds.

**As of:** 11-27-2026 (Friday, Pacific time). CPU, 26 seconds, \$0, no model calls. Protocol frozen first: [PROTOCOL.md](PROTOCOL.md), commit b487c4b.

## Setup

Frozen Experiment 28 verbalized forecasts of GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3 and DeepSeek V4.1 Flash on ChaosNLI SNLI (600 items), MNLI-m (600), alphaNLI (400), DICES-350 (350) and DICES-990 (400). Experiment 46's loader, probability floor, calibrator (multinomial logit on [log q, 1], ridge 1e-3, shrunk to the identity) and primary rule (Brier for NLI, normalized RPS for DICES), unchanged. 50 draws per case: group-level 50/50 split into a calibration pool and a test half; for n in {10, 25, 50, 100, 200}, n pool items with one randomly drawn rating each; the reference "full" map uses every rating of every pool item (about 100 per ChaosNLI item, 69 to 123 per DICES item). Skill $SS = 1 - S/S_{\text{base}}$ on the test half, with the base the pool's marginal rating histogram.

## Results

Full table: [results/tables.md](results/tables.md); raw values: `results/label_budget_frontier.json`.

| n single ratings | cases with mean calibrated skill below uncalibrated (of 25) | cases with mean calibrated skill below 0 | sets where every judge's mean calibrated skill > 0 |
|---|---|---|---|
| 10 | 25 | 20 | 1 of 5 |
| 25 | 21 | 15 | 2 of 5 |
| 50 | 12 | 5 | 4 of 5 |
| 100 | 8 | 0 | 5 of 5 |
| 200 | 2 | 0 | 5 of 5 |

- **DICES-350** (the set where four judges are below the base rate uncalibrated): at n = 25 every judge's mean calibrated skill is negative (−0.124 to −0.078); at n = 100 all are positive (+0.007 to +0.036); full calibration gives +0.049 to +0.072.
- **The reversal pair** (GPT-6 Astra vs GLM 5.3): calibrated Astra still has lower skill than calibrated GLM in 54% / 50% of draws at n = 25 (SNLI / MNLI), 22% / 22% at n = 100 and 18% / 10% at n = 200.
- **Distance to full calibration** at n = 200: 0.003 to 0.022 skill. The labelled deviation run (`results_deviation_all_pool_items/`, every pool item with one rating: 292 to 306 items on SNLI, 300 on MNLI, and the same 200 or 175 as n = 200 elsewhere) closes only part of that on the NLI sets: the remaining shortfall comes from using one rating per item rather than all of them.

## Predictions (PROTOCOL.md)

- **H1 (n\* ≤ 200 in at least half of the 23 cases with full gain ≥ 0.01): not supported.** n\* ≤ 200 on the five alphaNLI cases only (100 on one, 200 on four); on SNLI, MNLI and both DICES sets the single-rating map stays more than 0.01 below full calibration at every budget tested.
- **H2 (at n = 10 calibration hurts in more than 10% of draws in at least one case): supported**, in all 25 cases (86% to 100% of draws).

## Does this support or undercut the thesis?

It replaces a borrowed number with a measured one and supports the reading that recalibration is a proper-score validation step with a real label cost. On these frontier judges, about 100 held-out single ratings per task are needed before every judge's calibrated skill is above the base rate, and fewer than 50 usually leave the judge worse than not calibrating. The budget is of the same order as Experiment 47's for linear heads (n\* = 50 to 200 there, for a different criterion). Scope: one calibrator family, five verbalized-forecast judges at temperature 0, sets with many ratings per item (which this design subsamples to one).

## Deviation

`E56_BUDGETS` lets the script add n = 9999 (every pool item, one rating each); that run is labelled as a deviation and stored separately. The primary table uses the frozen budgets. Pool sizes: 292 to 306 items (SNLI), 300 (MNLI), 200 (alphaNLI, DICES-990) and 175 (DICES-350), so n = 200 is the whole pool on alphaNLI and DICES-990 and is capped at 175 on DICES-350, as the protocol allows.

## Reproduce

```
OMP_NUM_THREADS=1 python scripts/label_budget_frontier.py /path/to/e28/labels.json /path/to/e23_inputs results     # about 25 s
OMP_NUM_THREADS=1 E56_BUDGETS=10,25,50,100,200,9999 python scripts/label_budget_frontier.py ... results_deviation_all_pool_items
python scripts/make_table.py
```

`labels.json` is rebuilt by Experiment 28's `runners/prepare.py` (request bundle SHA-256 6004ef74…ee64). Seeds: SeedSequence([20261127, 56, set, judge, draw]). Single-threaded BLAS is recommended (with multithreaded BLAS the same run had not finished after 15 minutes on the box).
