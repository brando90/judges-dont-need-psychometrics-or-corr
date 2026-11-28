# Experiment 50: proper-score gaps, skill and correlation stratified by human agreement

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/50_stratification_by_human_agreement_proper_gap_vs_corr_start_27Nov2026/README.md>

**TLDR:** We split each test set into five equal-size bins by how much its human raters agreed, measuring agreement on half of each item's ratings and scoring on the other half. Three things came out of it.

1. **The raw gap behaves opposite to our prediction.** The proper-score penalty for training a judge on correlation instead of cross-entropy (CE) is largest on the items humans agree on, not on the contested ones. It is significantly larger in the highest-agreement bin on 9 of 12 views; the frozen prediction R1 held on only 2. Calibration removes the gap in every bin.
2. **Frontier LLM judges are worthless on contested items.** Their skill over the base rate falls from 0.16–0.91 on high-agreement items to about zero or below on the lowest-agreement fifth. Their Pearson correlation and top-1 agreement stay far above zero there, so correlation hides this.
3. **Rankings shift with agreement.** In 4 of 5 frontier sets the judge ranking by proper score is more stable across bins than the ranking by top-1 agreement. The best judge by Pearson and the best by proper score differ in 6 of 25 (set, bin) cells (point estimates, no tests).

**As of:** 11-27-2026 (Friday, Pacific time). CPU, \$0. Protocol frozen first: [PROTOCOL.md](PROTOCOL.md), commit 757d462.

## Setup (PROTOCOL.md)

- **Binning.** For every item with at least 4 ratings, the ratings are split at random (seed 20261127 + item position) into half A and half B. Each item gets the Shannon entropy of half A (nats), and items fall into equal-count entropy quintiles within each set (Q1 = most agreement). Every score in a bin uses half B only.
- **SummEval experts** have only 3 ratings per item. They are binned and scored on the same ratings, so those rows are descriptive.
- **Ties.** Many POPQUORN K = 2 items have entropy 0. Equal-count bins split those ties arbitrarily, so Q1–Q3 can share an entropy range.
- **Arm A (Experiment 41 heads).**
  - Covers 16 views and split seeds 0–4, using the preregistered heads (L2 0.01, at most 35 iterations).
  - Raw gap = test normalized RPS of the Pearson-trained head minus that of the CE-trained head (positive = CE-trained better). Matched gap = the same after Exp 41's calibration map.
  - Uncertainty: paired item bootstrap, 2,000 draws, 95% intervals.
- **Arm B (real judges).**
  - Covers the five frontier judges of Experiment 28 (Exp 46's panel) on ChaosNLI SNLI, MNLI and alphaNLI, DICES-350 and DICES-990, plus G-Eval GPT-4 on SummEval.
  - For each bin we report: the primary proper score S (Brier for NLI, normalized RPS for DICES and SummEval); skill = 1 − S/S_base, where S_base is a cross-fitted whole-set marginal scored in the bin; Pearson; top-1 agreement with the half-B mode.
- **Operationalizations.** These were fixed in `scripts/summarize_strata.py` after the protocol but before reading the tables:
  - R1 holds on a view if the seed-mean Q5 − Q1 raw gap is > 0 and at least 3 of 5 seed intervals lie above 0.
  - R2 is counted per (view, seed, bin, head).
  - R3 is "top-1 agreement falls Q1 → Q5 for every judge" plus Kendall τ between the Q1 and Q5 judge rankings, computed by top-1 and by S.

## Results

Full tables: [results/strata_tables.md](results/strata_tables.md).

**Arm A, raw gap by entropy quintile** (seed means; Q1 = most agreement):

| view | Q1 | Q3 | Q5 | Q5 − Q1 | seeds with CI > 0 / < 0 |
|---|---|---|---|---|---|
| POPQUORN offensiveness K5 | +0.148 | +0.105 | +0.057 | −0.090 | 0 / 5 |
| POPQUORN politeness K5 | +0.049 | +0.079 | +0.083 | +0.034 | 5 / 0 |
| SummEval expert K5 (same ratings) | +0.144 | +0.088 | +0.061 | −0.084 | 0 / 5 |
| DICES-990 K3 | +0.143 | +0.051 | −0.020 | −0.163 | 0 / 5 |
| ChaosNLI SNLI+MNLI K3 | +0.066 | +0.064 | +0.042 | −0.024 | 0 / 4 |

- **R1: not supported.** It holds on 2 of 12 views with signal (POPQUORN politeness K2 and K5). On 9 of 12 the gap is significantly *smaller* in the lowest-agreement bin. One reading: on high-agreement items the CE-trained head's near-base-rate forecast is close to the one-hot human distribution, while the uncalibrated Pearson-trained head's arbitrary scale is not. This explanation was not separately tested.
- **The matched gap stays near zero in every bin.** It is within ±0.0032 in Q1 and Q5 on every view, so calibration fixes the gap uniformly across agreement levels (in-domain).
- **R2: not supported as stated.** Within-bin Pearson is below the pooled Pearson in 494 of 800 Arm A cells (62%) and 76 of 135 Arm B cells (56%), not in every bin. For example, 4 of 5 frontier judges on DICES-990 have *higher* Pearson in Q5 than in Q1.

**Arm B, frontier judges, Q1 (most agreement) → Q5 (least):**

| set | skill range over 5 judges, Q1 → Q5 | Pearson range, Q1 → Q5 | top-1 range, Q1 → Q5 |
|---|---|---|---|
| ChaosNLI SNLI | 0.71–0.78 → −0.04–0.11 | 0.96–0.99 → 0.69–0.86 | 0.94–1.00 → 0.63–0.70 |
| ChaosNLI MNLI | 0.27–0.38 → −0.15–0.03 | 0.84–0.93 → 0.51–0.70 | 0.77–0.87 → 0.56–0.64 |
| ChaosNLI alphaNLI | 0.51–0.91 → −0.28–0.02 | 0.79–1.00 → 0.38–0.59 | 0.89–1.00 → 0.63–0.76 |
| DICES-350 | 0.16–0.27 → −0.38–−0.10 | 0.68–0.79 → 0.12–0.22 | 0.91–0.95 → 0.64–0.72 |
| DICES-990 | 0.25–0.30 → −0.23–0.04 | 0.21–0.48 → 0.37–0.49 | 0.99–1.00 → 0.60–0.67 |

- **R3, first half: supported on 6 of 7 sets.** Top-1 falls Q1 → Q5 for every judge on every frontier set and on SummEval consistency. It does not fall on SummEval relevance (G-Eval: 0.267 → 0.300).
- **R3, second half: supported on 4 of 5 frontier sets.** The judge ranking by S is more stable between Q1 and Q5 than the ranking by top-1. Kendall τ by S vs by top-1:

  | set | τ by S | τ by top-1 |
  |---|---|---|
  | MNLI | +0.40 | −0.20 |
  | alphaNLI | +0.20 | +0.11 |
  | DICES-350 | +0.60 | −0.20 |
  | DICES-990 | +0.60 | +0.12 |
  | SNLI (exception) | −0.60 | +0.20 |

  With 5 judges these τ values are coarse.
- **The top judge depends on the metric in some bins.** The best judge by Pearson differs from the best by S in 6 of 25 (set, bin) cells:
  - MNLI Q1: GPT-6 Astra by Pearson, GLM 5.3 by S.
  - DICES-350 Q3–Q5: GPT-6 Astra vs Claude Opus 5.5.
  - DICES-990 Q1: GLM 5.3 vs Claude Opus 5.5.
  - alphaNLI Q4: GPT-6 Astra vs Claude Opus 5.5.

  These are point estimates with no paired tests.

## Does this support or undercut the thesis?

Mostly support, with one honest miss.

- **Support:**
  - The skill score does exactly what the thesis says correlation cannot. It shows frontier judges add no value over the base rate on the most-contested fifth of items, even though Pearson on those same items stays at 0.38–0.86 on NLI.
  - The agreement-stratified report the paper promises is informative.
- **Miss:** Our prediction about *where* the training-objective gap lives was wrong. It concentrates on high-agreement items, so we should not claim that correlation-trained judges fail mainly on ambiguous items.

**Proposed paper text (not inserted):** "Stratifying by human agreement, with agreement measured on held-out ratings, shows that frontier judges' skill over the base rate falls from 0.16–0.91 on the highest-agreement fifth of items to −0.38–0.11 on the lowest. Their Pearson correlation on those contested items remains 0.38–0.86 on NLI. We recommend reporting skill by agreement stratum; a single pooled correlation hides where a judge stops being useful."

## Reproduce

```
export E40_DATA_CACHE=/path/to/cache
python scripts/agreement_strata.py /path/to/e28/labels.json /path/to/e23_inputs results   # imports Experiment 46 and 47 scripts
python scripts/summarize_strata.py results
```

Seeds: rating split 20261127 + item position; split seeds 0–4; bootstrap 2,000 draws. The inputs are as in Experiment 48 (Exp 28 `prepare.py` labels, request bundle SHA 6004ef74…ee64; Exp 23 SummEval npz files).
