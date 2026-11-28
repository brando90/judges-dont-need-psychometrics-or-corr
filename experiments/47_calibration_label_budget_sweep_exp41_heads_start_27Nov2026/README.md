# Experiment 47: how many held-out labels does post-hoc calibration need to close the CE-vs-Pearson gap?

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/47_calibration_label_budget_sweep_exp41_heads_start_27Nov2026/README.md>

> **Note 11-28-2026 (QA; results unchanged).** The two heads compared here are both scored by a proper rule, so the comparison is circular as evidence for the evaluation claim (Experiment 53); it measures how many held-out labels the calibration step needs for these linear heads. For the frontier judges the budget is Experiment 56: n = 100 is the smallest tested budget at which every judge's mean calibrated skill is above the base rate on all five sets, and 8 of 25 judge-set cases are still below their uncalibrated skill at n = 100.


**TLDR:** "Calibration closes the gap" (Experiment 41: at most 0.0024 normalized RPS) holds only once the calibrator has about 100 to 200 held-out human ratings. With n single ratings on n validation items, the Pearson-trained head plus its calibration map is still worse than the cross-entropy-trained (CE-trained) head used raw by +0.010 to +0.024 test normalized RPS at n = 25, and it is then worse than the constant base-rate forecast on 15 of 16 views. The gap reaches the 0.0024 bound at n\* = 50 (1 view), 100 (6 views) or 200 (9 views). With all ratings of each calibration item, n\* = 25 to 100 items. The calibration step needs labels, and only a proper score on held-out labels shows when it has worked; correlation-based validation asks for neither.

**As of:** 11-27-2026 (Friday, Pacific time). CPU, \$0. Protocol frozen first: [PROTOCOL.md](PROTOCOL.md), commit 83b9782.

## Setup

Experiment 41 unchanged: 16 views of 7 real multi-annotator rating sets (POPQUORN offensiveness and politeness, SummEval expert and crowd, DICES-990, ChaosNLI SNLI+MNLI, ChaosNLI alphaNLI; K = 2, 3, 5); group split 0.6/0.2/0.2 with split seeds 0 to 4; CE-, Pearson- and MSE-trained linear heads on 24 train-only principal components, ridge penalty 0.01, at most 35 iterations (the preregistered Exp 41 run); Exp 41's calibration map (multinomial logit on the standardized scalar score, fitted by cross-entropy). Check: at n = all with all ratings, M(n) equals Experiment 41's matched gap for every view to 6 decimals (seed 0).

What changes: the map is fitted on n labels drawn from the validation split only, n in {25, 50, 100, 200, 500, 1000, all} (capped at the 198 to 1,280 validation items). Design R1 (primary): n items, one rating each. Design RA: n items, all their ratings. 20 calibration draws per (view, seed, n). Test: the full test split and all its ratings.

- G(n) = test normalized RPS of the Pearson-trained head plus a map fitted on n labels, minus that of the CE-trained head used raw. Positive = the CE-trained raw head is better.
- M(n) = the same calibrated Pearson-trained head minus the CE-trained head with a map fitted on the same n labels (matched calibration).
- E(n) = the calibrated Pearson-trained head minus the constant train-marginal forecast. Positive = worse than predicting the base rate.

Normalized RPS = RPS / (K - 1), range 0 to 1, lower is better.

## Results, design R1 (one rating per calibration item)

Mean over 5 seeds x 20 draws; (SD over the 100 seed-draw cells). Full tables for R1 and RA, G and M: [results/label_budget_tables.md](results/label_budget_tables.md).

| view | n = 25 | n = 50 | n = 100 | n = 200 | n = all | n\* (G <= 0.0024) | E(25) |
|---|---|---|---|---|---|---|---|
| POPQUORN offensiveness K5 | +0.0168 (0.016) | +0.0075 (0.007) | +0.0026 (0.003) | +0.0012 (0.002) | +0.0004 (0.001) | 200 | +0.0161 |
| POPQUORN politeness K5 | +0.0117 (0.011) | +0.0011 (0.005) | -0.0023 (0.003) | -0.0041 (0.001) | -0.0055 (0.001) | 50 | +0.0057 |
| SummEval expert K5 | +0.0097 (0.007) | +0.0043 (0.006) | +0.0008 (0.003) | -0.0003 (0.002) | -0.0016 (0.001) | 100 | +0.0080 |
| SummEval crowd K5 | +0.0113 (0.009) | +0.0062 (0.006) | +0.0026 (0.002) | +0.0014 (0.002) | +0.0002 (0.001) | 200 | +0.0115 |
| DICES-990 K3 | +0.0165 (0.019) | +0.0082 (0.010) | +0.0036 (0.005) | +0.0006 (0.002) | +0.0006 (0.002) | 200 | +0.0138 |
| ChaosNLI SNLI+MNLI K3 | +0.0175 (0.018) | +0.0064 (0.005) | +0.0039 (0.003) | +0.0014 (0.002) | +0.0002 (0.001) | 200 | +0.0162 |
| ChaosNLI alphaNLI K2 | +0.0225 (0.023) | +0.0094 (0.013) | +0.0038 (0.006) | +0.0014 (0.003) | +0.0006 (0.002) | 200 | +0.0239 |

Over all 16 views (R1): G(25) ranges from +0.0097 to +0.0225; in 72% to 97% of seed-draw cells G(25) exceeds 0.0024, and at n = 100 still 4% to 59% do; the 95th percentile of G(25) over cells is 0.024 to 0.069. E(25) > 0 on 15 of 16 views (the exception is POPQUORN politeness K2, -0.0014). The matched gap M(n) is flat in n: it stays at Experiment 41's matched gap (+0.0023 to +0.0025 on POPQUORN politeness K2, +0.0013 to +0.0017 on K3, under 0.001 in absolute value on the other 14 views), so the label budget moves the calibrated level of both heads, not their difference.

## Skill-score terms (amendment A1, reporting only)

The advisor asked for skill-score differences rather than raw RPS gaps. Skill is $SS(h) = 1 - \mathrm{nRPS}(h)/S_{\text{const}}$, where $S_{\text{const}}$ is the test nRPS of the constant train-marginal forecast for each view and seed. Then

$$\Delta SS(n) = SS(\text{Pearson-trained head} + \text{map on } n \text{ labels}) - SS(\text{CE-trained head, raw}) = -G(n)/S_{\text{const}}.$$

Negative means the CE-trained raw head is better. Nothing was refitted; [scripts/skill_terms.py](scripts/skill_terms.py) rescales the stored per-draw values. Full tables, with the 2.5–97.5% range over the 100 seed-draw cells for every view, n and design, are in [results/skill_terms_tables.md](results/skill_terms_tables.md).

- **Context: these heads have almost no skill.** The CE-trained raw head's skill over the base rate is between −0.005 and +0.071 on the 16 views.
- **Small budgets cost more than the whole skill.** Under design R1 at n = 25, $\Delta SS$ = −0.066 to −0.116 skill points. That is larger than the CE-trained head's entire skill on every view. The calibrated Pearson-trained head has negative skill on 15 of 16 views. Example: POPQUORN offensiveness K5 has $\Delta SS$ = −0.113, with seed-draw range [−0.417, −0.016].
- **Larger budgets close the gap.** $\Delta SS$ is −0.021 to +0.013 at n = 100, −0.013 to +0.023 at n = 200, and −0.008 to +0.031 at n = all.
- **Labels needed.** The first n at which the seed-mean $\Delta SS \ge -0.01$ is 50 (1 view), 100 (5), 200 (7) or 500 (3).
- **With all labels, the calibrated Pearson head can win.** On POPQUORN politeness it ends significantly *ahead* (K5: +0.031 [+0.025, +0.038]). The training objective's disadvantage disappears once enough labels calibrate it, which is consistent with Experiment 41.

## Predictions (PROTOCOL.md)

- **Q1: supported.** On all 12 views with signal G(25) > 0.0024, and on every view G(n) <= 0.0024 by n = 200 (R1).
- **Q2: partly supported.** The SD falls faster than the $n^{-1/2}$ rate from 25 to 100 labels (2.3- to 6-fold for a 4-fold increase in n, where $n^{-1/2}$ predicts 2-fold) and then levels off at a split-seed floor of about 0.001 to 0.003.
- **Decision rule:** n\* (R1) = 50 on 1 view, 100 on 6, 200 on 9. Design RA: 25 on 3 (POPQUORN politeness), 50 on 7, 100 on 6.

## Does this support or undercut the thesis?

It supports the evaluation thesis and limits the "calibration fixes it" reading. Calibration closes the gap only with roughly 100 to 200 held-out human ratings per task, and below that the calibrated correlation-trained head can be worse than the base rate. A practitioner who validated a judge by correlation has, by construction, not collected or scored those labels with a proper score, so they cannot know whether their calibrator worked. Scope: weak linear heads on hashed text (Experiment 41's regime).

**Proposed paper text (not inserted):** "Post-hoc calibration is not free. On seven real rating sets, a calibrator fitted on 25 single held-out ratings leaves the correlation-trained judge 0.010 to 0.024 normalized RPS worse than the proper-score-trained judge, and worse than the base rate on 15 of 16 views. It needs 100 to 200 ratings to come within 0.0024. Calibration therefore relocates the validation problem: it still requires held-out human ratings scored with a proper rule."

## Reproduce

```
export E40_DATA_CACHE=/path/to/cache   # Experiment 36/41 loaders download the public files once (hash-checked)
python scripts/label_budget_sweep.py results --draws 20     # about 3.5 minutes on one core; heads cached by scripts/heads41.py
python scripts/summarize_label_budget.py results figures     # tables; figures kept on the box
```

Files: `scripts/heads41.py` (shared with Experiments 48 and 50: rebuilds Exp 41's views and heads), `scripts/label_budget_sweep.py`, `scripts/summarize_label_budget.py`; `results/label_budget_rows_rounded.json` (every seed-draw cell, values rounded to 1e-6), `results/label_budget_summary.json`, `results/label_budget_tables.md`, `scripts/skill_terms.py`, `results/skill_terms.json`, `results/skill_terms_tables.md`. Figures `label_budget_closure_gap.png` and `label_budget_sd.png` (`ax.grid(True)`) are on the box at `/workspace/calib_followups/figures/e47/` and rebuilt by the summary script; the push tool carries only text.
