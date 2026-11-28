# Experiment 51: Experiment 41's raw and matched gaps across a fixed ridge-penalty grid

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/51_exp41_fixed_ridge_penalty_grid_sweep_start_27Nov2026/README.md>

**TLDR:** Experiment 41's conclusions do not depend on the ridge penalty. Every head was refitted at 10 fixed penalties, $\lambda \in \{10^{-7}, \dots, 100\}$, with no validation selection.

- **The matched gap is small almost everywhere.** It stays within ±0.003 normalized RPS in 152 of 160 (view, λ) cells. The 8 exceptions are all POPQUORN politeness at λ ≤ 1e-5 (nearly unregularized), at most +0.0081 nRPS, or −0.034 in skill terms.
- **The raw gap favours the cross-entropy-trained (CE-trained) head nearly everywhere.** The raw gap (Pearson-trained raw minus CE-trained raw) is positive in 119 of 120 cells on views with signal, and in 115 of those at least 3 of 5 seed intervals lie above 0.
- **Shrinkage does not remove the raw gap.** On views with signal it is +0.13 to +0.38 at λ = 1e-7 and still +0.001 to +0.134 at λ = 100. Pearson correlation of the expected score is unchanged by rescaling or shifting the score, so the Pearson objective never pins down the forecast's scale; only calibration does.

This is a robustness check, not a new finding. Experiment 41 already ran an L2 sensitivity analysis with validation selection; this run removes the selection and widens the grid.

**As of:** 11-27-2026 (Friday, Pacific time). CPU, about 29 minutes, \$0. Protocol frozen first: [PROTOCOL.md](PROTOCOL.md), commit 757d462.

## Setup (PROTOCOL.md)

- **Data and heads.** Experiment 41's 16 views and split seeds 0–4. CE-, Pearson- and MSE-trained linear heads on Exp 41's 24 train-only principal components, at most 300 L-BFGS iterations.
- **Penalty.** Exp 41's common form `e40_l2.l2_settings`: $0.5\lambda\sum W^2$ added to the mean loss. Every head in a cell uses the same λ.
- **Calibration.** Matched map as in Exp 41: a multinomial logit on the standardized scalar score, fitted on validation.
- **Gaps.** Raw gap = test nRPS(Pearson-trained raw) − nRPS(CE-trained raw). Matched gap = the same with both heads calibrated. Positive = CE-trained better.
- **Uncertainty.** Seed means, with per-seed paired item bootstrap (2,000 draws).

## Results

Full tables (raw gap, matched gap, and CE-trained head vs constant at every λ): [results/grid_tables.md](results/grid_tables.md). Figure `fixed_ridge_gaps.png` (`ax.grid(True)`) is on the box at `/workspace/calib_followups/figures/e51/`.

| prediction | result |
|---|---|
| L1: \|matched gap\| ≤ 0.003 at every λ on every view | **Not supported as stated.** 152 of 160 cells pass. All 8 failures are POPQUORN politeness at small λ: K2 +0.0081, +0.0078, +0.0059 at λ = 1e-7, 1e-6, 1e-5; K3 +0.0075, +0.0064, +0.0047; K5 +0.0041, +0.0036 at λ = 1e-7, 1e-6. From λ = 1e-4 upward, every view is within ±0.0028. |
| L2: raw gap > 0 at every λ on views with signal | **Supported in 119 of 120 cells.** The exception is POPQUORN politeness K2 at λ = 0.01: −0.0037. |
| L3: best-λ CE-trained gain over the constant ≤ 0.02 | **Supported on 14 of 16 views.** Fails on POPQUORN politeness K2 (0.0268) and K3 (0.0203). Elsewhere the best gain is 0.000 to 0.015. |

**Skill-score terms (A1).** $\Delta SS = -\text{gap}/S_{\text{const}}$, with $S_{\text{const}}$ the test nRPS of the constant train-marginal forecast for each view and seed ([results/skill_terms.json](results/skill_terms.json)).
- Matched gap: −0.034 to +0.005 skill points over all 160 cells, and −0.010 to +0.000 at Exp 41's λ = 0.01.
- Raw gap: down to −2.97 skill points. At small λ, the uncalibrated Pearson-trained head is far worse than the base rate.

## Does this support or undercut the thesis?

It supports the thesis.
- **The proper-score penalty of correlation training is robust.** It is present at every penalty strength and comes from the forecast's unidentified scale.
- **In-domain calibration removes it** at every penalty from λ = 1e-4 up, consistent with Experiment 41.
- **The one blemish:** nearly unregularized heads on POPQUORN politeness keep a matched gap of up to 0.008 nRPS, the one view where these heads carry real signal (best skill over the constant 0.07).

**Proposed paper text (not inserted):** "Refitting all heads at ten fixed ridge penalties from 1e-7 to 100 leaves the conclusions unchanged. The raw proper-score gap favours the cross-entropy-trained head in 119 of 120 cells. After matched calibration the gap is within ±0.003 normalized RPS in 152 of 160 cells, with the exceptions confined to nearly unregularized fits on one dataset."

## Reproduce

```
export E40_DATA_CACHE=/path/to/cache
python scripts/fixed_ridge_grid.py results                       # about 29 minutes on one core; imports Exp 47's heads41.py and Exp 41's e40_l2.py
python scripts/summarize_grid.py results figures
```
