# Experiment 51 protocol: Experiment 41 raw and matched gaps across a fixed ridge-penalty grid (frozen 11-27-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/51_exp41_fixed_ridge_penalty_grid_sweep_start_27Nov2026/PROTOCOL.md>

**TLDR:** Experiment 41's sensitivity run chose the penalty per head on validation from 1e-5..1, and 21-49% of fits chose an end of the grid. Here every head is fitted at every penalty of a wider fixed grid, with no selection, to show how the raw and matched gaps depend on the penalty. Frozen before the script exists.

**As of:** 11-27-2026 (Friday, Pacific time). CPU, $0.

## Design

- Experiment 41 views (16), split seeds 0-4, CE-, Pearson- and MSE-trained heads, at most 300 L-BFGS iterations, the penalty applied in the common form of `41_*/scripts/e40_l2.py::l2_settings` (0.5 lambda sum W^2 on the mean loss for every head), lambda in {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}, the same lambda for all heads in a cell.
- Matched map as Exp 41 (validation-fitted, per head).
- Quantities per (view, lambda): raw gap and matched gap (Pearson-trained minus CE-trained test normalized RPS; positive = CE-trained better), MSE-trained minus CE-trained matched gap, CE-trained raw minus constant; seed mean, seed range, and per-seed paired item bootstrap intervals (2,000 draws).
- **Predictions.** L1: the matched gap stays within +-0.003 at every lambda on every view. L2: the raw gap is positive at every lambda on views with signal. L3: the best-lambda CE-trained head's gain over the constant is at most 0.02 on every view (weak-signal regime), so these heads cannot test large shape effects. Each reported as supported or not, with every failing cell listed.
