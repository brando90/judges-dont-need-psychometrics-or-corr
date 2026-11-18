# Experiment 11: MSE-Trained and Regularized Score-Then-Calibrate Toy Baselines vs Cross-Entropy

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/11_toy_mse_and_tuned_head_calibration_vs_ce_start_17Nov2026/README.md>

**Pre-registration (`workflows/ml-research-practice.md`, 11-17-2026):** Written before executing `scripts/run_sweep.py`. Directly addresses Sanmi Koyejo's Point 2 feedback (`paper_latex_and_notes/notes/2026-11-16_sanmi_feedback_corr_vs_ce.md`): testing (1) whether separately tuning slope regularization and cross-fitting Stage 2 eliminates the `Pearson + calibrated head` small-`n` spike (`n <= 32`) while preserving the 1D bottleneck gap at large `n`, and (2) how directly MSE-trained 78-weight judges compare against CE, Pearson, and score-then-calibrate baselines.

## Measurement Contract & Predictions

- **Data-generating process & judge family:** Reuses Experiment 08's exact 4 Beta-Binomial scenarios (`A`, `B`, `C`, `D` on `0..5`), exact sample sizes `n in {4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048}` (`k = 3` ratings/task), exact RNG (`1000003 * seed + 7919 * n + CODE[sc]`), `100` seeds, `500` held-out test tasks per seed scored against the true population distribution `p(r | x)`, and the same 78-weight cosine/sine Fourier softmax family (`13 x 6`) with `W0 = marginal_W0(Phi, P, seed)`.
- **Compared judges:**
  - `constant`: training-task empirical marginal histogram `P.mean(0)`.
  - `ce_fixed` (④) & `ce_valrps` (③): exact log-sum-exp CE (fixing `toy.py`'s clipped gradient; preserves Exp 08 fixed cells within `1e-12`). Ridge `c = 1` or 4-fold pooled CV RPS over `{0.01, 0.1, 1, 10, 100}`.
  - `pearson_fixed` (②): correlation training with ridge `(1/n)||W||^2` (`toy.py`).
  - `pearson_head_naive`: exact `toy.py` two-stage baseline (in-sample Stage-1 scores `s_tr`, Stage-2 `(a, b)` regularized identically by `(1/n)(||a||^2 + ||b||^2)`).
  - `pearson_head_in_tuned`: Stage-1 `pearson_fixed` in-sample scores, with Stage-2 slope ridge `c_b` tuned over `{0.01, 0.1, 1, 10, 100, 1e4, 1e8}` via end-to-end 4-fold pooled CV RPS (intercept `a` uses only a `1e-9 ||a - log(p_bar)||^2` gauge so `c_b -> inf` nests `constant`).
  - `pearson_head_tuned` (primary cross-fitted score-then-calibrate baseline): Stage-1 scores generated strictly out-of-fold across 4 training folds (`z_oof`), Stage-2 slope ridge `c_b` tuned over `{0.01, 0.1, 1, 10, 100, 1e4, 1e8}` via 4-fold CV RPS on `(z_oof, P)` and refitted on `(z_oof, P)` before evaluating on standardized full-train Stage-1 scores at test time.
  - `mse_fixed`: 78-weight softmax minimizing `(1/n) sum_i (m_W(x_i) - h_i)^2 + (1/n)||W||^2` (400 L-BFGS-B iters / 600 evals, perturbed log-histogram start).
  - `mse_valrps`: 78-weight MSE-trained softmax with ridge `c in {0.01, 0.1, 1, 10, 100}` chosen by 4-fold pooled CV RPS.
  - `mse_head_tuned`: cross-fitted 4-fold Stage-1 `mse_fixed` out-of-fold scores + tuned Stage-2 CE calibration head (same protocol as `pearson_head_tuned`).
- **Pre-registered quantitative predictions:**
  1. `constant` and `ce_fixed` match Exp 08 `ce_gradient_check.json` exact-CE values to `< 1e-11`; `c_b = 1e8` in Stage 2 reproduces `P.mean(0)` (`b -> 0`) to `< 1e-7`.
  2. `pearson_head_naive` spikes at `n <= 32` (`~0.086–0.248` RPS excess in A, B); `pearson_head_tuned` eliminates this small-`n` spike (`<= constant` up to fold finite-sample noise) by cross-fitting Stage 1 and letting heavy `c_b` fall back toward `constant`.
  3. At `n = 2048`, `mse_fixed` and `mse_valrps` fix Pearson's arbitrary affine scale/shift (`pearson_fixed` loses to `constant`), yet still underperform `ce_fixed` / `ce_valrps` on RPS excess and CE excess because MSE constrains only the 1st moment `E[R | x]`.
  4. In varying-spread `Scenarios A & B` at `n = 2048`, both 1D score-calibrated heads (`pearson_head_tuned`, `mse_head_tuned`) improve sharply over scalar Pearson yet retain a nonzero distributional gap vs `CE` (because `q_head(r | x)` sees only 1 scalar `z(x)`), whereas in constant-concentration `Scenarios C & D` (`kappa = exp(1.5)`), `mse_head_tuned` and `pearson_head_tuned` approach `CE` much more closely.

## Primary Metric Panel

- **Test RPS excess over true distribution:** `0` optimal, `0..1`, lower is better (reported alongside **true distribution RPS** baseline and **constant training-histogram** test RPS so readers see exact differences relative to both `truth` and `constant`).
- **Test CE excess (KL(true || judge), nats):** `0` upward, lower is better.
- **Test Pearson correlation with true conditional mean:** `-1..1`, higher is better (undefined counted as `0` only in aggregates with explicit count reported).

## How to run

```bash
python3 experiments/11_toy_mse_and_tuned_head_calibration_vs_ce_start_17Nov2026/scripts/smoke_check.py
python3 experiments/11_toy_mse_and_tuned_head_calibration_vs_ce_start_17Nov2026/scripts/run_sweep.py 9
python3 experiments/11_toy_mse_and_tuned_head_calibration_vs_ce_start_17Nov2026/scripts/make_report.py
```

**TLDR-end:** [judge objectives: experiment 11] Pre-registered baseline test adding cross-fitted + slope-regularized Stage-2 calibration heads (`pearson_head_tuned`, `mse_head_tuned`) and directly MSE-trained judges (`mse_fixed`, `mse_valrps`) on Exp 08's exact synthetic suite.
