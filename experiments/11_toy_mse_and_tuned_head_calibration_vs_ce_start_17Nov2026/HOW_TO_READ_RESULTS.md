# How to read the results of experiment 11

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/11_toy_mse_and_tuned_head_calibration_vs_ce_start_17Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Synthetic baseline follow-up testing Sanmi Koyejo's Point-2 baselines (`pearson_head_tuned`, `pearson_head_in_tuned`, `mse_fixed`, `mse_valrps`, and `mse_head_tuned`) on Experiment 08's exact four scenarios, sample sizes, and seeds. Scores compare held-out test distributions against both the true distribution (`RPS excess`, `CE excess`) and the constant training-histogram baseline (`Judge minus Constant`), without ratio-of-excess compression.

## What is compared

All multi-parameter weights belong to the exact same 78-weight softmax family (`13` cosine/sine harmonics × `6` logits) used in Experiments 06–08 and 13, and every two-stage calibrated head uses a 12-parameter ordinal logistic regression (`softmax(a_r + b_r z)` fitted by exact log-sum-exp CE):

| Judge | Training objective | Regularization & calibration | Notes |
| --- | --- | --- | --- |
| `constant` | None (training empirical marginal histogram) | None | Non-parametric baseline |
| `ce_fixed` (④) | Exact log-sum-exp CE | `(1/n) ||W||^2` (`c = 1`) | Convex |
| `ce_valrps` (③) | Exact log-sum-exp CE | 4-fold CV RPS chooses `c` in `{0.01..100}` | Convex at fixed `c` |
| `pearson_fixed` (②) | Maximize Pearson corr of expected ratings | `(1/n) ||W||^2` (`c = 1`) | Non-convex |
| `pearson_head_naive` | Stage 1 Pearson (`c = 1`), Stage 2 exact CE on `(1, z_tr)` | `(1/n)(||a||^2 + ||b||^2)` on in-sample `z_tr` | Reproduces `toy.py` head (with exact CE) |
| `pearson_head_in_tuned` | Stage 1 Pearson (`c = 1`), Stage 2 exact CE on `(1, z)` | End-to-end 4-fold CV RPS chooses `c_b` in `{0.01..1e8}` | Isolates slope-ridge tuning |
| `pearson_head_tuned` | 4-fold cross-fitted Stage-1 Pearson scores (standardized per fold) + Stage-2 exact CE | 4-fold CV RPS chooses slope ridge `c_b` in `{0.01..1e8}` on out-of-fold `z_oof`; intercept gauge `1e-9 ||a - log(p_bar)||^2` | Recovers `constant` when `c_b -> 1e8` (`b -> 0`) |
| `mse_fixed` | MSE between expected rating `m_W(x)` and sample mean | `(1/n) ||W||^2` (`c = 1`) | Anchors level & scale |
| `mse_valrps` | MSE between `m_W(x)` and sample mean | 4-fold CV RPS chooses `c` in `{0.01..100}` | Non-convex |
| `mse_head_tuned` | Cross-fitted Stage-1 MSE (`c = 1`) + Stage-2 exact CE | 4-fold CV RPS chooses slope ridge `c_b` in `{0.01..1e8}` on `z_oof` | Same Stage-2 protocol as `pearson_head_tuned` |

## Which data each number uses

- **Training tasks:** `n in {4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048}`, `k = 3` sampled ratings per task; Scenarios A–D generated identically to Experiment 08 (`seed = 0..99`).
- **Held-out test tasks:** `500` fresh tasks per seed never used in training, standardization, or cross-validation, scored against the true population distribution `p(r | x)`.

## Metrics and subtraction conventions

- **Raw Test RPS** (`0` to `1`, lower is better): expected ranked probability score against test truth; reported alongside **true distribution RPS** (`E[1/5 sum_j P_true(j)(1 - P_true(j))]`).
- **Test RPS excess** (`0` to `1`, `0` optimal, lower is better): `Judge RPS - True RPS`.
- **Test RPS relative to Constant** (`Judge − Constant`, paired by seed; **negative means the judge beats Constant**; **positive means the judge is worse than Constant**).
- **Test RPS relative to Val-CE** (`Judge − ③ Val-CE`, paired by seed; **positive means ③ Val-CE is better**).
- **Test CE excess** (`KL(true || judge)`, nats, `0` optimal, lower is better).
- **Uncertainty:** Paired `95%` percentile bootstrap intervals over `100` seeds (`10,000` draws).

**TLDR-end:** [judge objectives: experiment 11 HOW_TO_READ_RESULTS] Exact comparison contract across `constant`, exact-CE, Pearson, naive/tuned calibrated heads, and MSE-trained baselines on Experiment 08's four scenarios.
