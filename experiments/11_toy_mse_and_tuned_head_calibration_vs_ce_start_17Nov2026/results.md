# Experiment 11: results

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/11_toy_mse_and_tuned_head_calibration_vs_ce_start_17Nov2026/results.md>

**Status, 11-17-2026:** sweep complete (`4,400` rows: `4` scenarios × `11` training sizes × `100` seeds, `500` held-out test tasks/seed). Pre-registered smoke checks passed (`check_grad` errors `< 2.5e-7`, `c_b = 1e8` marginal limit `1.09e-9`, `Exp 08` seed-0 parity `< 1e-11`).

**TLDR:** Cross-fitting and tuning slope regularization on Stage-2 calibrated heads (`pearson_head_tuned`) cuts the naive calibrated head's small-`n` spike (`n <= 32`) by ~3–9× relative to `constant` (eliminating it completely by `n = 32`), and direct mean-squared-error (`MSE`) training fixes Pearson's scale/shift drift so it beats `constant` across all four scenarios at `n = 2,048` (`0.0076–0.0118` RPS excess). Yet directly fitting all 78 softmax weights by MSE leaves conditional higher moments unconstrained (`0.118–0.197` nats CE vs `0.0066–0.0224` for `③ Val-CE`), and in varying-disagreement Scenarios A and B (`κ(x)` varies), `③ Val-CE` beats every Stage-2 head at every tested sample size (`0.0007` vs `0.0078–0.0079` in A; `0.0036` vs `0.0071–0.0074` in B at `n = 2,048`). When concentration `κ` is constant (`C` and `D`, a 1-parameter exponential-family mean curve), properly regularized Stage-2 CE heads (`12` params on a sufficient scalar summary) slightly outperform unconstrained `78`-weight `Val-CE` from `n = 512–2,048` (`0.0015–0.0022` vs `0.0027–0.0029`) while still requiring **proper-loss CE calibration** in Stage 2.

## Pre-registered prediction verdicts

| Pre-registered prediction | Result |
| --- | --- |
| 1. `constant` and `ce_fixed` match `Exp 08` exact-CE within `1e-11`; `c_b = 1e8` yields `P.mean(0)` within `1e-7` | **Confirmed:** seed-0 parity `< 1e-11`, `check_grad <= 2.41e-7`, `c_b = 1e8` max diff `1.09e-9`. |
| 2. `pearson_head_tuned` eliminates the `n <= 32` spike seen in `pearson_head_naive` | **Mostly confirmed:** cuts max `n <= 32` excess over `constant` from `+0.1040..+0.1470` (`3.50–3.80` nats at `n=8`) down to `+0.0145..+0.0387`, and eliminates it by `n = 32` (`-0.0004..+0.0006`). |
| 3. At `n = 2,048`, `mse_fixed` and `mse_valrps` fix Pearson's affine drift yet still lose to `ce_fixed`/`ce_valrps` | **Confirmed:** `mse_valrps` beats `constant` (`-0.0020` to `-0.0552`), yet `mse_valrps − Val-CE` is `+0.0069` to `+0.0087` RPS excess (`95%` CIs all above zero) and `0.118–0.197` nats vs `0.0066–0.0224`. |
| 4. At `n = 2,048`, 1D calibrated heads keep an irreducible gap to `CE` in `A & B` while approaching `CE` in `C & D` | **Confirmed & sharpened:** in `A & B`, `Val-CE` beats `pearson_head_tuned` and `mse_head_tuned` at every `n >= 4`; in constant-`κ` `C & D`, 12-param proper-loss heads slightly beat 78-weight `Val-CE` at `n = 2,048` (`-0.0007` to `-0.0012` RPS). |

## Held-out test RPS relative to truth and constant baseline (`n = 2,048` tasks / `6,144` ratings; 100 seeds)

Reported on `500` fresh test tasks/seed (`true RPS` baseline: `A = 0.1869`, `B = 0.1363`, `C = 0.1270`, `D = 0.1465`; subtraction `Judge − Val-CE` positive means `③ Val-CE` is better; `95%` percentile bootstrap over `100` seeds, `10,000` draws):

| Scenario | True RPS | Constant raw (ex.) | ③ Val-CE raw (ex., CE nats) | Pearson+tuned head raw (ex., diff vs ③) | MSE val-RPS raw (ex., diff vs ③) | MSE+tuned head raw (ex., diff vs ③) |
| --- | --- | --- | --- | --- | --- | --- |
| A: flat mean, κ varies | `0.1869` | `0.1965` (`0.0096`) | `0.1876` (`0.0007`, `0.0066`) | `0.1948` (`0.0079`, `+0.0072 [+0.0071,+0.0073]`) | `0.1945` (`0.0076`, `+0.0069 [+0.0069,+0.0070]`) | `0.1947` (`0.0078`, `+0.0071 [+0.0070,+0.0071]`) |
| B: curved mean, κ varies | `0.1363` | `0.2033` (`0.0670`) | `0.1400` (`0.0036`, `0.0224`) | `0.1437` (`0.0074`, `+0.0037 [+0.0036,+0.0038]`) | `0.1481` (`0.0118`, `+0.0082 [+0.0080,+0.0083]`) | `0.1434` (`0.0071`, `+0.0034 [+0.0033,+0.0035]`) |
| C: curved mean, κ const | `0.1270` | `0.1910` (`0.0640`) | `0.1299` (`0.0029`, `0.0202`) | `0.1292` (`0.0022`, `-0.0007 [-0.0008,-0.0006]`) | `0.1386` (`0.0116`, `+0.0087 [+0.0086,+0.0088]`) | `0.1289` (`0.0019`, `-0.0010 [-0.0011,-0.0009]`) |
| D: linear mean, κ const | `0.1465` | `0.1914` (`0.0449`) | `0.1492` (`0.0027`, `0.0183`) | `0.1483` (`0.0018`, `-0.0009 [-0.0010,-0.0008]`) | `0.1569` (`0.0104`, `+0.0077 [+0.0077,+0.0077]`) | `0.1480` (`0.0015`, `-0.0012 [-0.0013,-0.0011]`) |

For comparison, raw fixed-ridge `② Pearson` at `n = 2,048` reaches `0.2122` (`0.0253` ex), `0.2314` (`0.0951`), `0.2163` (`0.0893`), and `0.1989` (`0.0523`) raw RPS, worse than `constant` in every scenario (`+0.0074` to `+0.0281`).

Full tables: [results/tables.md](results/tables.md). Compact machine summary: [results/summary.json](results/summary.json). Figure: [fig_mse_and_tuned_head_vs_n.png](fig_mse_and_tuned_head_vs_n.png). Reading guide: [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md).

**TLDR-end:** [judge objectives: experiment 11 results] Cross-fitted slope regularization eliminates the naive calibration-head spike (`n <= 32`) and direct MSE removes Pearson's scale drift, yet end-to-end CE still strictly dominates whenever conditional disagreement varies (`A, B`) while 12-param proper-loss heads match or edge out 78-wt CE when concentration is constant (`C, D`).
