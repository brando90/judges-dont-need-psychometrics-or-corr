# How to read the results of experiment 10

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Every table, plot, and caption in `10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026` compares real frontier artificial intelligence (AI) judges trained, prompt-searched (`iii.a`), or harness-configured (`iii.b`) under different objectives, then scored with the **same held-out test metrics** on the **same held-out task families**. Judge names always describe **how a judge was built or chosen**, never the metric used to score it.


**Reference provenance amendment (11-16-2026):** Phase 1 records 15/75 reference-file hash matches (60 mismatches) against annotation metadata, alongside 75/75 candidate and 75/75 Python matches. Saved evidence does not resolve stale reference metadata versus changed bytes; identity with human-delivered reference files is unverified.



**Failure handling amendment (11-16-2026):** Saved evaluation retains uniform replacements after parsing failures: the selected cross-entropy prompt has 11/75 replacements overall, including 7/30 held-out test artifacts. Harnesses also retain uniform and upstream scaffold fallbacks. Cache-recovered status records and split denominators are in the Phase 3/4 result JSON files and the main report; no forecasts or metrics were changed.


## 1. Which data each number uses

- **Dataset:** 75 VeriBench Lean 4 formal-specification artifacts across 25 Python tasks grouped into **19 task families**, with **195 expert human ratings** on `{0, 1, 2, 3, 4, 5}` (2 or 3 expert raters per artifact; 20 ordinary tasks = 60 artifacts, and 5 `True`-placeholder security tasks = 15 artifacts). No differential testing (no DT) is run.
- **Prior exposure disclosure:** All 195 human ratings come from the historical VeriBench annotation collection (`cached_v1/dataset.json`, SHA-256 `b3a05699...`). Although our 15 new native-probability frontier judges (`P0`, `P1`, `P2`, `L0`) and our blind `iii.a`/`iii.b` prompt and harness searches never view held-out test items during prompt proposal, selection, or fitting, the underlying 75-artifact human rating set was used in earlier repository studies (`02`, `11`, `12`, `14`), so this study is **retrospective at the dataset level**.
- **Splits (all strictly grouped by the 19 task families — zero task-family leakage):**
  1. **Repeated grouped task-family splits (`250` splits, `iii.a`/`iii.b`/pool/learning-curve analysis):** Each split assigns `15` task families to training (and inner three-fold task-family cross-validation over ridge c in {0.1, 1, 10} for feature-head Formulations ① and ③) and holds out `4` unseen task families for test evaluation. Learning curves vary nested training sizes `n_fam in {3, 5, 7, 9, 12, 15}` while keeping the 4 held-out test families identical.
  2. **`02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026` 5x5 grouped task-family cross-validation (`25` outer folds):** Exact historical 5-partition x 5-fold task-family splits for direct 1-to-1 comparison with Experiment 02.
  3. **Locked 3-way grouped task-family split for blind prompt & harness search (`9` train families = 27 artifacts, 69 ratings / `5` validation families = 18 artifacts, 45 ratings / `5` held-out test families = 30 artifacts, 81 ratings):** Proposers in `iii.a` see ONLY the 9 training families; validation selection (①, ③) uses ONLY the 5 validation families; final test numbers come from the 5 held-out test families. Any in-sample diagnostic is explicitly labeled `in-sample (train)`.

## 2. Metrics (all evaluated on held-out test artifacts unless labeled in-sample)

| Metric | First-use expansion & one-line definition | Range / units | Better | Reference scale on this dataset |
|---|---|---|---|---|
| **RPS** (primary) | **Ranked probability score:** `(1/5) sum_{k=0}^4 (Q_i(k) - 1{y <= k})^2` averaged over artifact `i`'s human ratings `y`, then averaged equally across test artifacts (`Q_i(k) = sum_{r<=k} q_i(r)`) | `0` to `1` (probability-squared / threshold) | **Lower** | Empirical human disagreement noise floor `~0.052`; constant training histogram `~0.187`; good frontier judge `~0.11–0.14` |
| **CE** (secondary) | **Cross-entropy (log loss):** average `-ln q_smooth(y)` over each artifact's human ratings with pre-specified floor `epsilon = 1e-6`, averaged across test artifacts | `0` upward, in **nats** | **Lower** | Constant training histogram `~1.93` nats; 3-sample historical scalar judge `~5.49` nats; calibrated native-prob judge `~1.45–1.75` nats |
| **TVD** (secondary) | **Total variation distance:** `0.5 * sum_{r=0}^5 |q_i(r) - p_hat_i(r)|` against artifact `i`'s empirical human rating histogram `p_hat_i` (sparse-histogram diagnostic, not a proper score) | `0` to `1` | **Lower** | Constant training histogram `~0.70`; 3-rating sampling floor even for truth `~0.23–0.29` |
| **Brier** | **Six-category Brier score:** average `sum_{r=0}^5 (q_i(r) - 1{y == r})^2` over observed human ratings | `0` to `2` | **Lower** | Constant training histogram `~0.87` |
| **Pearson `r` / Spearman `rho` / Kendall `tau-b`** | Linear, rank, and pairwise-concordance correlation between judge expected rating `E_q[R]` and artifact human mean rating across test artifacts | `-1` to `+1` | **Higher** | Leave-one-rater-out human Spearman ceiling `~0.79`; dumb theorem-count ratio `0.543` |
| **Intraclass correlation (ICC): ICC(A,1) / ICC(C,1)** | Shrout–Fleiss absolute agreement `ICC(2,1)` (penalizes level offset) and consistency `ICC(3,1)` (ignores offset) between `E_q[R]/5` and human mean | `-1` to `+1` | **Higher** | Gap `ICC(C,1) - ICC(A,1)` diagnoses systematic leniency/harshness |
| **Mean squared error (MSE) decomposition & signed bias** | Exact identity on `0–5` scale: `MSE(x, y) = (x_bar - y_bar)^2 [bias^2] + (s_x - s_y)^2 [scale_mismatch^2] + 2 s_x s_y (1 - r_xy) [unexplained_variance]`; signed `bias = x_bar - y_bar` (positive = lenient) | Rating-squared (`0` to `25`); bias in rating points (`-5` to `+5`) | **Lower** MSE; `0` bias | Separates level error and spread compression (invisible to Pearson!) from ordering error |

## 3. How to read a difference (never subtract two different metrics)

Every difference subtracts **the exact same metric on the exact same held-out test artifacts** between two judges that were trained or selected differently:

- **For lower-is-better loss/distance metrics (`RPS`, `CE`, `TVD`, `Brier`, `MSE`):**
  `Delta = (test metric of correlation-trained/selected judge) minus (test metric of proper-loss-trained/selected judge)`.
  **Positive (`> 0`) means the proper-loss judge is better** (had lower test error); **negative (`< 0`) means the correlation judge is better**.
- **For higher-is-better agreement metrics (`Pearson`, `Spearman`, `Kendall`, `ICC(A,1)`):**
  Stated explicitly in each column header, e.g., `test Spearman of proper-loss-trained judge minus test Spearman of correlation-trained judge (positive favors proper loss)`.
- Never read or write a bare `"CE minus Pearson"`.

## 4. Uncertainty

- **Task-family intervals:** Pool, prompt, harness, and disagreement contrasts use 95% task-family cluster-bootstrap percentile intervals (`10,000` draws), preserving artifacts within each drawn family and holding fitted predictions fixed. They do not refit models or include training-search uncertainty.
- **Feature-head learning curves:** The saved 95% intervals resample `250` overlapping split averages (`10,000` draws). They measure conditional split Monte Carlo variability on this fixed dataset, not population uncertainty over new task families. This differs from the frozen task-family-bootstrap plan; no population significance or rating threshold is established.
- **Tail fractions:** `2 * min(Pr[Delta* <= 0], Pr[Delta* >= 0])` is a descriptive uncentered bootstrap sign-tail fraction, not a calibrated null-test p-value. Reports label it `bootstrap-tail`; population `p-val=n/a`. No multiple-comparison adjustment was applied.

**Execution deviations:** Feature-head ridge selection used three inner folds and three c values, whereas the frozen plan named five folds and five values. The `L0` logprob configuration uses returned top-five tokens, normalizes observed rating-digit mass, and assigns zero to missing digits before flooring; it is a truncated estimate, not exact six-category conditional probabilities. See the transparent amendment in `FREEZE.json`.

**Ensemble comparison:** The probability-average H4 uses five forecasts (Sol, mini, Codestral, H2, H3), while the three-vote H4 uses only Sol, mini, and Codestral. Their loss difference changes ensemble membership and aggregation together and does not isolate discretization.

## 5. How each judge was chosen (search space, objective, optimizer, guarantee)

| Design axis & class | Search space `Theta` | Formulations & objectives | Optimizer & budget | Global optimum guaranteed? |
|---|---|---|---|---|
| **Single base judge (`M1`)** | 15 native-prob frontier judges, 5 historical scalar judges, or 20 combined | ④ Train CE/RPS; ③ Val RPS; ② Train Pearson/Spearman/Kendall/Mix; ① Val Pearson | Exhaustive evaluation over all `M` candidates | **Yes** (exhaustive) |
| **Simplex mixture (`M2`)** | Nonnegative weights `w in Delta^{M-1}` summing to 1 | `CE` (④/③), `RPS` (④/③), `Pearson` (②/①), `Spearman`, `Kendall`, `1/2 P + 1/2 S` | `CE` & `RPS`: SLSQP + Newton polishing with Frank–Wolfe convex gap `<= 1e-8`; `Pearson`: exact cone/NNLS solver; rank objectives: 5 L-BFGS-B starts (`TAU=0.1`) + 4,000 Dirichlet draws | **Yes** for `CE`, `RPS`, and `Pearson` (checked convex/cone certificates); **No** for smooth rank mixtures (reflects surrogate + budget) |
| **Softmax feature head (`M3`)** | `48` weights `W in R^{8 x 6}` on pool covariates + ridge `(c/n)||W||^2` | All 7 objectives + `Pearson + calibrated CE head`; fixed `c=1` (②/④) or three-fold inner cross-validation over `c in {0.1, 1, 10}` (①/③) | L-BFGS-B (`400` iters, 1 start) checked against **best-of-5 starts x `2,000` iters** (`Exp 07` check) | The CE objective is strictly convex; solver convergence is checked numerically, not by a saved optimality certificate. No global guarantee for RPS/correlation in W; the restart check measures budget sensitivity. |
| **`iii.a` Blind prompt search** | Natural-language rubric prompts (`<= 420` words) starting from `P0_human_mirror` | `Track_Corr` (① val Pearson, ② train Pearson) vs `Track_CE` (③ val RPS, ④ train CE) | `gpt-6.1-sol` blind proposer and `gpt-5.4-mini` judge, 4 iterations x 2 candidates = 8 prompts/track | **No** (discrete LLM prompt search; reflects matched 4x2 proposer budget) |
| **`iii.b` Harness search & calibration** | 4 harness architectures (`H1`, `H2`, `H3`, `H4`) + 2-param calibration `(alpha, beta)` | Discrete harness selection (①–④) + `softmax(alpha log q + beta r/5)` fitted by `CE` vs `Pearson` | Exhaustive over `H1..H4`; L-BFGS-B on `(alpha, beta) in R^2` | Discrete candidate selection is exhaustive. CE in `(alpha, beta)` is convex; finite numerical fitting is not itself an optimality certificate. Pearson calibration has no global guarantee. |

**TLDR-end:** [judges-corr: experiment 10 reading guide] All comparisons hold out entire task families, score competing training/prompt/harness objectives with the same test RPS/CE/TVD metrics, and write differences so positive RPS/CE/TVD gaps favor proper-loss training.
