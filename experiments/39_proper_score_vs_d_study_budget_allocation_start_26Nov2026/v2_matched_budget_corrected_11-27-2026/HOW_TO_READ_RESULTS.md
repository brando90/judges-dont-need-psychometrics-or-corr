# How to read Experiment 39, Study A v2 (corrected re-run)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/39_proper_score_vs_d_study_budget_allocation_start_26Nov2026/v2_matched_budget_corrected_11-27-2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Every number in `results/` is a held-out test score of a forecast trained on a small, simulated rating budget, on frozen test items of five public multi-annotator tables. A "gain" subtracts the mean test score of one selection rule from another's on the same items and replications; positive means the first rule is better. "Beats" means a lower mean and a Holm-adjusted two-way bootstrap p-value below 0.05. Full results: [RESULTS_V2.md](RESULTS_V2.md).

## 1. Which data

1. **Test items (held out).** The frozen 30% item split of five public tables, drawn before any design was evaluated: ChaosNLI SNLI (180 test items), ChaosNLI MNLI-m (180), ChaosNLI alphaNLI (120), DICES-350 (105) and DICES-990 (120). The test ratings and test features are never used to pick a design, a recalibration, or a mixture weight. Prior exposure: the five frontier-judge probability forecasts were collected in Experiment 28 earlier and are frozen; the human rating tables are public.
2. **Development items (used for training and pilots).** The other 70% (420, 420, 280, 245 and 280 items). In every replication a pilot of 80 items × 16 ratings is drawn from them, the selection rule sees only that pilot table, and the chosen design then draws its deployment training ratings (n_items × n_raters, at most 1,024) from the full development pool without replacement.
3. **What is random.** The pilot, the deployment draw, and the cross-validation folds, redrawn in each of 150 replications per dataset (750 rows in total, `results/study_a_v2_replications.csv`). The test items are fixed.
4. **Raters.** ChaosNLI has per-item label counts and no rater identities; DICES-350 is a full 350 × 123 crossing; DICES-990 is incomplete (at least 69 ratings per selected item). The split is by item: every rating of a test item stays in the test split. Raters are not split (DICES raters appear on both development and test items), so rater-level generalization is not tested.
5. **Not a human-rating claim.** This is a design-selection experiment on cached frozen forecasts, not a conference-style claim about a new judge.

## 2. Which metric

1. **RPS (primary).** The normalized individual-rating ranked probability score: for each test item, $(1/(K-1))\sum_{k=0}^{K-2}\big(F(k)-\mathbf 1\{y\le k\}\big)^2$ averaged over that item's ratings $y$, then averaged equally over test items. $F$ is the forecast cumulative distribution and $K$ is 3 (SNLI, MNLI-m, DICES) or 2 (alphaNLI). Lower is better. Range 0 to 1.
2. **Reference values.** The forecast-independent floor (the best any forecast could do against that item's rating spread) is 0.0961 (SNLI), 0.1387 (MNLI-m), 0.0878 (alphaNLI), 0.1890 (DICES-350) and 0.1806 (DICES-990). The base-rate forecast (a constant: the development-pool category frequencies with 0.5 added per category, which is Jeffreys smoothing) and the neutral oracle design's RPS are listed per cell in Table 1 of `results/tables_v2.md`. Differences of the order of 0.001 to 0.003 RPS are 1% to 3% of total RPS and 4% to 13% of the reducible excess (RPS minus floor).
3. **Secondary.** Cross-entropy (CE, in nats: $-\sum_k h(k)\log q(k)$ against the item's empirical rating histogram $h$, with the forecast $q$ floored at $10^{-6}$, averaged equally over test items) and total variation distance (TVD, $0.5\sum_k|q(k)-h(k)|$, 0 to 1), both lower is better. The NLI labels are nominal, so RPS depends on the label order imposed; CE and TVD do not.
4. **Analytic sanity rows (Table A).** Item-mean and grand-mean squared error are universe-score quantities, computed on the deployment items, never mixed into an RPS comparison.

## 3. What is subtracted

1. A **gain of A over B** is test RPS of the design chosen by B minus test RPS of the design chosen by A, on the same test items, in the same replication, averaged over replications. Positive means A is better.
2. The families are, in order: F1 `fix_items` over `fix_raters`; F2 `ps` over `fix_raters`; F3 `ps` over `fix_items`; F4 `ds_rel80` over `fix_raters`; F5 `ds_rel80` over `fix_items`; F6 `ps` over `ds_rel80`; F7 `ds_rel90` over `fix_items`. A negative gain in F3, F5 or F7 means the fixed most-items rule is better than the pilot-based rule.
3. Methods are named by how the design was chosen (a fixed rule, a pilot D-study, a pilot cross-validated proper-score selector), never by a bare metric name.

## 4. Uncertainty

1. **Interval and test.** Two-way bootstrap, resampling replications and test items independently, 10,000 draws, percentile 95% interval, two-sided p-value with a +1 correction. Paired within replication.
2. **Multiplicity.** Holm-Bonferroni across the 20 comparable dataset-budget cells (5 datasets × budgets 64, 128, 256, 512; budgets 32 and 1,024 have a single design) within each family. No correction across families. Raw counts are given next to the Holm counts in `RESULTS_V2.md`.
3. **Not covered.** Development-pool sampling and rater-population sampling are held fixed.

## 5. How the design and the forecast were chosen

1. **Forecast head.** For each candidate design, the head is the member of a 27-element family of recalibrations of the five-judge mixture (temperature in {0.5, 1, 2} times linear and quadratic category shifts in {−1, 0, 1}) with the lowest cross-entropy on that design's training ratings. This is an exhaustive search over a discrete family, so it is an exact minimizer within the family and ties go to the first member in a fixed order. The discreteness is why many designs have exactly equal test RPS.
2. **Design choice.** Candidate designs are n_items ∈ {16, 32, 64} times n_raters ∈ {2, 4, 8, 16} with n_items × n_raters equal to the budget. `fix_raters` picks the most raters, `fix_items` the most items, `ds_rel80` / `ds_rel90` the smallest number of raters whose estimated generalizability coefficient reaches 0.80 / 0.90 on the pilot table, `ps` the design with the lowest 5-fold × 8-draw cross-validated RPS of the CE-fitted head on the pilot table. The neutral RPS-oracle (uses test labels) is a diagnostic.
3. **Pilot cost.** The 80 × 16 pilot (1,280 ratings) is 2.5 to 20 times the deployment budget and is not netted out. A pilot-using selector therefore consumes the pilot plus the budget and a fixed rule only the budget, so a win by a pilot-using selector is an upper bound on its net value.
4. **Not guaranteed optimal.** The pilot-based rules are estimates from a noisy table; they carry no guarantee, and the oracle is the only per-replication optimum.

**TLDR-end:** [judge objectives: experiment 39 v2] Positive gain means the first-named rule has the lower held-out test RPS; "beats" needs a Holm-adjusted two-way bootstrap p below 0.05; effects are 1% to 3% of total RPS, and the pilot cost is not netted out.
