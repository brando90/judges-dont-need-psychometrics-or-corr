# Experiment 44 protocol: is the rater pool the construct, are peaky forecasts right on consensus items, and what do binned JSD and k_test = 1 power say about real judges? (frozen 11-27-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/44_pool_as_construct_consensus_sharpness_binned_jsd_power_start_27Nov2026/PROTOCOL.md>

**TLDR:** Four small CPU analyses of the five frozen frontier judges of Experiment 28 on public multi-annotator rating sets. (P) Does changing the rater pool that defines p* change the score or the judge ranking beyond sampling noise? (S) Do proper scores reward a sharper forecast on consensus items and penalize it on contested items? (J) Where does binned JSD (Elangovan et al., ICLR 2025) agree or disagree with proper scores, and is it blind to item-level forecast quality? (M) How many held-out items does a paired judge comparison need at one rating per item? Frozen before any analysis script exists.

**As of:** 11-27-2026 (Friday, Pacific time). Written by Claude (Sonnet 5.5) in Claude Code for Brando Miranda. CPU only, $0, no API calls.

## Construct (stated before any metric)

The target of judge validation is a conditional distribution, p*(y | c, R, rho): the distribution of the rating y that a rater drawn from the pool R would give item c under the rubric rho. The pool R has three parts: who is eligible, how raters are weighted, and which question and scale they answer. Changing R or rho changes p*, so the choice of R and rho is the construct and is stated before any score is computed. The empirical stand-in used here is the per-item rating histogram of the raters in the stated pool, with uniform weights, written p_hat(. | c, R).

Credit and scope. Pavlick and Kwiatkowski (TACL 2019) and Plank (EMNLP 2022) established that a label is a distribution and argued for evaluating models against it; Nie et al. (EMNLP 2020, ChaosNLI), Uma et al. (2020) and Kurniawan et al. (2025) proposed soft metrics for NLP models. This repository does not claim the target. The claim is about the ruler for LLM-judge validation: how a judge should be scored once the target is accepted, and what current practice does instead (Experiments 43 and 45). On consensus items p* is peaked and a good judge is peaked too; a proper score rewards that, which is why scores are also reported by agreement stratum (Arm S, Experiment 50).

## What the analyst had seen before freezing

- Experiment 28's `results/raw_judge_scores.csv` (RPS, CE, TVD and top-1 of each of the five judges on every item of each set) was opened before this protocol was written. The all-item rankings of the five judges under those four metrics are therefore known, including that top-1 agreement and the proper scores order the judges differently on ChaosNLI MNLI-m, alphaNLI and DICES-990. No quantity in Arms P, S, J and M below was computed before freezing.
- The frozen protocols of Experiments 46 to 52 (written by a parallel session on 11-27-2026) were read. This protocol does not repeat their registered tests: rank reversal between correlation/agreement and proper scores, the skill score, strata by human entropy, calibration and the estimator simulation belong to them. Experiment 36 already holds power and minimum detectable difference (MDD) curves for trained heads; Arm M here does the same for frozen judge pairs.
- The structure of Experiment 39's processed bundle (sizes, K, rater-id availability) and the DICES demographic value counts were inspected. The elicitation prompts of Experiment 28 were read: the judges were told to forecast the share of a whole crowd ("a crowd of human annotators", "a diverse crowd of human raters"), not of a sub-pool.
- Elangovan et al. (arXiv 2410.03775) was read for the definition of binned JSD (Arm J).

## Data and judges

- Judges: the five frozen frontier judges of Experiment 28 (gpt-6-astra, claude-opus-5-5, kimi-k3, glm-5.3, deepseek-v4.1-flash), temperature 0, one verbalized-distribution forecast per item, from the committed `28_*/forecasts/forecasts_11-23-2026.csv`. Forecasts are floored as in Experiment 28: q <- (1 - K*1e-6) q + 1e-6. An invalid forecast is replaced by the uniform forecast (Experiment 28 and 46 rule, primary); the sensitivity analysis drops every item on which any judge was invalid.
- Sets and items (same item selection as Experiment 39's `prepare_data.py`, which is imported, not re-implemented): ChaosNLI SNLI 600, MNLI-m 600 (3 classes, nominal, 100 labels per item, no rater ids), alphaNLI 400 (2 classes, 100 labels per item), DICES-350 350 items (3 classes ordinal No=0, Unsure=1, Yes=2, 123 raters per item, rater ids and demographics), DICES-990 400 items (same coding, 69 to 76 raters per item). Public data are fetched by `39_*/src/fetch_raw_data.py` with SHA-256 checks.
- No fitting happens in this experiment, so there is no train/test split: every item is scored once.
- Primary proper score S, matched to the encoding, lower is better, unbiased per-sample risk (mean over each item's ratings of S(q, y), then mean over items): multiclass Brier, sum_k (q_k - 1{y = k})^2, for the nominal sets and alphaNLI; ranked probability score normalized by K - 1, (K-1)^-1 sum_{k<K-1} (Q_k - 1{y <= k})^2 with Q the cumulative forecast, for DICES. Secondary: CE with the 1e-6 floor.
- Seeds: numpy `SeedSequence([20261127, 44, arm, ...])`. Intervals are percentile intervals of item-cluster bootstrap draws (items resampled with replacement, 4,000 draws). A two-sided p-value is p = min(1, 2 min(P*(Delta <= 0), P*(Delta >= 0))) with the (B+1) correction. Holm-Bonferroni is applied inside each family named below.

## Arm P: does the pool that defines p* change the score or the ranking?

Sets: DICES-350 and DICES-990 only (the only sets with rater ids and demographics). Three pool contrasts, each between two disjoint pools of the raters who rated the item: gender (Man versus Woman), age group (gen z versus millenial plus gen x+), race (White versus Black/African American), using the CSV columns `rater_gender`, `rater_age`, `rater_race`. An item enters a contrast only if at least 5 raters of each pool rated it (counts reported). Per contrast and judge: the per-sample risk R_A(judge) and R_B(judge) against pool A and pool B on the eligible items, and the all-raters risk on the same items.

- **P1 (distributions).** Mean over eligible items of the total variation distance TVD(p_hat_A, p_hat_B), against a permutation null that shuffles pool labels among each item's raters (group sizes preserved), 500 permutations. p = (1 + #{null >= observed}) / 501. Holm over the 6 (set, contrast) cells. H-P1: in each set at least one contrast has Holm-adjusted p < 0.05.
- **P2 (absolute score).** Delta_pool(judge) = R_A(judge) - R_B(judge), paired item bootstrap, Holm over the 30 (set, contrast, judge) cells. H-P2: in each set at least one cell has Holm-adjusted p < 0.05.
- **P3 (ranking).** For each contrast, Kendall tau-b between the judge rankings under pool A and pool B (descriptive). For each judge pair (a, b): D(a, b) = [R_A(a) - R_A(b)] - [R_B(a) - R_B(b)], paired item bootstrap, Holm over the 60 (set, contrast, pair) cells. A **pool-dependent ordering** is a pair with Holm-adjusted p < 0.05 for D and with the pair's order reversed between the two pools by point estimate. Reported either way; no direction is predicted. "Pool-robust ranking" is the reading if no cell is Holm-significant and tau-b >= 0.8 in all six contrasts.
- The judges were asked to forecast the whole crowd. Scoring them against a sub-pool is a deliberate shift of the evaluation target relative to what they were told; the question is how far conclusions move, not whether a judge is wrong for a sub-pool.
- Not run: ChaosNLI (no rater ids), rubric shifts (documented in the README, not tested), SummEval expert versus crowd (no frozen frontier forecasts there).

## Arm S: are peaked forecasts rewarded on consensus items?

All five sets. Independent halves: each item's ratings are shuffled with seed SeedSequence([20261127, 44, 2, set, item]); the first floor(n/2) form half A (defines agreement), the rest half B (scores). Agreement a_i = largest category share in half A. Within each set the items are ordered by (a_i, item index) and the lowest third is the **contested** stratum and the highest third the **consensus** stratum (floor(N/3) items each; the middle third is unused). The temperature transform is q_tau proportional to q^(1/tau) renormalized and re-floored, tau in {0.5, 0.75, 1, 1.5, 2} (tau < 1 sharpens; the argmax, hence top-1 agreement, is unchanged for every tau). Per judge, stratum and tau: the primary-score risk on half B (CE secondary).

- Sharpening gain G(judge, stratum) = R(tau = 1) - R(tau = 0.5); positive means sharpening helps. Contrast Delta_G(judge) = G(consensus) - G(contested), with the two strata resampled independently (4,000 draws).
- **H-S1.** Delta_G > 0 by point estimate in at least 20 of the 25 (set, judge) cells, with the 95% interval excluding 0 in at least 15.
- Descriptive: the grid tau minimizing the risk in each stratum; the median a_i of each stratum (ChaosNLI SNLI and MNLI-m items were selected for moderate original disagreement, so consensus there is milder).

## Arm J: binned JSD

Definition (Elangovan et al., ICLR 2025), adapted to probabilistic forecasts. Bins: ordinal (DICES) by the human median rating (lower median of the sorted ratings), nominal and binary by the most frequent human label (ties to the lowest index). Per bin: the human PMF is the pooled share of the ratings of the bin's items; the machine PMF is the mean of the judge's forecasts over the bin's items (the expectation of the sampled-label PMF used in the paper). Per bin JSD with log base 2 (half KL of each PMF to their midpoint); binned JSD (JSb) is the sum over bins weighted by the bin's share of items. Lower is better; 0 means the pooled PMFs match. All items, all ratings.

- **H-J1 (invariance, exact).** JSb is unchanged, to 1e-12, when each judge's forecast vectors are permuted among the items of the same human bin. Checked over 200 seeded permutations per (set, judge).
- **H-J2.** The same permutation worsens the primary-score risk in all 25 (set, judge) cells (mean over the 200 permutations minus the original risk is positive). A judge whose forecasts carried no item-level information inside a bin would show about 0; the cells where the difference is not positive are listed.
- Descriptive: Kendall tau-b between the judge ranking by JSb and by the primary score within each set, with item-bootstrap 95% intervals of JSb. No hypothesis; the full-set proper-score rankings are known (see above).
- The permutation is a diagnostic of the metric, not a judge: bins are defined from human labels.

## Arm M: how many held-out items does a paired comparison need at k_test = 1?

All five sets, all 10 judge pairs per set, primary score. For a pair (a, b) and a rating y of item i, d_i,y = S(q_a,i, y) - S(q_b,i, y). Item means D_i, within-item variances s_i^2 (ddof 1), sigma_w^2 = mean_i s_i^2, sigma_b^2 = max(0, Var_i(D_i) - mean_i s_i^2 / n_i). Variance of the paired mean over N items with k ratings each: V(N, k) = (sigma_b^2 + sigma_w^2 / k) / N. MDD(N, k) = (1.96 + 0.8416) sqrt(V(N, k)) (two-sided 5%, 80% power). Grid N in {25, 50, 100, 200, 400, 800}, k in {1, 2, 5, 10}.

- Reported: MDD(N, 1) per pair, the observed |R(a) - R(b)| on all items, and N_80 = (1.96 + 0.8416)^2 (sigma_b^2 + sigma_w^2) / (observed difference)^2 for each pair; the share of the 10 pairs resolvable (N_80 <= 400) per set. At a fixed total budget B = N k the variance is (sigma_b^2 k + sigma_w^2) / B, which does not decrease in k.
- **H-M1 (formula check).** For N in {50, 100, 200}, k in {1, 5}, 2,000 subsamples (items without replacement, k ratings per item without replacement) give an empirical variance of the paired mean between 0.6 and 1.1 times V(N, k) for every checked cell of every set and pair median (named tolerance; below 1 is expected from sampling without replacement).
- **H-M2 (iso-budget).** At equal total ratings B the analytic variance is nondecreasing in k for every pair: checked as an identity, not tested.
- Descriptive, no hypothesis: whether k_test = 1 resolves the observed judge differences at N = 400.

## Outputs

`results/arm_p.csv`, `arm_s.csv`, `arm_j.csv`, `arm_m.csv`, `summary.json`; `RESULTS.md`; `HOW_TO_READ_RESULTS.md`; figures with `ax.grid(True)` on every axis; `tests/` (pytest: score identities, exact unbiasedness with k = 1 by enumeration, temperature leaves argmax unchanged, JSb permutation invariance, MDD formula on a hand-computed case); `src/verify.py` (an independent per-rating loop version of the scores and of JSb). Every number in the README comes from a script in `src/`. Runs are deterministic: two runs must produce byte-identical `summary.json`.

## Not run (and why)

Fixed-budget item-versus-rating split in training (needs training runs; Experiment 39 covers the selection side), judge drift across model generations, any API or GPU job. A conclusion that depends on them is not drawn here. Amendments after this freeze go in `AMENDMENT_A1.md` with a date; this text is not edited.
