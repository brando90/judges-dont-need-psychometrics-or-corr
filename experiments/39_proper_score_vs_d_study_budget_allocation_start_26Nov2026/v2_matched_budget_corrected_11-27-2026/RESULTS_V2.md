# Experiment 39, Study A v2: corrected matched-information re-run (results)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/39_proper_score_vs_d_study_budget_allocation_start_26Nov2026/v2_matched_budget_corrected_11-27-2026/RESULTS_V2.md>

**Date:** 11-27-2026 (America/Los_Angeles). **Machine:** local CPU, `$0`, no model calls. **Run:** started 18:41:56 PT, 247.78 s, 150 replications × 5 datasets = 750 rows. **Git order (checkable):** the frozen protocol and the code were committed in `4065d0f` before the run; the results, analysis and figures are in `b1ddf6e`.

**TLDR:** With the four defects found by the independent review fixed, the old headline shrinks. On held-out ranked probability score (RPS), choosing the design with the most items per rating budget has a lower mean RPS than the most-raters design in 18 of 20 dataset-budget cells, but the difference is significant after Holm correction in only 5 of 20 cells and never exceeds 3.0% of total RPS. A pilot-based proper-score selector beats the most-raters rule in 2 of 20 cells after Holm (the first version said 10 of 20) and beats the most-items rule in 0 of 20. A pilot-based D-study with a reliability target does not beat the most-items rule in any cell. All six predictions written before the run held; decision rules D2, D3 and D4 are not met.

**Predictions caveat.** The six predictions were written after the v1 results and the review were known, so they test whether the corrected protocol behaves as the review implied. They are not blind forecasts of the data.

## 1. What was re-run, and the four fixes

The independent review of commit `1cd4fc6` (`experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/qa_of_experiment_39_push_1cd4fc6_11-27-2026/REVIEW_by_independent_opus_subagent_11-27-2026.md`) found four defects in the first version of Study A. Study B (the restricted-maximum-likelihood recovery study) was not re-run; its numbers are unchanged and only its wording was corrected.

1. **Ties.** v1 resolved exact ties in the oracle "by more items", which inflated agreement with a more-items selector. v2 draws the pick uniformly from the tied set with a seeded rule that favours neither side, and reports tie-inclusive agreement next to it.
2. **A D-study that did not depend on the data.** At a fixed budget the v1 selectors U1 (per-item) and U2 (grand mean) always picked the most-raters and the most-items design, whatever the pilot said. v2 keeps them as labelled analytic corner rules (they equal the fixed rules in 100% of replications, checked) and adds `ds_rel80`, a D-study that picks the smallest number of raters per item whose estimated generalizability coefficient reaches 0.80, so its choice does depend on the pilot variance components.
3. **Hidden information.** v1's proper-score selector read the full rating pool of its pilot items (69 to 123 ratings per item), more than the advertised 80 × 16 pilot, and the pilot (1,280 ratings) is 2.5 to 20 times the deployment budget. v2 gives every pilot-using selector only the 80 × 16 table and reports the pilot cost next to the budget. The pilot cost is not netted out: a pilot-using selector consumes the pilot plus the budget, a fixed rule only the budget. A literal equal-total comparison is impossible on this grid, because 1,280 exceeds the largest design (64 × 16 = 1,024). So any win for a pilot-using selector is an upper bound on its net value, and a loss is stronger evidence than a loss at equal total.
4. **Statistics.** v1 used a replication-only Student-t interval and no multiplicity correction. v2's primary interval and p-value come from a replication × test-item two-way bootstrap (10,000 draws), with Holm-Bonferroni across the 20 comparable dataset-budget cells inside each comparison family.

## 2. What is measured

**Data.** Frozen test items of five public multi-annotator sets: ChaosNLI SNLI (180 test items), ChaosNLI MNLI-m (180), ChaosNLI alphaNLI (120), DICES-350 (105) and DICES-990 (120); the development pools hold 420, 420, 280, 245 and 280 items. The test ratings are never used to choose a design, a transform or a mixture weight. The items are the subsets already used in Experiment 28, whose five frozen frontier-judge forecasts were produced earlier without training on these ratings. ChaosNLI has no rater identities; DICES-350 is a full 350 × 123 crossing and DICES-990 is incomplete.

**Metric.** "RPS" is the normalized individual-rating ranked probability score: for each test item, the squared distance between the forecast cumulative distribution and the cumulative indicator of each of the item's ratings, averaged over ratings and then over items (equal item weights). Lower is better. It contains a forecast-independent floor. The per-cell floor, the reducible excess (RPS minus floor) and the RPS of a constant base-rate forecast are listed in Table 1 of `results/tables_v2.md`. For example, ChaosNLI SNLI has floor 0.0961, base-rate RPS 0.1886, and the selectors and the oracle in the table lie between 0.1109 and 0.1177. The NLI labels are nominal, so RPS depends on the label order imposed; cross-entropy (CE) and total variation distance (TVD) use the same procedure as order-free secondary checks.

**What is subtracted.** A gain of selector A over selector B is the mean held-out test RPS of B minus the mean held-out test RPS of A on the same test items and replications; a positive gain means A is better. "Beats" means a lower mean and a Holm-adjusted two-way-bootstrap p-value below 0.05.

**Uncertainty.** Two-way bootstrap (replications and test items resampled independently, 10,000 draws, percentile 95% interval, two-sided p with a +1 correction); Holm-Bonferroni across the 20 cells, separately within each family; no adjustment across families. The bootstrap does not cover development-pool or rater-population sampling.

**How the forecast head was chosen.** For each replication and each candidate design (n_items ∈ {16, 32, 64} times n_raters ∈ {2, 4, 8, 16}), the head is the 27-member family of recalibrations of the frozen five-judge mixture (temperature ∈ {0.5, 1, 2} times linear and quadratic category shifts ∈ {−1, 0, 1}). The head with the lowest training cross-entropy on that design's training ratings is selected by exhaustive search over the 27 (an exact global minimizer within the family; ties go to the first in the fixed order). This discreteness is why many designs have exactly equal test RPS.

## 3. Selectors compared

| Name | Rule | Uses the pilot? |
|---|---|---|
| `fix_raters` | the design with the most raters per item at the budget | no |
| `fix_items` | the design with the most items at the budget | no |
| `ds_rel80` | smallest raters per item with estimated generalizability coefficient ≥ 0.80 (else the largest) | yes, the 80 × 16 table |
| `ds_rel90` | the same at 0.90 (sensitivity) | yes |
| `ps` | lowest mean cross-validated RPS of the CE-fitted head on the pilot table, 5 folds × 8 inner draws | yes, the table only |
| neutral RPS-oracle | the design with lowest deployed test RPS, ties broken neutrally (diagnostic, not a selector) | uses test labels |

## 4. Pre-declared predictions and decision rules (scored by `src/score_predictions_v2.py`)

| # | Prediction (from `PROTOCOL_V2_11-27-2026.md`) | Result | Verdict |
|---|---|---|---|
| 1 | F1: `fix_items` has lower mean RPS than `fix_raters` in a majority of the 20 cells, Holm-significant in at most 6, gains below 3% of total RPS | lower in 18 of 20 cells; Holm-significant in 5; largest gain 2.97% of total RPS (alphaNLI, B = 256) | holds (the 3% bound was met with 0.03 points to spare) |
| 2 | F3: `ps` does not significantly beat `fix_items` in any cell | 0 of 20 (`fix_items` is lower in mean in 16 of 20 cells, `ps` in 3, one exact tie) | holds |
| 3 | F6: `ps` does not beat `ds_rel80` in at least 11 of 20 cells | 0 of 20 after Holm (12 of 20 lower in mean, none significant) | holds |
| 4 | Adaptivity: `ds_rel80` has at least two distinct modal designs across datasets at ≥ 2 of the 4 budgets | 3 of 4 budgets (B = 64: all datasets pick 16 × 4) | holds |
| 5 | Unique-best share of the RPS-oracle is below 0.5 overall | 0.250 | holds |
| 6 | `ds_u1` = `fix_raters` and `ds_u2` = `fix_items` in 100% of replications; deployed scores equal v1 | 100% in every cell; deployed cell means equal the v1 summary means to 2.2e-16 (the v1 replication-level file was never committed, so per-replication bit-identity could not be tested) | holds, with that limit |

| Decision rule | Outcome |
|---|---|
| D1: report the Holm count of cells where `fix_items` beats `fix_raters`, not a general law | 5 of 20 after Holm (11 of 20 raw); 18 of 20 lower in mean; CE 6, TVD 4 after Holm |
| D2: proper-score selection adds value only if `ps` beats `fix_items` in ≥ 1 cell | **not met** (0 cells) |
| D3: a pilot D-study adds value only if `ds_rel80` beats both fixed rules in ≥ 1 cell | **not met** (0 cells; `ds_rel80` beats `fix_raters` in 2 cells, and `fix_items` beats `ds_rel80` in 3) |
| D4: `ps` tracks the distributional target better than the D-study only if it beats `ds_rel80` in ≥ 11 cells | **not met** (0 cells) |

## 5. Results by family (full tables: `results/tables_v2.md`)

Counts are over the 20 cells (5 datasets × B ∈ {64, 128, 256, 512}), RPS, two-way bootstrap, Holm within the family.

| Family (gain of first over second) | first lower in mean | first beats, raw | first beats, Holm | second beats, Holm | CE Holm (first beats) | TVD Holm (first beats) |
|---|---:|---:|---:|---:|---:|---:|
| F1 `fix_items` over `fix_raters` | 18 | 11 | **5** | 0 | 6 | 4 |
| F2 `ps` over `fix_raters` | 17 | 6 | **2** | 0 | 3 | 2 |
| F3 `ps` over `fix_items` | 3 | 0 | **0** | 0 | 0 | 0 |
| F4 `ds_rel80` over `fix_raters` | 10 (the other 10 cells tie exactly) | 5 | **2** | 0 | 1 | 2 |
| F5 `ds_rel80` over `fix_items` | 2 | 0 | **0** | 3 | 0 | 0 |
| F6 `ps` over `ds_rel80` | 12 | 2 | **0** | 0 | 0 | 0 |
| F7 `ds_rel90` over `fix_items` | 2 | 0 | **0** | 3 | 0 | 0 |

Effect sizes. The largest F1 gain is +0.00335 RPS (ChaosNLI alphaNLI, B = 256), which is 2.97% of total RPS and 13.3% of the reducible excess; the largest F2 gain is +0.00277 (alphaNLI, B = 256; 2.45% of total, 11.0% of excess). In DICES-990 at B = 512 the four selectors have identical per-item test losses (every gain is exactly 0 with a degenerate interval), so that cell carries no information. In the three cells where `fix_items` beats `ds_rel80` after Holm (SNLI B = 64, MNLI-m B = 256, DICES-990 B = 256), the losses are 2.35%, 0.86% and 0.96% of total RPS.

Sensitivity. Under the replication-only paired t (the v1 method) with Holm, `fix_items` beats `ps` in 1 of 20 cells on RPS (the two-way bootstrap, which is primary, gives 0), and `fix_items` beats `ds_rel80` in 4 (bootstrap: 3). The secondary Amendment A1 mixture selector `ps_mix` (replication-only t with Holm, because per-item mixture losses are not rebuilt; this interval is the more liberal one) has a lower mean RPS than `fix_raters` in 18 of 20 cells, significantly so after Holm in 7, and a lower mean than `fix_items` in only 4 of 20 cells, none significant; `fix_items` beats `ps_mix` in 1 cell after Holm. So the mixture variant recovers a larger win over the max-raters rule than the primary head does, and still never beats the max-items rule.

Analytic sanity rows (Table A; mean-estimation targets, not RPS). `fix_raters` has the lower item-mean squared error in 20 of 20 cells and `fix_items` the lower grand-mean squared error in 18 of 20, the same as v1: these two targets have opposite optimal corners, which is a property of the formulas, not of a pilot.

## 6. What this changes in the v1 statements

| v1 statement (commit `1cd4fc6`) | Status after v2 |
|---|---|
| The proper-score selector significantly lowers held-out RPS against the max-raters rule in 10 of 20 cells and never loses | **Withdrawn as stated.** Replication-only t-test, no correction, and hidden full-pool access. v2: 6 of 20 raw (the v1-style replication-only t also gives 6), 2 of 20 after Holm. The comparator `fix_raters` is the same rule with identical deployed scores in both versions, so the drop from 10 to 6 comes from the changed `ps` selector (pilot-table-only information and the neutral tie rule); the drop from 6 to 2 comes from the multiplicity correction. |
| The proper-score selector does not beat the max-items rule (0 wins, 2 losses, 18 ties) | **Kept, reworded.** 0 wins and 0 Holm-significant losses; `fix_items` is lower in mean in 16 of 20 cells; no evidence that pilot-based proper-score selection improves on a fixed rule at this pilot size. The v1 "ties" were mostly shared head choices. |
| The better split for held-out RPS is more items, fewer raters | **Kept, weakened.** Lower mean RPS in 18 of 20 cells, significant in 5, effect at most 3.0% of total RPS. A small effect, not a general law: the grid has at least 2 raters per item and at most 64 items, and the head has three effective parameters. |
| A D-study picks the design (U1, U2) | **Reworded.** U1 and U2 are data-independent corner rules at fixed budget. The adaptive `ds_rel80` responds to the data across datasets (3 of 4 budgets) but never beats the max-items rule. |
| U1 matches the item-mean oracle; U1 beats U2 on item-mean error (20/20); U2 beats U1 on grand-mean error (18/20) | **Kept** (analytic; reproduced in Table A). |

The corrected statement for the paper: proper scores make the target of a rating allocation explicit; they do not choose an allocation by themselves. For the held-out-forecast target on this grid, more items mildly beats more raters, and a 1,280-rating pilot did not pay for itself against that fixed rule. A universe-score target keeps its own optimum (more raters per item), as variance-component theory says.

## 7. Limits

- One forecast head (27 recalibrations of five frozen judges), one pilot size (80 × 16), a grid without single-rating designs and with at most 64 items; Experiment 40 covers the one-rating-per-item regime.
- ChaosNLI has no rater identities, so crossed rater effects are not modelled by the one-facet D-study.
- The bootstrap omits development-pool and rater-population sampling.
- RPS depends on the label order for nominal NLI labels; CE and TVD are reported for that reason; they show the same qualitative pattern (columns in section 5), and none of F3, F5, F6 or F7 has a significant win for a pilot-using selector on any of the three metrics.
- Predictions were written after v1 was known (caveat above).

## 8. Files

`PROTOCOL_V2_11-27-2026.md` (frozen before the run), `src/study_a_v2.py`, `src/analyze_v2.py`, `src/verify_v2.py`, `src/score_predictions_v2.py`, `src/make_figures_v2.py`; `results/study_a_v2_replications.csv` (750 rows), `results/study_a_v2_meta.json`, `results/analysis_v2.json`, `results/cells_v2.csv`, `results/tables_v2.md`, `results/verification_v2.json` (32 of 32 checks pass), `results/prediction_scorecard_v2.json`, run logs; `figures/fig1_v2_test_rps_by_selector.svg`, `figures/fig2_v2_paired_gains_holm.svg`; `HOW_TO_READ_RESULTS.md`; the paste-ready note for the Experiment 39 Google Doc is `../CORRECTION_NOTE_EXP39.md`.

Reproduce: `python3 src/study_a_v2.py` (about 4 minutes), then `python3 src/analyze_v2.py`, `python3 src/verify_v2.py`, `python3 src/score_predictions_v2.py`, `python3 src/make_figures_v2.py`, all from this folder after fetching the raw data (`../src/fetch_raw_data.py`).
