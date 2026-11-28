# Experiment 53: Experiment 41's heads as skill scores over the human base rate, with item-clustered intervals (27 Nov 2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/53_exp41_heads_as_skill_scores_with_item_bootstrap_cis_start_27Nov2026/README.md>

**TLDR:** Expressed as skill over the human base rate (1 = perfect, 0 = base rate), Experiment 41's "up to 0.128 normalized RPS" gap is mostly a Pearson-trained head that is *worse than the base rate*: its raw skill is below zero on 15 of 16 views (as low as -1.11), while the cross-entropy-trained (CE-trained) head's raw skill is between -0.005 and +0.071. After each head's validation-fitted calibration map, the CE-minus-Pearson skill difference is -0.0003 to +0.0098 on all 16 views. The headline 0.1282 is on SummEval crowd K = 5, where no head beats the base rate (CE-trained calibrated skill -0.0020). As the advisor noted, scoring a CE fit against a correlation fit with a proper rule favors the CE fit by construction, and a near-zero post-calibration gap is what $r^2 = 1 - \mathrm{MSE}(\text{best affine fit})/\mathrm{Var}(y)$ predicts. So this result supports "select by correlation, then recalibrate with labeled data". It does not show an artifact. The costs of that route are in other experiments: 50 to 200 held-out ratings per target ([Experiment 47](../47_calibration_label_budget_sweep_exp41_heads_start_27Nov2026/README.md)) and maps that do not transfer under shift ([Experiment 48](../48_calibration_under_dataset_rubric_and_rater_pool_shift_start_27Nov2026/README.md)).

**As of:** 11-27-2026 (Friday, Pacific time). CPU, \$0, no model calls. Protocol frozen first: [PROTOCOL.md](PROTOCOL.md), commit fe2e7a7.

## Question

The advisor's note S8 on the abstract asked for skill-score differences in place of the maximum raw RPS gap, so readers can judge whether 0.128 is large. It also pointed out the circularity of the comparison.

## Design

No new training design. [`scripts/skill_reanalysis.py`](scripts/skill_reanalysis.py) imports Experiment 41's code read-only and rebuilds its preregistered K-sweep: 7 rating sets re-binned to 16 views, CE-, Pearson- and MSE-trained linear heads, ridge 0.01, at most 35 L-BFGS iterations, group-level split seeds 0 to 4, and each head's own validation-fitted multinomial-logit calibration map. It keeps the per-item test scores and checks every (view, seed, head) mean against Experiment 41's `k_sweep_matched_calibration_maxiter35.json`.

- **Data split:** held-out test items of each seed's group split (no prompt group in two splits); all test ratings of each item; test items were also test items in Experiment 41.
- **Metric:** normalized ranked probability score (nRPS = RPS/(K-1), 0 to 1, lower is better). Skill = 1 - nRPS(head)/nRPS(base rate), where the base rate is the training-split marginal rating distribution. Higher is better: 1 is perfect, 0 is the base rate, and negative is worse than the base rate.
- **Difference:** skill of the CE-trained head minus skill of the Pearson-trained head, on the same items (positive = CE-trained better).
- **Uncertainty:** item-clustered percentile bootstrap 95% intervals (2,000 draws, seed 20261127; test items resampled within each split seed, the same draw for every head; the statistic is the mean over the five seeds). Holm-Bonferroni across the 16 views.

## Results

Full table: [`results/skill_tables.md`](results/skill_tables.md). Raw output: [`results/skill_reanalysis.json`](results/skill_reanalysis.json). Run log: [`logs/skill_reanalysis_run1_11-27-2026.txt`](logs/skill_reanalysis_run1_11-27-2026.txt).

| view | base-rate nRPS | skill, CE-trained raw | skill, Pearson-trained raw | skill difference raw (CE minus Pearson) [95%] | skill difference after calibration [95%] |
|---|---|---|---|---|---|
| POPQUORN offensiveness K=2 | 0.1156 | +0.0031 | -1.1141 | +1.1173 [+1.0116, +1.2586] | +0.0007 [+0.0003, +0.0011] |
| POPQUORN offensiveness K=5 | 0.1501 | +0.0048 | -0.6938 | +0.6986 [+0.6487, +0.7529] | -0.0002 [-0.0011, +0.0006] |
| POPQUORN politeness K=2 | 0.2489 | +0.0712 | +0.0862 | -0.0149 [-0.0189, -0.0111] | +0.0098 [+0.0081, +0.0114] |
| POPQUORN politeness K=5 | 0.1758 | +0.0340 | -0.3911 | +0.4251 [+0.4046, +0.4450] | +0.0053 [+0.0044, +0.0062] |
| SummEval expert K=5 | 0.1449 | +0.0120 | -0.6720 | +0.6840 [+0.6517, +0.7185] | +0.0046 [+0.0017, +0.0077] |
| SummEval crowd K=5 (no signal) | 0.1277 | -0.0009 | -1.0058 | +1.0049 [+0.9769, +1.0353] | -0.0003 [-0.0006, +0.0001] |
| DICES-990 K=3 | 0.1980 | +0.0138 | -0.3103 | +0.3240 [+0.2839, +0.3694] | +0.0032 [+0.0016, +0.0047] |
| ChaosNLI SNLI+MNLI K=3 | 0.1950 | +0.0066 | -0.2883 | +0.2950 [+0.2784, +0.3119] | -0.0001 [-0.0009, +0.0006] |

(8 of 16 views shown. All 16 are in the full table.)

- Raw skill difference: -0.0149 to +1.1173, positive on 15/16 views. The Pearson-trained head's raw skill is below zero on 15/16 views.
- The CE-trained head has little skill to begin with: raw -0.0054 to +0.0712, calibrated -0.0050 to +0.0988. These are weak linear heads on frozen text features, so the raw gap is large compared with the available headroom.
- After calibration, the CE-minus-Pearson skill difference is -0.0003 to +0.0098 on all 16 views. The MSE-trained head with the same map is within -0.0105 to +0.0033 of the CE-trained head.
- Preregistered predictions: P1 (raw difference > 0 with Holm p < 0.05 wherever Experiment 41's raw gap exceeds 0.01) held on 14/14 views. P2 (|calibrated difference| <= 0.025) held on 16/16. P3 (descriptive: CE-trained raw skill < 0.10) held on 16/16.

## Reading, and the circularity

1. A proper score that one head was trained on will favor that head. The informative part is the size of the gap on a bounded scale: the Pearson-trained head's raw forecast is often worse than ignoring the input. Correlation cannot see this, because it is invariant to the affine level and scale errors that cause it.
2. One validation-fitted map per head removes almost all of the gap, as $r^2 = 1 - \mathrm{MSE}(\text{best affine fit})/\mathrm{Var}(y)$ predicts for heads with the same discrimination. That supports selecting by correlation and then recalibrating, provided labels are available. [Experiment 47](../47_calibration_label_budget_sweep_exp41_heads_start_27Nov2026/README.md) finds the map needs 50 to 200 single held-out ratings per view; with 25, the recalibrated Pearson-trained head is worse than the base rate on 15/16 views. [Experiment 48](../48_calibration_under_dataset_rubric_and_rater_pool_shift_start_27Nov2026/README.md) finds a map fitted on one set reopens the gap on another in 53 of 82 frozen-judge cases (five frontier judges plus one G-Eval judge). Only a proper score on held-out labels shows whether recalibration has worked.
3. This is not evidence that apparent judge quality is an artifact. That needs real judges ranked differently by correlation and by proper-score skill, which is [Experiment 46](../46_real_llm_judge_rank_reversal_corr_vs_proper_scores_start_27Nov2026/README.md). Its interim frontier panel finds two Holm-significant Spearman-vs-Brier-skill reversals among five frontier judges, which were trained on neither objective, so that comparison is not circular.

## Protocol deviations

- The sanity tolerance was 1e-9 in the protocol. Re-running Experiment 41's code with this machine's numerical libraries reproduces its means to at most 1.3e-6 (non-convex Pearson fit at a 35-iteration cap) and 3.1e-7 (CE-trained head). The script therefore enforces 1e-3 and records the maximum deviation for each head in the JSON. No reported value changes at the 4-decimal precision used here.
- The uncertainty-term check (below) was compiled in Lean before the Python check ran. The repository's rule is numerics first. Both pass.

## The uncertainty term (advisor's note S7), checked in Lean and Python

With one rating per item, the mean per-item proper score is unbiased for the judge's expected score against $p^*$, which is enough to rank judges. The distance to $p^*$ is the expected score minus the uncertainty term $\mathbb{E}_x S(p^*_x, p^*_x)$. For the normalized Brier score that term is $\tfrac12(1-\sum_k p_k^2)$. For one RPS threshold $j$ it is $P_j(1-P_j)$. Both are functions of how often two independent ratings of the same item agree, so they can be estimated from pairs of ratings. With one rating per item they cannot be identified: two item populations with the same single-rating law $(1/2, 1/2)$ have mean Brier uncertainty $1/4$ and $0$.

- Lean: [`lean/UncertaintyTerm.lean`](lean/UncertaintyTerm.lean), four theorems with zero `sorry` and standard axioms only. Compiler and `#print axioms` output: [`lean/UncertaintyTerm_compile_output_11-27-2026.txt`](lean/UncertaintyTerm_compile_output_11-27-2026.txt). It was compiled with `lake env lean` inside the Point 2 Lake project (Lean 4.36.0, Mathlib v4.36.0, [`docs/notes/point2_support_27Nov2026/lean`](../../docs/notes/point2_support_27Nov2026/lean)).
- Python: [`scripts/check_uncertainty_term.py`](scripts/check_uncertainty_term.py) uses exact fractions on 80 laws plus a seeded Monte Carlo of the pair estimator, and includes a claimed-false case. It passes with identical output on both runs ([run 1](logs/check_uncertainty_term_run1_11-27-2026.txt), [run 2](logs/check_uncertainty_term_run2_11-27-2026.txt)).

## Reproduce

```bash
cd experiments/53_exp41_heads_as_skill_scores_with_item_bootstrap_cis_start_27Nov2026/scripts
E40_DATA_CACHE=/path/to/data_cache python3 skill_reanalysis.py --boot 2000   # ~7 min CPU; needs numpy, scipy
python3 make_tables.py
python3 check_uncertainty_term.py
```

`E40_DATA_CACHE` is the pinned public-data cache that Experiment 41's loaders download into (POPQUORN, SummEval, DICES and ChaosNLI at pinned revisions and checksums).
