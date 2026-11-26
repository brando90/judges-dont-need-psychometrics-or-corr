# Audit of judge experiments, and the generalizability result of 11-25-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/RESULTS.md>

**TLDR:** Synthetic studies change analytic forecasts or toy weights, not a language-model prompt. Open-weight work already covers prompts, harnesses, and low-rank adaptation. Frontier work covers native probabilities, prompt search, and harness search. The new result, experiment 34, is a CPU simulation: a lenient point mass with generalizability coefficient 1 is 0.0492 ranked probability score (RPS) worse than a constant forecast, and a same-moment shape change leaves a 0.0205 RPS gap that cross-entropy closes.

**As of:** 11-25-2026. Numbers below are copied from each experiment's own results file or from [experiments/README.md](README.md). This page does not recompute them.

## (i) Synthetic and toy cross-entropy versus correlation

These studies do not change a language-model prompt or a language-model weight. They pick an analytic distribution or fit a small toy head.

| Folder | What changed | What the comparison was |
| --- | --- | --- |
| `00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026` | Two fixed score rules on one synthetic sample | Pearson stays high for a squashed judge (0.877) while Kullback–Leibler divergence separates it (3.420 versus 0.001). One seed, no interval |
| `01_conditional_binomial_distribution_identification_start_08Nov2026` | Choice among analytic conditional distributions | Cross-entropy identifies the known conditional distribution; correlation does not |
| `05_cross_entropy_vs_correlation_information_gap_start_13Nov2026` | A derivation and a numerical check | What correlation constrains versus what cross-entropy constrains |
| `06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026` | Toy-head weights, Spearman, Kendall, or a mixture with Pearson | None matched the cross-entropy-trained judge on held-out distribution metrics |
| `07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026` | Optimizer budget and five random starts | Held-out RPS crossovers were stable. No global-optimum certificate for the nonconvex objectives |
| `08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026` | Ridge penalty chosen on validation | Correlation training won at low data only in the weak-mean setting |
| `11_toy_mse_and_tuned_head_calibration_vs_ce_start_17Nov2026` | Mean-squared-error training and a second-stage calibrator, 4,400 grid points | Calibration removes some small-sample error. End-to-end cross-entropy still leads when the shape varies |
| `archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026` | A flexible toy head on nonlinear synthetic ratings | Pearson versus cross-entropy training. Not human-data evidence |
| `17_flexible_scalar_calibrator_vs_1d_bottleneck_claim_start_20Nov2026` | A flexible map from a scalar score to a distribution | On the true mean it matches or beats end-to-end cross-entropy |
| `18_when_does_ce_beat_correlation_notebook_page_math_and_sweep_start_21Nov2026` | Data size, flexibility, regularization, rater noise | When the cross-entropy advantage appears |
| `20_trained_judge_ce_vs_pearson_limited_data_synthetic_and_59_judge_pools_start_21Nov2026` | The same toy judge, two objectives; also selection on cached pools | Cross-entropy wins above about 16 items in the synthetic sweep and can lose at 4 to 8 items |
| `33_moment_derivation_checks_what_correlation_and_ce_identify_start_25Nov2026` | No ratings and no fit | Checks the moment identities used by the technical memo |

`02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026` and archived experiments 11, 12, and 14 mix fixed cached judge outputs. The trained object is a weight on those outputs, not a prompt and not a network weight.

## (ii) Open-weight judges

| Folder | Prompt | Harness | Weights |
| --- | --- | --- | --- |
| `09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026` | Yes. Six prompts and a blind search | Yes. Six harnesses | Yes. PyTorch low-rank adaptation, about 1.85M to 2.91M parameters, on mercury2 at $0. Ten models, 75 artifacts. Native six-category cross-entropy 1.629 nats (verbalized) and 1.606 nats (log-probability) versus a 1.887-nat constant forecast |
| `21_fresh_task_families_matched_open_closed_judges_start_21Nov2026` | The same grading message for both models | Frozen models. Calibration of the six reported probabilities, 27 settings | No weight update. Qwen3.5-9B cross-entropy fit, test RPS 0.1548 at 4 ratings versus Pearson 0.2263 |
| `24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026` | Fixed | Fixed | Yes. Qwen3-4B-Instruct-2507 low-rank adaptation on HelpSteer2, matched budgets |
| `25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026` | Fixed | Fixed | Yes. Larger Qwen budget and Llama-3.1-8B |
| `29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026` | Fixed | Fixed | Yes. Cross-entropy, RPS, mean squared error, and Pearson, on Qwen3-4B and Llama-3.1-8B. 1,401 runs |

Archived experiment 14 selects among cached open-weight outputs. It does not update weights.

## (iii) Frontier judges

| Folder | What was varied | Scale |
| --- | --- | --- |
| `10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026` | Native six-category probabilities, blind prompt search, harness search | $17.85. Native-pool cross-entropy 1.416 nats versus 5.49 nats for the historical scalar pool and 1.928 for a constant forecast. Simplex test RPS, Pearson-trained minus cross-entropy-trained, +0.0775 [+0.0437, +0.1135]. Discrete prompt and harness selection was uncertain |
| `26_five_new_frontier_judges_on_fresh_homework_and_independent_recheck_start_23Nov2026` | Five frontier judges, calibration on fresh homework | Cross-entropy fit beat Pearson fit on held-out RPS in 35 of 35 judge-by-budget pairs. About $1.34 |
| `28_public_rating_sets_chaosnli_dices_frontier_judges_ce_vs_pearson_start_23Nov2026` | Five frontier judges on ChaosNLI and DICES | Cross-entropy fit beat Pearson fit on held-out RPS after 4–8 ratings at one rating per item, or 10–20 at five, in all 25 dataset-by-judge pairs |

Experiments 30, 31, and 32 replot or refit scaling curves on the saved outputs of 26, 28, and 29. They do not call a new model.

## Experiment 34, the new result

Folder: `34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026`. Full tables: [results.md](34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026/results.md). Derivation: [g_d_vs_ce_derivation.pdf](34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026/derivation/g_d_vs_ce_derivation.pdf). Google Doc: <https://docs.google.com/document/d/1QbJYWT1rrRFSQFFZpncQt6pN7xAl9wKHlR06xpvQMS0/edit>. The Drive connector cannot grant anyone-with-the-link access, so the Doc is shared with brando.science@gmail.com as a commenter.

Known ordinal distributions, 400 replications, CPU, 32.2 seconds, $0. No model and no human rating.

- Binomial items, 256 ratings: binomial-projection RPS minus add-one cross-entropy RPS is −0.000187 [−0.000194, −0.000181]. Both beat the pooled Laplace forecast by about 0.075.
- Leniency, 16 ratings: the point mass has judge-only reliability 1, bias +0.998 [0.988, 1.008], Pearson 0.961 [0.959, 0.962], and RPS 0.049183 [0.047743, 0.050622] worse than the pooled Laplace forecast (400/400 replications). Maximum reliability selects it 400/400 times. Validation RPS never selects it and beats the constant forecast by 0.0721 RPS.
- Same mean and variance, 256 ratings: coefficients match, and the projection remains 0.020508 [0.020476, 0.020541] RPS worse than add-one, with smoothed cross-entropy +0.690 nats and TVD +0.486.
- Six raters, 64 ratings each: lenient point mass minus pooled Laplace is +0.153902 [0.153007, 0.154797] RPS (positive means the constant forecast is better). Add-one on all raters minus pooled Laplace is −0.052124 [−0.052129, −0.052118]. The neutral rater, which has the highest stochastic reliability, also beats the constant forecast.

This does not replace experiment 04's finding that variance components can choose a rater-versus-occasion design. No GPU run is required; see [NEEDS_GPU.md](NEEDS_GPU.md).

**TLDR-end:** [judge objectives: audit and experiment 34] Synthetic work is toy weights or analytic forecasts, open-weight work already includes prompts, harnesses, and low-rank adaptation, and the new simulation shows reliability 1 with a 0.049 RPS loss to a constant forecast.
