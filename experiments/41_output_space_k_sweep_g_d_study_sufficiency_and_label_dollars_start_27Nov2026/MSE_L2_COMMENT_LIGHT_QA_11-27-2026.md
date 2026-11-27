# Did we address Sanmi's MSE plus L2 comment? A light QA

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/MSE_L2_COMMENT_LIGHT_QA_11-27-2026.md>

**TLDR:** Yes, we addressed it, and the verdict is "refined, not disproved". Sanmi is right that a regularized squared-error objective is a reasonable way to train a judge: an MSE-trained head, given the same per-head calibration step as every other head, is within 0.0004 normalized ranked probability score (RPS) of the cross-entropy-trained head on all 16 views. He is not right that this makes correlation a good evaluation metric, because correlation is the squared error after both sides are standardized, which throws away exactly the level and spread that the raw 0.05 to 0.13 gaps measure. Light QA means my own checks, not the Opus max review.

## 1. What he said (from the 11-25 transcript text)

1. Correlation is "really a proxy for squared error", with an L2 norm or "the regularization".
2. Squared error is "effectively Brier score", and Brier is proper.
3. In discrete settings "you're fine", because the mean of the indicator equals the distribution.
4. Brando's reply: "Brier and correlation are not the same", and he would do the math offline. Sanmi agreed the check was worth doing before leading with correlation.

## 2. The mathematical pieces, and how each was checked

1. **Correlation is squared error only after standardizing.** For standardized scores, the expected squared difference equals 2 times (1 minus the correlation). Lean: `standardized_mse_corr` in `experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/BrierCorr.lean`. The full decomposition of ordinary squared error into a bias term, a scale term and a correlation term is `mse_decomposition` in the same file. Correlation is unchanged by any positive affine map (`pearson_affine_invariant`), so it cannot see the bias or the scale term at all.
2. **Squared error equals Brier only when K = 2.** Two-class Brier is twice the scalar squared error (`binary_brier_eq_two_l2` in `experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/SanmiPitch.lean`). For K of 3 or more, Brier is a squared distance between probability vectors, and a scalar rating's squared error is a different object (`brier_not_local_three` in `experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/lean/E40_theorems.lean`).
3. **The mean of the indicator is the distribution only for K = 2 or for the full one-hot vector.** For a scalar ordinal score with K of 3 or more it is not (`mean_not_identifying`, `binary_mean_identifies`). Brando's margin comment on the Gemini notes calls the contrary claim "false obviously"; it is false for the scalar score and true in the two cases above.
4. **"L2" names two different things.** The L2 distance inside the squared-error loss is one; the weight penalty (ridge) on the model is the other. The ridge penalty is an optimization setting, and it mattered here for the wrong reason (section 4).
5. All of these have paired deterministic numerical checks (`experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/scripts/numerics_theorems.py`, `experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/check_math.py`).

## 3. The experiment

1. **Design.** Five training objectives on one family of heads (a 24-component feature map into a softmax head; the MSE head regresses the item's mean rating): cross-entropy, Brier, RPS, Pearson, and MSE. Seven public multi-annotator datasets, 16 views at K = 2, 3 and the native K, five seeds each, prompt-level train, validation and test split, item-clustered bootstrap 95% intervals. Scripts: `experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/scripts/k_sweep_l2_sensitivity.py` and `experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/scripts/analyze_l2_sensitivity.py`; tables in `experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/results/k_sweep_l2_sensitivity_tables.md`.
2. **Penalty.** The ridge weight is chosen per head on the validation split from six values (1e-5 to 1), not fixed. This analysis was not preregistered; it was added after the Opus max review found that a fixed penalty of 0.01 over-shrinks the cross-entropy heads on these small features.
3. **Metric.** Held-out normalized RPS (RPS divided by K minus 1, between 0 and 1, lower is better), averaged over each test item's ratings first. A difference is always the same metric on the same test split for two heads, and a positive value means the cross-entropy-trained head is better.

## 4. Results

1. **Raw output, no calibration step.** The Pearson-trained head is worse than the cross-entropy-trained head by 0.0053 to 0.1008 normalized RPS on 15 of 16 views (the one exception is a no-signal binary view, where the gap is within noise). Correlation training does not identify the level or spread of the distribution, and that is what the gap measures. This does not depend on the penalty.
2. **After each head gets its own calibration map** (standardize the head's scalar mean with validation moments, then fit a multinomial-logit map on the validation split, separately per head, of the same family): Pearson-trained minus cross-entropy-trained is at most 0.0022 in absolute value (largest: POPQUORN politeness, K = 2); MSE-trained minus cross-entropy-trained is at most 0.0004 in absolute value on every view, and its interval straddles zero on most. So a squared-error objective plus calibration matches cross-entropy on these sets, which supports Sanmi's point, and correlation plus calibration is close but keeps a small significant deficit on the POPQUORN politeness views.
3. **The earlier "MSE beats cross-entropy" signal was an artifact.** A fixed 0.01 penalty over-shrank the cross-entropy heads while the MSE head's ridge, written on the sum form, was about half of N times weaker. With an N-scaled ridge and validation-chosen penalties the difference disappears (`experiments/41_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/scripts/e40_l2.py`).
4. **Size of the signal.** On the views where a head beats the constant (train-marginal) forecast, the whole learnable gain is small (the constant scores 0.116 to 0.250, and the gain is usually a few thousandths), so a residual of 0.002 is a real fraction of it on some views. Between 21% and 49% of the 80 fits per head choose an end value of the penalty grid. The lower end (1e-5) is effectively no regularization; the upper end (1.0) is the one that suggests the grid may be too narrow, and it is chosen in 6 to 24 of the 80 fits per head (most often for the Pearson-trained head's raw forecast, 24).

## 4b. What to say to Sanmi (two sentences)

Correlation is the squared error between standardized scores, so it inherits squared error's mean-matching but deletes bias and scale, and the 0.05 to 0.13 raw gap is exactly that deletion; once every head is calibrated the objectives are within a few thousandths, so the case for cross-entropy rests on evaluation and calibration rather than on a large training advantage. We checked this with five training objectives on seven public sets, and the squared-error head ties cross-entropy.

## 5. Verdict and what is not established

1. **Verdict:** refined, not disproved. The claim "correlation is a proxy for squared error" is true with the standardization caveat; "squared error is Brier" is true at K = 2 only; "discrete means are distributions" is false for scalar ordinal scores; "a regularized squared-error objective is a fine training loss" is supported empirically here.
2. **Not established:** that cross-entropy training beats squared-error training on these sets (it does not, after calibration); that the result holds for a fine-tuned language-model judge (the heads here are small feature-map heads, and a LoRA run is the open test); and that the penalty grid is wide enough everywhere.
