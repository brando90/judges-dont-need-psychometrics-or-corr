# Experiment 40: predictions written before any head was trained (11-27-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/40_output_space_k_sweep_g_d_study_sufficiency_and_label_dollars_start_27Nov2026/PREREGISTERED_PREDICTIONS_11-27-2026.md>

**TLDR:** Seven predictions, a decision rule for each, and the exact design, all fixed before `scripts/k_sweep_matched_calibration.py` or `scripts/gd_study_and_allocation.py` produced a number. This file is committed locally before those scripts run; its git commit time precedes every file under `results/`.

**As of:** 11-27-2026 (Friday, Pacific time). Local CPU, \$0, no model calls.

## Question

Sanmi Koyejo's objection on the 11-25 call was that in discrete settings the mean is the distribution, so correlation is close to the Brier score. The repository has already shown that the claim is false for scalar ordinal scores with K of 3 or more, and for any Pearson-type score, even at K = 2. What remains open is empirical: on real multi-annotator data, how does the output-space size K change the comparison between a judge trained by Pearson correlation and a judge trained by cross-entropy? Experiment 36 compared the two at each dataset's native K only.

## Design (fixed)

- **Data.** Experiment 36's loaders, unchanged: POPQUORN offensiveness and politeness (1–5), SummEval expert and crowd (1–5), VeriBench FSC `R_v1` all-five-raters (0–5), DICES-990 (3-level), ChaosNLI SNLI+MNLI (3-class) and ChaosNLI alphaNLI (binary). Re-binned views: K = 2 (value at or above the cut that Experiment 36 pre-registered, 4 for the 1–5 and 0–5 scales; DICES "Yes" versus not), K = 3 (1–5 as {1,2}, {3}, {4,5}; 0–5 as {0,1}, {2,3}, {4,5}) and the native K.
- **Split.** Prompt-level groups, fractions 0.6 / 0.2 / 0.2, split seeds 0, 1, 2, 3, 4. No rater of a prompt is in two splits.
- **Heads.** Experiment 36's linear softmax on 24 train-only principal components of a 256-dimensional signed token hash, L2 penalty 0.01, L-BFGS from zero, at most 35 iterations (the pre-registered cap of Experiment 36). Trained by cross-entropy (CE-trained head), by the sampled-judge Pearson surrogate (Pearson-trained head), and by scalar mean squared error (MSE-trained head). A constant forecast (the train marginal) is the reference.
- **Robustness run.** The same design with at most 300 iterations, to test whether the 35-iteration cap handicaps the Pearson-trained head.
- **Matched calibration.** Each head's scalar mean score is standardized with its validation moments, and one multinomial-logit map on [standardized score, 1] is fitted by cross-entropy to the validation ratings, then applied to the test items. The same map family is used for every head.
- **Primary metric.** Normalized ranked probability score (RPS divided by K − 1; range 0–1; lower is better) on held-out test items, averaged over each item's ratings first and then over items (the unbiased per-sample risk). Secondary: cross-entropy (CE) in nats. Reference value: the constant forecast.
- **Difference, in words.** "Test normalized RPS of the Pearson-trained head minus test normalized RPS of the CE-trained head; positive means the CE-trained head is better." Computed raw, and after both heads pass through the shared calibration map ("matched").
- **Uncertainty.** Paired item-clustered percentile bootstrap, 2,000 draws, seed 1, 95% level, per split seed; across split seeds we report the mean and the range.
- **Entropy strata.** For test items with four or more ratings, the ratings are split at random (fixed seed) into half A and half B. Half A gives the item's empirical Shannon entropy; items at or below the median entropy are "consensus", the rest "disputed". Scores use half B only, so the stratum is not chosen from the ratings being scored.
- **G-study and D-study.** One-way random-effects (items as the random effect, ratings within items) variance components by the unbalanced analysis-of-variance estimator. The single-rating reliability is the intraclass correlation ICC(1); the D-study reliability for k ratings per item is σ_c² / (σ_c² + σ_w² / k). The smallest k with reliability of at least 0.80 is the D-study recommendation k\*.
- **Allocation test.** At a fixed total rating budget B = N × k, the CE-trained head is trained with k = 1 (N = B items) and with the D-study k\* (N = B / k\* items), on the same fixed test items.

## Predictions and decision rules

1. **P1 (theory, K = 2).** On the binary views, a scalar score plus a calibration map spans the whole Bernoulli family, so the matched gap is zero up to estimation noise. *Supported* if the absolute matched gap, averaged over the five split seeds, is below 0.003 normalized RPS on every binary view.
2. **P2 (raw gap).** The raw gap favors the CE-trained head on every view that carries signal (CE-trained head beats the constant forecast), as Experiment 36 reported at the native K. *Supported* if the seed-mean raw gap is positive on at least 90% of such views.
3. **P3 (shape).** For K of 3 or more, the matched gap stays below 0.003 on at least 80% of views, because a 24-component linear head on hashed text has no capacity to predict item-specific shape beyond the mean. If P3 holds, the data say the cross-entropy advantage on these heads is calibration, not shape. *Not supported* (and shape learning is indicated) if some dataset shows a matched gap above 0.005 whose interval excludes 0 at K of 3 or more while its K = 2 view shows about 0.
4. **P4 (cap).** Raising the iteration cap from 35 to 300 changes no matched gap by more than 0.003 normalized RPS. *Supported* if so.
5. **P5 (strata).** The raw gap is positive in both entropy strata, and the matched gap is below 0.003 in both.
6. **P6 (D-study).** On POPQUORN and SummEval crowd, whose single-rating reliability is low, the D-study k\* for 0.80 is 8 or more.
7. **P7 (allocation).** At the same total budget, the CE-trained head trained at k = 1 is not worse on held-out normalized RPS than the same head trained at the D-study k\*, within the bootstrap interval. If P7 fails on a dataset, the failure is reported, not dropped.

Failures, non-convergence and missing cells are reported, not dropped.

## What this folder does not claim

It does not claim that cross-entropy training is useless relative to Pearson training in practice. It tests one specific hypothesis: whether, on the same features and the same calibration step, the training objective still matters. Any result here is for linear heads on hashed text, one seed-0 head per split seed, and says nothing about LoRA fine-tuning of language models.
