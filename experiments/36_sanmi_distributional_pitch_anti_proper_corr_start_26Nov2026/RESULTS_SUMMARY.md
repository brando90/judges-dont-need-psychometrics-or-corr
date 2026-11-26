# Results summary — Sanmi distributional pitch (11-26-2026)

**As of 11-26-2026.**

**Bottom line.** Lean 4.36.0 checks the algebraic claims with no `sorry`. Python checks the same claims numerically, including 0.50 versus \(\sqrt{1/2}\) for the anti-proper correlation. The human-data tables below are held-out test prompts from SummEval, ChaosNLI, and DICES. POPQUORN and VeriBench functional semantic correctness were not available here and are not imputed.

## Formally proved (Lean) and numerically checked

Binary witness, four paired items: Pearson correlation of the calibrated probabilities and of the affine map \(0.1q+0.45\) are both 1.000. Mean Brier 0.0800 versus 0.4418. Mean cross-entropy 0.2231 versus 0.6349 nats. Anti-proper Monte Carlo (n=200000): correlation of the true sampled judge 0.5002 (theory 0.5) and of \(0.1\mu+4\) 0.7074 (theory \(\sqrt{1/2}\approx 0.7071\)). Ordinal witness: both laws have mean 2.0; \(D_B=1.5\); normalized RPS gap 0.2000; cross-entropy of the split forecast on the consensus outcome is infinite.

Theorem names and the axiom printout are in `lean_check.txt`. The numerical assertions are `proofs_numeric.py` (every check returned ok).

Fair-coin couplings: \(\mathbb E[(X-Y)^2]\) is 0, 1/2, and 1 while \(D_B=0\). Plug-in \(D_B(\hat p_N\Vert q)\) sits above \(D_B(p^*\Vert q)\) by \((1/N)(1-\|p^*\|_2^2)\) in expectation. The test tables do not use that plug-in.

## What the held-out numbers say

The CE, Brier, and RPS heads agree with each other. The Pearson head's held-out cross-entropy is much larger on every ordinal and 3-class set (SummEval expert about 4.73 versus about 1.21; DICES-990 about 1.16 versus about 0.74; ChaosNLI SNLI+MNLI about 4.02 versus about 1.05). Its correlation is not higher than the CE head's. The MSE two-point readout has infinite cross-entropy on those sets because it assigns probability 0 off the two adjacent scores. On binary views the same readout is finite and competitive, which is the K=2 case where the mean is the law. ChaosNLI abductive NLI is a negative control: every head matches the uniform forecast, so the text hash does not predict that task. On DICES-990, splitting each test prompt's raters in half gives a within-prompt oracle, not a deployable judge. A draw from the add-one distribution of the first half has correlation 0.190 with the second half. The collapsed score 0.1·mean+4 has correlation 0.410. Held-out-half cross-entropy is 0.6764 versus 6.7936, and Brier is 0.3881 versus 0.9835. Higher correlation, worse proper scores.

## Real multi-annotator results (this machine)

Proper-score cells are the unbiased prompt-average of per-rating losses. Pearson r is the correlation of the judge's scalar (expected score, or the ridge mean) with individual ratings, item-clustered bootstrap. Spearman and Kendall are correlations of the prompt-level human mean with that scalar, same bootstrap. Lower is better for cross-entropy, sqrt(RPS), Brier, and normalized RPS. Higher is better for the three correlations.

### summeval_expert

SummEval expert annotations on four aspects (coherence, consistency, fluency, relevance), native grades 1–5 stored as class values [1,2,3,4,5]. Every summary has [3] expert annotation records. The jsonl does not publish a stable rater id, so rater_id is the within-item vote index, not a person. Split group is the source article (filepath), so every system and aspect of one article stays in one split. Binary views use the pre-registered cut grade ≥ 4.

Split by prompt group, seed 0: train 3840, validation 1280, test 1280. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 1.210 [1.171, 1.247] | 0.749 [0.731, 0.764] | 0.615 [0.595, 0.634] | 0.1402 | 0.253 | 0.331 | 0.238 |
| brier-trained | 1.213 [1.174, 1.251] | 0.750 [0.733, 0.766] | 0.617 [0.597, 0.636] | 0.1408 | 0.256 | 0.345 | 0.247 |
| rps-trained | 1.213 [1.174, 1.251] | 0.750 [0.732, 0.765] | 0.617 [0.597, 0.636] | 0.1405 | 0.245 | 0.309 | 0.223 |
| pearson-trained | 4.730 [4.509, 4.961] | 0.998 [0.994, 1.002] | 0.905 [0.883, 0.929] | 0.2489 | 0.225 | 0.272 | 0.196 |
| mse-trained | inf [undefined, undefined] | 0.902 [0.887, 0.918] | 1.291 [1.266, 1.316] | 0.2035 | 0.220 | 0.274 | 0.197 |

Uniform forecast held-out test CE 1.6094. Train-marginal forecast held-out test CE 1.2210. CE head refit after shuffling train features: held-out test CE 1.2217.

### summeval_expert_binary

Binary view of summeval_expert: class 1 iff the original value ≥ 4. This is a threshold of the same ratings, not a separately collected binary task. SummEval expert annotations on four aspects (coherence, consistency, fluency, relevance), native grades 1–5 stored as class values [1,2,3,4,5]. Every summary has [3] expert annotation records. The jsonl does not publish a stable rater id, so rater_id is the within-item vote index, not a person. Split group is the source article (filepath), so every system and aspect of one article stays in one split. Binary views use the pre-registered cut grade ≥ 4.

Split by prompt group, seed 0: train 3840, validation 1280, test 1280. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.575 [0.556, 0.593] | 0.440 [0.430, 0.449] | 0.387 [0.370, 0.404] | 0.1934 | 0.200 | 0.254 | 0.195 |
| brier-trained | 0.576 [0.558, 0.593] | 0.440 [0.431, 0.450] | 0.388 [0.371, 0.404] | 0.1939 | 0.200 | 0.253 | 0.194 |
| rps-trained | 0.578 [0.560, 0.596] | 0.441 [0.432, 0.451] | 0.390 [0.373, 0.407] | 0.1948 | 0.201 | 0.254 | 0.195 |
| pearson-trained | 0.689 [0.686, 0.693] | 0.498 [0.496, 0.500] | 0.496 [0.492, 0.500] | 0.2481 | 0.202 | 0.257 | 0.197 |
| mse-trained | 0.567 [0.549, 0.585] | 0.436 [0.427, 0.446] | 0.381 [0.364, 0.398] | 0.1904 | 0.193 | 0.247 | 0.189 |

Uniform forecast held-out test CE 0.6931. Train-marginal forecast held-out test CE 0.5819. CE head refit after shuffling train features: held-out test CE 0.5814.

### chaosnli_snli_mnli

Pooled ChaosNLI-SNLI and ChaosNLI-MNLI-matched. ChaosNLI SNLI or MNLI-matched, 3 classes, 100 votes per item in the author files. label_count order is entailment, neutral, contradiction. Ranked probability score uses that order as an imposed order (entailment < neutral < contradiction); it is not a native Likert scale. rater_id is a within-item vote index expanded from the counts, not a person. Each uid is its own split group. Item counts by file are kept distinct by the group prefix.

Split by prompt group, seed 0: train 1867, validation 622, test 624. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 1.053 [1.040, 1.068] | 0.627 [0.619, 0.634] | 0.637 [0.629, 0.647] | 0.1963 | 0.078 | 0.111 | 0.074 |
| brier-trained | 1.053 [1.040, 1.068] | 0.627 [0.619, 0.634] | 0.638 [0.629, 0.647] | 0.1964 | 0.081 | 0.114 | 0.076 |
| rps-trained | 1.054 [1.041, 1.068] | 0.627 [0.619, 0.634] | 0.638 [0.630, 0.647] | 0.1965 | 0.082 | 0.114 | 0.076 |
| pearson-trained | 4.022 [3.867, 4.176] | 0.711 [0.707, 0.715] | 0.940 [0.919, 0.962] | 0.2530 | 0.086 | 0.118 | 0.078 |
| mse-trained | inf [undefined, undefined] | 0.694 [0.682, 0.706] | 0.903 [0.872, 0.934] | 0.2409 | 0.063 | 0.079 | 0.052 |

Uniform forecast held-out test CE 1.0986. Train-marginal forecast held-out test CE 1.0569. CE head refit after shuffling train features: held-out test CE 1.0554.

### chaosnli_alpha

ChaosNLI abductive NLI, native two-option labels. The release stores counts, not person ids; rater_id is the within-item vote index obtained by expanding label_count. Order is option 1 then option 2. Each item is its own split group.

Split by prompt group, seed 0: train 919, validation 306, test 307. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.696 [0.692, 0.699] | 0.501 [0.500, 0.503] | 0.502 [0.499, 0.506] | 0.2512 | -0.057 | -0.064 | -0.045 |
| brier-trained | 0.696 [0.692, 0.699] | 0.501 [0.500, 0.503] | 0.502 [0.499, 0.506] | 0.2512 | -0.058 | -0.064 | -0.045 |
| rps-trained | 0.694 [0.692, 0.696] | 0.501 [0.500, 0.502] | 0.501 [0.499, 0.503] | 0.2505 | -0.049 | -0.057 | -0.040 |
| pearson-trained | 0.696 [0.689, 0.703] | 0.501 [0.498, 0.505] | 0.503 [0.496, 0.510] | 0.2514 | -0.034 | -0.049 | -0.036 |
| mse-trained | 0.708 [0.699, 0.718] | 0.507 [0.503, 0.512] | 0.515 [0.506, 0.524] | 0.2573 | -0.083 | -0.087 | -0.060 |

Uniform forecast held-out test CE 0.6931. Train-marginal forecast held-out test CE 0.6930. CE head refit after shuffling train features: held-out test CE 0.6916.

### dices990

DICES-990 rater-level Q_overall. Yes means unsafe (DICES-990 README; DICES-350 uses the same question family). Order for ranked probability score: No < Unsure < Yes, values 0,1,2. Dropped 7217 rows whose rater_id is in the README quality-filter list (12 raters). rater_id is the released person id. Split group is item_id. Raters are crossed, so the same person can appear in train and test on different conversations; no conversation's raters are split.

Split by prompt group, seed 0: train 594, validation 198, test 198. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.743 [0.708, 0.778] | 0.628 [0.608, 0.647] | 0.442 [0.415, 0.469] | 0.1969 | 0.127 | 0.280 | 0.189 |
| brier-trained | 0.744 [0.710, 0.778] | 0.628 [0.608, 0.647] | 0.443 [0.416, 0.470] | 0.1972 | 0.124 | 0.278 | 0.187 |
| rps-trained | 0.744 [0.710, 0.779] | 0.628 [0.608, 0.647] | 0.443 [0.416, 0.470] | 0.1972 | 0.124 | 0.276 | 0.186 |
| pearson-trained | 1.164 [1.118, 1.210] | 0.717 [0.709, 0.726] | 0.565 [0.552, 0.578] | 0.2572 | 0.110 | 0.255 | 0.172 |
| mse-trained | inf [undefined, undefined] | 0.720 [0.699, 0.739] | 0.825 [0.781, 0.866] | 0.2593 | 0.150 | 0.312 | 0.210 |

Uniform forecast held-out test CE 1.0986. Train-marginal forecast held-out test CE 0.7488. CE head refit after shuffling train features: held-out test CE 0.7492.

### dices990_binary_yes

Binary view of dices990: class 1 iff the original value ≥ 2. This is a threshold of the same ratings, not a separately collected binary task. DICES-990 rater-level Q_overall. Yes means unsafe (DICES-990 README; DICES-350 uses the same question family). Order for ranked probability score: No < Unsure < Yes, values 0,1,2. Dropped 7217 rows whose rater_id is in the README quality-filter list (12 raters). rater_id is the released person id. Split group is item_id. Raters are crossed, so the same person can appear in train and test on different conversations; no conversation's raters are split.

Split by prompt group, seed 0: train 594, validation 198, test 198. SVD components used: 24. Intervals are item-clustered percentile 95% intervals, B draws in `results.json`.

| Objective | CE | sqrt(RPS) | Brier | norm. RPS | Pearson r | Spearman rho | Kendall tau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce-trained | 0.557 [0.527, 0.586] | 0.431 [0.415, 0.446] | 0.371 [0.344, 0.397] | 0.1856 | 0.127 | 0.293 | 0.196 |
| brier-trained | 0.558 [0.528, 0.587] | 0.431 [0.415, 0.446] | 0.372 [0.345, 0.398] | 0.1858 | 0.125 | 0.291 | 0.196 |
| rps-trained | 0.560 [0.530, 0.589] | 0.432 [0.416, 0.447] | 0.373 [0.346, 0.400] | 0.1865 | 0.118 | 0.278 | 0.185 |
| pearson-trained | 0.710 [0.698, 0.722] | 0.508 [0.502, 0.514] | 0.516 [0.505, 0.529] | 0.2582 | 0.110 | 0.270 | 0.180 |
| mse-trained | 0.552 [0.521, 0.583] | 0.428 [0.412, 0.443] | 0.367 [0.340, 0.393] | 0.1835 | 0.149 | 0.323 | 0.216 |

Uniform forecast held-out test CE 0.6931. Train-marginal forecast held-out test CE 0.5627. CE head refit after shuffling train features: held-out test CE 0.5631.


### Crossover

### ChaosNLI alphaNLI binary

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 0.6966 | 0.5017 | 0.5035 | -0.066 |
| 2 | ce-trained | 0.6964 | 0.5016 | 0.5032 | -0.065 |
| 4 | ce-trained | 0.6960 | 0.5014 | 0.5028 | -0.056 |
| 8 | ce-trained | 0.6959 | 0.5014 | 0.5027 | -0.061 |
| 16 | ce-trained | 0.6958 | 0.5013 | 0.5026 | -0.067 |
| 32 | ce-trained | 0.6955 | 0.5012 | 0.5024 | -0.057 |
| 1 | pearson-trained | 0.6987 | 0.5027 | 0.5055 | -0.052 |
| 2 | pearson-trained | 0.6967 | 0.5018 | 0.5036 | -0.043 |
| 4 | pearson-trained | 0.6960 | 0.5014 | 0.5028 | -0.030 |
| 8 | pearson-trained | 0.6968 | 0.5018 | 0.5035 | -0.037 |
| 16 | pearson-trained | 0.6963 | 0.5016 | 0.5031 | -0.042 |
| 32 | pearson-trained | 0.6959 | 0.5013 | 0.5027 | -0.034 |

### SummEval crowd Likert 1-5

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 1.3276 | 0.7187 | 0.7077 | -0.051 |
| 2 | ce-trained | 1.3273 | 0.7186 | 0.7073 | -0.049 |
| 4 | ce-trained | 1.3272 | 0.7183 | 0.7072 | -0.047 |
| 1 | pearson-trained | 8.6508 | 1.0120 | 1.2395 | -0.057 |
| 2 | pearson-trained | 7.3791 | 1.0110 | 1.2388 | -0.057 |
| 4 | pearson-trained | 7.4974 | 1.0092 | 1.2378 | -0.052 |

### DICES-990 3-level ordinal

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 0.7467 | 0.6294 | 0.4445 | 0.087 |
| 2 | ce-trained | 0.7447 | 0.6283 | 0.4432 | 0.112 |
| 4 | ce-trained | 0.7438 | 0.6279 | 0.4427 | 0.128 |
| 8 | ce-trained | 0.7437 | 0.6279 | 0.4426 | 0.122 |
| 16 | ce-trained | 0.7433 | 0.6276 | 0.4422 | 0.129 |
| 32 | ce-trained | 0.7429 | 0.6274 | 0.4420 | 0.134 |
| 1 | pearson-trained | 1.0533 | 0.6883 | 0.5246 | 0.087 |
| 2 | pearson-trained | 1.1894 | 0.7208 | 0.5704 | 0.104 |
| 4 | pearson-trained | 1.1820 | 0.7208 | 0.5704 | 0.117 |
| 8 | pearson-trained | 1.1619 | 0.7174 | 0.5655 | 0.107 |
| 16 | pearson-trained | 1.1598 | 0.7158 | 0.5633 | 0.112 |
| 32 | pearson-trained | 1.1757 | 0.7164 | 0.5641 | 0.116 |

### ChaosNLI SNLI+MNLI 3-class

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 1.0533 | 0.6268 | 0.6373 | 0.061 |
| 2 | ce-trained | 1.0523 | 0.6265 | 0.6369 | 0.072 |
| 4 | ce-trained | 1.0523 | 0.6264 | 0.6370 | 0.081 |
| 8 | ce-trained | 1.0529 | 0.6266 | 0.6375 | 0.079 |
| 16 | ce-trained | 1.0530 | 0.6267 | 0.6375 | 0.074 |
| 32 | ce-trained | 1.0529 | 0.6266 | 0.6374 | 0.079 |
| 1 | pearson-trained | 4.9262 | 0.7103 | 0.9387 | 0.066 |
| 2 | pearson-trained | 3.9205 | 0.7105 | 0.9387 | 0.081 |
| 4 | pearson-trained | 4.3048 | 0.7112 | 0.9399 | 0.089 |
| 8 | pearson-trained | 3.7622 | 0.7112 | 0.9396 | 0.087 |
| 16 | pearson-trained | 3.9491 | 0.7105 | 0.9389 | 0.084 |
| 32 | pearson-trained | 4.3641 | 0.7107 | 0.9392 | 0.087 |


### Iso-budget

| N | k | B | objective | held-out test CE | sqrt(RPS) | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1 | 1024 | ce | 1.0543 | 0.6270 | 0.6385 |
| 1024 | 1 | 1024 | pearson | 4.5795 | 0.7175 | 0.9490 |
| 256 | 4 | 1024 | ce | 1.0560 | 0.6278 | 0.6400 |
| 256 | 4 | 1024 | pearson | 5.0037 | 0.7167 | 0.9478 |
| 64 | 16 | 1024 | ce | 1.0615 | 0.6290 | 0.6423 |
| 64 | 16 | 1024 | pearson | 6.0508 | 0.7092 | 0.9372 |
| 32 | 32 | 1024 | ce | 1.1013 | 0.6445 | 0.6660 |
| 32 | 32 | 1024 | pearson | 5.8740 | 0.7363 | 0.9764 |


### Single pass versus rollouts

- single_pass: held-out test CE 1.2097, Brier 0.6149, sqrt(RPS) 0.7488 (the softmax itself, no floor)
- M1: held-out test CE 9.6059, Brier 1.2516, sqrt(RPS) 1.0679 (mean over 8 floored histograms; one unlucky draw can still put a zero on a realized class)
- M5: held-out test CE 3.9603, Brier 0.7348, sqrt(RPS) 0.8152 (mean over 8 floored histograms; one unlucky draw can still put a zero on a realized class)
- M20: held-out test CE 1.7095, Brier 0.6409, sqrt(RPS) 0.7653 (mean over 8 floored histograms; one unlucky draw can still put a zero on a realized class)

### Within-prompt oracle

{'dataset': 'dices990', 'ok': True, 'labeled': 'within-prompt oracle, not a held-out judge', 'n_items': 198, 'n_pairs': 6534, 'corr_sampled_halfA': 0.19014077203550955, 'corr_collapsed_affine': 0.40998839462035036, 'sqrt_gap': True, 'mean_ce_distributional': 0.676435839928489, 'mean_ce_collapsed': 6.793578522305762, 'mean_brier_distributional': 0.38805111770986345, 'mean_brier_collapsed': 0.9834681239729421}

## Synthetic ground truth, labeled as such

The identities in the bottom line are not estimated from human ratings. The Monte Carlo anti-proper check draws from a known two-context law. It is the verification of Proof C, not a human result.

## Positioning against Kohli et al. 2026, arXiv:2605.29797

Kohli reports metric-dependent annotation saturation for soft-label NLI training: on ChaosNLI, KL is largely in place by about 10 annotators (paper table: 91±3% of the hard-to-soft-100 KL improvement) while entropy correlation keeps moving through roughly 20–50 annotators; the DICES-990 exploratory table is 90% versus 62% at 3 annotators for those two metrics. This experiment does not repeat that fine-tune. It compares objectives on a shared head and scores held-out proper risk. Experiment 28 remains the frontier-judge break-even (CE-fit ahead of Pearson-fit on held-out RPS after 4–8 ratings at one per item, or 10–20 at five per item, across its 25 dataset-by-judge cells). The shaded band on the new curves cites that result. The heads here are hashed-text linear heads, not those frontier judges.

## GPU scripts not run

`scripts/snap_lora_train.py` and `scripts/snap_llm_logit_readout.py` are ready for SNAP or Sherlock. This VM has no GPU (`nvidia-smi` absent). No LoRA number in this folder comes from a run.

## Why not psychometrics (G-theory / D-theory)?

**G-theory.** Generalizability theory (Cronbach, Gleser, Nanda, and Rajaratnam, 1972) writes a scalar rating as an additive variance split: prompt variance \(\sigma^2_c\), rater variance \(\sigma^2_r\), and the prompt-by-rater residual \(\sigma^2_{cr}\). The generalizability coefficient \(G\) is a multi-facet form of an intraclass correlation. Like Pearson correlation, it is a second-moment summary. The usual analysis of variance treats the components as additive and homoscedastic. On a bounded 0–5 or 1–5 rubric that summary does not score calibration, and it does not see a second mode. It is not a divergence from \(p^*(y \mid c)\).

**D-studies.** A decision study pushes those variance components forward and asks how many raters \(k\) would bring \(G\) to a threshold such as \(0.80\). Under a proper scoring rule the same allocation question is the held-out risk \(R(N, k)\) of a judge trained on \(N\) prompts with \(k\) ratings each. The curves in this folder answer it in held-out cross-entropy, in the square root of the ranked probability score (RPS), and in Brier score, and they convert \(B = N \times k\) into dollars at a stated price. They do not pass through a Gaussian analysis of variance.

**Where a generalizability coefficient still helps.** It is an optional diagnostic of the human panel. A large rater main effect is harshness or leniency. A large residual is disagreement about the item or the rubric. That diagnosis plans the annotation. It is not a training loss for the judge, and it is not the test metric.

**What the \((N, k)\) curves cost on the sets this folder actually scored.** Two unit prices, both labeled as assumptions. \(\$0.05\) per rating is the published POPQUORN crowd rate, used here as a unit price. It is not a wage we observed for SummEval, ChaosNLI, or DICES. \(\$5\) per rating is an assumed expert rate. POPQUORN ratings were not loaded. CertJudge / VeriBench functional semantic correctness ratings were not on this machine. Neither appears in the table.

Train sizes are the prompt-level split, seed 0: SummEval crowd \(N=3840\) items (same article split as the expert matrix), DICES-990 \(N=594\), ChaosNLI SNLI+MNLI \(N=1867\), ChaosNLI abductive NLI \(N=919\). The \(k\)-sweep trained a cross-entropy head and a Pearson head. On SummEval crowd (grades 1–5), DICES-990, and ChaosNLI SNLI+MNLI, the cross-entropy head has lower held-out cross-entropy, lower sqrt(RPS), and lower Brier at \(k=1\), the smallest budget we trained, and it stays lower at every larger \(k\) through 4 or through 32. There is no later \(k\) at which the Pearson head becomes the better proper-score trainer.

| Dataset | \(k\) at which CE is ahead on CE, sqrt(RPS), and Brier | Train ratings \(B=N k\) | Crowd dollars at \$0.05 | Expert dollars at \$5 | Held-out CE, CE-trained vs Pearson-trained |
| --- | --- | --- | --- | --- | --- |
| SummEval crowd, grades 1–5 | 1 (also 2 and 4) | 3,840 | \$192 | \$19,200 | 1.328 vs 8.651 |
| DICES-990, three-level | 1 (through 32) | 594 | \$29.70 | \$2,970 | 0.747 vs 1.053 |
| ChaosNLI SNLI+MNLI, three-class | 1 (through 32) | 1,867 | \$93.35 | \$9,335 | 1.053 vs 4.926 |
| ChaosNLI abductive NLI, binary | point estimates differ by about 0.002; both heads sit on \(\log 2 \approx 0.693\) | 919 | \$45.95 | \$4,595 | 0.697 vs 0.699 |
| POPQUORN | not scored |  |  |  |  |
| CertJudge / VeriBench FSC, 0–5 | not scored |  |  |  |  |

At \(k=1\) the other two proper scores move with cross-entropy. DICES-990 sqrt(RPS) is 0.629 versus 0.688 and Brier is 0.445 versus 0.525. SummEval crowd sqrt(RPS) is 0.719 versus 1.012 and Brier is 0.708 versus 1.240. ChaosNLI SNLI+MNLI sqrt(RPS) is 0.627 versus 0.710 and Brier is 0.637 versus 0.939. The abductive panel is the negative control: the hash does not predict the task, so that row is not a budget recommendation.

The iso-budget cut holds \(B=1024\) ratings on ChaosNLI SNLI+MNLI, which is \$51.20 at \$0.05 and \$5,120 at \$5. Held-out cross-entropy of the cross-entropy head is 1.054 at \((N,k)=(1024,1)\), 1.056 at \((256,4)\), 1.062 at \((64,16)\), and 1.101 at \((32,32)\). The Pearson head on the same four cells is 4.580, 5.004, 6.051, and 5.874. Inside this grid, spending the fixed bill on more prompts with one rating beats fewer prompts with many ratings.

The \(k\)-sweep did not train the mean-squared-error head. On the full training ratings, that head's two-point readout has infinite held-out cross-entropy on SummEval expert, on ChaosNLI SNLI+MNLI, and on DICES-990, because a rating off the two adjacent scores has probability 0. SummEval expert uses all 3 grades on each of 3,840 train items (11,520 grades). ChaosNLI SNLI+MNLI uses all 100 votes on each of 1,867 train items (186,700 votes). DICES-990 uses every filtered rating on each of 594 train conversations. The file has 60–69 ratings per conversation and 64,886 ratings in total. The train-slice total was not stored as its own field.

This driver finished in about 105 seconds on CPU and spent \$0 on model calls. Researcher salary was not metered, so this note does not invent one. A generalizability coefficient computed on the same ratings would still not be a held-out cross-entropy, Brier score, or RPS. That is the distinction the dollar rows are buying. On known synthetic ordinal laws, not on these human sets, experiment 34 already recorded a lenient point-mass judge with generalizability coefficient 1 and Pearson 0.961 whose expected RPS was worse than a constant forecast by 0.0492 [0.0477, 0.0506].



## Complete Sanmi Transcript Coverage

As of 11-26-2026. These five points are the ones the 11-25 call left open. The table is the taxonomy; the sentences under it are the boundaries.

| Point | Output space or claim | What is true | Where it is checked |
| --- | --- | --- | --- |
| 1a | Binary, \(K=2\) | One degree of freedom. The scalar mean is the full probability mass function, and scalar \(L_2\) equals the Brier score (two-class Brier is twice scalar squared error). Pearson correlation still fails calibration, because \(\operatorname{Corr}(aX+b,Y)=\operatorname{Corr}(X,Y)\) for \(a>0\), and it is strictly anti-proper for a sampled judge: \(\sqrt{\rho}>\rho\). | Lean `binary_brier_eq_two_l2`, `pearson_affine_invariant`, `binary_witness_corr_tie`, `anti_proper_from_parts`. Python `binary_l2`, `binary_affine_witness`, `anti_proper`. Real binary panels: SummEval cut at grade \(\ge 4\), ChaosNLI abductive NLI, DICES Yes-versus-not. |
| 1b | Finite categorical or ordinal, \(K\ge 3\) (a 0–5 rubric is this case) | \(K-1\) degrees of freedom. This is ordinary discrete classification. The multinomial frequency error is \(\mathcal{O}(K/N)\), from \(\mathbb{E}[\|\hat q_N-p^*\|_2^2]=(1/N)(1-\|p^*\|_2^2)\), not a nonparametric curse of dimensionality. A scalar mean drops spread and multimodality. Cross-entropy, Brier, and ranked probability score (RPS) see them. | Lean `empirical_l2_rate`, `plugin_DB_bias`, `ordinal_same_mean_different_law`. Python `empirical_rate`, `mean_ne_pmf`. Real panels: SummEval grades 1–5 and DICES-990 three-level `Q_overall`. VeriBench functional semantic correctness (0–5) was not on this machine. |
| 1c | Continuous, \(Y\in\mathbb{R}\) | The target is an infinite-dimensional density \(p^*(y\mid c)\). Sanmi's nonparametric sample-complexity warning applies here. A usable judge then needs a parametric head (Gaussian, Beta, or quantiles), the continuous ranked probability score (CRPS), or a discretization into \(K\) bins. | Conceded. This folder does not estimate a continuous density. |
| 2 | Paired versus unpaired; agreement versus fidelity | Correlation has one narrow use as an unpaired or arbitrary-scale diagnostic. Inter-rater agreement, for example Krippendorff's \(\alpha\), measures human-to-human consensus when no target distribution has been defined. A proper scoring rule measures judge fidelity to a defined target \(p^*(y\mid c)\). Those are different questions. | The experiments in this folder score fidelity on held-out prompts. They do not report Krippendorff's \(\alpha\). |
| 3 | Sanmi's RPS framing | RPS is the squared distance between forecast and realized cumulative probabilities. It charges an ordering of how wrong the forecast is — distance from the realized rating — over the whole distribution. It is not a ranking of items. | Lean `bernoulli_square_excess` (one threshold) and `ordinal_same_mean_different_law` (the 0–5 gap). The reported column is the square root of the unbiased mean per-sample RPS; normalized RPS (divided by \(K-1\)) is stored beside it. |
| 4 | Surrogate-loss literature | Proper scoring rules and surrogate losses are classic: Brier (1950), Epstein (1969) for the ranked probability score, Gneiting and Raftery (2007). What is new here is the application to an LLM-as-a-judge, the strict anti-properness \(\sqrt{\rho}>\rho\) for independent sampled judges, the single-pass token-logit readout, and the rating-budget and dollar scaling curves. | This note, `SanmiPitch.lean`, and the figures under `figures/`. |
| 5 | Sharpen the rubric versus match \(p^*\) | Tightening a rubric so that \(p^*(y\mid c)\) becomes peakier, and training a judge to match \(p^*(y\mid c)\), are complementary. What fails is pretending a non-peaky target has zero conditional variance. | Proof B (consensus on 2 versus a 50/50 split on \(\{0,4\}\), same mean 2) and Proof C (the collapsed judge \(X_\delta=a\mu_Y(C)+b\)). |


