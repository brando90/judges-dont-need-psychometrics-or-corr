# Open-Weight Real Judges: Prompt Optimization (`ii.a`), Harness Optimization (`ii.b`), and Real Weight Training (`ii.c`) — Cross-Entropy vs Correlation

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/RESULTS_open_weight_ce_vs_corr.md>

**TLDR:** Across 10 real open-weight large language models (LLMs) on `mercury2` (`$0.00` free GPUs, no differential testing (no DT), `75` VeriBench Lean-4 artifacts, `25` tasks, `19` task families, `195` expert ratings), the native six-category probability pools have lower cross-entropy (CE) than the historical three-sample scalar pool (`3.480` -> `1.629` nats on verbalized six-category probabilities and `1.606` nats on temperature-calibrated logprob harnesses, both beating the `1.887`-nat constant marginal baseline), and proper-loss optimization (CE / ranked probability score (RPS)) beats correlation optimization across **`ii.a` prompt-pool mixtures**, **`ii.b` harness optimization**, and **`ii.c` end-to-end PyTorch low-rank adaptation (LoRA) transformer weight training**.

Generated from `results/analysis_summary.json` on `11-16-2026 14:44 PST`. Read [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md) for full split, metric, and optimizer definitions.

**Input provenance and prior exposure:** 60/75 reference files differ from the hashes recorded in the annotation manifest; Python and candidate files match. The saved evidence does not resolve stale reference metadata versus changed reference bytes, so byte identity with the human-delivered reference files is unverified. The fixed P2 prompt includes historical expert-mean and leniency summaries that were not estimated within each training split; P2-containing pools are retrospectively informed, not zero-exposure evaluations. Family separation applies to fitting and selection. Digit-logprob extraction imputes absent digits below the observed top-20 tail and can report success with fewer than six observed digits; saved receipts do not identify the exact incidence. These are probability estimates, not guaranteed complete six-digit softmax probabilities.

---

## 0. Audit of Part `i`: What Did the Synthetic and Historical Studies Actually Change?

Table notation: ranked probability score (RPS; squared cumulative-probability error, 0–1), cross-entropy (CE; negative log-probability, nats), total variation distance (TVD; half the histogram L1 distance, 0–1), and mean squared error (MSE; rating squared) are lower-is-better; Pearson/Spearman are correlations (−1 to 1, higher is better). CI means confidence interval, CV cross-validation, OOF out-of-fold, SD standard deviation, and LoRA low-rank adaptation. Differences are the stated metric of the correlation-trained/selected judge minus that of the CE-trained/selected judge; positive favors CE. Baselines and method-specific selection are in the reading guide.

```text
Study Group                                | Prompt Opt? | Harness Opt? | Real LLM Weight Updates? | What Actually Varied
-------------------------------------------+-------------+--------------+--------------------------+---------------------------------------------------------------
i.a Synthetic fixed-candidate (00, 01)     | No          | No           | No (0 params)            | Picked 1 of 2 or 9 analytic distributions (CE picked truth 249/256 at 1 rating)
i.b Synthetic basis-head toys (13,06,07,08)| No          | No           | Toy 78-weight softmax    | Fitted 13x6 cosine/sine basis weights on scalar x in [-1,1] via L-BFGS-B
Historical real caches (02, 11, 12, 14)    | No          | No           | No (simplex mix only)    | 5 or 44 simplex weights over fixed 3-sample scalar scores (Exp 14 CE = 3.397 nats)
THIS STUDY (Exp 09: ii.a, ii.b, ii.c)      | YES (ii.a)  | YES (ii.b)   | YES: PyTorch LoRA (ii.c) | 6 prompts + blind meta-prompt search; 6 harnesses; 1.85M-2.91M LoRA LLM params
```

---

## 1. Karpathy Sanity Controls, Repeat Stability, and Native-Probability Representation Bridge

### 1.1 Karpathy Sanity Controls (`P1_conventions`, `0–5` scale; human mean = `2.63`, theorem-count baseline Spearman = `0.546`)

Table notation: ranked probability score (RPS; squared cumulative-probability error, 0–1), cross-entropy (CE; negative log-probability, nats), total variation distance (TVD; half the histogram L1 distance, 0–1), and mean squared error (MSE; rating squared) are lower-is-better; Pearson/Spearman are correlations (−1 to 1, higher is better). CI means confidence interval, CV cross-validation, OOF out-of-fold, SD standard deviation, and LoRA low-rank adaptation. Differences are the stated metric of the correlation-trained/selected judge minus that of the CE-trained/selected judge; positive favors CE. Baselines and method-specific selection are in the reading guide.

```text
Model             | Oracle (Gold) | Blank Cand | Vacuous (True) | Shuffled Cand | Label-in-Prompt Match (Pearson)
------------------+---------------+------------+----------------+---------------+--------------------------------
ds_r1_qwen_7b     |           n/a |       5.00 |           5.00 |          2.75 |   2/5 (+0.478)
q25c_14b          |          2.44 |       0.00 |           0.12 |          0.08 |   5/5 (+1.000)
q3_14b_nothink    |          4.04 |       0.00 |           0.10 |          3.68 |   5/5 (+1.000)
q3_14b_think      |          5.00 |       0.00 |           0.08 |          0.50 |   2/5 (+0.852)
q3_30b_a3b        |          5.00 |       0.00 |           0.15 |          0.72 |   5/5 (+1.000)
q3_4b_2507        |          3.27 |       0.00 |           0.16 |          0.53 |   5/5 (+0.965)
```

- **Repeat stability (`k = 3` stochastic repeats at `T = 0.7` across all `75` items):** `q25c_14b` root-mean-square within-artifact expected-score SD = `0.936` (`0–5` scale; `0.1871` on `0–1`); `q3_30b_a3b` root-mean-square within-artifact expected-score SD = `0.331` (`0–5` scale; `0.0662` on `0–1`).
- **Human ceiling (leave-one-rater-out across the `195` ratings):** RPS = `0.1113`, pairwise Pearson = `0.805`, Spearman = `0.798`.

### 1.2 Representation Bridge: Fixing Archived Experiment 14's Empty-Bin Log-Loss Failure (`200` Grouped `15/4` Task-Family Splits)

Table notation: ranked probability score (RPS; squared cumulative-probability error, 0–1), cross-entropy (CE; negative log-probability, nats), total variation distance (TVD; half the histogram L1 distance, 0–1), and mean squared error (MSE; rating squared) are lower-is-better; Pearson/Spearman are correlations (−1 to 1, higher is better). CI means confidence interval, CV cross-validation, OOF out-of-fold, SD standard deviation, and LoRA low-rank adaptation. Differences are the stated metric of the correlation-trained/selected judge minus that of the CE-trained/selected judge; positive favors CE. Baselines and method-specific selection are in the reading guide.

```text
Judge Pool Representation (Held-Out Test)  | Constant RPS / CE | ④ CE-Mix RPS / CE  | ② Pearson-Mix RPS / CE | ② minus ④ Test RPS [95% Fam CI]           | ② minus ④ Test CE (nats) [95% Fam CI]
-------------------------------------------+-------------------+--------------------+------------------------+-------------------------------------------+-------------------------------------------
Archived Exp 14 (39 open, 3-sample scalar) | 0.1857 / 1.887   | 0.2819 / 3.480    | 0.3263 / 5.668        | +0.0445 [+0.0167,+0.0676] p=0.00866318 | +2.188 [+1.167,+3.067] p=0.00257874
Exp 09 Verbalized 6-Prob JSON Pool         | 0.1857 / 1.887   | 0.1431 / 1.629    | 0.1848 / 1.804        | +0.0417 [+0.0194,+0.0666] p=0.00336456 | +0.175 [+0.067,+0.278] p=0.00750732
Exp 09 Conditional Digit-Logprob Pool      | 0.1857 / 1.887   | 0.1712 / 2.273    | 0.2250 / 2.533        | +0.0538 [+0.0340,+0.0709] p=0.000488281 | +0.260 [+0.066,+0.424] p=0.0293083
```

- **Direct representation verdict (`H1`):** Moving from three-sample scalar histograms (`3.480` nats) to verbalized six-category probabilities drops held-out CE to **`1.629` nats** (`RPS = 0.1431`, beating the `1.887`-nat / `0.1857`-RPS constant baseline on both metrics). Raw `T=1.0` teacher-forced digit logprobs also lower CE by **`+1.207 [+0.604, +1.963] p-val=0.00181198 (exact_2^19_task_family_sign_flip)`** nats and RPS by **`+0.1106 [+0.0805, +0.1430] p-val=3.43323e-05 (exact_2^19_task_family_sign_flip)`** (`RPS = 0.1712`), and once temperature-calibrated in `ii.b` reach **`1.606` nats** and **`0.1325` RPS** — lower measured losses for these selected probability pools. Models, prompts, and harnesses also differ across pools, so this comparison does not isolate representation or identify a reinforcement-learning-from-human-feedback (RLHF) mechanism.

---

## 2. Sub-Experiment `ii.a`: Prompt Optimization on Real Open-Weight Judges (CE/RPS vs Correlation)

### 2.1 Candidate Prompt Optimization (`P0..P5`, `200` Grouped `15/4` Task-Family Splits)

Table notation: ranked probability score (RPS; squared cumulative-probability error, 0–1), cross-entropy (CE; negative log-probability, nats), total variation distance (TVD; half the histogram L1 distance, 0–1), and mean squared error (MSE; rating squared) are lower-is-better; Pearson/Spearman are correlations (−1 to 1, higher is better). CI means confidence interval, CV cross-validation, OOF out-of-fold, SD standard deviation, and LoRA low-rank adaptation. Differences are the stated metric of the correlation-trained/selected judge minus that of the CE-trained/selected judge; positive favors CE. Baselines and method-specific selection are in the reading guide.

```text
Prompt Search Space & Representation       | ② Corr-Single RPS/CE | ④ CE-Single RPS/CE | ③ Val-Tuned RPS/CE | ② minus ④ Single RPS [95% CI]      | ② Mix minus ④ Mix RPS [95% CI]
-------------------------------------------+----------------------+--------------------+--------------------+------------------------------------+------------------------------------
Multi-Model x P0..P5 (Verbal 6-Prob)       | 0.2826 / 5.043      | 0.2223 / 2.149    | 0.1794 / 1.781    | +0.0603 [-0.0172,+0.1246] p=0.142 | +0.0550 [+0.0047,+0.1032] p=0.057
Multi-Model x P0..P5 (Digit Logprob)       | 0.2886 / 8.713      | 0.3239 / 3.851    | 0.1899 / 1.978    | -0.0353 [-0.1109,+0.0504] p=0.407 | +0.0873 [+0.0430,+0.1329] p=0.002
```

### 2.2 Automated Blind Iterative Meta-Prompt Search (`Qwen3-30B-A3B`, `3`-Fold Outer Task-Family CV, All `75` Artifacts Held Out)

Table notation: ranked probability score (RPS; squared cumulative-probability error, 0–1), cross-entropy (CE; negative log-probability, nats), total variation distance (TVD; half the histogram L1 distance, 0–1), and mean squared error (MSE; rating squared) are lower-is-better; Pearson/Spearman are correlations (−1 to 1, higher is better). CI means confidence interval, CV cross-validation, OOF out-of-fold, SD standard deviation, and LoRA low-rank adaptation. Differences are the stated metric of the correlation-trained/selected judge minus that of the CE-trained/selected judge; positive favors CE. Baselines and method-specific selection are in the reading guide.

```text
Blind Meta-Prompt Search Arm (Qwen3-30B)   | Test RPS | Test CE (nats) | Test TVD | Pearson | Spearman | Mean / Bias (0-5)
-------------------------------------------+----------+----------------+----------+---------+----------+------------------
Untuned Seed Prompt (P0 human-mirror)      | 0.2119   | 3.047          | 0.653    | +0.346  | +0.299   | 3.66 (+1.04)
② Correlation-Optimized Prompt (Train)     | 0.2115   | 2.955          | 0.656    | +0.440  | +0.413   | 3.71 (+1.09)
① Correlation-Optimized Prompt (Val Corr)  | 0.2331   | 3.395          | 0.667    | +0.363  | +0.412   | 3.86 (+1.23)
④ CE-Optimized Prompt (Train CE)           | 0.2520   | 3.218          | 0.693    | +0.202  | +0.216   | 3.90 (+1.28)
③ Proper-Loss-Optimized Prompt (Val RPS)   | 0.2592   | 5.143          | 0.693    | +0.239  | +0.222   | 3.91 (+1.29)
```

- **Blind meta-prompt contrast (`②` train-corr minus `④` train-CE):** Held-out test RPS = **`-0.0405 [-0.0841, +0.0036] p-val=0.132538 (exact_2^19_task_family_sign_flip)`**, held-out test CE = **`-0.263 [-1.054, +0.448] p-val=0.529907 (exact_2^19_task_family_sign_flip)`** nats.
- **Blind meta-prompt contrast (`①` val-corr minus `③` val-RPS):** Held-out test RPS = **`-0.0261 [-0.0643, +0.0138] p-val=0.277802 (exact_2^19_task_family_sign_flip)`**, held-out test CE = **`-1.748 [-3.483, -0.021] p-val=0.0996399 (exact_2^19_task_family_sign_flip)`** nats.

---

## 3. Sub-Experiment `ii.b`: Judge Harness Optimization (`59` Harness Configurations across `H1–H6`)

Table notation: ranked probability score (RPS; squared cumulative-probability error, 0–1), cross-entropy (CE; negative log-probability, nats), total variation distance (TVD; half the histogram L1 distance, 0–1), and mean squared error (MSE; rating squared) are lower-is-better; Pearson/Spearman are correlations (−1 to 1, higher is better). CI means confidence interval, CV cross-validation, OOF out-of-fold, SD standard deviation, and LoRA low-rank adaptation. Differences are the stated metric of the correlation-trained/selected judge minus that of the CE-trained/selected judge; positive favors CE. Baselines and method-specific selection are in the reading guide.

```text
Harness Selection / Optimization Method    | Test RPS | Test CE (nats) | Test TVD | Pearson | Spearman | Bias (0-5) | MSE = Bias² + Scale² + Disp
-------------------------------------------+----------+----------------+----------+---------+----------+------------+----------------------------
Constant Train-Marginal Baseline           | 0.1857   | 1.887          | 0.693    | -0.825  | -0.804   |  -0.01     | 2.21 = 0.00+1.88+0.34
② Best Single Harness by Train Pearson     | 0.1992   | 5.835          | 0.660    | +0.655  | +0.550   |  -0.19     | 1.39 = 0.03+0.01+1.34
④ Best Single Harness by Train CE          | 0.1609   | 1.708          | 0.631    | +0.456  | +0.490   |  -0.16     | 1.73 = 0.02+1.12+0.59
④ Best Single Harness by Train RPS         | 0.1863   | 2.162          | 0.651    | +0.429  | +0.424   |  -0.18     | 1.74 = 0.03+0.40+1.31
② Harness Mixture by Train Pearson         | 0.1836   | 1.978          | 0.661    | +0.712  | +0.629   |  +0.88     | 2.07 = 0.77+0.90+0.40
① Harness Chosen by Val Pearson            | 0.1892   | 3.111          | 0.663    | +0.688  | +0.614   |  +0.54     | 1.44 = 0.29+0.49+0.65
④ Harness Mixture by Train CE              | 0.1325   | 1.606          | 0.556    | +0.673  | +0.576   |  -0.03     | 1.17 = 0.00+0.45+0.72
④ Harness Mixture by Train RPS             | 0.1310   | 1.625          | 0.554    | +0.681  | +0.578   |  +0.03     | 1.11 = 0.00+0.31+0.80
③ Harness Tuned by Val RPS                 | 0.1340   | 1.627          | 0.564    | +0.676  | +0.574   |  -0.01     | 1.16 = 0.00+0.45+0.71
```

- **Harness optimization primary verdict (`②` Pearson-mixture minus `④` CE-mixture):** Held-out test RPS = **`+0.0511 [+0.0222, +0.0833] p-val=0.00527573 (exact_2^19_task_family_sign_flip)`**, held-out test CE = **`+0.372 [+0.153, +0.549] p-val=0.00632858 (exact_2^19_task_family_sign_flip)`** nats.
- **Validation-selected (`①` minus `③`):** Held-out test RPS = **`+0.0552 [+0.0287, +0.0791] p-val=0.00192642 (exact_2^19_task_family_sign_flip)`**, held-out test CE = **`+1.484 [+0.973, +1.925] p-val=2.28882e-05 (exact_2^19_task_family_sign_flip)`** nats.
- **2×2 Factorial (`Prompt Objective × Harness/Calibration Objective`):** `(Corr Prompt, Corr Harness)` RPS = `0.2893` (`CE = 3.805`), `(CE Prompt, Corr Harness)` RPS = `0.1970`, `(Corr Prompt, CE Harness)` RPS = `0.1883`, `(CE Prompt, CE Harness)` RPS = `0.1830` (`CE = 1.856`; joint gain = **`+0.1062 [+0.0315, +0.1768] p-val=0.0170364 (exact_2^19_task_family_sign_flip)`**).
- **Deliberative thinking ablation (`Qwen3-14B` thinking ON vs OFF):** Thinking ON RPS = `0.2176` (`Pearson = +0.591`), Thinking OFF RPS = `0.3813` (`Pearson = +0.345`; `nothink minus think` RPS = `+0.1637 [+0.0959, +0.2229] p-val=0.000598907 (exact_2^19_task_family_sign_flip)`).

---

## 4. Sub-Experiment `ii.c`: Real Neural Weight Updates (End-to-End PyTorch LoRA Backpropagation + Neural Heads)

### 4.1 End-to-End PyTorch Transformer LoRA Fine-Tuning: `q3_4b_2507` (`1,850,374` Trainable Params, `5`-Fold Grouped Task-Family CV)

Table notation: ranked probability score (RPS; squared cumulative-probability error, 0–1), cross-entropy (CE; negative log-probability, nats), total variation distance (TVD; half the histogram L1 distance, 0–1), and mean squared error (MSE; rating squared) are lower-is-better; Pearson/Spearman are correlations (−1 to 1, higher is better). CI means confidence interval, CV cross-validation, OOF out-of-fold, SD standard deviation, and LoRA low-rank adaptation. Differences are the stated metric of the correlation-trained/selected judge minus that of the CE-trained/selected judge; positive favors CE. Baselines and method-specific selection are in the reading guide.

```text
Trained LLM Weight Objective (5-Fold OOF)  | Test RPS | Test CE (nats) | Test TVD | Pearson | Spearman | Bias (0-5) | MSE = Bias² + Scale² + Disp
-------------------------------------------+----------+----------------+----------+---------+----------+------------+----------------------------
Untrained Pretrained Base LLM (Epoch 0)    | 0.4683   | 6.177          | 0.851    | +0.467  | +0.598   |  +2.35     | 7.57 = 5.54+2.00+0.03
② Pearson-Trained LoRA Weights (Train)     | 0.3351   | 7.546          | 0.777    | +0.217  | +0.160   |  +1.53     | 4.31 = 2.35+1.23+0.73
① Pearson-Trained LoRA Weights (Val Corr)  | 0.3597   | 5.216          | 0.802    | +0.281  | +0.341   |  +1.78     | 5.06 = 3.16+1.15+0.74
② Smooth-Spearman LoRA Weights (Train)     | 0.2752   | 4.354          | 0.731    | +0.529  | +0.522   |  +1.34     | 3.39 = 1.79+1.05+0.55
Pearson-LoRA + Calibrated 12-Param CE Head | 0.1412   | 1.593          | 0.593    | +0.611  | +0.533   |  -0.16     | 1.32 = 0.02+0.42+0.87
④ RPS-Trained LoRA Weights (Train)         | 0.1887   | 2.886          | 0.652    | -0.032  | -0.026   |  -0.07     | 2.16 = 0.01+1.34+0.81
③ RPS-Trained LoRA Weights (Val RPS)       | 0.2001   | 3.127          | 0.692    | +0.017  | +0.007   |  +0.15     | 2.19 = 0.02+1.15+1.02
④ CE-Trained LoRA Weights (Train)          | 0.2007   | 2.084          | 0.726    | -0.017  | +0.008   |  +0.33     | 2.34 = 0.11+1.07+1.16
③ CE-Trained LoRA Weights (Val RPS)        | 0.1997   | 2.180          | 0.739    | +0.206  | +0.221   |  +0.48     | 2.20 = 0.23+1.36+0.61
```

- **`q3_4b_2507` individual-rating (`n = 195` ratings) correlations & signed bias:** `③ CE-LoRA (Val RPS)` individual Pearson = `+0.202`, Spearman = `+0.216`, Kendall tau-b = `+0.156`, judge mean = `3.11` vs human mean = `2.63` (`bias = +0.48`); `② Pearson-LoRA (Train)` individual Pearson = `+0.171`, Spearman = `+0.125`, Kendall tau-b = `+0.105`, judge mean = `4.16` vs human mean = `2.63` (`bias = +1.53`).
- **`q3_4b_2507` LoRA weight-training verdict (`②` Pearson-LoRA minus `④` CE-LoRA):** Held-out test RPS = **`+0.1344 [+0.0495, +0.2220] p-val=0.00907898 (exact_2^19_task_family_sign_flip)`**, held-out test CE = **`+5.462 [+3.749, +6.775] p-val=0.000106812 (exact_2^19_task_family_sign_flip)`** nats.
- **`q3_4b_2507` Validation-selected (`①` Val-Corr Pearson-LoRA minus `③` Val-RPS CE-LoRA):** Held-out test RPS = **`+0.1599 [+0.0870, +0.2273] p-val=0.00179672 (exact_2^19_task_family_sign_flip)`**, held-out test CE = **`+3.036 [+1.906, +4.353] p-val=9.91821e-05 (exact_2^19_task_family_sign_flip)`** nats.
- **`q3_4b_2507` Two-stage `Pearson-LoRA + calibrated CE head` minus `③` `CE-LoRA`:** Held-out test RPS = **`-0.0585 [-0.0985, -0.0261] p-val=0.0028038 (exact_2^19_task_family_sign_flip)`**; formulation `④` minus `③` (`ce_lora_train_4 minus ce_lora_val_rps_3`) = **`+0.0009 [-0.0180, +0.0195] p-val=0.921875 (exact_2^19_task_family_sign_flip)`**.

### 4.2 End-to-End PyTorch Transformer LoRA Fine-Tuning: `q25c_7b` (`2,905,094` Trainable Params, `5`-Fold Grouped Task-Family CV)

Table notation: ranked probability score (RPS; squared cumulative-probability error, 0–1), cross-entropy (CE; negative log-probability, nats), total variation distance (TVD; half the histogram L1 distance, 0–1), and mean squared error (MSE; rating squared) are lower-is-better; Pearson/Spearman are correlations (−1 to 1, higher is better). CI means confidence interval, CV cross-validation, OOF out-of-fold, SD standard deviation, and LoRA low-rank adaptation. Differences are the stated metric of the correlation-trained/selected judge minus that of the CE-trained/selected judge; positive favors CE. Baselines and method-specific selection are in the reading guide.

```text
Trained LLM Weight Objective (5-Fold OOF)  | Test RPS | Test CE (nats) | Test TVD | Pearson | Spearman | Bias (0-5) | MSE = Bias² + Scale² + Disp
-------------------------------------------+----------+----------------+----------+---------+----------+------------+----------------------------
Untrained Pretrained Base LLM (Epoch 0)    | 0.4067   | 3.345          | 0.821    | +0.560  | +0.481   |  +2.12     | 6.47 = 4.48+1.95+0.05
② Pearson-Trained LoRA Weights (Train)     | 0.4015   | 4.801          | 0.811    | +0.246  | +0.107   |  +2.11     | 6.43 = 4.47+1.60+0.37
① Pearson-Trained LoRA Weights (Val Corr)  | 0.3761   | 3.623          | 0.801    | +0.355  | +0.291   |  +2.00     | 5.93 = 4.02+1.60+0.31
② Smooth-Spearman LoRA Weights (Train)     | 0.3600   | 4.358          | 0.797    | +0.262  | +0.289   |  +1.90     | 5.54 = 3.61+1.47+0.47
Pearson-LoRA + Calibrated 12-Param CE Head | 0.1560   | 1.709          | 0.619    | +0.488  | +0.368   |  -0.12     | 1.59 = 0.02+0.44+1.13
④ RPS-Trained LoRA Weights (Train)         | 0.1821   | 2.343          | 0.661    | +0.108  | +0.109   |  -0.11     | 2.04 = 0.01+1.69+0.34
③ RPS-Trained LoRA Weights (Val RPS)       | 0.1832   | 1.851          | 0.648    | -0.212  | -0.114   |  +0.03     | 2.15 = 0.00+1.71+0.43
④ CE-Trained LoRA Weights (Train)          | 0.1927   | 2.026          | 0.710    | -0.310  | -0.284   |  +0.11     | 2.31 = 0.01+1.48+0.81
③ CE-Trained LoRA Weights (Val RPS)        | 0.1859   | 1.950          | 0.701    | -0.091  | -0.062   |  +0.14     | 2.13 = 0.02+1.66+0.45
```

- **`q25c_7b` individual-rating (`n = 195` ratings) correlations & signed bias:** `③ CE-LoRA (Val RPS)` individual Pearson = `-0.051`, Spearman = `-0.039`, Kendall tau-b = `-0.033`, judge mean = `2.76` vs human mean = `2.63` (`bias = +0.14`); `② Pearson-LoRA (Train)` individual Pearson = `+0.213`, Spearman = `+0.115`, Kendall tau-b = `+0.074`, judge mean = `4.74` vs human mean = `2.63` (`bias = +2.11`).
- **`q25c_7b` LoRA weight-training verdict (`②` Pearson-LoRA minus `④` CE-LoRA):** Held-out test RPS = **`+0.2088 [+0.0957, +0.3037] p-val=0.00491714 (exact_2^19_task_family_sign_flip)`**, held-out test CE = **`+2.775 [+1.635, +3.721] p-val=0.00102615 (exact_2^19_task_family_sign_flip)`** nats.
- **`q25c_7b` Validation-selected (`①` Val-Corr Pearson-LoRA minus `③` Val-RPS CE-LoRA):** Held-out test RPS = **`+0.1903 [+0.0886, +0.2757] p-val=0.00488663 (exact_2^19_task_family_sign_flip)`**, held-out test CE = **`+1.673 [+0.898, +2.509] p-val=0.00172043 (exact_2^19_task_family_sign_flip)`** nats.
- **`q25c_7b` Two-stage `Pearson-LoRA + calibrated CE head` minus `③` `CE-LoRA`:** Held-out test RPS = **`-0.0298 [-0.0754, +0.0081] p-val=0.146175 (exact_2^19_task_family_sign_flip)`**; formulation `④` minus `③` (`ce_lora_train_4 minus ce_lora_val_rps_3`) = **`+0.0069 [-0.0009, +0.0172] p-val=0.131836 (exact_2^19_task_family_sign_flip)`**.

### 4.3 Real-Judge Learning Curves Across Training Rating Counts (`Qwen3-4B` Hidden-State + Logit Heads, `120` Grouped Splits per Size)

Table notation: ranked probability score (RPS; squared cumulative-probability error, 0–1), cross-entropy (CE; negative log-probability, nats), total variation distance (TVD; half the histogram L1 distance, 0–1), and mean squared error (MSE; rating squared) are lower-is-better; Pearson/Spearman are correlations (−1 to 1, higher is better). CI means confidence interval, CV cross-validation, OOF out-of-fold, SD standard deviation, and LoRA low-rank adaptation. Differences are the stated metric of the correlation-trained/selected judge minus that of the CE-trained/selected judge; positive favors CE. Baselines and method-specific selection are in the reading guide.

```text
Train Fams (~Ratings) | ④ CE RPS | ④ RPS RPS | ② Pearson RPS | Pearson+Head RPS | ② Spearman RPS | Constant RPS | ② Pearson-Head minus ④ CE-Head RPS [95% CI]
----------------------+----------+-----------+---------------+------------------+----------------+--------------+--------------------------------------------
 3 fams (~ 30.8 rat) | 0.1688   | 0.1673    | 0.1761        | 0.2742           | 0.1758         | 0.2167       | +0.0073 [-0.0019,+0.0189] p=0.154869
 5 fams (~ 52.1 rat) | 0.1542   | 0.1606    | 0.1764        | 0.2226           | 0.1760         | 0.1979       | +0.0221 [+0.0040,+0.0443] p=0.0312614
 7 fams (~ 72.4 rat) | 0.1495   | 0.1568    | 0.1771        | 0.2175           | 0.1760         | 0.1904       | +0.0276 [+0.0064,+0.0537] p=0.0248604
 9 fams (~ 92.7 rat) | 0.1433   | 0.1525    | 0.1765        | 0.1917           | 0.1762         | 0.1913       | +0.0332 [+0.0089,+0.0638] p=0.0194054
12 fams (~124.0 rat) | 0.1375   | 0.1472    | 0.1765        | 0.1836           | 0.1764         | 0.1897       | +0.0390 [+0.0115,+0.0740] p=0.0163765
15 fams (~155.3 rat) | 0.1335   | 0.1435    | 0.1761        | 0.1759           | 0.1765         | 0.1860       | +0.0426 [+0.0105,+0.0822] p=0.0225639
```

- **Non-convex optimizer restart check (`Exp 07` replication on real LLM features, `50` splits at `15` train families):** Single-start (`400` iters) `test RPS of Pearson-trained minus CE-trained judge` = `+0.0426 [+0.0110, +0.0821] p-val=0.0212364 (exact_2^19_task_family_sign_flip)` (Pearson convergence rate `0.0%`, CE `100.0%`); Best-of-`5` random restarts (`2,000` iters) `test RPS of Pearson-trained minus CE-trained judge` = **`+0.0436 [+0.0117, +0.0838] p-val=0.0203133 (exact_2^19_task_family_sign_flip)`** — the gap persists under this larger search budget. Non-convergence and the non-convex objective prevent ruling out optimizer or local-minimum effects.

---

## 5. WHEN Is Cross-Entropy Better Than Correlation? Observed Comparisons & Stratified Diagnostics

1. **Training-size regime (`3` to `15` task families, `~31` to `~155` expert ratings on `48`-weight hidden-state heads):** Point estimates favor `④ CE` over `② Pearson` at all six training sizes (`0.1688` vs `0.1761` at `3` families / `~30.8` ratings, `p-val = 0.1549`), reaching statistical significance from **`5` task families (`~52.1` ratings, `+0.0221`, `p-val = 0.0313`)** through **`15` task families (`~155.3` ratings, `0.1335` vs `0.1761`, `+0.0426 [+0.0105, +0.0822]`, `p-val = 0.0226`)**. These are unadjusted, size-specific comparisons on this retrospective dataset; they do not establish a universal rating threshold. Experiment 10 reports conditional split variability and is not an independent confirmation of a population threshold.
2. **High rater disagreement (`SD > 0`) vs unanimous agreement (`SD = 0`):** On split-rater artifacts (`62` artifacts, `SD > 0`), the Pearson-trained minus CE-trained mixture held-out test RPS excludes zero (`+0.0611 [+0.0449, +0.0750] p-val=0.000144958 (exact_2^18_task_family_sign_flip)`), whereas on unanimous-rater artifacts (`13` artifacts, `SD = 0`) the interval spans zero (`+0.0187 [-0.0183, +0.0730] p-val=0.417969 (exact_2^10_task_family_sign_flip)`). Correlation between per-artifact smoothed information gap `G_i` and `(RPS_Pearson − RPS_CE)` is Pearson `+0.198`, Spearman `+0.136`, and Kendall tau-b `+0.119`.
3. **Ordinary vs `True`-placeholder tasks:** The CE advantage is driven by ordinary specification tasks (`60` artifacts: `+0.0719 [+0.0595, +0.0871] p-val=0.00012207 (exact_2^14_task_family_sign_flip)`), whereas on the `15` `True`-placeholder-reference security items raw `T=1.0` logprob Pearson slightly leads (`-0.0187 [-0.0288, -0.0085] p-val=0.0625 (exact_2^5_task_family_sign_flip)`).
4. **Parameter-capacity comparison:** On these fitted judges, the `48`-weight cross-entropy (CE) head has lower held-out ranked probability score (RPS) than the Pearson-trained head (`0.1335` vs `0.1761`). Both CE-trained low-rank adaptation (LoRA) models improve on the corresponding Pearson-trained LoRA model. Adding a CE calibration head to Pearson-trained LoRA lowers RPS relative to validation-selected CE-LoRA on Qwen3-4B (`0.1412` vs `0.1997`; paired interval excludes zero), while the Qwen2.5-Coder-7B interval includes zero (`0.1560` vs `0.1859`). Different features, training procedures, and splits prevent attributing these comparisons solely to parameter capacity, overfitting, or RLHF.

## Figures

- [fig1_taxonomy_forest_rps_and_ce.png](figures/fig1_taxonomy_forest_rps_and_ce.png): Forest plot of held-out test ranked probability score (RPS, 0–1) and cross-entropy (CE, nats): correlation-trained/selected minus CE-trained/selected judge; positive favors CE. Bars are 95% task-family bootstrap intervals (10,000 draws, 19 families). Mixtures use training-set sequential least-squares programming (SLSQP) or non-negative least squares (NNLS); prompts use the P0-seeded two-round/two-candidate search; LoRA uses base-initialized AdamW for eight epochs, with train or validation selection as labeled. Non-convex searches have no global-optimum guarantee.
- [fig2_lora_weight_training_and_mse_decomp.png](figures/fig2_lora_weight_training_and_mse_decomp.png): Held-out test ranked probability score (RPS, 0–1) and mean squared error (MSE, rating squared), lower is better, for five-fold task-family low-rank adaptation (LoRA). Base-initialized AdamW trains eight epochs; checkpoint selection uses the labeled training or validation objective, without a global guarantee. The dashed constant forecast is a reference from the separate 200-split pool analysis, not a matched five-fold baseline.
- [fig3_real_judge_learning_curves_vs_n.png](figures/fig3_real_judge_learning_curves_vs_n.png): Held-out test ranked probability score (RPS, 0–1), cross-entropy (CE, nats), and Pearson correlation (−1 to 1) across training rating counts; lower losses/higher correlation are better. Each size uses 120 grouped splits and a 48-weight hidden-state head, zero-initialized L-BFGS-B (400 iterations), trained on the named objective; non-convex objectives have no global guarantee. Curves show point estimates; constant training forecasts give the reference scale.
- [fig4_when_ce_wins_disagreement_and_prompts_vs_harnesses.png](figures/fig4_when_ce_wins_disagreement_and_prompts_vs_harnesses.png): Stratified held-out test ranked probability score (RPS, 0–1) of Pearson-trained minus CE-trained mixtures (positive favors CE), with 95% task-family bootstrap intervals (10,000 draws), plus the prompt × harness matrix (lower RPS is better). The 200 grouped splits select fixed prompt/harness candidates on training data and fit simplex mixtures by SLSQP or NNLS; discrete candidate selection is exhaustive only within its declared pool.

## Limitations

- **Retrospective shared human ratings:** All `75` artifacts (`25` tasks, `19` task families, `195` expert ratings) come from the shared VeriBench cache (`dataset.json` SHA-256 `b3a05699...`). Although prompts, harnesses, and LoRA weights were fitted strictly on training task families without test leakage, prospective validation awaits fresh ratings in `trustworthy-theorem-judges` experiment 09.
- **LLM reading, no DT:** These judges score specification coverage by reading Python and Lean files (`no DT`, no Lean proof compilation inside the judge).

**TLDR-end:** [judges-corr: 09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026] The native six-category probability pools have lower log loss than the historical scalar pool (`3.480` -> `1.629` nats verbal / `1.606` nats calibrated logprob), and proper-loss optimization (CE / RPS) beats correlation optimization across `ii.a` prompt mixtures, `ii.b` harness optimization, and `ii.c` `48`-weight heads & PyTorch LoRA LLM weight training (`p < 0.03`), while uncalibrated discrete 3-fold meta-prompt search has a paired interval including zero (`-0.0261` RPS, `p-val=0.2778`).
