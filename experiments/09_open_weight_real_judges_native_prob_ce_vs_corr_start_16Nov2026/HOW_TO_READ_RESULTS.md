# How to Read Results: 09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Reference sheet answering the five mandatory result-clarity questions (`AGENTS.md`, 11-13-2026) for every table, plot, and headline contrast in `experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/`.

## 1. Which data, split, and prior exposure

**Input provenance and prior exposure:** 60/75 reference files differ from the hashes recorded in the annotation manifest; Python and candidate files match. The saved evidence does not resolve stale reference metadata versus changed reference bytes, so byte identity with the human-delivered reference files is unverified. The fixed P2 prompt includes historical expert-mean and leniency summaries that were not estimated within each training split; P2-containing pools are retrospectively informed, not zero-exposure evaluations. Family separation applies to fitting and selection. Digit-logprob extraction imputes absent digits below the observed top-20 tail and can report success with fewer than six observed digits; saved receipts do not identify the exact incidence. These are probability estimates, not guaranteed complete six-digit softmax probabilities.

- **Evaluation dataset:** `75` Lean 4 candidate formalization artifacts across `25` Python tasks grouped into `19` task families, rated by qualified Lean experts (`195` individual integer ratings on `0..5`, `60` ordinary non-placeholder artifacts and `15` `True`-placeholder-reference artifacts; `62` artifacts have non-zero expert disagreement (standard deviation, SD) `SD > 0` and `13` have unanimous expert ratings `SD = 0`).
- **Held-out splits:** Every reported score (except the one explicitly labeled in-sample single-judge reference table) is measured strictly on **held-out test task families** never seen during training, prompt proposal, harness selection, low-rank adaptation (LoRA) backpropagation, or hyperparameter selection:
  - **Pool / prompt / harness evaluations:** `200` stratified grouped splits holding out `4` of the `19` task families (`~16` test artifacts) and fitting on the remaining `15` task families (`~59` training artifacts; inner `3`-family validation split for formulations ① and ③). Hidden-state head learning curves use `120` grouped splits per training size `n_fam in {3, 5, 7, 9, 12, 15}`, holding out `4` test families per split (with the remaining `15 - n_fam` families unused when `n_fam < 15`).
  - **End-to-end PyTorch LoRA weight training (`ii.c`):** `5`-fold grouped task-family cross-validation (`11–12` inner-train families, `4` inner-validation families, `4` or `3` outer held-out test families per fold; every one of the `75` artifacts is scored strictly out-of-fold once).
  - **Blind automated meta-prompt search (`ii.a` on `Qwen3-30B-A3B`):** `3`-fold grouped task-family cross-validation (`9–10` inner-train families, `4` inner-validation families, `6–7` untouched outer held-out test families; every artifact scored out-of-fold once).
- **Prior exposure disclosure:** These `75` artifacts and `195` expert ratings come from the historical VeriBench cache (`dataset.json` SHA-256 `b3a05699...`) inspected in earlier studies (`experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/`, `archive/11, 12, 14`). Unlike historical `H1`/`H2` prompts, all new prompts (`P0` and blind meta-prompts) and LoRA weights here were constructed without accessing test-split human ratings, but the dataset itself remains retrospective.

## 2. Metrics, units, direction, and baselines

| Metric | One-line definition | Range / Units | Better | Baseline reference (`constant` train marginal) |
| --- | --- | --- | --- | --- |
| Ranked probability score (RPS) — **primary** | `(1/5) sum_{k=0..4} (Q(k) - 1{rating <= k})^2`, averaged within artifact then equally across held-out test artifacts | `[0, 1]` | Lower | `0.1857` (leave-one-rater-out human ceiling: `0.1113`) |
| Cross-entropy (CE) — **secondary** | `-ln q(rating)` in natural-log units (`1e-6` probability floor), averaged within artifact then equally across test artifacts | `[0, ~13.8]` nats | Lower | `1.887` nats |
| Total variation distance (TVD) — **secondary** | `0.5 sum_{r=0..5} |q(r) - p_hat(r)|` against each artifact's empirical human histogram (noisy 2–3 rating diagnostic) | `[0, 1]` | Lower | `0.693` |
| Six-category Brier score | `sum_{r=0..5} (q(r) - 1{rating = r})^2` averaged over ratings and artifacts | `[0, 2]` | Lower | `0.857` |
| Mean squared error (MSE) & signed bias (`0–5`) | `mean((mu_judge - mu_human)^2)` and `mean(mu_judge - mu_human)` (`>0` = judge is lenient); exact split `MSE = bias^2 + (sx - sy)^2 + 2 sx sy (1 - r)` | `[0, 25]` / `[-5, +5]` | Closer to `0` | Human mean `2.63` (`SD = 1.43`) |
| Pearson, Spearman, Kendall tau-b | Linear, rank, and pairwise concordance correlation between expected judge rating `E_q[R]` and empirical human mean rating | `[-1, +1]` | Higher | Leave-one-rater-out human pair Pearson `0.805`, Spearman `0.798` |

## 3. Subtraction order in every difference

Every difference subtracts **the exact same metric on the exact same held-out test artifacts** for two differently trained/selected judges:
- Default headline convention: **`test <metric> of correlation-trained judge minus test <metric> of CE-trained judge`**.
- Because lower is better for RPS, CE, and TVD, **a positive (`+`) difference means the CE-trained judge is better (has lower held-out loss)**, and a negative (`−`) difference means the correlation-trained judge is better.

## 4. Uncertainty and hypothesis tests

- **Intervals:** `95%` percentile cluster bootstrap (`10,000` draws) resampling the **`19` task families** with replacement (weighting held-out artifacts equally). Resamples fixed per-artifact out-of-fold losses averaged over overlapping splits without refitting. It does not capture training/search variability, new solver populations, or fresh rater panels.
- **P-values:** Exact two-sided paired task-family sign-flip permutation test (`2^19 = 524,288` sign configurations when all `19` families are active) under the null that family-level differences are sign-exchangeable (a stronger assumption than zero mean alone). P-values are nominal exploratory comparisons without multiple-comparison adjustment; overlapping training splits limit population interpretation.

## 5. How each judge was chosen (search space, optimizer, certificates)

- **Simplex mixtures (`ii.a`, `ii.b`, `ii.c`):** Non-negative weights summing to `1`. CE uses convex sequential least-squares programming (SLSQP) + active-face Newton refinement with checked Frank-Wolfe duality gap `<= 1e-8` (global optimum guaranteed). RPS uses convex quadratic SLSQP on the simplex with duality gap `<= 1e-7` (global optimum guaranteed). Pearson uses exact non-negative least-squares (NNLS) cone projection (global optimum guaranteed). Spearman uses best of `1,500` Dirichlet draws + simplex vertices (approximate).
- **Blind meta-prompt search (`ii.a`):** `Qwen3-30B-A3B` proposes `2` candidate rubric addenda per round for `2` rounds (`5` candidates total including `P0`), guided strictly by inner-train correlation vs inner-train CE/RPS; selected by inner-train (`②`/`④`) or inner-validation (`①`/`③`). Non-exhaustive prompt space.
- **End-to-end PyTorch LoRA weight training (`ii.c`):** `28` LoRA projection adapters (`rank=8, alpha=16`) across the top `4` transformer blocks + `6`-logit rating head (`1.85M` trainable parameters in `Qwen3-4B`, `2.91M` in `Qwen2.5-Coder-7B`), initialized to reproduce the unadapted layer-streamed digit-logit path (`epoch 0` max difference `< 3e-7`; this is not an independent full-model forward equivalence check), optimized by full-batch `AdamW` (`lr=2e-4, weight_decay=0.01`, `8` epochs on 5-fold CV and `6` epochs on learning curves; non-convex in transformer weights).
- **Hidden-state neural softmax heads (`ii.c`):** Leak-free training-fold `5` principal-component (PC) Gram features + base score/entropy + intercept (`8 x 6 = 48` weights), fitted by `L-BFGS-B` (`400` iterations single start, checked against best-of-`5` random starts × `2,000` iterations). CE is convex in head weights; RPS and correlation surrogates (`tau=0.1`) are non-convex.

**TLDR-end:** [judges-corr: 09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026 › how to read] Defines splits, metrics, subtraction order, 19-family bootstrap/sign-flip uncertainty, and optimizer guarantees for every reported result.
