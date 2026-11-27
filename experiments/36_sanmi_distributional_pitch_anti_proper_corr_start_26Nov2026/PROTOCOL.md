# Measurement contract — 11-26-2026

Written before `run_all.py` printed a held-out test metric.

## Estimator

For a proper loss L in {cross-entropy, one-hot Brier, ranked probability score}, the test risk is

R(q) = (1/|D_test|) sum_c (1/k_c) sum_j L(q(c), y_{c,j}).

Plug-in D_B of the test histogram is not this number. Its bias against the truth is (1/k)(1-||p*||^2) in expectation.

## Split

Prompt groups, seed 0, fractions 0.6 / 0.2 / 0.2. SummEval groups are source articles. ChaosNLI and DICES groups are item ids. Validation does not choose hyperparameters.

## Interval

Item-clustered percentile bootstrap, B=1000, seed 1. The interval is for the fixed trained head.

## Head

Hash dimension 256. 24 principal components on train prompts only, plus an intercept. L2 penalty 1e-2 on non-intercept weights. L-BFGS-B from zero, at most 35 iterations, for cross-entropy, Brier, RPS, and the Pearson sample-correlation surrogate. MSE is ridge, closed form, two-point readout. Cross-entropy is convex in the linear logits. The other softmax objectives are not, and are reported as this start and this budget.

## Predictions

- Uniform CE equals log K.
- Shuffling train features does not lower held-out CE.
- Known two-context law: true sampled correlation near 0.50, collapsed 0.1*mean+4 near 0.707.
- Binary: Brier = 2 * scalar squared error, and an affine forecast can tie correlation while Brier and CE worsen.
- Ordinal consensus-on-2 versus 50/50 on {0,4}: same mean, D_B = 1.5.
- Within-prompt oracle (labeled, not a held-out judge): collapsed affine correlation above the sampled-distribution correlation when within-item variance is positive.

## Dollars

Top axis uses $0.05 per rating (POPQUORN's published crowd rate) and $5 per rating (assumed expert rate). Neither is a measured wage for SummEval, ChaosNLI, or DICES. Experiment 23 found no verified per-grade dollar for those releases.

## Not in the contract because the files were initially absent (resolved in Amendment 1)

POPQUORN ratings. VeriBench FSC / CertJudge R_v1. Live LLM logits. LoRA.

## Amendment 1 (11-26-2026) — Added datasets and CertJudge bridge (before running new cells)

Before running `run_all.py` or `eval_power_mdd.py` on the newly collected/resolved datasets, we extend the same pre-registered pipeline (identical seed-0 `0.6/0.2/0.2` group split, identical 256-dim signed hash + 24-component train-only SVD + intercept, identical L2 penalty `1e-2`, identical `L-BFGS-B` from zero with `maxiter=35`, identical item-clustered bootstrap `B=1000` seed 1, and identical binary cut `value >= 4`) to:

1. **POPQUORN** (`Jiaxin-Pei/Potato-Prolific-Dataset`, pinned commit `dbb118c29b3d9bcce5aa172d8249ded9ece4df84`):
   - `popquorn_offensiveness`: 1–5 integer ratings (`13,036` ratings across `1,500` items, `262` raters; SHA-256 `fb0589093d45da76bf4ef4d0abff679b540bdd5b17af5764913b3ccd0a069e1e`), split by `instance_id`, plus binary view (`>= 4`).
   - `popquorn_politeness`: 1–5 integer ratings (`25,042` ratings across `3,718` items, `506` raters; SHA-256 `beb7c68c6366ab27819cdf40fec7eb76676f5903c99dad95c13d4bd2bc28365f`), split by `instance_id`, plus binary view (`>= 4`).
   - `question_answering` and `email_rewriting` in that repository are free-text/QA tasks without a 1–5 ordinal scale and are excluded.
2. **VeriBench Functional Semantic Correctness (FSC) `R_v1`** (`veribench` commit `d1c473559d3a7d5f7e7b4f708a9d7b9469afd779`, `items.jsonl` SHA-256 `ad9ed552fc2ffac6328e6010bf6be82215dd74f1eddd791622d433126877a96c`):
   - `veribench_fsc_rv1` (`all5`, `75` artifacts across `25` tasks, `345` ratings on 0–5), `veribench_fsc_rv1_expert3` (`expert3`, `75` artifacts, `195` ratings), and binary view (`>= 4`). Split group is `task` so all 3 candidate samples of one task remain together.
3. **VeriBench FSC `native_main` (live synced 11-26-2026)** (`trustworthy-theorem-judges` Experiment 09 commit `460466dd10b735c5ff25f9e9cb21f0441fcf960b`):
   - `veribench_fsc_native_main`: `100` artifacts across `25` tasks (`250` valid numeric `0–5` non-coordinator ratings across `6` active raters; blanks and `N/A` excluded, never treated as `0`; coordinator `A1` has 0 numeric main ratings), split by `task_folder`.
4. **VeriBench FSC Pooled (`R_v1` + `native_main`)**:
   - `veribench_fsc_pooled` (`175` artifacts across `50` disjoint task groups, `595` ratings on 0–5) and `veribench_fsc_pooled_binary` (`>= 4`). Note: `native_main` has 74.0% ratings at `5` (185/250; QA Stage 1 11-26-2026 corrected an earlier 72.8%, recomputed by `load_veribench_fsc_pooled`) vs `R_v1`'s `mean 2.97`, so pooled results reflect both sets' distributions.
5. **Test-set power & MDD extension (`eval_power_mdd.py`)**:
   - Add `popquorn_offensiveness` (`k_test in {1, 2, 4}`) and `popquorn_politeness` (`k_test in {1, 2}`) alongside `dices990` using the same outer/inner bootstrap (`200 x 200`, seed 11).
6. **CertJudge Real LLM Judge Panels Bridge (read-only verification from `cert-judge` commit `f6047849853e65fd533a6e5df2416d6a9d9447d6`)**:
   - Summarize and verify `study_a_e63_proper_scores.json` (6 `Qwen3-8B` arms with native 6-class probabilities on `R_v1` held-out test), `study_b_e39_proper_scores.json` (56 LLM judge configs on 45 held-out `R_v1` items, `CE-fit` vs `Pearson-fit` vs raw), and Experiment 67 `interim_11-24-2026` (3 headline judges × 3 repeats on the 100 `native_main` artifacts). `$0` extra API spend.
7. **Remaining unrun GPU items**: `scripts/snap_lora_train.py` and `scripts/snap_llm_logit_readout.py` remain unrun on this CPU-only machine; `T30` (Max Sieb) Google Sheet remains HTTP 401 unverified without renewed OAuth.
