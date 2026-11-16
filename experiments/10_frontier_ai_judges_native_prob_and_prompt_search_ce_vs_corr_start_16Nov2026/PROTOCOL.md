# Folder 10 — PROTOCOL (LOCKED before any expensive run)

This file freezes the protocol before any LLM-budget-spending run begins. It is written PRIOR to seeing any results. Any deviation observed during execution is documented in `RESULTS_frontier_ce_vs_corr.md` → "Deviations and open questions", not by editing this file.

## Scientific question (locked)

On real frontier large language model (LLM) judges predicting human expert formal-specification-coverage ratings on `{0, 1, 2, 3, 4, 5}` (75 VeriBench artifacts, 25 Python tasks, 19 task families, 195 expert ratings; no differential testing (no DT)):

1. **Native probabilities versus coarse scalar samples:** Does eliciting native 6-category probability vectors `q = (q_0, ..., q_5)` (and exact single-token logprobs over `0..5`) across OpenAI and Mistral frontier models repair the `5.49`-nat cross-entropy (CE) failure of 3-sample scalar frontier judges (`02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026`) relative to the `~1.93`-nat constant training-histogram baseline, and when does proper-loss training/selection beat correlation training/selection across Formulations ①, ②, ③, and ④?
2. **Axis `iii.a` — Blind prompt optimization (`CE/RPS` versus `Correlation`):** When an agentic prompt optimizer iteratively refines a frontier judge prompt (`gpt-6.1-sol`, starting from verbatim human instructions `P0_human_mirror`, 4 iterations x 2 candidates = 8 candidates per track, identical `<= 420`-word budget, strict training-only feedback firewall), does optimizing against individual-rating proper losses (③ validation ranked probability score (RPS) / ④ training CE) outperform optimizing against correlation (① validation Pearson / ② training Pearson) on held-out task families, before and after post-hoc calibration?
3. **Axis `iii.b` — Frontier judge harness optimization (`CE/RPS` versus `Correlation`) and `2x2` Prompt x Harness interaction:** Across four distinct frontier judge harness architectures — `H1_single_call`, `H2_ast_obligation_harness` (programmatic Python/Lean abstract syntax tree (AST) theorem and obligation pre-extractor), `H3_two_step_decompose_aggregate` (two-stage obligation audit -> calibrated aggregator), and `H4_multi_family_ensemble_calibration_harness` (cross-family OpenAI + Mistral ensembles, simplex mixtures, 2-parameter convex temperature/tilt calibration, and two-stage `Pearson + calibrated CE head`) — does proper-loss harness selection/fitting beat correlation harness selection/fitting on held-out task families, and how much held-out RPS improvement comes from the prompt-search objective (`iii.a`) versus the harness/calibration objective (`iii.b`)?
4. **Regime boundary ("WHEN is CE better than correlation?"):** At what real training sample size (`n_fam in {3, 5, 7, 9, 12, 15}` training task families, ~24 to ~155 human ratings) and on which artifact strata (positive human rater disagreement `SD > 0` / larger Dirichlet-smoothed information gap `G_i` versus unanimous `SD = 0`; ordinary tasks versus `True`-placeholder tasks) does proper-loss optimization establish a held-out test RPS advantage (`95%` task-family bootstrap interval of `test RPS of correlation-trained judge minus test RPS of proper-loss-trained judge` strictly above `0`)?

**Operational primary endpoint and decision rule (`EVALUATION_CRITERION_11-09-2026.md`):**
- **Primary held-out metric:** Normalized individual-rating **ranked probability score (RPS)** on `{0..5}`, averaged over observed human ratings within each artifact and weighted equally across held-out test artifacts (`range [0, 1]`, lower is better).
- **Primary signed contrast:** `Delta_RPS = test RPS of correlation-trained judge minus test RPS of proper-loss-trained judge` on identical held-out test artifacts (`Delta_RPS > 0` means the proper-loss-trained judge is better). A proper-loss advantage is established in a condition when the paired `95%` task-family cluster-bootstrap percentile interval (`10,000` draws) excludes `0` (`lo > 0`). Ties (`0` inside `[lo, hi]`) and correlation wins (`hi < 0`) are reported symmetrically without relabeling.
- **Secondary held-out metrics:** Individual-rating **cross-entropy (CE)** in nats (with pre-specified probability floor `epsilon = 1e-6`, plus unsmoothed `epsilon = 0` zero-miss diagnostics), **total variation distance (TVD)** (`0.5 * sum_r |q_i(r) - p_hat_i(r)|`), 6-category Brier score, Pearson `r`, Spearman `rho`, Kendall `tau-b`, intraclass correlation coefficients `ICC(A,1)` and `ICC(C,1)`, mean absolute error (MAE), and the exact **mean squared error (MSE) decomposition** `MSE = bias^2 + scale_mismatch^2 + 2 s_x s_y (1 - r)`.

## Pre-registered measurement contract (`ml-research-practice.md` Step 1)

Committed before running any model calls. `E[R]/5` denotes normalized expected rating in `[0, 1]`; `RPS` is normalized ranked probability score in `[0, 1]` (lower is better); `CE` is smoothed individual-rating cross-entropy in nats with `epsilon = 1e-6` (lower is better).

| Condition / check | Code path | Metric(s) | Predicted range / behavior before run | Decision rule if violated |
|---|---|---|---|---|
| **(a) Oracle control** (gold reference judged against itself, 4 non-placeholder smoke tasks) | `scripts/run_phase0_and_sanity.py` | Mean `E[R]/5`; `RPS` vs `5` | `E[R]/5 >= 0.80` (`>= 4.0/5`), `q_4 + q_5 >= 0.75`, `RPS <= 0.12` | Stop and inspect rendered prompt/files before scaling |
| **(b) Intended behavior** (untuned `P0_human_mirror` on `gpt-6.1-sol`, 75 items) | `scripts/run_arm_a_pool.py` | `RPS`, `CE`, Spearman `rho`, modal share | `0.11 <= RPS <= 0.19`, `1.25 <= CE <= 2.20` nats (beating `5.49`-nat 3-sample scalar CE), `rho >= 0.45`, modal share `< 0.70` | Stop and inspect parser / probability calibration instructions |
| **(c) Literal compliance** (candidate covering all gold theorems cleanly with `C=No` spam) | `scripts/run_phase0_and_sanity.py` | `E[R]/5` | `E[R]/5 >= 0.80` (`q_4 + q_5 >= 0.75`); instructions and metric agree (no `sorry`-mandate contradiction) | Fix prompt-metric mismatch before main run |
| **(d1) Degenerate: blank candidate** (`""`, 5 smoke tasks) | `scripts/run_phase0_and_sanity.py` | `E[R]/5`; `RPS` vs `0` on ordinary tasks | Ordinary tasks: `E[R]/5 <= 0.15` (`q_0 + q_1 >= 0.80`, `RPS <= 0.10`) | Stop: prompt fails to penalize empty submissions |
| **(d2) Degenerate: vacuous claim** (`theorem main_claim : True := trivial`, 4 non-placeholder smoke tasks) | `scripts/run_phase0_and_sanity.py` | `E[R]/5` | `E[R]/5 <= 0.20` (`q_0 + q_1 >= 0.75`) | Stop: metric is gameable by trivial theorems |
| **(d3) Degenerate: shuffled cross-task candidate** (5 smoke tasks) | `scripts/run_phase0_and_sanity.py` | `E[R]/5` | Ordinary tasks: `E[R]/5 <= 0.25` | Stop: judge ignores task alignment |
| **(e) Label-in-prompt reproduction** (15 smoke artifacts given exact expert ratings in prompt) | `scripts/run_phase0_and_sanity.py` | Pearson `r`, Spearman `rho`, `TVD` | `r >= 0.98`, `rho >= 0.97`, `TVD <= 0.05` | Pipeline/alignment/parser bug; halt immediately |
| **(f1) Dumb baseline: theorem-count ratio** `min(1, n_cand / max(1, n_ref))` | `scripts/run_phase0_and_sanity.py` | Spearman `rho` (all 75 / 60 ordinary) | `rho ~ 0.50–0.58` on all 75 (`0.543` in VeriBench exp 100) | Verify exact AST theorem-statement counter |
| **(f2) Constant marginal baseline** (training human rating histogram) | `scripts/analyze_all.py` | Held-out `RPS`, `CE`, `TVD` | `RPS ~ 0.180–0.192`, `CE ~ 1.85–1.96` nats, `TVD ~ 0.67–0.72` | Verify split and metric parity against `Exp 02` (`0.1872` RPS, `1.9280` CE) |
| **(f3) Historical 5 cached 3-sample scalar frontier judges** (`H0/H1/H2` Sol/Opus) | `scripts/analyze_all.py` | Held-out `RPS`, `CE` (`Exp 02` 5x5 CV) | `CE ~ 5.49` (CE-mixture) and `5.75` (Pearson-mixture); `RPS ~ 0.1391` vs `0.1436` | Must match `Exp 02` saved numbers to `< 1e-5` |
| **(f4) Human ceiling & disagreement floor** (leave-one-expert-out on 75 items) | `scripts/run_phase0_and_sanity.py` | LORO Spearman `rho`, `ICC(A,1)`, empirical RPS noise floor | LORO `rho ~ 0.75–0.83`, `ICC(A,1) ~ 0.72–0.82`, empirical RPS disagreement floor `~ 0.048–0.058` | Verify rater columns in `dataset.json` |

## Locked parameters

- **Hard spend cap & billing scope:**
  - `HARD_CAP_USD = 18.00` enforced via file-locked (`fcntl.flock`) reserve-before-call ledger (`private/cost_ledger.jsonl`, with public sanitized summary `results/cost_summary.json`).
  - Approved keys ONLY: `~/keys/openai_bm_key_koyejolab.txt` (`koyejolab-openai-bm`) and `~/keys/mistral_api_key.txt` (`mistral`). Zero calls to Vals, Allen Gemini, or Anthropic keys.
- **Probability parsing and smoothing policy (`EVALUATION_CRITERION_11-09-2026.md`):**
  - Native-probability JSON schema requires `{"probabilities": [q_0, q_1, q_2, q_3, q_4, q_5], "rationale": "..."}` with 6 finite values in `[0, 1]` whose sum is in `[0.99, 1.01]`, normalized to exact unit sum `sum_r q_r = 1.0` (exact pre-normalization sum deviation recorded per call).
  - Pre-specified probability floor for individual-rating `CE` and log-space calibration: `epsilon = 1e-6` (`q_smooth = (1 - 6e-6) * q + 1e-6`), matching `02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026` exactly. Unsmoothed zero-probability bin counts (`q_r == 0` on observed human ratings) are also reported directly.
- **Phase 1 — Karpathy smoke set (`5` task families, `15` artifacts) & repeat stability (`k = 3`):**
  - Smoke tasks (fixed seed `20261116`, 4 ordinary + 1 placeholder): `0_bisect_right`, `13_MyUppercase`, `4_json_decoder`, `humaneval_13__gcd`, `0_unsafeCopy_safe`.
  - Repeat stability: `k = 3` independent API evaluations of `gpt-6.1-sol` (`P0_human_mirror`, `effort="low"`) across all `15` smoke artifacts.
- **Phase 2 — Multi-family frontier native-probability pool (`15` new judges across `7` models in `2` families, all `75` artifacts, zero split exposure):**
  1. `gpt6_astra__P0`: `gpt-6-astra` (`effort="low"`), prompt `P0_human_mirror`
  2. `gpt61_sol__P0`: `gpt-6.1-sol` (`effort="low"`), prompt `P0_human_mirror`
  3. `gpt61_sol__P1`: `gpt-6.1-sol` (`effort="low"`), prompt `P1_conventions`
  4. `gpt61_sol__P2`: `gpt-6.1-sol` (`effort="low"`), prompt `P2_disagreement_aware`
  5. `gpt56_sol__P0`: `gpt-5.6-sol` (`effort="none"`), prompt `P0_human_mirror`
  6. `gpt54_mini__P0`: `gpt-5.4-mini` (`effort="low"`), prompt `P0_human_mirror`
  7. `gpt54_mini__P1`: `gpt-5.4-mini` (`effort="low"`), prompt `P1_conventions`
  8. `gpt54_mini__P2`: `gpt-5.4-mini` (`effort="low"`), prompt `P2_disagreement_aware`
  9. `gpt54_mini__L0_logprobs`: `gpt-5.4-mini` (`effort="none"`, `/v1/chat/completions` `logprobs=True, top_logprobs=20, max_tokens=1`), exact softmax over rating tokens `"0".."5"`
  10. `mistral_med35__P0`: `mistral-medium-3.5` (`reasoning_effort="none"`), prompt `P0_human_mirror`
  11. `mistral_med35__P1`: `mistral-medium-3.5` (`reasoning_effort="none"`), prompt `P1_conventions`
  12. `mistral_med35__P2`: `mistral-medium-3.5` (`reasoning_effort="none"`), prompt `P2_disagreement_aware`
  13. `magistral_med__P0`: `magistral-medium-latest` (`reasoning_effort="high"`), prompt `P0_human_mirror`
  14. `codestral_2508__P0`: `codestral-2508`, prompt `P0_human_mirror`
  15. `codestral_2508__P2`: `codestral-2508`, prompt `P2_disagreement_aware`
  - Plus historical cached coarse scalar pool (`5` judges, `3` scalar votes each): `H0_opus`, `H0_sol`, `H1_sol`, `H2_opus`, `H2_sol`.
- **Phase 3 (`iii.a`) — Locked 3-way grouped task-family split (`9` train / `5` validation / `5` held-out test) & blind prompt search:**
  - **Train families (`9` families, `9` tasks, `27` artifacts, `69` ratings):** `0_unsafeCopy_safe`, `13_MyUppercase`, `18_MyFibonacci`, `1_MyAdd`, `33_MyMinimum`, `43_replace_spaces`, `4_json_decoder`, `4_unsafeCounter_safe`, `humaneval_8_sum_product`.
  - **Validation families (`5` families, `6` tasks, `18` artifacts, `45` ratings):** `1_unsafeMultiply_safe`, `9_pathTraversal_safe`, `heap_family`, `humaneval_13__gcd`, `humaneval_31_is_prime`.
  - **Held-out test families (`5` families, `10` tasks, `30` artifacts, `81` ratings):** `5_unsafeFormatString_safe`, `binary_search_family`, `humaneval_3_below_zero`, `humaneval_47_median`, `sorting_family`.
  - **Search budget & firewall:** Starting from `P0_human_mirror`, proposer `gpt-6.1-sol` (`effort="medium"`) runs **4 iterations x 2 candidates per iteration = 8 prompt candidates per track** (`<= 420` words per rubric block), evaluated on `gpt-6.1-sol` (`effort="low"`):
    - `Track_Corr` sees ONLY training `Pearson / Spearman / Kendall` and top-6 standardized `z`-score / rank residuals on the 27 training items (zero raw ratings, zero bias/scale/histograms). Selects `P_corr_tr` (②, max train Pearson) and `P_corr_val` (①, max validation Pearson).
    - `Track_CE` sees ONLY training individual-rating `CE / RPS` and top-6 CE-loss training items (`q` vs training rating counts). Selects `P_ce_tr` (④, min train CE) and `P_ce_val` (③, min validation RPS).
- **Phase 4 (`iii.b`) — Frontier judge harness space & `2x2` Prompt x Harness interaction:**
  - Four architectural harnesses evaluated across all 75 artifacts (and scored strictly on the locked `9/5/5` split AND across `250` repeated grouped task-family splits):
    1. `H1_single_call`: Direct single-call native-probability judge (`gpt-6.1-sol` and `mistral-medium-3.5` across `P0`, `P_corr_val`, `P_ce_val`, plus `P_corr_tr`, `P_ce_tr`).
    2. `H2_ast_obligation_harness`: Deterministic Python/Lean AST pre-extractor (`gold` public theorem statements, `candidate` theorem statements, placeholder detection, `sorry` counts, theorem-count ratio, lexical signature alignment) injected as structured scaffold before `gpt-6.1-sol` (`P0`, `P_corr_val`, `P_ce_val`) and `mistral-medium-3.5` (`P0`).
    3. `H3_two_step_decompose_aggregate`: Step 1 `gpt-6.1-sol` per-obligation coverage mapper (`COVERED / PARTIAL_PRECOND / IMPL_RESTATEMENT / MISSING` + spam audit, zero rating commitment) -> Step 2 `gpt-6.1-sol` calibrated 6-category distribution aggregator under `P0`, `P_corr_val`, `P_corr_tr`, `P_ce_val`, and `P_ce_tr`.
    4. `H4_multi_family_ensemble_calibration_harness`: Cross-family OpenAI + Mistral harness combining `{H1, H2, H3}` forecasts via (a) equally weighted cross-family average, (b) simplex mixture (`CE`, `RPS`, `Pearson`, `Spearman`, `Kendall`, `1/2 Pearson + 1/2 Spearman`), (c) 2-parameter convex temperature + ordinal-tilt calibration `softmax(alpha * log q(r|x) + beta * (r/5))`, (d) two-stage `Pearson + calibrated CE head`, and (e) discrete 3-vote scalar histogram baseline.
- **Repeated grouped task-family splits, objectives, restarts, and learning curves (`scripts/analyze_all.py`):**
  - **Primary split suite:** `250` seeded grouped task-family splits (`15` train families / `4` held-out test families, seed `20261116`, stratified so every held-out test split contains `>= 1` ordinary multi-rating/multi-task family and `>= 12` artifacts), plus `Exp 02`'s exact 5-partition x 5-fold grouped task-family CV (`25` outer folds, seeds `[20261106..20261110]`).
  - **Inner validation for ① and ③:** 5-fold grouped task-family CV inside each training split to select shrinkage `lambda in {0, 0.05, 0.15, 0.30, 0.50, 0.75, 1.0}`, ridge `c in {0.01, 0.1, 1, 10, 100}`, or discrete prompt/harness choice by validation Pearson (①) versus validation RPS (③).
  - **Optimizer restart check (`Exp 07` on real frontier features):** On the 48-parameter (`8 features x 6 classes`) softmax feature head, compare `1 start x 400 L-BFGS-B iterations` against `best-of-5 starts x 2,000 L-BFGS-B iterations` across 50 grouped splits for all 7 objectives.
  - **Training-size learning curve:** `n_fam in {3, 5, 7, 9, 12, 15}` nested training task families evaluated on the identical 4 held-out test families across all `250` splits.
  - **Bootstrap:** `10,000` task-family cluster bootstrap draws (`seed = 20261116`).

## Scale verification (no further normalization)

- Raw human ratings in `dataset.json["rows"][i]["ratings"]` are integers in `{0, 1, 2, 3, 4, 5}`. Do NOT divide raw rating integers before computing rating histograms or category indices!
- Normalized expected scores `mu = sum_{r=0}^5 (r / 5.0) * q_r` lie in `[0, 1]` (`Exp 02` convention); raw expected ratings `5 * mu = sum_{r=0}^5 r * q_r` lie in `[0, 5]`. Both scales are explicitly labeled whenever MSE, MAE, or signed bias (`judge minus human`) is reported; `RPS`, `CE`, and `TVD` are scale-invariant proper/distribution metrics on the 6 ordered categories `0..5`.

## Verified data layout

- `/Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/private/historical_exp06/cached_v1/dataset.json` (SHA-256 `b3a05699a26d3ffd9f28e793178811219885df78319b3aa57f9d393c68724ba0`):
  - `rows`: 75 dicts with `item`, `task`, `family` (19 unique), `ordinary` (60 `True`, 15 `False`), `ambiguous`, `split`, `ratings` (list of 2–3 ints in `0..5`, total 195; **PRIVATE — never commit**), `ratings_by_expert` (**PRIVATE — never commit**).
  - `outputs`: dict mapping `H0_opus`, `H0_sol`, `H1_sol`, `H2_opus`, `H2_sol` -> list of 75 lists of 3 integer ratings in `0..5`.
- `/Users/brandomiranda/veribench/experiments/100_karpathy_recipe_tc_judge_search_human_train_val_test/inputs/items.jsonl` (75 rows matching `dataset.json` `item` 1-to-1):
  - `candidate_path` (`75/75` on-disk SHA-256 match `candidate_sha256`), `py_ann_path` (`75/75` on-disk SHA-256 match `py_ann_sha256`), `ref_ann_path` (on-disk files read directly as in `exp 100/12/14`, recording both on-disk SHA-256 and `HEAD`/`items.jsonl` SHA-256 in our input manifest).

## Pre-conditions for run

- Python `3.11+` with `numpy`, `scipy`, `matplotlib`, `httpx` / `urllib.request` verified on local Mac.
- `.gitignore` present in experiment root ignoring `private/`.
- Runtime-readable key files `~/keys/openai_bm_key_koyejolab.txt` and `~/keys/mistral_api_key.txt` (verified HTTP 200).
- `FREEZE.json` committed in local Git before Phase 1–4 paid API calls execute.

## Estimated wall-clock

- Phase 1 (sanity checks & `k=3` repeats, ~64 calls concurrent): ~2 minutes.
- Phase 2 (15 multi-family frontier judges x 75 items = 1,125 calls, 12-way concurrency): ~8–12 minutes.
- Phase 3 (`iii.a` 4-iteration blind prompt search, ~840 calls): ~8–10 minutes.
- Phase 4 (`iii.b` `H2` + `H3` + cross-family harnesses, ~675 calls): ~6–8 minutes.
- Phase 5 (CPU bootstrap/learning-curve/restart analysis + plots + deterministic verification): ~3–4 minutes.
- Total estimated wall-clock: ~28–36 minutes; expected API spend `~$13.50–$15.80` (`<= $18.00` hard cap).

## Pass/fail / abort rules

- **Pre-scaling smoke gate (Phase 1):** Before launching Phase 2–4 batches, verify on the 5-task smoke set that:
  1. `label_in_prompt` achieves Pearson `r >= 0.95` and `TVD <= 0.08`;
  2. `oracle` mean expected score exceeds `blank`, `vacuous`, and `shuffled` by at least `+0.50` on `[0, 1]` (`+2.5` rating points on `0..5`);
  3. Valid probability JSON parse rate is `>= 95%`.
  If any smoke gate fails, STOP before Phase 2, diagnose the bug, and document any fix transparently.
- **Full-run abort rules:** Stop new admissions immediately if cumulative reserved + settled spend reaches `HARD_CAP_USD = 18.00`, or if authentication returns HTTP 401/403 on an approved key. Retain every attempted, failed, or unattempted item in the full 75-artifact denominator.

## Identity / contact crib (verified)

- Author / repository owner: Brando Miranda (`brando90`, `brandojazz@gmail.com` / `brando9@stanford.edu` / `brando.science@gmail.com`, verified from `git log` and `~/agents-config/personal/contacts.md`).
- Billing scope: `koyejolab-openai-bm` (`~/keys/openai_bm_key_koyejolab.txt`) and `mistral` (`~/keys/mistral_api_key.txt`), Koyejo Lab / Sanmi collaboration study; notifications and `origin/main` push handled solely by parent coordinator (`7fa0b312-b867-45d4-b095-82ed9c135b7b`).
