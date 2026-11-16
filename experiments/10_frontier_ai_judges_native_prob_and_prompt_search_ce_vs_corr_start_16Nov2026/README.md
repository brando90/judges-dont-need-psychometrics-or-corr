# 10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/README.md>

**TLDR:** Answers Part `iii` ("frontier artificial intelligence (AI) judges too") of the three-part comparison of proper-loss optimization versus correlation optimization for building trustworthy AI judges on human ratings: **`iii.a` prompt optimization** and **`iii.b` harness optimization**, plus their **2x2 interaction** and a **multi-family native-probability frontier judge pool** (OpenAI GPT-6 Astra, GPT-6.1 Sol, GPT-6.1 Terra, GPT-5.6 Sol, GPT-5.6 Terra, GPT-5.4 mini; Mistral Codestral 25.08) versus historical 3-sample scalar frontier judges across the 19 VeriBench task families (75 artifacts, 25 Python tasks, 195 expert ratings). No differential testing (no DT) is run; all human ratings are existing retrospective annotations and raw ratings remain strictly private.


**Reference provenance amendment (11-16-2026):** Phase 1 records 15/75 reference-file hash matches (60 mismatches) against annotation metadata, alongside 75/75 candidate and 75/75 Python matches. Saved evidence does not resolve stale reference metadata versus changed bytes; identity with human-delivered reference files is unverified.



**Failure handling amendment (11-16-2026):** Saved evaluation retains uniform replacements after parsing failures: the selected cross-entropy prompt has 11/75 replacements overall, including 7/30 held-out test artifacts. Harnesses also retain uniform and upstream scaffold fallbacks. Cache-recovered status records and split denominators are in the Phase 3/4 result JSON files and the main report; no forecasts or metrics were changed.


## Scientific context across the three-experiment program (11-16-2026)

1. **`i. Synthetic / toy / simulated` (`00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026`, `01_conditional_binomial_distribution_identification_start_08Nov2026`, `06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026`, `07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026`, `08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026`, and `archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026`):** Evaluated fixed candidate distributions (`00`, `01`) or fitted 78 linear-basis softmax weights (`13`, `06`, `07`, `08`) on synthetic beta-binomial targets, with **zero prompt or harness changes**.
2. **`ii. Real judges — open weights` (`09_...`, sister study on cluster graphics processors):** Evaluates `ii.a` prompt optimization, `ii.b` harness optimization, and `ii.c` open-weight low-rank adaptation (LoRA) parameter updates.
3. **`iii. Real judges — frontier AI judges` (this folder, `10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026`):** Closed frontier application programming interfaces (APIs) do not expose model weights (`ii.c`), so practitioners optimize frontier judges along two real design axes — **`iii.a` prompt optimization** and **`iii.b` harness optimization** — plus post-hoc mixture/calibration heads over multi-family frontier probability forecasts.

## Goal, hypotheses, and decision criteria (fixed before evaluation, 11-16-2026)

Earlier real-rating comparisons (`02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026`, `archive/11_formulation4_train_only_ce_mixture_cv_start_10Nov2026_end_10Nov2026`, `archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026`, and `archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026`) were inconclusive on frontier judges because they (a) formed rating histograms from three discrete scalar samples (`5.49` nats cross-entropy (CE) versus `1.93` nats for a constant forecast), (b) split by task rather than task family in archived studies `11`, `12`, and `14`, and (c) never searched over prompts (`iii.a`) or judge harnesses (`iii.b`).

We test four pre-registered hypotheses using normalized individual-rating **ranked probability score (RPS; range 0 to 1, lower is better)** as primary held-out metric and **cross-entropy (CE; nats, lower is better)** and **total variation distance (TVD; range 0 to 1, lower is better)** as secondary held-out metrics (`EVALUATION_CRITERION_11-09-2026.md`):

- **H1 (Native probabilities versus 3-sample scalar histograms):** Eliciting native 6-category probability forecasts `q = (q_0, ..., q_5)` from frontier judges reduces held-out test CE below the constant training-histogram baseline (`~1.93` nats) and improves held-out test RPS over the 5 historical 3-sample scalar frontier judges on identical task-family splits.
- **H2 (`iii.a` Blind prompt optimization — proper loss versus correlation):** Under matched proposer model (`gpt-6.1-sol`), iteration budget (4 iterations x 2 candidates = 8 prompt candidates per track), word cap (`<= 420` words), and strict training-only feedback firewall, prompts optimized and selected by proper loss (Formulations ③ validation RPS and ④ training CE) achieve lower held-out test RPS and lower held-out test CE than prompts optimized and selected by correlation (Formulations ① validation Pearson and ② training Pearson), because correlation feedback hides level bias, scale compression, and rater disagreement.
- **H3 (`iii.b` Frontier harness optimization & 2x2 interaction):** Across four frontier judge harness architectures — `H1_single_call` (direct single-call native probability), `H2_ast_obligation_harness` (programmatic Python/Lean abstract syntax tree (AST) theorem and obligation pre-extractor scaffold), `H3_two_step_decompose_aggregate` (two-stage obligation mapper -> calibrated probability aggregator), and `H4_multi_family_ensemble_calibration_harness` (cross-family OpenAI + Mistral ensemble and calibration architectures, including two-stage `Pearson + calibrated CE head`) — choosing and fitting the harness by proper loss (③/④) achieves lower held-out test RPS and test CE than choosing and fitting by correlation (①/②), and combining proper-loss prompt optimization with proper-loss harness optimization dominates the 2x2 interaction table.
- **H4 (When proper loss beats correlation):** Held-out test RPS gains (`test RPS of correlation-trained judge minus test RPS of proper-loss-trained judge; positive favors proper loss`) grow with training task-family count `n_fam in {3, 5, 7, 9, 12, 15}` and are concentrated on artifacts with positive human rater disagreement (empirical standard deviation (SD) `> 0` / larger Dirichlet-smoothed information gap `G_i`). Detailed thresholds and abort rules are locked in [PROTOCOL.md](PROTOCOL.md).

## Directory structure

```text
experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/
├── .gitignore                        # Excludes private/ so raw human ratings and keys never enter Git
├── README.md                         # Goal, hypotheses, taxonomy (i, ii, iii.a, iii.b), method, status
├── PROTOCOL.md                       # Locked protocol + Measurement Contract table (committed pre-run)
├── HOW_TO_READ_RESULTS.md            # Data splits, metric definitions, subtraction order, uncertainty
├── FREEZE.json                       # Pre-run cryptographic SHA-256 manifest of protocol and scripts
├── CKPT_frontier_ce_vs_corr.md       # Timestamped resumable checkpoint (Rule 44)
├── cc.md                             # Self-contained runnable agent prompt
├── results.md                        # Live status ledger (Rule 37)
├── RESULTS_frontier_ce_vs_corr.md    # Canonical reader-facing report with tables and figure links
├── writing.tex                       # % Status: DRAFT paper subsection for AAAI-27 workshop draft
├── scripts/
│   ├── common.py                     # Hash audit, API clients (OpenAI + Mistral), $18.00 atomic ledger
│   ├── prompts.py                    # Zero-exposure prompts (P0, P1, P2, L0), iii.a proposers, H2/H3 scaffolds
│   ├── metrics_and_solvers.py        # RPS/CE/TVD, mean squared error (MSE) decomp, intraclass correlation (ICC), convex/cone & head solvers
│   ├── run_phase0_and_sanity.py      # Karpathy Steps 1 & 2.1–2.7: 225-hash audit, leave-one-rater-out (LORO) ceiling, controls, k=3 repeat SD
│   ├── run_arm_a_pool.py             # Multi-family native-prob + logprob frontier pool (7 models, all 75 artifacts)
│   ├── run_arm_b_prompt_search.py    # iii.a: Blind 4x2 iterative prompt optimization (Corr track vs CE track)
│   ├── run_arm_c_harnesses.py        # iii.b: Frontier judge harnesses (H1, H2, H3, H4) & 2x2 interaction
│   ├── analyze_all.py                # Grouped family splits (250 splits + Exp 02 5x5 CV), learning curves, G_i, restarts
│   ├── make_figures_and_report.py    # Publication-grade figures and RESULTS_frontier_ce_vs_corr.md builder
│   └── verify_results.py             # Independent deterministic recomputation and privacy/secret leak verifier
├── results/                          # Public JSON tables, judge_forecasts.json (zero human ratings), receipts
└── figures/                          # Publication-grade PNG plots with split-and-metric-named axes and gridlines
```

## Method (numbered procedure)

1. **Pre-register and lock (`PROTOCOL.md` + `FREEZE.json`):** Commit all predictions in the Measurement Contract table, split definitions, budgets (`HARD_CAP_USD = 18.00`), and script hashes before running measured evaluations.
2. **Karpathy sanity checks (`scripts/run_phase0_and_sanity.py`):** Verify all 225 input files against VeriBench experiment 100, compute leave-one-rater-out (LORO) human ceilings and dumb baselines at `$0`, then run 5-task smoke controls (`oracle`, `vacuous`, `blank`, `shuffled`, `label_in_prompt`) and `k=3` repeat stability (`15` artifacts) on `gpt-6.1-sol`.
3. **Multi-family native-probability frontier pool (`scripts/run_arm_a_pool.py`):** Run 15 zero-exposure frontier judges across 7 models in 2 families (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.1-terra`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.4-mini`; `codestral-2508`; the mini logprob configuration uses truncated top-five probabilities, not exact six-category probabilities) on all 75 artifacts, and compare directly against the 5 cached historical 3-sample scalar frontier judges (`H0_opus`, `H0_sol`, `H1_sol`, `H2_opus`, `H2_sol`).
4. **Axis `iii.a` — Blind agentic prompt optimization (`scripts/run_arm_b_prompt_search.py`):** On a locked grouped 3-way split of the 19 task families (`9` train / `5` validation / `5` held-out test), run two matched 4-iteration x 2-candidate prompt-optimization tracks (`gpt-6.1-sol` proposer and `gpt-5.4-mini` judge, identical word cap, strict feedback firewall: standardized correlation/rank residuals versus individual-rating CE/RPS losses) and evaluate Formulations ①, ②, ③, and ④ on held-out test families with and without post-hoc calibration.
5. **Axis `iii.b` — Frontier judge harness optimization & 2x2 interaction (`scripts/run_arm_c_harnesses.py`):** Execute `H1_single_call`, `H2_ast_obligation_harness`, `H3_two_step_decompose_aggregate`, and `H4_multi_family_ensemble_calibration_harness` across untuned `P0`, correlation-optimized prompts (`P_corr`), and proper-loss-optimized prompts (`P_ce`), comparing harness selection and parameter fitting under Formulations ①–④ and computing the full 2x2 `Prompt Objective x Harness Objective` decomposition.
6. **Learning curves, optimizer restarts, and disagreement diagnostics (`scripts/analyze_all.py`):** Evaluate all 7 objectives (`CE`, `RPS`, `Pearson`, `Spearman`, `Kendall`, `1/2 Pearson + 1/2 Spearman`, `Pearson + calibrated CE head`), 1-start (400 iterations) versus best-of-5 starts (2,000 iterations), training-size curves `n_fam in {3, 5, 7, 9, 12, 15}`, exact mean squared error (MSE) decomposition (`bias^2 + scale_mismatch^2 + 2 s_x s_y (1 - r)`), and per-artifact information-gap `G_i` stratification.
**Analysis & harness-key amendment (11-16-2026):** The implemented feature-head grid uses three inner folds and ridge c in {0.1, 1, 10}; its bands resample overlapping split averages and quantify conditional Monte Carlo variability, not task-family population uncertainty. In `phase4_arm_c_harnesses.json`, legacy keys `H2_ast_gpt61__*` and `H3_twostep_gpt61__*` executed `gpt-5.4-mini` (the optional `gpt-6.1-sol` Step-2 aggregator in `PROTOCOL_AMENDMENT_01.md` was omitted within the `$18.00` cap), and `H1_single_mistral35__*` / `H2_ast_mistral35__*` executed `codestral-2508`. Frozen pre-run plans remain in `PROTOCOL.md`; effective QA hashes are recorded in `FREEZE.json`.

7. **Deterministic verification (`scripts/verify_results.py`):** Independently recompute the native judges’ in-sample metrics, compare saved historical entries against the archived Experiment 02 parity reference, check mean squared error (MSE) identities and freeze hashes, and scan public artifacts for private ratings or secrets. Numerical differences must remain within the verifier’s tolerances; these checks do not independently reproduce all held-out fits or certify every optimizer optimum.

## Dependencies

- Read-only canonical private human ratings and cached scalar judges: `/Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/private/historical_exp06/cached_v1/dataset.json` (SHA-256 `b3a05699a26d3ffd9f28e793178811219885df78319b3aa57f9d393c68724ba0`).
- Read-only annotation-era Python, Gold Lean, and Candidate Lean artifacts: `/Users/brandomiranda/veribench/experiments/100_karpathy_recipe_tc_judge_search_human_train_val_test/inputs/`.
- Runtime-only in-scope API keys (never printed or committed): `~/keys/openai_bm_key_koyejolab.txt` (`koyejolab-openai-bm`) and `~/keys/mistral_api_key.txt` (`mistral`), capped at `HARD_CAP_USD = 18.00`.

## Status table (updated 11-16-2026)

| Step | Status | Notes |
|---|---|---|
| Pre-registration (`PROTOCOL.md`, `FREEZE.json`) | Done | Frozen in `FREEZE.json` (isolated pre-run worktree `aa09730`, integrated in `a346d5c`) on 11-16-2026 prior to paid runs (`PROTOCOL_AMENDMENT_01.md` documents zero-cost wire/endpoint remedies) |
| Phase 1: Karpathy sanity controls & repeat SD | Done | `79/79` parsed (`$0.50`), LIP `r=1.0000`, `oracle=1.000` vs negatives `0.000`, `k=3` repeat SD `0.0178` (intraclass correlation coefficient, `ICC=0.9957`) |
| Phase 2: Multi-family frontier pool (7 models) | Done | 15 native-prob/logprob judges (`OpenAI` + `Mistral`) on 75 items vs 5 historical scalar judges (`results/judge_forecasts.json`) |
| Phase 3 (`iii.a`): Blind prompt optimization | Done | 4 iterations x 2 candidates per track (`Track_Corr` ①/② vs `Track_CE` ③/④); `③ Loss-Val` held-out test `RPS = 0.1410` vs `① Corr-Val = 0.1828` |
| Phase 4 (`iii.b`): Frontier harness optimization & 2x2 | Done | `H1` (`gpt-6.1-sol`, `gpt-5.4-mini`, `codestral-2508`), `H2` (`gpt-5.4-mini`, `codestral-2508`), `H3` (`gpt-5.4-mini`), and `H4` cross-family harnesses (`$17.85` cumulative API spend `<= $18.00` cap) |
| Phase 5: Analysis, figures, report & verification | Done | `RESULTS_frontier_ce_vs_corr.md`, 5 PNG figures, `writing.tex`, and `verify_results.py` (`PASS`, numerical differences within tolerance, `0` detected leaks) |

**TLDR-end:** [judges-corr: experiment 10] Pre-registered experiment testing proper-loss (CE/RPS) versus correlation optimization on real multi-family frontier AI judges across both `iii.a` blind prompt optimization and `iii.b` harness optimization under an `$18.00` hard spend cap.
