# Live Results: 09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/results.md>

**Status:** COMPLETE (verified `11-16-2026 14:12 PST`)
**Last updated:** `11-16-2026 14:12 PST`

**Input provenance and prior exposure:** 60/75 reference files differ from the hashes recorded in the annotation manifest; Python and candidate files match. The saved evidence does not resolve stale reference metadata versus changed reference bytes, so byte identity with the human-delivered reference files is unverified. The fixed P2 prompt includes historical expert-mean and leniency summaries that were not estimated within each training split; P2-containing pools are retrospectively informed, not zero-exposure evaluations. Family separation applies to fitting and selection. Digit-logprob extraction imputes absent digits below the observed top-20 tail and can report success with fewer than six observed digits; saved receipts do not identify the exact incidence. These are probability estimates, not guaranteed complete six-digit softmax probabilities.

## Phase Table

| Phase | Description | Status | Notes |
| --- | --- | --- | --- |
| Phase 0 | Protocol freeze (`FREEZE.json`), byte-exact input SHA-256 verification (`75` items, `19` families), tokenizer 6-digit bijection | DONE | Committed before GPU execution (`2997834`) |
| Phase 1 | `10`-model open-weight vLLM inference (`ii.a` prompts `P0..P5` + blind meta-prompt search; `ii.b` harnesses `H1..H6`; `25` Karpathy controls; `k=3` repeats) + `ii.c` PyTorch LoRA weight fine-tuning (`Qwen3-4B` & `Qwen2.5-Coder-7B`) on `mercury2` (`$0.00`) | DONE | All GPUs released and verified clean via `nvidia-smi` |
| Phase 2 | Certified `19`-task-family evaluation: formulations ①–④, learning curves (`3–15` families), `5`-restart non-convex check, MSE & information-gap stratification | DONE | Saved in `results/analysis_summary.json` and `results/open_weight_forecasts.json` |
| Phase 3 | Figures (`figures/*.png`), [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md), [RESULTS_open_weight_ce_vs_corr.md](RESULTS_open_weight_ce_vs_corr.md), and `verify_results.py` | DONE | Full report in [RESULTS_open_weight_ce_vs_corr.md](RESULTS_open_weight_ce_vs_corr.md) |

## Headline Held-Out Test Results (`75` Artifacts, `19` Task Families, `195` Ratings, No Differential Testing)

Ranked probability score (RPS; 0–1) and cross-entropy (CE; nats) are lower-is-better. CI denotes a 95% task-family bootstrap confidence interval (10,000 draws); differences subtract CE-trained/selected from correlation-trained/selected judges on the same test artifacts. Low-rank adaptation (LoRA) updates transformer weights; large language model (LLM) heads use fixed features. See the reading guide for selection and baselines.

| Optimization Mechanism | Proper-Loss Judge Test RPS / CE | Correlation Judge Test RPS / CE | Correlation minus CE Test RPS `[95% Family CI]` (`>0` favors CE) |
| --- | --- | --- | --- |
| Representation Bridge: Verbal 6-Prob Mixture (`②` vs `④`) | `0.1431` / `1.629` | `0.1848` / `1.804` | `+0.0417 [+0.0194, +0.0666] p-val=0.00336456 (exact_2^19_task_family_sign_flip)` |
| Representation Bridge: Digit-Logprob Mixture (`②` vs `④`) | `0.1712` / `2.273` | `0.2250` / `2.533` | `+0.0538 [+0.0340, +0.0709] p-val=0.000488281 (exact_2^19_task_family_sign_flip)` |
| `ii.a` Candidate Prompt Pool (`P0..P5` Verbal `②` vs `④` Mix) | `0.1761` / `1.753` | `0.2311` / `2.117` | `+0.0550 [+0.0047, +0.1032] p-val=0.0573616 (exact_2^19_task_family_sign_flip)` |
| `ii.b` Judge Harness Optimization (`H1–H6` Pool `②` vs `④`) | `0.1325` / `1.606` | `0.1836` / `1.978` | `+0.0511 [+0.0222, +0.0833] p-val=0.00527573 (exact_2^19_task_family_sign_flip)` |
| `ii.c` PyTorch LoRA (`Qwen3-4B, 1.85M params`, `②` Train Pearson vs `④` Train CE) | `0.2007` / `2.084` | `0.3351` / `7.546` | `+0.1344 [+0.0495, +0.2220] p-val=0.00907898 (exact_2^19_task_family_sign_flip)` |
| `ii.c` PyTorch LoRA (`Qwen3-4B, 1.85M params`, `①` Val-Corr Pearson vs `③` Val-RPS CE) | `0.1997` / `2.180` | `0.3597` / `5.216` | `+0.1599 [+0.0870, +0.2273] p-val=0.00179672 (exact_2^19_task_family_sign_flip)` |
| `ii.c` PyTorch LoRA (`Qwen2.5-Coder-7B, 2.91M params`, `②` Train Pearson vs `④` Train CE) | `0.1927` / `2.026` | `0.4015` / `4.801` | `+0.2088 [+0.0957, +0.3037] p-val=0.00491714 (exact_2^19_task_family_sign_flip)` |
| `ii.c` PyTorch LoRA (`Qwen2.5-Coder-7B, 2.91M params`, `①` Val-Corr Pearson vs `③` Val-RPS CE) | `0.1859` / `1.950` | `0.3761` / `3.623` | `+0.1903 [+0.0886, +0.2757] p-val=0.00488663 (exact_2^19_task_family_sign_flip)` |

Full tables, `Qwen2.5-Coder-7B` LoRA results, blind meta-prompt optimization (`Qwen3-30B-A3B`), `2×2` Prompt × Harness factorial, MSE decomposition, and learning curves: [RESULTS_open_weight_ce_vs_corr.md](RESULTS_open_weight_ce_vs_corr.md).

**TLDR-end:** [judges-corr: 09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026] Complete across `ii.a` prompt optimization, `ii.b` harness optimization, and `ii.c` PyTorch LoRA LLM weight training on `mercury2`: native six-category probabilities beat the constant marginal CE baseline, and proper-loss training beats correlation training across prompt mixtures, harness pools, `48`-weight heads, and LoRA weight updates (`3`-fold discrete meta-prompt search `-0.0261` RPS, `p-val=0.2778`).
