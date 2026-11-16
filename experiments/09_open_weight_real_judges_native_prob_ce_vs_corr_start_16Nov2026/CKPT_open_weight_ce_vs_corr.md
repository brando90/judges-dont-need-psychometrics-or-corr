# Checkpoint: Open-Weight Real Judges Native-Probability CE vs Correlation

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/CKPT_open_weight_ce_vs_corr.md>

Created: 11-16-2026 10:35 PST
Last updated: 11-16-2026 14:57 PST

## Goal
- Test whether and when cross-entropy (CE) and ranked probability score (RPS) optimization beat correlation optimization on 10 real open-weight large language model (LLM) judges across `ii.a` prompt optimization, `ii.b` harness optimization, and `ii.c` real PyTorch low-rank adaptation (LoRA) neural weight updates over 19 task families (`$0.00` free GPUs on `mercury2`, no differential testing (no DT)).
## What ran
- Completed 10-model vLLM 0.11.0 inference across `P0..P5` prompts (`q_verbal` + `q_logprob`), `H3_ast` and `H4_two_call` harnesses, 25 Karpathy sanity controls, `k=3` repeat stability, and 3-fold blind iterative meta-prompt search on `Qwen3-30B-A3B`.
- Completed end-to-end PyTorch LoRA transformer weight fine-tuning (`ii.c`) on `Qwen3-4B-Instruct-2507` (`1,850,374` trainable params) and `Qwen2.5-Coder-7B-Instruct` (`2,905,094` trainable params) across 5 grouped task-family folds and 4 learning-curve sizes, plus `48`-weight hidden-state head learning curves (`3..15` task families) and 5-restart non-convex optimizer verification; verified all 10 `mercury2` GPUs released (`0%`).
- Integrated companion Experiment 10 (`ac1a9c9`, `$17.85 / $18.00`) and passed deterministic verifiers on both experiments; current receipts are in each `results/verification.json`.
## Results
- Native verbalized six-category mixtures have lower held-out cross-entropy than the historical scalar pool (`3.480` to `1.629` nats; calibrated digit-logprob mixture `1.606` nats, constant baseline `1.887` nats). Model/prompt pool composition also changes, so this does not isolate representation.
- Cross-entropy fits have lower measured held-out ranked probability score in the reported mixture, harness, head, and low-rank adaptation comparisons; uncertainty is method-specific. The verbal prompt-mixture comparison has `p=0.0574`, while discrete meta-prompt search (four validation families per fold) has `p=0.2778` and an interval containing zero. These are nominal exact task-family sign-flip tests; see the full report for paired 95% intervals and denominators.
- [Full results](RESULTS_open_weight_ce_vs_corr.md) and [reading guide](HOW_TO_READ_RESULTS.md) disclose historical prompt exposure, reference-byte uncertainty, scaffold misclassification, optimizer initialization/search choices, and split-loss versus averaged-forecast aggregation. No universal significance or rating threshold is established.
## Stuck / blocked
- nothing
## Next steps
- Complete Stage 3 local verification and record its review. Publication remains outside this review task.
## Brando to-do
- none

## Log
11-16-2026 10:35 PST -- Created isolated worktree `exp09-open-weight-1116`, verified inputs and `mercury2` vLLM 6-digit logprob extraction, froze `PROTOCOL.md`, dispatched Exp 10 subagent, and sent `[RUN START]` email.
11-16-2026 14:01 PST -- Finished all 10 `mercury2` vLLM models, `Qwen3-30B` meta-prompt search, and `Qwen3-4B` + `Qwen2.5-Coder-7B` PyTorch LoRA weight training; released all GPUs (`0%`); integrated Exp 10; passed `verify_results.py` (`391` checks); starting 3-stage Mega QA.

11-16-2026 14:57 PST -- Stage 3 review corrected remaining reporting and scaffold defects; historical forecasts and numerical analyses preserved. See qa/mega_qa_11-16-2026/qa3_codex_vals.md for final verification.
