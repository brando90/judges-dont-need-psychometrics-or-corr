# Checkpoint: Open-Weight Real Judges Native-Probability CE vs Correlation

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/CKPT_open_weight_ce_vs_corr.md>

Created: 11-16-2026 10:35 PST
Last updated: 11-16-2026 14:01 PST

## Goal
- Test whether and when cross-entropy (CE) and ranked probability score (RPS) optimization beat correlation optimization on 10 real open-weight large language model (LLM) judges across `ii.a` prompt optimization, `ii.b` harness optimization, and `ii.c` real PyTorch low-rank adaptation (LoRA) neural weight updates over 19 task families (`$0.00` free GPUs on `mercury2`, no differential testing (no DT)).
## What ran
- Completed 10-model vLLM 0.11.0 inference across `P0..P5` prompts (`q_verbal` + `q_logprob`), `H3_ast` and `H4_two_call` harnesses, 25 Karpathy sanity controls, `k=3` repeat stability, and 3-fold blind iterative meta-prompt search on `Qwen3-30B-A3B`.
- Completed end-to-end PyTorch LoRA transformer weight fine-tuning (`ii.c`) on `Qwen3-4B-Instruct-2507` (`1,850,374` trainable params) and `Qwen2.5-Coder-7B-Instruct` (`2,905,094` trainable params) across 5 grouped task-family folds and 4 learning-curve sizes, plus `48`-weight hidden-state head learning curves (`3..15` task families) and 5-restart non-convex optimizer verification; verified all 10 `mercury2` GPUs released (`0%`).
- Integrated companion Experiment 10 (`ac1a9c9`, `$17.85 / $18.00`) and passed deterministic verifiers on both experiments (`391` checks in Exp 09; `0` discrepancies / `0` leaks in Exp 10).
## Results
- Native verbalized 6-category probabilities drop open-weight CE mixture held-out test CE from `3.480` nats (`Exp 14` 3-sample scalar) to `1.629` nats (`1.606` nats on temperature-calibrated logprob harnesses), beating the `1.887`-nat constant baseline.
- Proper-loss optimization beats correlation optimization across `ii.a` prompt mixtures (`+0.0550` verbal / `+0.0873` logprob RPS), `ii.b` harness mixtures (`+0.0511` RPS, `p=0.0053`; `2x2` joint `+0.1062` RPS, `p=0.0170`), `ii.c` `48`-weight hidden-state heads (`+0.0426` RPS at 15 families, `p=0.0226`, significant from 5 families / ~52 ratings), and `ii.c` PyTorch LoRA weight updates (`+0.1344` RPS on `Qwen3-4B`, `p=0.0091`; `+0.2088` RPS on `Qwen2.5-Coder-7B`, `p=0.0049`), while 3-fold uncalibrated discrete meta-prompt search on 12 validation items is noisy (`-0.0261` RPS, `p=0.2778`).
## Stuck / blocked
- nothing
## Next steps
- [p 10/10] Run 3-stage cross-model Mega QA (`Codex gpt-6-astra ultra` -> Stage 2 -> fresh `Codex gpt-6-astra ultra`), land on `origin/main`, update `agent-sync`, and send `[RUN DONE]` email.
## Brando to-do
- none

## Log
11-16-2026 10:35 PST -- Created isolated worktree `exp09-open-weight-1116`, verified inputs and `mercury2` vLLM 6-digit logprob extraction, froze `PROTOCOL.md`, dispatched Exp 10 subagent, and sent `[RUN START]` email.
11-16-2026 14:01 PST -- Finished all 10 `mercury2` vLLM models, `Qwen3-30B` meta-prompt search, and `Qwen3-4B` + `Qwen2.5-Coder-7B` PyTorch LoRA weight training; released all GPUs (`0%`); integrated Exp 10; passed `verify_results.py` (`391` checks); starting 3-stage Mega QA.
