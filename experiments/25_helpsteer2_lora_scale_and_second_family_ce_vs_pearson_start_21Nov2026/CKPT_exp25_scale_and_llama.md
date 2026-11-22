# Checkpoint: Experiment 25 — HelpSteer2 LoRA Scale (`B_total <= 8,192`) and Llama-3.1-8B Replication

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026/CKPT_exp25_scale_and_llama.md>

Created: 11-21-2026 20:54 PST
Last updated: 11-21-2026 20:54 PST

**TLDR:** Running Experiment 25 Study A (`Qwen/Qwen3-4B-Instruct-2507`, `20` draws, `B_total` `512–8,192`) and Study B (`unsloth/Llama-3.1-8B-Instruct`, `10` draws, `B_total` `16–2,048`) on `mercury2` (`$0.00`), followed by paper/one-pager integration and `agy` Neon Orchestrator Light QA.

## Goal
- Test two-way crossed (`draw x prompt`) RPS superiority up to `B_total = 8,192` (`20` draws) on `Qwen3-4B-Instruct-2507` and replicate CE-vs-Pearson LoRA thresholds on `unsloth/Llama-3.1-8B-Instruct`.

## What ran
- Prepared `9,230`-item master HelpSteer2 bundle (`results/data_manifest.json`) and froze `PROTOCOL_11-21-2026.md` prior to any training or test scoring.

## Results
- Pending sanity checks and GPU runs on `mercury2`.

## Stuck / blocked
- nothing

## Next steps
- Commit & push frozen protocol, send `[RUN START]`, build & sanity-check batched `train_exp25_lora.py`, run Study A then Study B on `mercury2`.

## Brando to-do
- none

## Log
11-21-2026 20:54 PST -- Created Experiment 25 bundle (`9,230` items, SHA-256 `1f7046e0441a532a...`) and froze `PROTOCOL_11-21-2026.md` before any training run.
