# Checkpoint: Experiment 25 — HelpSteer2 LoRA Scale (`B_total <= 8,192`) and Llama-3.1-8B Replication

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026/CKPT_exp25_scale_and_llama.md>

Created: 11-21-2026 20:54 PST
Last updated: 11-22-2026 11:40 PST

**TLDR:** Completed Experiment 25 Study A (`Qwen/Qwen3-4B-Instruct-2507`, `20` draws, `B_total` `512–8,192`) and Study B (`unsloth/Llama-3.1-8B-Instruct`, `10` draws, `B_total` `16–2,048`) on `mercury2` (`$0.00`), verified all provenance and metric invariants (`verify_exp25.py`), and synced results to Google Drive Desktop.

## Goal
- Test two-way crossed (`draw x prompt`) RPS superiority up to `B_total = 8,192` (`20` draws) on `Qwen3-4B-Instruct-2507` and replicate CE-vs-Pearson LoRA thresholds on `unsloth/Llama-3.1-8B-Instruct`.

## What ran
- Prepared `9,230`-item master HelpSteer2 bundle (`results/data_manifest.json`) and froze `PROTOCOL_11-21-2026.md` (`41fa0d9`) prior to any training or test evaluation.
- Verified live full-model prefix vs cached `H^(split)` 2-step batched (`m=4`) `AdamW` and Epoch-0 HF wrapper equivalence (`0.00` max prob diff on both models; `bfloat16` GEMV-vs-GEMM batching drift `<= 7.3e-3`) and sanity checks (`qwen3_4b_sanity.json` `1,847,813` trainable params; `llama31_8b_sanity.json` `2,641,925` trainable params due to grouped-query attention `k_proj`/`v_proj` dimensions — noted here while keeping frozen `PROTOCOL_11-21-2026.md` unchanged; Llama 1-batch Pearson sanity swept pre-registered `LRS` and hit `0.6246` at `lr=2e-4` vs `0.5494` saturated at `lr=8e-4`).
- Ran Study A (`20` draws x `5` budgets x `2` objectives = `200` runs) and Study B (`10` draws x `8` budgets x `2` objectives = `160` runs) across 10 `mercury2` RTX A4000 GPUs; transferred & SHA-256 verified NPZs (`10.10 MB` and `4.09 MB`); cleaned `qwen_cache` and `llama_cache`; ran `calibrate_and_score_exp25.py` and `plot_and_report_exp25.py`.

## Results
- **Study A (`Qwen3-4B-Instruct-2507`, `20` draws):** Without calibration, raw CE LoRA crushes raw Pearson LoRA at all 5 budgets (e.g., `B=8,192`: RPS `0.15119` vs `0.17090`, crossed sim. 95% CI `[+0.01509, +0.02432]`, `p=0.0002`, `20/20` draws). After fitting the shared proper-loss 7-param CE calibrator on `S_cal`, calibrated CE LoRA maintains lower RPS at all `5/5` budgets (`0.15080` vs `0.15089` at `B=8,192`, `11/20` draws, approaching leave-one-vote-out human reference `0.14567`, beating `base_calibrated_all_dev` `0.15481` with crossed `p=0.0002`), wins `16/20` draws on CE (`1.3231` vs `1.3285`) and `16/20` on TVD (`0.4558` vs `0.4580`), and wins `18/20` on both CE (`cp=0.0342`) and TVD (`cp=0.0052`) at `B=2,048`, while calibrated RPS narrows as `N_cal -> 2,048` and does not clear strict two-way crossed `draw x prompt` simultaneous 95% (`N* = None`).
- **Study B (`unsloth/Llama-3.1-8B-Instruct`, `10` draws):** Uncalibrated raw CE LoRA again beats raw Pearson LoRA on mean RPS at all `8/8` budgets (crossed `p=0.0002` from `B=32` onward, `10/10` draws for `B=64..2,048`; `0.16031` vs `0.19409` at `B=2,048`, crossed `95% CI [+0.01477, +0.05280]`, `10/10` draws). After post-hoc CE calibration on `S_cal`, neither objective clears simultaneous significance on RPS (`crossed p >= 0.68`; `-0.00112 [-0.00410, +0.00185]`, `4/10` CE wins on RPS and `6/10` on CE/TVD at `B=2,048`), showing backbone strength and small-`N_cal` calibration noise mediate calibrated differences.

## Stuck / blocked
- nothing

## Next steps
- Run `verify_exp25.py`, sync Drive Desktop, update paper + one-pager v28, run `agy` Neon Orchestrator Light QA, push to `main`, clean `/tmp`, send `[RUN DONE]` email.

## Brando to-do
- none

## Log
11-21-2026 20:54 PST -- Created Experiment 25 bundle (`9,230` items, SHA-256 `1f7046e0441a532a...`) and froze `PROTOCOL_11-21-2026.md` before any training run.
11-22-2026 11:40 PST -- Completed Study A (`20` draws, `B_total <= 8,192`) and Study B (`10` draws, `Llama-3.1-8B-Instruct`, `B_total <= 2,048`), merged & verified outputs, cleared `mercury2` caches, and ran `calibrate_and_score_exp25.py` + `plot_and_report_exp25.py`.
