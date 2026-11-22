# Checkpoint: Experiment 24 — Open-Weight LoRA Judge CE vs Pearson on HelpSteer2

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026/CKPT_helpsteer2_lora.md>

Created: 11-21-2026 17:02 PST
Last updated: 11-21-2026 17:02 PST

## Goal
- Measure at which human rating budget $N_{\text{train}}$ (and total development labels $B_{\text{total}} = N_{\text{train}} + N_{\text{val}} + N_{\text{cal}}$) real PyTorch low-rank adaptation (LoRA) weight fine-tuning of `Qwen/Qwen3-4B-Instruct-2507` with cross-entropy (CE) beats matched LoRA fine-tuning with Pearson correlation under identical post-hoc calibration on `nvidia/HelpSteer2` helpfulness ratings (`0–4`, prompt-disjoint splits, `$0.00` on `mercury2` GPUs).

## What ran
- Prepared `nvidia/HelpSteer2` (`990b2711a36180dd19d9c94b8627844866f8982a`) into prompt-disjoint pools (`train_pool=1,600`, `val_pool=800`, `cal_pool=800`, `test_set=1,038` across `519` prompts) and froze `PROTOCOL_11-21-2026.md` prior to any training or test evaluation.

## Results
- Pending training and calibration runs on `mercury2`.

## Stuck / blocked
- nothing

## Next steps
- Commit and push frozen protocol, send `[RUN START]` email, run sanity checks + prefix caching + 10-draw LoRA training sweep on `mercury2`, then calibrate and score locally.

## Brando to-do
- none

## Log
11-21-2026 17:02 PST -- Created Experiment 24 folder, prepared prompt-disjoint HelpSteer2 splits and 10 nested draw schedules (`results/data_manifest.json`), and wrote `PROTOCOL_11-21-2026.md` before any training or test scoring.
