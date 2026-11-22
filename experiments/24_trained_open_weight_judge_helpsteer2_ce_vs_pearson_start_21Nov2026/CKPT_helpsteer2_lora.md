# Checkpoint: Experiment 24 — Open-Weight LoRA Judge CE vs Pearson on HelpSteer2

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026/CKPT_helpsteer2_lora.md>

Created: 11-21-2026 17:02 PST
Last updated: 11-21-2026 20:33 PST

## Goal
- Measure at which human rating budget $N_{\text{train}}$ (and $B_{\text{total}} = N_{\text{train}} + N_{\text{val}} + N_{\text{cal}}$) real PyTorch low-rank adaptation (LoRA) weight fine-tuning of `Qwen/Qwen3-4B-Instruct-2507` with cross-entropy (CE) beats matched LoRA fine-tuning with Pearson correlation under identical post-hoc calibration on `nvidia/HelpSteer2` (`0–4`, prompt-disjoint `1,038` test responses / `519` prompts, `$0.00` on `mercury2`).

## What ran
- Completed Phase A (`--phase sanity`, 0.0 full-vs-cached equivalence diff, 5/5 literal compliance, `0.160` blank mean), Phase B (`--phase extract`, `4,238` items cached on GPUs `0..9`), Phase C (`--phase train`, all `10` draws x `11` budgets/depths x `2` objectives = `220` runs, `exit=0`), Phase D merge (`predictions.npz` + `training_provenance.json`), local calibration (`calibrate_and_score.py`), and figure/report generation (`plot_and_report.py`).

## Results
- Calibrated CE-trained LoRA beats calibrated Pearson-trained LoRA across simultaneous conditional 95% bands from `B_total=512 (N_train=256)` onward on primary held-out ranked probability score (RPS); at `B_total=2,048 (N_train=1,024)`: `0.15200` vs `0.15309` (`+0.00110 [+0.00017, +0.00203]`, `p-val=0.0177`, `9/10` draws; CE `10/10` draws; TVD `10/10` draws), and beats calibrated base `Qwen3-4B` (`base_calibrated_all_dev` `0.15514`) at `B_total=2,048` (`+0.00315`). Pearson point-estimate RPS was slightly lower only at noisy tiny budgets `B_total in [16, 32]`.

## Stuck / blocked
- nothing

## Next steps
- Completed all planned training, calibration, figure verification, independent `verify.py` checks, Drive Desktop sync, and GitHub push.

## Brando to-do
- Optional: refresh Google Drive OAuth (`~/keys/gspread_authorized_user.json` or Antigravity `google-drive` MCP) and re-run `scripts/sync_gdoc_and_drive.py` if a native Google Doc copy is wanted alongside the synced `.md`/`.html`/`.png`/`.csv`/`.zip` files in Drive folder `1QKHpBRLFOaYIw5IHwv2yAqZaHvBq-moG`.

## Log
11-21-2026 17:02 PST -- Created Experiment 24 folder, prepared prompt-disjoint HelpSteer2 splits and 10 nested draw schedules (`results/data_manifest.json`), and wrote `PROTOCOL_11-21-2026.md` before any training or test scoring.
11-21-2026 20:33 PST -- Completed all 10 parallel GPU draws (`220` runs, `$0.00`), calibrated and scored `1,038` held-out responses (`RESULTS_11-21-2026.md`), and verified figure clarity.
