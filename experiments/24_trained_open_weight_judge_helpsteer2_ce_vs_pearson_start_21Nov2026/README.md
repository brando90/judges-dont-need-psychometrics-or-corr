# Experiment 24 (`24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026`): Trained Open-Weight LoRA Judge (CE vs Pearson) on HelpSteer2 (11-21-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026/README.md>

**Bottom line (11-21-2026):** Updating real `Qwen/Qwen3-4B-Instruct-2507` LoRA weights (`1,847,813` trainable parameters) with cross-entropy (CE) beats matched Pearson correlation LoRA weight training on held-out ranked probability score (RPS) under identical post-hoc calibration on `nvidia/HelpSteer2` (`1,038` responses / `519` prompts) at sustained threshold **B_total=512 (N_train=256)** (two-way crossed threshold: **None within tested budgets**). At `B_total = 2,048` (`N_train = 1,024`, `N_val = 512`, `N_cal = 512`), calibrated CE LoRA reaches test RPS **0.15200** vs **0.15309** for calibrated Pearson LoRA (paired difference = **+0.00110** `[+0.00017, +0.00203]`, `p-val=0.0177`, **9/10** draws won) and **+0.00374** better RPS than the calibrated unmodified base judge on `N_cal` labels (`0.15573` -> `0.15200`).

## Files
- [RESULTS_11-21-2026.md](RESULTS_11-21-2026.md): Full verdict, budget learning curve, calibration comparison, and provenance.
- [BUDGET_TABLES_11-21-2026.md](BUDGET_TABLES_11-21-2026.md): Every budget table (`d=1` and `d=2`), CE/TVD/bias/ranking diagnostics, and crossed uncertainty.
- [PROTOCOL_11-21-2026.md](PROTOCOL_11-21-2026.md): Frozen pre-run protocol (SHA-256 in `results/protocol.sha256`).
- [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md): Metric definitions, splits, calibration, and optimizer disclosures.
- [CKPT_helpsteer2_lora.md](CKPT_helpsteer2_lora.md): Run checkpoint and verification log.
- `scripts/`: `prepare_helpsteer2.py`, `train_lora_helpsteer2.py`, `calibrate_and_score.py`, `plot_and_report.py`, `verify.py`, `send_run_email.py`.
- `figures/`: `helpsteer2_lora_learning_curves.{png,pdf}`, `helpsteer2_calibration_and_depth.{png,pdf}`.
