# Experiment 25 (`25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026`): HelpSteer2 Scale & Llama-3.1-8B Replication (11-21-2026) | QA update 11-26-2026

**As of 11-26-2026:** read the [dated QA corrections and evidence limits](QA_CORRECTIONS_11-26-2026.md). Historical values and scored tables below are preserved.

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026/README.md>

**Bottom line (11-21-2026):** In Study A (`Qwen3-4B-Instruct-2507`, `20` draws, `B_total` up to `8,192`), calibrated cross-entropy (CE) low-rank adaptation (LoRA) reaches held-out ranked probability score (RPS) **0.15080** vs **0.15089** for calibrated Pearson LoRA at `B_total = 8,192` (diff = **+0.00009**, cond. sim. threshold **None within tested budgets**, crossed sim. threshold **None within tested budgets**). In Study B (`unsloth/Llama-3.1-8B-Instruct`, `10` draws, `B_total = 16..2,048`), calibrated CE LoRA reaches **0.16005** vs **0.15893** at `B_total = 2,048` (diff = **-0.00112**, cond. sim. threshold **None within tested budgets**, crossed sim. threshold **None within tested budgets**).

## Files
- [RESULTS_11-21-2026.md](RESULTS_11-21-2026.md): Full verdict, tables, cross-family figures, sanity & provenance summary.
- [BUDGET_TABLES_11-21-2026.md](BUDGET_TABLES_11-21-2026.md): Complete Study A & Study B tables across RPS, CE, TVD, bias, and pair accuracy.
- [PROTOCOL_11-21-2026.md](PROTOCOL_11-21-2026.md): Pre-registered protocol (SHA-256 in `results/protocol.sha256`).
- [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md): Metric definitions, splits, calibration, and optimizer disclosures.
- [CKPT_exp25_scale_and_llama.md](CKPT_exp25_scale_and_llama.md): Execution and verification checkpoint.
- `scripts/`: `prepare_exp25_bundle.py`, `train_exp25_lora.py`, `calibrate_and_score_exp25.py`, `plot_and_report_exp25.py`, `verify_exp25.py`, `send_run_email.py`, `sync_gdoc_and_drive_exp25.py`.
- `figures/`: `exp25_qwen3_4b_scale_curves.{png,pdf}`, `exp25_llama31_8b_and_cross_family.{png,pdf}`.
