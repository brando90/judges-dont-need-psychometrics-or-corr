# Experiment 28: ChaosNLI and DICES with five frontier judges, CE-fit vs Pearson-fit, break-even number of human ratings | 11-23-2026 | QA update 11-26-2026

**As of 11-26-2026:** read the [dated QA corrections and evidence limits](QA_CORRECTIONS_11-26-2026.md). Historical values and scored tables below are preserved.

Run by ultimate_brando9 for Brando Miranda. Orchestration and analysis: Claude Sonnet 5.5 in Claude Code, headless. Judges (Stanford AI API Gateway chat completions): GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3, DeepSeek V4.1 Flash. Task AT-3292; follows experiments 22 (non-circular protocol), 21 and 26 (same 27-transform head).

**Read first:** `report/REPORT_11-23-2026.pdf` (bottom line, derivation, tables, figures, caveats, catalog of 26 public rating sets). Same text: `report/REPORT_11-23-2026.md` and `.tex`. Catalog with checked facts: `CATALOG_11-23-2026.md`, `catalog/catalog_checked.json`.

Bottom line, in the report's words: CE-fit beats Pearson-fit and uniform on held-out RPS after 4 to 8 ratings (one per item) or 10 to 20 (five per item) in all 25 set-by-judge cells; Pearson-fit is significantly better in 19 of 325 cells, mostly at two training items. Against the raw judge, CE-fit wins sustained only in some cells, and Pearson-then-calibration is better than CE-fit on DICES from 64 ratings.

| Path | What |
|---|---|
| `PROTOCOL_11-23-2026.md` | frozen before any judge call and any label join |
| `requests.jsonl`, `manifest.json` | the 2,350 label-free judge requests (SHA-256 in the protocol) |
| `receipts/` | raw judge responses with token usage |
| `forecasts/` | parsed forecasts and costs, committed before any label join |
| `results/` | `scores_absolute.csv`, `contrasts.csv` (every comparator, with `_reverse` rows), `breakeven_summary.csv`, `metric_agreement.csv`, `comparator_wins.csv`, `correlations.csv`, `raw_judge_scores.csv`, `forecast_status.csv`, `verify_log_11-23-2026.txt` |
| `figures/` | learning curves (RPS, CE) and CE advantage; `ax.grid(True)` on every axis |
| `analysis/` | `collect_forecasts.py`, `lib28.py`, `run_analysis.py`, `summarize.py`, `verify.py` (independent loop check), `make_figures.py`, `build_tables.py`, `build_report.py` |
| `runners/` | request preparation and the judge runner |

Labels are not in the repository (sealed table kept off-repo); counts per item are public in the source datasets.

Reproduce: `collect_forecasts.py receipts requests.jsonl forecasts/forecasts_11-23-2026.csv forecasts/costs_11-23-2026.csv`, then `run_analysis.py forecasts/forecasts_11-23-2026.csv LABELS.json results` (about 14 minutes on 4 CPUs), `summarize.py results`, `make_figures.py results figures`, `build_tables.py .`, `build_report.py .`.

Cost: judge calls about $19.69 at configured list rates ($39.0 with a 1.4/4.4 stand-in for the open models), plus $1.34 in experiment 26: under the $80 stop of the shared $100 cap.
