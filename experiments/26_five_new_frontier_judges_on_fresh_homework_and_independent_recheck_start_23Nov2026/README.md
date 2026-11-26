# Experiment 26: five new frontier judges on fresh homework answers, plus an independent recheck | 11-23-2026 | QA update 11-26-2026

**As of 11-26-2026:** read the [dated QA corrections and evidence limits](QA_CORRECTIONS_11-26-2026.md). Historical values and scored tables below are preserved.

**TLDR:** [Experiment 21](../21_fresh_task_families_matched_open_closed_judges_start_21Nov2026/README.md) already answered the layer-4 question for two judges. This adds five frontier judges (closed: GPT-6 Astra, Claude Opus 5.5; open weights: Kimi K3, GLM 5.3, DeepSeek V4.1 Flash) on the identical 160 UNT prompts, and a second implementation of the analysis that matches experiment 21's numbers to 3.5e-11. CE fit has lower held-out RPS than Pearson fit in 35 of 35 judge-by-budget cells; the pre-registered descending test passes for GPT-6 Astra and Kimi K3 (256 down to 8 training ratings) and fails at 256 for the other three. At 4 training ratings CE is not significantly better in RPS for four of five judges. New spend about $1.34 of the $100 cap.

As of 11-23-2026. Harness and models: judge calls through the Stanford AI API Gateway (one call per answer, same prompts for every judge); analysis in ub9sc's Python (numpy/scipy), cross-checked against Codex's experiment 21 code.

## Read first

- [`METHOD_AND_RESULTS_11-23-2026.pdf`](METHOD_AND_RESULTS_11-23-2026.pdf) (`.tex`, `.md` alongside): what the result does and does not show (including that Pearson sees only the per-answer mean), the derivations, results, caveats, all tables.
- [`HOW_TO_READ_RESULTS.md`](HOW_TO_READ_RESULTS.md): data, metrics, sign convention, intervals, how fits were chosen.
- [`PROTOCOL_11-23-2026.md`](PROTOCOL_11-23-2026.md) and [`AMENDMENT_1_11-23-2026.md`](AMENDMENT_1_11-23-2026.md): frozen before any grade was read.

## Headline (two ratings per answer, held-out RPS, lower is better)

| judge | CE fit at 256 | Pearson fit at 256 | uniform | formal descending test (alpha 0.01) |
|---|---|---|---|---|
| gpt-6-astra (closed) | 0.0660 | 0.0843 | 0.2497 | passes 256 to 8, fails at 4 |
| claude-opus-5-5 (closed) | 0.0639 | 0.0675 | 0.2497 | fails at 256 |
| kimi-k3 (open) | 0.0712 | 0.0996 | 0.2497 | passes 256 to 8, fails at 4 |
| glm-5.3 (open) | 0.0757 | 0.0952 | 0.2497 | fails at 256 (one-sided bound -0.0021) |
| deepseek-v4.1-flash (open) | 0.0827 | 0.0867 | 0.2497 | fails at 256 |

Every case where the Pearson fit is not beaten is in Table 4 of the PDF, including held-out CE and TVD at 4 ratings and TVD for Claude Opus 5.5 at 256 and 128.

## Files

- `runners/run_judges.py`, `analysis/collect_forecasts.py`, `analysis/recheck.py` (independent implementation), `analysis/report.py` (tables and figure).
- `receipts/` raw call receipts; `forecasts/` parsed forecasts and cost table.
- `results/new5/` and `results/exp21_recheck/`: `scores.csv`, `contrasts.csv`, `decisions.csv`; `results/tables_11-23-2026.md`, `results/fig_learning_curves_11-23-2026.png`.

## Not covered

Fresh VeriBench ratings (not yet collected); judges fine-tuned rather than calibrated; other corpora. One course, two graders, ten dependent folds, one run per judge.
