# Live Results: 09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/results.md>

**Status:** RUNNING (started 11-16-2026 10:35 PST)
**Last updated:** 11-16-2026 10:35 PST

## Phase Table

| Phase | Description | Status | Notes |
| --- | --- | --- | --- |
| Phase 0 | Protocol freeze, byte-exact input SHA-256 verification (`75` items, `19` families), tokenizer 6-digit bijection check | DONE | All 10 models verified 1-to-1 single-token `0..5`; `dataset.json` SHA-256 `b3a05699...` verified |
| Phase 1 | Karpathy sanity controls (`oracle`, `vacuous`, `blank`, `shuffled`, `label_in_prompt`) + `10`-model × `3`-prompt native-probability (`q_logprob`, `q_verbal`) vLLM inference on `mercury2` (`$0` free GPUs) | IN PROGRESS | Launching across GPUs `0..7` on `mercury2.stanford.edu` |
| Phase 2 | Certified task-family evaluation: formulations ①–④, all 6 objectives + `Pearson + calibrated CE head`, 5-restart optimizer check, training-size learning curves, MSE & information-gap stratification, Prompts-vs-Weights 2×2 | PENDING | Runs after Phase 1 outputs transfer back |
| Phase 3 | Plots, `HOW_TO_READ_RESULTS.md`, `RESULTS_open_weight_ce_vs_corr.md`, deterministic `verify_results.py`, and 3-stage Mega QA | PENDING | Runs after Phase 2 |

**TLDR-end:** [judges-corr: 09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026] Protocol frozen and input/tokenizer checks passed; launching 10-model open-weight native-probability inference on `mercury2`.
