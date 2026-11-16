# Mega QA of Experiments 09 and 10: Builder Notes

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/qa/mega_qa_11-16-2026/BUILDER_NOTES.md>

**TLDR:** Records the 3-stage cross-company Mega QA chain (`Codex gpt-6-astra ultra` -> Stage 2 reviewer -> fresh `Codex gpt-6-astra ultra`), exact model/effort/account provenance, resolution of every material finding, and post-QA deterministic verification for Experiments 09 and 10 on 11-16-2026.

## Chain (In Progress)

| Stage | Reviewer (harness, model, effort, account) | Time (PST) | Findings: critical / major / minor | Verdict | Commit |
| --- | --- | --- | --- | --- | --- |
| Builder | Antigravity `Gemini 4 Carbon (High)` (parent + subagent `4b9ddc42`) | 10:35–13:05 | Pre-QA self-audit fixed `verify_results.py` assertion, `make_reports.py` column alignment, and `RESULTS_frontier_ce_vs_corr.md` null-contrast wording | — | pending |

**TLDR-end:** [judges-corr: mega QA 11-16-2026 builder notes] Tracks reviewer provenance, finding resolutions, and deterministic verification across Experiments 09 and 10.
