# Mega QA of Experiments 09 (Open-Weight Real Judges) and 10 (Frontier AI Judges): Checkpoint

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/qa/mega_qa_11-16-2026/CKPT_mega_qa_11-16-2026.md>

**TLDR:** Completed the 3-stage sequential cross-company Mega QA chain (`OpenAI Codex gpt-6-astra ultra` -> `Gemini 4 Carbon High` -> fresh `OpenAI Codex gpt-6-astra ultra`) on 11-16-2026 covering Experiment 09, Experiment 10, and `experiments/README.md` from base `233d5047d20da8591b4609ab52242ceed8717152`. Final verdict: **ACCEPT (`CRITICAL_ISSUES: 0`)**; ready to land on `origin/main`.

Created: 11-16-2026 12:51 PST
Last updated: 11-16-2026 15:04 PST

## State
- **Coordinator / builder:** Antigravity `Gemini 4 Carbon (High)` (parent `7fa0b312-b867-45d4-b095-82ed9c135b7b` + Exp 10 subagent `4b9ddc42-4d76-48d9-abc4-15e7b8f53abc`).
- **Original base commit (`QA_BASE`):** `233d5047d20da8591b4609ab52242ceed8717152`
- **Stages:**
  - **Stage 1 (`qa1_codex_vals.md` + `../full_base_review_11-16-2026/REVIEW.md`):** COMPLETE (`c20517c`) — OpenAI Codex CLI `v0.169.0`, `gpt-6-astra`, `ultra` (`CODEX_HOME=~/.codex-vals`), `0` critical, `13` major fixed.
  - **Stage 2 (`qa2_carbon.md`):** COMPLETE (`9546ab0`) — Fresh Antigravity `Gemini 4 Carbon (High)` subagent `24c24eed`, `0` critical, `4` major fixed (`916` checks passing).
  - **Stage 3 (`qa3_codex_vals.md`):** COMPLETE — Fresh OpenAI Codex CLI `v0.169.0`, `gpt-6-astra`, `ultra` (`CODEX_HOME=~/.codex-vals`, session `01a1e68f`), `0` critical, `6` major fixed (`917` checks passing in Exp 09; `14` freeze hashes & `28` status matrices passing in Exp 10).

## Log
- 11-16-2026 14:01 PST -- Pre-QA snapshot committed (`a346d5c`); Stage 1 Codex `gpt-6-astra` `ultra` launched.
- 11-16-2026 14:20 PST -- Stage 1 finished (`FIXED`, 13 fixes); verified and committed (`c20517c`). Attempted `claude-fable-5-1` (`~/.claude-su` expired; `~/.claude` Fable limit until 23:00 PST); dispatched fresh `Gemini 4 Carbon (High)` for middle builder-review Stage 2 per `qa-correctness.md`.
- 11-16-2026 14:51 PST -- Stage 2 finished (`FIXED`, 4 fixes); verified and committed (`9546ab0`); launched fresh Stage 3 Codex `gpt-6-astra` `ultra`.
- 11-16-2026 15:04 PST -- Stage 3 finished (`FIXED`, 6 fixes, `0` critical); builder verification passed across both experiments; landing on `origin/main`.

**TLDR-end:** [judges-corr: mega QA 11-16-2026 checkpoint] All 3 Mega QA stages complete and verified (`0` critical issues); ready for landing on `origin/main`.
