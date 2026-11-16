# Mega QA of Experiments 09 (Open-Weight Real Judges) and 10 (Frontier AI Judges): Checkpoint

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/qa/mega_qa_11-16-2026/CKPT_mega_qa_11-16-2026.md>

**TLDR:** Brando Miranda requested a 3-stage sequential multi-model Mega QA ("At the end do a mega qa... with a different model of course!") on 11-16-2026 covering Experiment 09 (`ii.a` prompt optimization, `ii.b` harness optimization, `ii.c` PyTorch low-rank adaptation (LoRA) transformer weight updates on open-weight judges) and Experiment 10 (`iii.a` blind prompt optimization and `iii.b` frontier harness optimization on frontier artificial intelligence (AI) judges) plus `experiments/README.md`. Nothing lands on `origin/main` until Stage 3 accepts and deterministic verifiers pass.

Created: 11-16-2026 12:51 PST
Last updated: 11-16-2026 12:51 PST

## State
- **Coordinator / builder:** Antigravity `Gemini 4 Carbon (High)` (parent session `7fa0b312-b867-45d4-b095-82ed9c135b7b` + subagent `4b9ddc42-4d76-48d9-abc4-15e7b8f53abc` for Experiment 10). Sole email notifier: parent coordinator (`RUN START` sent 11-16-2026; `RUN DONE` pending after landing).
- **Isolated integration worktree:** `/Users/brandomiranda/.worktrees/judges-dont-need-psychometrics-or-corr/exp09-open-weight-1116`
- **Original base commit (`QA_BASE`):** `233d5047d20da8591b4609ab52242ceed8717152`
- **Owned paths (`QA_PATHS`):**
  - `experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026`
  - `experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026`
  - `experiments/README.md`
- **Stages (sequential, each reviewing the improved tree from `QA_BASE`):**
  - **Stage 1 (`qa1_codex_vals.md`):** PENDING — OpenAI Codex CLI (`gpt-6-astra`, `ultra`, `CODEX_HOME=~/.codex-vals`).
  - **Stage 2 (`qa2_*.md`):** PENDING — Anthropic Claude CLI (`claude-fable-5-1` / `claude-opus-5-5`, `max`) or builder review (`Gemini 4 Carbon (High)`) per `~/agents-config/workflows/qa-correctness.md`.
  - **Stage 3 (`qa3_codex_vals.md`):** PENDING — Fresh-context OpenAI Codex CLI (`gpt-6-astra`, `ultra`, `CODEX_HOME=~/.codex-vals`).

## Next steps
1. Collect final `lora_weight_training_q3_4b_2507.json`, `lora_weight_training_q25c_7b.json`, and `meta_prompt_opt_q3_30b.json` from `mercury2`, verify all `mercury2` GPUs are released (`0%`), and run Experiment 09's `analyze_open_judges.py`, `make_plots.py`, `make_reports.py`, and `verify_results.py`.
2. Commit pre-QA state in the isolated integration worktree and run Stage 1 -> Stage 2 -> Stage 3 sequentially, verifying deterministically after every stage.
3. Rebase/fast-forward onto `origin/main`, push, clean up worktrees, log to `~/agent-sync/coding/log.md`, and send `[RUN DONE]` email.

**TLDR-end:** [judges-corr: mega QA 11-16-2026 checkpoint] Tracks the 3-stage cross-company Mega QA chain for Experiments 09 and 10 prior to landing on `main`.
