# Experiment 10 Agent Runbook — Frontier AI Judges: Native Probabilities, Prompt Search (`iii.a`), and Harness Search (`iii.b`)

**TLDR:** Execute the pre-registered Experiment 10 pipeline (`10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026`) testing whether and when proper-loss optimization (cross-entropy (CE) and ranked probability score (RPS)) beats correlation optimization on real multi-family frontier AI judges (OpenAI GPT-6/5.6/5.4 + Mistral Medium/Magistral/Codestral) across both `iii.a` blind prompt optimization and `iii.b` judge harness optimization (`H1`–`H4`). Pass/fail requires passing the Phase 1 Karpathy sanity gates (`label_in_prompt` `r >= 0.95`, `oracle > degenerate` by `>= +0.50`), completing all declared 75-artifact evaluations within `HARD_CAP_USD = 18.00`, and passing `scripts/verify_results.py` with zero numerical discrepancies and zero leaked private ratings or credentials. Deliverables are `results/judge_forecasts.json`, `RESULTS_frontier_ce_vs_corr.md`, `figures/*.png`, `writing.tex`, and `results/verification.json`. Never commit raw human ratings (`private/`) or print API keys (`~/keys/`).

Read `~/agents-config/INDEX_RULES.md` and `PROTOCOL.md` before running any step.

> Solve the entire assigned task, including every required file, question and subtask. Produce the complete required deliverable in the specified output location or response format; an outline, partial draft, progress report or claim of completion is not a substitute. Write/save the output, inspect the actual saved artifact or final response, and run the allowed checks, tests or compilation required by the task. Within the declared time, token, call and tool limits, continue working and fixing errors until the requirements are met or a declared terminal condition is reached. Follow the fixed continuation procedure without resetting budgets. If anything remains unresolved, preserve the best current deliverable and report the exact remaining failures and checks that did not pass; never claim success or invent verification.

## Execution sequence

```bash
cd experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026
python3 scripts/run_phase0_and_sanity.py
python3 scripts/run_arm_a_pool.py
python3 scripts/run_arm_b_prompt_search.py
python3 scripts/run_arm_c_harnesses.py
python3 scripts/analyze_all.py
python3 scripts/make_figures_and_report.py
python3 scripts/verify_results.py
```

TL;DR: Runs the locked 5-phase frontier-judge evaluation (OpenAI + Mistral APIs under an `$18.00` atomic cap, CPU bootstrap/solver analysis, figure/report generation, and deterministic verification) and writes all public artifacts into `results/` and `figures/`.
