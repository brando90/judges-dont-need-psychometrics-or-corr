# Light QA Brief — Round 2 Phase B (`11-23-2026`, Complete 6-Study Exp 29)

## Scope Under Audit (`d6930ed` vs current working tree)

1. **Experiment 29 Complete (`1,401/1,401` GPU runs: `921` Qwen3-4B-Instruct-2507 + `480` Llama-3.1-8B-Instruct per `AMENDMENT_1_11-23-2026.md`):**
   - New Phase B artifacts: `experiments/29_.../results/{chaos3_llama_predictions.npz,chaos3_llama_provenance.json,dices3_llama_predictions.npz,dices3_llama_provenance.json}` + re-scored `scores.csv`, `contrasts.csv`, `calibration_logs.csv`, `summary.json` (plus `AMENDMENT_1_11-23-2026.md`).
   - Updated reports & figures: `experiments/29_.../{RESULTS_11-23-2026.md,BUDGET_TABLES_11-23-2026.md,CKPT_exp29.md,README.md,HOW_TO_READ_RESULTS.md}`, `figures/fig1_four_objectives_and_k2_control.{png,pdf}`, `figures/fig2_disagreement_slices_and_logit_rescue.{png,pdf}`.
2. **Paper, Status, Indexes, & Version 29 One-Pager:**
   - Paper: `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/{6-case-study.tex,7-discussion.tex,10-supplementary.tex,00_main_aaai_2027.pdf,00_main_arxiv.pdf,figures/fig*.pdf}`
   - Top-level docs: `STATUS_11-23-2026.md`, `README.md`, `experiments/README.md`
   - One-pager: `paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/{judge-objectives-status.html,README.md}`

## Audit Checks Requested

1. Verify every Llama-3.1-8B (`chaos3_llama`, `dices3_llama`) & Qwen3-4B metric, 95% crossed/conditional CI, win count, and sustained threshold in `STATUS_11-23-2026.md`, `RESULTS_11-23-2026.md`, `6-case-study.tex`, `10-supplementary.tex`, `experiments/README.md`, and `judge-objectives-status.html` matches `results/{scores.csv,contrasts.csv,summary.json,*_provenance.json}` (`1,401/1,401`).
2. Verify all Phase A Light QA disclosures (`Exp 28` prior exposure `265/713`, `110/412`, `172/300`; `Exp 24` HelpSteer2 CE/Pearson reuse & parity `0.17098 vs 0.16886`; `AMENDMENT_1` `chaos3_llama` `320` runs; honest partial `H2–H4` verdicts incl. `chaos3_llama` `mse_raw - ce_raw` crossed `[-0.00066,+0.00565]` and `dices3_llama` `pearson_raw - mse_raw = -0.01610`) remain intact with zero stale `"in progress"` or `"Phase A"` contradictions.
3. Verify Brando 5 reporting questions, zero banned words, anonymization (`00_main_aaai_2027.pdf` & `.tex` inputs), Page-4 `Conclusion` fit (`References` Page 5), and `verify_exp29.py` PASS (`1401/1401`).

Output structured `P0/P1/P2` findings (if any) and verified `PASS` checks concisely with exact file paths and line numbers. Do NOT edit files.
