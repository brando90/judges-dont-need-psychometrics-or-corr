# Light QA Brief — Round 2 Phase A (`11-23-2026`)

## Scope Under Review (`HEAD` vs current working tree)

1. **Experiment 29 Phase A (`Qwen3-4B-Instruct-2507`, `921/921` GPU runs complete; `Llama-3.1-8B-Instruct` `480` runs still active on `mercury2`):**
   - Scripts: `experiments/29_.../scripts/{prepare_exp29_bundle.py,train_exp29_lora.py,calibrate_and_score_exp29.py,plot_and_report_exp29.py,verify_exp29.py}`
   - Reports & Figures: `experiments/29_.../{RESULTS_11-23-2026.md,HOW_TO_READ_RESULTS.md,BUDGET_TABLES_11-23-2026.md,CKPT_exp29.md,README.md}`, `figures/fig1_four_objectives_and_k2_control.{png,pdf}`, `figures/fig2_disagreement_slices_and_logit_rescue.{png,pdf}`
   - Artifacts: `results/{data_manifest.json,protocol.sha256,qwen3_4b_sanity.json,llama31_8b_sanity.json,scores.csv,contrasts.csv,calibration_logs.csv,summary.json,*_qwen*.{npz,json}}`
2. **Project Tech Report, Status, Root README, Experiment Index, & Version 29 One-Pager:**
   - Paper: `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/{00_main.bib,6-case-study.tex,7-discussion.tex,10-supplementary.tex,00_main_aaai_2027.pdf,00_main_arxiv.pdf}`
   - Status & Indexes: `STATUS_11-23-2026.md`, `README.md`, `experiments/README.md`
   - One-Pager: `paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/{build_bundle.py,judge-objectives-status.html,README.md}`

## Specific Checks Requested

1. **Numerical consistency & source faithfulness:** Verify every cited metric, confidence interval, and win count in `STATUS_11-23-2026.md`, `RESULTS_11-23-2026.md`, `6-case-study.tex`, `10-supplementary.tex`, `judge-objectives-status.html`, `README.md`, and `experiments/README.md` matches the saved CSV/JSON outputs (`experiments/26_...`, `experiments/27_...`, `experiments/28_...`, `experiments/29_.../results/{scores.csv,contrasts.csv,summary.json}`).
2. **Brando reporting rules (11-13-2026) & anonymization:** Check that all 5 reporting questions (which data/split, metric definition/direction/reference, subtraction order `comparator - target`, 4,000-draw cluster bootstrap 95% CI, search space/selection method) are answered cleanly; verify zero anonymization leaks (`brando`, `/Users/`, `mercury2`, personal names) in `00_main_aaai_2027.tex` inputs (`6-case-study.tex`, `7-discussion.tex`, `00_main.bib`, `10-supplementary.tex`); verify zero banned words (`arms`, `cells`, `contracts`, `untouched`, `holdouts`, `compact`, `CE minus Pearson`, `bit flip`, `north star`, `killer demo`, `velocity`).
3. **Framing honesty:** Verify Phase A clearly marks `Qwen3-4B-Instruct-2507` (`921/921` runs) complete while `Llama-3.1-8B-Instruct` (`480` runs) is in progress, and preserves Exp 24 `hs2` parity disclosure (`0.17098` vs `0.16886` RPS).
4. **Code & build checks:** Confirm AST limits (`LOC <= 55, CC <= 10`) on `experiments/29_.../scripts/*.py`, `ax.grid(True)` on figures, and clean `pdftotext` Page-4 `Conclusion` placement in `00_main_aaai_2027.pdf`.

Report findings concisely ranked `P0` (must fix), `P1` (should fix), `P2` (minor/nice-to-fix), with exact file paths, line numbers, and concrete fixes.
