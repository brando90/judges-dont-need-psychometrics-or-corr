# Checkpoint: Open-Weight Real Judges Native-Probability CE vs Correlation

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/CKPT_open_weight_ce_vs_corr.md>

Created: 11-16-2026 10:35 PST
Last updated: 11-16-2026 10:35 PST

## Goal
- Test whether and when cross-entropy (CE) and ranked probability score (RPS) training beat correlation training on 10 real open-weight LLM judges using native six-category probabilities (`q_logprob` and `q_verbal`) across 19 task families.
## What ran
- Verified `dataset.json` (`b3a05699...`, 75 artifacts, 25 tasks, 19 families, 195 expert ratings) and byte-exact join to VeriBench experiment-100 annotation-era files.
- Verified `mercury2` 10x RTX A4000 free GPUs (`vLLM 0.11.0`), `continue_final_message` + `logit_bias` exact 6-digit logprob API, and single-token `0..5` bijection across all 10 open-weight models.
- Dispatched companion subagent (`4b9ddc42-4d76-48d9-abc4-15e7b8f53abc`) for Experiment 10 (frontier AI judges) with `Correlation_vs_Loss_Judges.zip`; sent `[RUN START]` email.
## Results
- none yet
## Stuck / blocked
- nothing
## Next steps
- [p 10/10] Build prompts, commit `FREEZE.json`, and run `scripts/run_mercury2_open_judges.py` across 3 GPU slots (`0,1`, `2,3`, `4,5,6,7`) on `mercury2`.
- [p 9/10] Run `scripts/analyze_open_judges.py`, `scripts/make_plots.py`, and `scripts/verify_results.py`, merge with Experiment 10, and run 3-stage Mega QA.
## Brando to-do
- none

## Log
11-16-2026 10:35 PST -- Created isolated worktree `exp09-open-weight-1116`, verified inputs and `mercury2` vLLM 6-digit logprob extraction, froze `PROTOCOL.md`, dispatched Exp 10 subagent, and sent `[RUN START]` email.
