# CKPT — Experiment 10: Frontier Artificial Intelligence (AI) Judges (`iii.a` Prompt Search + `iii.b` Harness Search, CE vs Correlation)

Created: 11-16-2026 10:51 PST
Last updated: 11-16-2026 14:57 PST

## Goal
- Test whether and when proper-loss optimization (cross-entropy (CE) / ranked probability score (RPS), Formulations ③/④) beats correlation optimization (Formulations ①/②) on real multi-family frontier AI judges across native probability elicitation, `iii.a` blind prompt search, and `iii.b` judge harness optimization (`H1`–`H4`).
## What ran
- Completed all 5 phases (`4,389` settled application programming interface (API) calls across `koyejolab-openai-bm` and `mistral`, `$17.8511` total spend `<= $18.00` cap; `verify_results.py` `PASS`, `0` secret/privacy leaks, `Exp 02` parity diff `0.0`).
## Results
- Native 15-judge mixtures across seven models use 25 outer task-family folds (five partitions × five folds): cross-entropy-trained test cross-entropy `1.4156` nats versus Pearson-trained `1.9710`; test ranked probability score of Pearson-trained minus cross-entropy-trained is `+0.0775 [+0.0437, +0.1135]`. This is a 95% task-family percentile bootstrap interval (10,000 draws, fixed predictions); `p-val=n/a`. Historical scalar-pool and constant reference losses are `5.4914` and `1.9280` nats; pool composition also changes.
- Feature-head learning curves use 250 overlapping task-family splits. At 15 training families, test ranked probability score of Pearson-trained minus cross-entropy-trained is `+0.0250 [+0.0238, +0.0262]`; the 95% interval resamples split averages (10,000 draws) and measures conditional Monte Carlo variability, not population uncertainty; `p-val=n/a`. The implemented disagreement diagnostic is a smoothed ranked probability score disagreement floor, not Experiment 05's information gap.
- Prompt/harness comparisons use a locked 9/5/5 family split (30 held-out artifacts). Matched `gpt-5.4-mini` two-step harnesses have test ranked probability score `0.1136–0.1174` for the selected prompts; [full results](RESULTS_frontier_ce_vs_corr.md) give paired uncertainty, model mappings and retained failures. No general population rating threshold or isolated aggregation advantage is established.
## Stuck / blocked
- nothing (complete and deterministically verified).
## Next steps
- Complete Stage 3 local verification and record its review in Experiment 09. Publication remains outside this review task.
## Brando to-do
- none.

## Log
11-16-2026 12:20 PST -- Completed Phase 4 (`iii.b` harnesses `H1`–`H4`, `$17.8511` final spend) and Phase 5 (`analyze_all.py`, 5 figures, `RESULTS_frontier_ce_vs_corr.md`, `writing.tex`, and `verify_results.py` PASS with `0` leaks and `0.0` Exp-02 parity diff).
11-16-2026 11:48 PST -- Completed Phase 3 (`iii.a` 4x2 blind prompt search: held-out test RPS `③ Loss-Val = 0.1410` vs `① Corr-Val = 0.1828` and `P0 = 0.1812`; cumulative `$14.34` spent); launched Phase 4 (`iii.b` harnesses `H1`–`H4`).
11-16-2026 11:36 PST -- Completed Phase 2 (`15` frontier judges across `7` OpenAI & Mistral models, `results/judge_forecasts.json` saved, `$11.37` cumulative spend; `PROTOCOL_AMENDMENT_01.md` recorded); launched Phase 3 (`iii.a` blind 4x2 prompt search).
11-16-2026 11:15 PST -- Committed freeze `aa09730`; Phase 1 Karpathy sanity checks passed (`79/79` parsed, LIP `r=1.0000`, `oracle=1.000` vs negatives `0.000`, `k=3` repeat SD `0.0178`, `$0.4988` spent); launched Phase 2 multi-family frontier pool.
11-16-2026 10:51 PST -- Initialized Experiment 10 folder, `.gitignore` (`private/`), `README.md`, `PROTOCOL.md` (with Measurement Contract and Brando's 10:44 PST `iii.a` prompt + `iii.b` harness + `2x2` design), `HOW_TO_READ_RESULTS.md`, `cc.md`, and `results.md`; `$0.00` spent.

11-16-2026 14:57 PST -- Stage 3 review corrected current inference, diagnostic definitions, reference scales, and optimizer descriptions; historical log and numerical evidence preserved. See Experiment 09 qa/mega_qa_11-16-2026/qa3_codex_vals.md.
