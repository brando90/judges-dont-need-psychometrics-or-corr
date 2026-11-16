# CKPT — Experiment 10: Frontier AI Judges (`iii.a` Prompt Search + `iii.b` Harness Search, CE vs Correlation)

Created: 11-16-2026 10:51 PST
Last updated: 11-16-2026 12:20 PST

## Goal
- Test whether and when proper-loss optimization (cross-entropy (CE) / ranked probability score (RPS), Formulations ③/④) beats correlation optimization (Formulations ①/②) on real multi-family frontier AI judges across native probability elicitation, `iii.a` blind prompt search, and `iii.b` judge harness optimization (`H1`–`H4`).
## What ran
- Completed all 5 phases (`4,389` settled API calls across `koyejolab-openai-bm` and `mistral`, `$17.8511` total spend `<= $18.00` cap; `verify_results.py` `PASS`, `0` secret/privacy leaks, `Exp 02` parity diff `0.0`).
## Results
- Native 6-category probabilities (`15` frontier judges across `7` OpenAI & Mistral models, `5x5` grouped family CV): `④ CE-trained` mixture held-out `CE = 1.4156` nats (`RPS = 0.1137 [0.0895, 0.1411]`) vs `② Pearson-trained` `CE = 1.9710` (`RPS = 0.1912`, `② minus ④` RPS `+0.0775 [+0.0437, +0.1135] p-val=0.0000`), fixing the historical `5.4914`-nat 3-sample scalar failure and beating the `1.9280`-nat constant baseline.
- Learning curve (`250` grouped family splits, 48-param feature head): `④ CE-trained` beats `② Pearson-trained` at every training size from `5` task families (`~51.7` ratings) onward (`n_fam=15`: `+0.0250 [+0.0238, +0.0262] p-val=0.0000`; `③ Val-RPS-tuned CE = 0.1290` beats `④ fixed CE = 0.1569` by `+0.0280 p-val=0.0000`); `G_i` correlates with `(RPS_Pearson - RPS_CE)` at `r = +0.5425 [+0.2634, +0.7176] p-val=0.0022`.
- Axes `iii.a` & `iii.b` (locked `9/5/5` family split, 30 held-out test artifacts): `iii.a` `③ Loss-Val` prompt reaches `0.1410` RPS (`0.1590` on `gpt-6.1-sol`) vs `① Corr-Val` `0.1828` (`0.2071` on `gpt-6.1-sol`); `iii.b` two-step audit harness `H3` lowers held-out RPS to `0.1136–0.1174` and cross-family `H4` at `③ Loss` reaches `0.1284` (vs `6.591`-nat CE for 3-vote scalar voting).
## Stuck / blocked
- nothing (complete and deterministically verified).
## Next steps
- [p 9/10] Parent coordinator merges Experiment 09 and Experiment 10 into `main`, updates `experiments/README.md` and `paper_latex_and_notes/`, and runs the requested 3-stage Mega QA.
## Brando to-do
- none.

## Log
11-16-2026 12:20 PST -- Completed Phase 4 (`iii.b` harnesses `H1`–`H4`, `$17.8511` final spend) and Phase 5 (`analyze_all.py`, 5 figures, `RESULTS_frontier_ce_vs_corr.md`, `writing.tex`, and `verify_results.py` PASS with `0` leaks and `0.0` Exp-02 parity diff).
11-16-2026 11:48 PST -- Completed Phase 3 (`iii.a` 4x2 blind prompt search: held-out test RPS `③ Loss-Val = 0.1410` vs `① Corr-Val = 0.1828` and `P0 = 0.1812`; cumulative `$14.34` spent); launched Phase 4 (`iii.b` harnesses `H1`–`H4`).
11-16-2026 11:36 PST -- Completed Phase 2 (`15` frontier judges across `7` OpenAI & Mistral models, `results/judge_forecasts.json` saved, `$11.37` cumulative spend; `PROTOCOL_AMENDMENT_01.md` recorded); launched Phase 3 (`iii.a` blind 4x2 prompt search).
11-16-2026 11:15 PST -- Committed freeze `aa09730`; Phase 1 Karpathy sanity checks passed (`79/79` parsed, LIP `r=1.0000`, `oracle=1.000` vs negatives `0.000`, `k=3` repeat SD `0.0178`, `$0.4988` spent); launched Phase 2 multi-family frontier pool.
11-16-2026 10:51 PST -- Initialized Experiment 10 folder, `.gitignore` (`private/`), `README.md`, `PROTOCOL.md` (with Measurement Contract and Brando's 10:44 PST `iii.a` prompt + `iii.b` harness + `2x2` design), `HOW_TO_READ_RESULTS.md`, `cc.md`, and `results.md`; `$0.00` spent.
