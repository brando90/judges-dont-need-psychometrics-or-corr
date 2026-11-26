# QA phase 2 summary (11-26-2026)
Date: 11-26-2026.
Harness: Codex CLI, GPT-6 Astra ultra (Vals account). CPU verification only.
Fix commit range: 4ce401c4711cc5eb944c86ea0e5cac862ed02356 -> e95a2c7f1968b099144a0b73d341b2c934150a13; report-only publication follows.
89 findings: 88 CONFIRMED, 1 PARTLY, 0 NOT CONFIRMED; 63 fixed, 26 open.
HIGH: 16 confirmed; 16 fixed; 0 open. All 16 individual verification scripts exit 0.
MEDIUM: 41 confirmed; 15 fixed; 26 open for missing evidence or chronology.
LOW: 31 confirmed, 1 partly; 32 fixed; 0 open.
Correction 1 — E18: old 22/40 significant head wins -> 22/42 (same numerator).
Correction 2 — E11: old fixed MSE beats constant everywhere -> A/2048 RPS worse by +0.00899056 [0.00816237,0.00981270].
Correction 3 — E29: old registered H1 confirmed -> partly supported; CE RPS crossed wins 22/27 across K=3 budgets, 7/7 at maximum budgets.
Correction 4 — verbal mixtures: old CE 1.629 vs 3.480 -> 1.629 vs 1.804 nats; 3.480 belongs to scalar CE.
Correction 5 — DICES: old sustained calibrated-Pearson RPS advantage from 64 universally -> DICES-990 128, DICES-350 64 at one rating/item.
Tests: 31 pytest tests passed; E10 verifier exit1 missing data, E18 exit1 float equality (tolerance check0), E20 pytest5/no tests (direct0); recovered E30/build attempts retained; dual PDFs build0.
Open: 26 evidence/chronology findings detailed in results.md; raw outputs, scored tables, locked protocols and deposits preserved; no external posts/email.
