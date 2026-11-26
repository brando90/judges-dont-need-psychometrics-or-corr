# Experiment 18 QA corrections (11-26-2026)

Date: 11-26-2026. CPU Python verification of retained records; no new model calls or scientific runs.

**g4-H1 — CONFIRMED.** Old scored prediction: “22 of 40” J=1 C/D cells. Corrected 11-26-2026: **22 of 42**, from 2 scenarios × 7 training sizes × 3 rating depths. Replaying the original 2,000-draw paired seed bootstrap from 5,040 raw records gives 22 head wins, 9 CE wins, 6 ties and 5 mixed cells. Both RPS and CE-excess intervals must favor the winner. Original scored table retained.

**g4-H2 — CONFIRMED.** Old verdict: “never significantly worse.” Corrected 11-26-2026: **the saved task-bootstrap analysis has one exception favoring CE selection**, at 44 judges, floor 0.05, four training tasks: held-out RPS of Pearson-selected minus CE-selected = 0.196008196152 − 0.185281801070 = **+0.010726395081**, saved 95% interval **[+0.001612285381, +0.019335559001]**. The headline and TLDR's general “not worse” wording require this exception. The aggregate subtraction is recomputed; private rating-level interval inputs are unavailable.

**g4-H3 — CONFIRMED.** Old reconciliation verdict: “never flips a sign” and “the split protocol and the ratings budget are not the cause.” Corrected 11-26-2026: **the direction depends on the pool, and H5 point-estimate signs also differ across protocols/budgets**. H5, one rating per artifact: at three training families under C the RPS difference (Pearson-selected minus CE-selected) is **+0.000065279598**; at four training tasks under M it is **−0.002903707496**. H5 has 3 positive C estimates out of 9 and 0 positive M estimates out of 12. Both example saved intervals include zero, so this is not evidence of a statistically resolved reversal or an isolated causal effect of split protocol.

**g4-M1 — CONFIRMED chronology discrepancy, unresolved provenance.** Old log says preregistration at 12:20 before any sweep cell, but completion at approximately 12:05 on 11-21-2026. Corrected 11-26-2026: **the written chronology is inconsistent by approximately 15 minutes; preregistration precedence cannot be established from these two timestamps**. Neither timestamp is guessed or replaced. The README's preregistration text is preserved unchanged; this adjacent correction qualifies the retrospective claim wherever repeated in the results.

**g4-M2 — CONFIRMED evidence limitation, still open.** Historical rating-budget, real training-size and reconciliation reports retain aggregate scores and intervals. Original individual ratings, fitted per-split/per-artifact outputs and resampling draws are absent. The stored CE-selected RPS 0.15005 and contrast interval [+0.01677, +0.05730] are historical aggregate results, not independently replayed measurements in phase 2.

**g4-L1 — CONFIRMED.** Old status: “Not reconciled.” Corrected 11-26-2026: **reconciliation completed**, with 105 aggregate comparisons in [the reconciliation report](reconcile_single_judge_selection_21Nov2026/RECONCILE_11-21-2026.md); its sign wording is qualified above.

**g4-L2 — CONFIRMED.** Old review link target `paired_rps.csv` is absent inside `review/`. Corrected 11-26-2026: `../historical_rating_budget/paired_rps.csv`.

Verification: repository scripts `docs/qa_11-25-2026/phase2/verify_g4-H1.py`, `verify_g4-H2.py`, `verify_g4-H3.py` and their `.txt` outputs. No scored table, raw record or preregistration was edited.

**Test receipt (11-26-2026):** The existing conditional-shape verifier exits 1 at exact string equality when regenerating the first/last replications. The follow-up numerical diagnosis finds 719 serialized numeric differences, maximum absolute error 4.85722573273506e-16; all are below 1e-12. The original verifier is still reported as failing. Neither raw records nor that verifier were changed. See `docs/qa_11-25-2026/phase2/verify_e18_float_replay_11-26-2026.txt`.
