# Experiment 07: verified QA corrections (11-26-2026)

**Date: 11-26-2026.** Codex CLI, GPT-6 Astra ultra (Vals account). CPU Python/raw-output checks; no Lean checking, no new model calls. Original protocols, result tables and raw outputs remain preserved.

**g2-H3 — CONFIRMED, historical Kendall crossover.** Old preregistered README sentence says CE has lower test “RPS and CE excess … (8 for Kendall).” Corrected 11-26-2026, without editing the preregistration: in scenario A, experiment 06's CE-trained judge is better than the Kendall-trained judge from **8 tasks on ranked probability score (RPS)** and **16 tasks on cross-entropy (CE)** at every larger tested size. At 8 tasks, Kendall-trained minus CE-trained RPS excess is **+0.000632366683 [0.000068241804, 0.001193281966]**; the same-order CE excess difference is **−0.004369102997 [−0.008969781531, 0.000300092721]** nats, so a CE advantage at 8 is not established. Intervals reproduce the original 10,000-draw paired 95% seed-percentile bootstrap over 100 seeds, 500 held-out test tasks/seed. RPS/CE excess have truth as the zero reference and lower is better. Experiment 07's own 30-seed tables are unchanged.

**g2-M2 — CONFIRMED evidence limitation; historical reconstruction remains open.** Old text: reproduction checks matched “about 1e-17” and later exact-CE refits preserved results. Corrected interpretation 11-26-2026: the earlier per-fit reproduction receipt and the later individual refit outputs are not preserved. The experiment 08 aggregate receipt is internally consistent (338 moved tuned fits, 95,291/95,291 converged exact fits), but does not independently reconstruct historical per-fit differences. This is an evidence gap, not proof the recorded precision is wrong.

## Recomputed evidence and matched metric panel (11-26-2026)

Date: 11-26-2026. Read-only CPU reconstruction; no model calls.

g2-M2: changed tuned fits 338 = {'0.01': 337, '0.1': 1, '1': 0, '10': 0, '100': 0} ; converged 95291 / 95291
Historical individual refit outputs and earlier 1e-17 reproduction receipt absent; this verifies receipt arithmetic only.


Reproduce the evidence with `python3 docs/qa_11-25-2026/phase2/verify_g1_g2_remaining.py 7` from the repository root. HIGH-item checks are separately saved as `verify_g1-H1.py` (experiment 05), `verify_g2-H1.py` / `verify_g2-H2.py` (experiment 11), and `verify_g2-H3.py` (experiment 07), with their printed `.txt` outputs.
