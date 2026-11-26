# Experiment 20 QA corrections (11-26-2026)

Date: 11-26-2026. CPU Python verification of retained records; no new model calls or scientific runs.

**g4-H4 — CONFIRMED.** Old verdict: CE “loses at 4 to 8 items or 3 families.” Corrected 11-26-2026: **small-budget results depend on the target, pool and rating depth; a blanket loss is unsupported**. At eight items and three ratings per item on the mean/spread/shape simulation, Pearson-trained minus CE-trained RPS excess is **+0.002422443302 [−0.001866928915, +0.006711815519]**, an unresolved difference with point estimate favoring CE. At three training families, all ratings, the historical closed five-judge pool has RPS **0.160971627504** for the CE head versus **0.181594349853** for the Pearson head, difference **+0.020622722349 [+0.007155529018, +0.040690918450]**. The latter supports a CE-head advantage on RPS in the saved analysis. The old wording is preserved in the README. Differences compare the same metric on the same held-out data; positive favors CE.

**g4-M3 — CONFIRMED evidence limitation, still open.** Old headline: +0.034 [0.031, 0.036], 96% of 200 seeds. The README explicitly omits the 48 MB synthetic per-seed output; `summary_synth.json` and `out_real_lam*.json` contain aggregate records. Private human ratings and per-split training outputs are absent. The H4 interval values above are saved, not regenerated; arithmetic and internal contradiction checks do not reproduce the historical training runs. No replacement simulation was run.

Verification: `docs/qa_11-25-2026/phase2/verify_g4-H4.py` and its printed output.
