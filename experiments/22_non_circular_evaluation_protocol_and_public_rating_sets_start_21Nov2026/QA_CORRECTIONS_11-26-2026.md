# Experiment 22 QA corrections (11-26-2026)

Date: 11-26-2026. CPU Python verification of retained records; no new model calls or scientific runs.

**g4-H5 — CONFIRMED.** Old protocol verdict (lines 7/47): A is “no better than uniform on RPS.” Corrected 11-26-2026: **A is slightly better than uniform on RPS, and worse on CE**. Exact finite expectations over the five specified Binomial distributions give A RPS **0.189097890096**, uniform RPS **0.192222222222**, improvement **0.003124332126**. These deterministic expectations have no sampling interval. B still has lower RPS than A. The locked `EVAL_PROTOCOL_11-21-2026.md` remains unchanged.

**g4-L4 — CONFIRMED.** Old locked table order: Pearson, RPS, CE, TVD. Corrected 11-26-2026 display below orders ranked probability score (RPS; normalized cumulative squared error, 0–1), cross-entropy (CE; nats) and total variation distance (TVD; half L1 to the known target, 0–1) first, all lower-is-better. Pearson is secondary. No training, selection, resampling or human-data claim occurs in this deterministic illustration.

| Forecast | RPS | CE (nats) | TVD | Pearson |
| --- | --- | --- | --- | --- |
| Truth | 0.095278179 | 1.234262164598 | 0 | 1 |
| A | 0.189097890096 | 1.982686981985 | 0.39670960896 | 1 |
| B | 0.10829322332 | 1.341131587632 | 0.169138 | 0.948683298051 |
| Uniform | 0.192222222222 | 1.791759469228 | 0.427184 | undefined |

**g4-M5 — CONFIRMED local evidence limitation, still open.** The protocol cites external releases for ChaosNLI 4,645 examples, DICES 43,050/72,103 rows, and HelpSteer2 23,652 disagreement rows. Full original releases are absent from this checkout. Arithmetic 350 × 123 = 43,050 and 1,162 + 16,390 + 1,314 + 4,724 + 62 = 23,652 is consistent, but cannot validate release inventories. Later experiment subsets and processed manifests are not the complete original releases. No external inventory was silently treated as a local reproduction.

Verification: `docs/qa_11-25-2026/phase2/verify_g4-H5.py` and its printed output.
