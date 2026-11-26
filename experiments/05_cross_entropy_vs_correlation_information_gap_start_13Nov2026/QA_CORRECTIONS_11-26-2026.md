# Experiment 05: verified QA corrections (11-26-2026)

**Date: 11-26-2026.** Codex CLI, GPT-6 Astra ultra (Vals account). CPU Python/raw-output checks; no Lean checking, no new model calls. Original protocols, result tables and raw outputs remain preserved.

**g1-H1 — CONFIRMED, forecast construction.** Old text: “two-point forecast on {0, 5} with the human mean.” Corrected 11-26-2026: a **smoothed endpoint mixture**, formed by adding 0.001 to all six categories and normalizing. Every artifact has **six positive categories**. Its expected rating is `(human_mean + 0.015) / 1.006`, so the mean is not matched exactly: mean absolute error **0.006099765046**, maximum **0.011182902584** rating points over 11 artifacts. The source comment “re-center mean to m exactly via tilt on the two points” describes an operation that is not executed; the original executable source is preserved. RPS excess **0.076333630945**, CE **5.039282445779** nats, TVD **0.711562051303**, KL **3.635311444282** nats and Pearson **1.000000000000** reproduce. Pearson remains 1 because the changed means are a positive affine transformation of human means. The distribution-identification counterexample remains valid, with the corrected description.

**g1-L8 — CONFIRMED, metric order.** Old three forecast rows reported Pearson before CE. Corrected 11-26-2026: use the population panel below with ranked probability score (RPS), cross-entropy (CE) and total variation distance (TVD) first, then Spearman and Pearson. RPS is expected squared cumulative error averaged across five thresholds [0,1], CE is expected negative log probability (nats), TVD is half the absolute probability difference from the known distribution [0,1], lower better. These are exact scores on 11 synthetic A1 artifacts, with no train/test split; the true-distribution row is the reference. Correlations are of forecast means against true means, [-1,1], higher better. No old table or scored output was changed.

**g1-M3 — historical evidence caveat (11-26-2026).** The experiment 02 interval quoted in the original reading paragraph is supported by saved aggregates, not reconstructable private ratings/bootstrap draws; see experiment 02's correction note.

## Recomputed evidence and matched metric panel (11-26-2026)

Date: 11-26-2026. Read-only CPU reconstruction; no model calls.

g1-L8: exact population distribution panel, 11 A1 artifacts; smoothed endpoints have six positive categories and affine-transformed means.

| Dataset / judge | RPS | CE (nats) | TVD | Spearman | Pearson |
|---|---:|---:|---:|---:|---:|
| True distribution | 0.116707954 | 1.403971001 | 0.000000000 | 1.000000000 | 1.000000000 |
| Maximum entropy with matching mean | 0.123568696 | 1.521294513 | 0.198893418 | 1.000000000 | 1.000000000 |
| Smoothed endpoint mixture | 0.193041585 | 5.039282446 | 0.711562051 | 1.000000000 | 1.000000000 |


Reproduce the evidence with `python3 docs/qa_11-25-2026/phase2/verify_g1_g2_remaining.py 5` from the repository root. HIGH-item checks are separately saved as `verify_g1-H1.py` (experiment 05), `verify_g2-H1.py` / `verify_g2-H2.py` (experiment 11), and `verify_g2-H3.py` (experiment 07), with their printed `.txt` outputs.
