# How to read experiment 20 results (11-26-2026)

Date: 11-26-2026. Added for finding g4-L6.

Synthetic sweeps vary 4–512 training items with 200 seeds; historical results reuse 75 artifacts and 250 family splits. CE and correlation heads share features and optimization setup; local optimization is not a global-optimum certificate. Compare differences of the same held-out metric, Pearson-trained minus CE-trained; positive favors CE. Intervals and win shares are saved aggregates: omitted per-seed outputs/private ratings prevent replay. At eight items/three ratings the example RPS difference is unresolved, and at three families the closed five-judge CE head has an RPS advantage.

Ranked probability score (RPS) is normalized cumulative squared error (0–1, lower better). Cross-entropy (CE) is negative log probability (nats, lower better); CE excess in synthetic studies is KL to the known truth. Total variation distance (TVD) is half the L1 distance to the stated target distribution (0–1, lower better); sparse empirical histograms are diagnostics rather than known populations. Report RPS, CE and TVD first and Spearman/Pearson as secondary. Keep original scored tables and protocols unchanged; use the [dated corrections and evidence limits](QA_CORRECTIONS_11-26-2026.md) alongside them.
