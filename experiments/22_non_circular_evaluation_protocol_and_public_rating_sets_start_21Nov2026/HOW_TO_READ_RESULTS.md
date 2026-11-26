# How to read experiment 22 results (11-26-2026)

Date: 11-26-2026. Added for finding g4-L6.

This folder proposes evaluation rules and gives a deterministic five-item Binomial illustration; it reports no new human-data experiment. The locked protocol defines held-out families, budgets, baselines, exhaustive calibration-grid selection and planned uncertainty. Toy expectations have no sampling interval. The uniform forecast is the reference; A improves its RPS by 0.003124332126 while worsening CE. External dataset inventories are not verified from local source releases.

Ranked probability score (RPS) is normalized cumulative squared error (0–1, lower better). Cross-entropy (CE) is negative log probability (nats, lower better); CE excess in synthetic studies is KL to the known truth. Total variation distance (TVD) is half the L1 distance to the stated target distribution (0–1, lower better); sparse empirical histograms are diagnostics rather than known populations. Report RPS, CE and TVD first and Spearman/Pearson as secondary. Keep original scored tables and protocols unchanged; use the [dated corrections and evidence limits](QA_CORRECTIONS_11-26-2026.md) alongside them.
