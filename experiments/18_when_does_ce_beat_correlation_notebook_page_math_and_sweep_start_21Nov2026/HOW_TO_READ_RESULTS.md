# How to read experiment 18 results (11-26-2026)

Date: 11-26-2026. Added for finding g4-L6.

Synthetic sweep A uses 20 seeds per cell and 500 fresh items per seed; sweep B uses 100 pools per cell. Joint 95% paired seed-bootstrap intervals on RPS and CE excess determine the winner. Historical analyses reuse 75 artifacts, 25 tasks, 19 families and 195 ratings, with protocol-specific task/family bootstrap intervals. Synthetic candidate/grid and softmax fits are documented in the preserved README; historical selection is over fixed pools. No universal superiority follows. The chronology and raw-data limits are in the correction note.

Ranked probability score (RPS) is normalized cumulative squared error (0–1, lower better). Cross-entropy (CE) is negative log probability (nats, lower better); CE excess in synthetic studies is KL to the known truth. Total variation distance (TVD) is half the L1 distance to the stated target distribution (0–1, lower better); sparse empirical histograms are diagnostics rather than known populations. Report RPS, CE and TVD first and Spearman/Pearson as secondary. Keep original scored tables and protocols unchanged; use the [dated corrections and evidence limits](QA_CORRECTIONS_11-26-2026.md) alongside them.
