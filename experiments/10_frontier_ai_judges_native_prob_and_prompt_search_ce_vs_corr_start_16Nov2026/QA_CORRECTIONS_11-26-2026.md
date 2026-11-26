# Experiment 10 QA corrections (11-26-2026)

Date: 11-26-2026. CPU Python verification of retained records; no new model calls or scientific runs.

**g3-M3 — CONFIRMED evidence limitation, still open.** Old summaries such as repeat SD 0.0178, native-pool CE 1.416 and prompt RPS difference +0.0418 [−0.0200, +0.1813] match retained aggregate outputs. Private human ratings, raw smoke/repeat responses, fitted per-split forecasts and bootstrap draws are absent here. Aggregate agreement does not reproduce the underlying measurements.

**g3-L5 — CONFIRMED.** Old presentation: the sanity table put correlations before proper scores and the live headline gave “r = 1.0000” alone. Corrected 11-26-2026 display below puts ranked probability score (RPS), cross-entropy (CE) and total variation distance (TVD) first. RPS is mean normalized cumulative squared error (0–1), CE negative log probability (nats), TVD half the histogram L1 distance (0–1); lower is better. Spearman/Pearson are secondary correlations (−1 to 1). These are saved diagnostic aggregates, without resampling intervals or new fitting; the human RPS entry is the empirical disagreement reference, while its correlations are leave-one-rater-out diagnostics, so they are not a single common scoring procedure.

| Diagnostic and data | RPS | CE (nats) | TVD | Spearman | Pearson |
| --- | --- | --- | --- | --- | --- |
| label-in-prompt, 15 smoke artifacts | 0.034074 | 0.481883 | 1.8e-05 | 1.0 | 1.0 |
| human empirical disagreement reference, 75 artifacts | 0.045481 | not saved | not saved | 0.764187 | 0.778589 |
