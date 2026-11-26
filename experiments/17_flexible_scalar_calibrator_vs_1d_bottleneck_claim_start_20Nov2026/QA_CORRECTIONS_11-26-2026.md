# Experiment 17 QA corrections (11-26-2026)

Date: 11-26-2026. CPU Python verification of retained records; no new model calls or scientific runs.

**g3-M5 — CONFIRMED evidence limitation, still open.** Old run claims: 22:46–23:12, 1,508 seconds, 16 processes and named package versions. The raw file contains fitted metrics without timestamp/environment receipts; no run log is retained. Corrected scope of verification: all 4,400 unique scenario/sample-size/seed cells exist and the uncompressed SHA-256 is `ecd3789f6d32ea84ceeeef6c94182d4e000399263b9f98ef36dfd31629dc819a`; these facts do not establish elapsed runtime or environment.

**g3-L4 — CONFIRMED.** Old table orders Spearman/Pearson before CE excess. Corrected 11-26-2026 display below derives the n=2,048 metrics directly from 100 seed records per row, preserving old tables in `results.md` and `results/summary.md`. Data: 500 fresh synthetic test items per seed, scenarios A–D. Ranked probability score (RPS; normalized cumulative squared error, 0–1) and cross-entropy (CE) excess over truth (KL, nats) are lower-is-better. The available total variation distance (TVD) is explicitly **distance from the training constant**, not TVD to truth; it is a diagnostic with no accuracy ordering. Spearman/Pearson compare the fitted and true means and are secondary (−1 to 1). These means add no uncertainty claim; fit/selection settings remain the original frozen experiment's settings.

| Scenario | Method | RPS | CE excess (nats) | TVD from constant (diagnostic) | Spearman | Pearson |
| --- | --- | --- | --- | --- | --- | --- |
| A | ce_valrps | 0.187590 | 0.006594 | 0.201578 | 0.902193 | 0.891601 |
| A | pearson_head_flex | 0.192777 | 0.087494 | 0.096959 | 0.834184 | 0.830413 |
| A | truemean_head_flex | 0.187160 | 0.003293 | 0.201122 | 0.978619 | 0.975103 |
| A | truemean_head_affine | 0.194143 | 0.116011 | 0.048537 | 1.000000 | 0.999914 |
| B | ce_valrps | 0.139968 | 0.022436 | 0.350651 | 0.948626 | 0.970395 |
| B | pearson_head_flex | 0.138925 | 0.025529 | 0.343772 | 0.970990 | 0.986018 |
| B | truemean_head_flex | 0.136685 | 0.008313 | 0.356675 | 0.999292 | 0.999065 |
| B | truemean_head_affine | 0.141996 | 0.101516 | 0.259012 | 1.000000 | 0.995417 |
| C | ce_valrps | 0.129885 | 0.020247 | 0.296973 | 0.958922 | 0.977861 |
| C | pearson_head_flex | 0.129049 | 0.013733 | 0.294141 | 0.974013 | 0.987213 |
| C | truemean_head_flex | 0.127156 | 0.002430 | 0.301310 | 1.000000 | 0.999518 |
| C | truemean_head_affine | 0.127358 | 0.003772 | 0.297726 | 1.000000 | 0.998286 |
| D | ce_valrps | 0.149189 | 0.018285 | 0.247178 | 0.973687 | 0.971474 |
| D | pearson_head_flex | 0.148303 | 0.011590 | 0.244986 | 0.983304 | 0.983434 |
| D | truemean_head_flex | 0.146668 | 0.002283 | 0.251319 | 0.999997 | 0.999416 |
| D | truemean_head_affine | 0.146706 | 0.001934 | 0.250125 | 1.000000 | 0.998936 |
