# Experiment 16 QA corrections (11-26-2026)

Date: 11-26-2026. CPU Python verification of retained records; no new model calls or scientific runs.

**g3-M4 — CONFIRMED evidence limitation, still open.** Old claims: correlations, optimizer-failure counts 4/52/78 and “22:14 to 22:16 … 21 s to 100 s.” Saved aggregates retain correlations and those failure counts, but no split correlation vectors, solver receipts or launcher/run logs are tracked. The 12 saved `seconds` values range 2.224164–20.549976 s and sum to 84.402701 s; they cannot establish launch/end times. Proper-score arrays exist in `per_artifact.json`; the missing evidence concerns the other measurements.

**g3-L1 — CONFIRMED.** Old relative target `archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/README.md` incorrectly resolved inside experiment 16. Corrected 11-26-2026: [experiment 14](../archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/README.md). The README also contains preregistered predictions, so its original broken link remains unchanged; this adjacent note supplies the repaired route.

**g3-L3 — CONFIRMED.** Old display: standalone Spearman/Pearson-only table. Corrected 11-26-2026: matched proper-score-first display below from the same saved floor-1e-6 aggregates. Historical 75 artifacts/195 ratings, 1,000 task-grouped 20/5 train/test splits; fitting and selection follow the original methods. Ranked probability score (RPS; normalized cumulative squared error, 0–1), cross-entropy (CE; negative log probability, nats) and total variation distance (TVD; half histogram L1 distance, 0–1) are lower-is-better. Correlations are secondary means across splits; undefined values stay undefined. These are point summaries, not new uncertainty estimates.

| Pool | Method | RPS | CE (nats) | TVD | Spearman | Pearson |
| --- | --- | --- | --- | --- | --- | --- |
| old5 | constant | 0.183566 | 1.851901 | 0.681795 | undefined | undefined |
| old5 | equal_mixture | 0.149558 | 5.431086 | 0.531111 | 0.608381 | 0.703263 |
| old5 | corr_single | 0.179688 | 8.103857 | 0.600423 | 0.719131 | 0.749695 |
| old5 | corr_single_spearman | 0.179258 | 8.097936 | 0.599999 | 0.721050 | 0.751664 |
| old5 | corr_mixture_pearson | 0.141865 | 5.787498 | 0.522182 | 0.669010 | 0.730467 |
| old5 | corr_mixture_spearman | 0.179178 | 8.094723 | 0.599842 | 0.720898 | 0.751423 |
| old5 | ce_single | 0.205062 | 8.934255 | 0.653991 | 0.608890 | 0.643807 |
| old5 | ce_mixture | 0.136239 | 5.413018 | 0.509815 | 0.605947 | 0.717438 |
| old5 | rps_single | 0.206785 | 8.611242 | 0.633959 | 0.574861 | 0.602834 |
| old5 | rps_mixture | 0.131331 | 5.773240 | 0.497871 | 0.647876 | 0.726012 |
| open | constant | 0.183566 | 1.851901 | 0.681795 | undefined | undefined |
| open | equal_mixture | 0.310663 | 3.216012 | 0.762051 | -0.171656 | -0.173504 |
| open | corr_single | 0.367969 | 10.537520 | 0.770492 | 0.208938 | 0.190402 |
| open | corr_single_spearman | 0.369982 | 10.489335 | 0.772339 | 0.117492 | 0.124060 |
| open | corr_mixture_pearson | 0.327405 | 5.724534 | 0.739314 | 0.219988 | 0.196456 |
| open | corr_mixture_spearman | 0.366600 | 10.315919 | 0.769768 | 0.108596 | 0.115497 |
| open | ce_single | 0.315111 | 9.419235 | 0.744508 | 0.111682 | 0.023901 |
| open | ce_mixture | 0.277254 | 3.388648 | 0.730609 | 0.031103 | 0.008631 |
| open | rps_single | 0.340205 | 10.122934 | 0.775573 | 0.039194 | -0.039721 |
| open | rps_mixture | 0.260124 | 4.471324 | 0.729255 | 0.029193 | 0.021633 |
| all | constant | 0.183566 | 1.851901 | 0.681795 | undefined | undefined |
| all | equal_mixture | 0.273270 | 2.687663 | 0.724646 | 0.112575 | 0.113527 |
| all | corr_single | 0.179688 | 8.103857 | 0.600423 | 0.719131 | 0.749695 |
| all | corr_single_spearman | 0.179258 | 8.097936 | 0.599999 | 0.721050 | 0.751664 |
| all | corr_mixture_pearson | 0.206328 | 3.915181 | 0.626433 | 0.600221 | 0.653627 |
| all | corr_mixture_spearman | 0.179258 | 8.097936 | 0.599999 | 0.721050 | 0.751664 |
| all | ce_single | 0.211389 | 9.018506 | 0.660144 | 0.594751 | 0.626725 |
| all | ce_mixture | 0.165809 | 2.622906 | 0.591178 | 0.568762 | 0.656506 |
| all | rps_single | 0.206785 | 8.611242 | 0.633959 | 0.574861 | 0.602834 |
| all | rps_mixture | 0.128865 | 4.803047 | 0.525582 | 0.649697 | 0.724897 |
