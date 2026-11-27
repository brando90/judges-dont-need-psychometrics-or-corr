# G-study variance components

Train-split components drive the D-study. All-splits components are in-sample descriptions of the rating design.
σ² is on the rating-point scale (the dataset's class values, not a z-score).

| Dataset | Scope | Facet | σ²_c raw | σ²_r raw | σ²_res | Φ(1) | smallest k with Φ≥0.80 |
|---|---|---|---:|---:|---:|---:|---|
| summeval_expert | train | nested_in_item | 0.9659 | unidentified | 0.3756 | 0.7200 | k=2 (hit) |
| summeval_expert | all splits, in-sample | nested_in_item | 0.9510 | unidentified | 0.3722 | 0.7187 | k=2 (hit) |
| summeval_crowd | train | nested_in_item | 0.4056 | unidentified | 0.4383 | 0.4806 | k=5 (hit) |
| summeval_crowd | all splits, in-sample | nested_in_item | 0.4217 | unidentified | 0.4353 | 0.4920 | k=5 (hit) |
| popquorn_offensiveness | train | crossed_person | 0.3925 | 0.3049 | 0.7541 | 0.2704 | k=10 (not hit) |
| popquorn_offensiveness | all splits, in-sample | crossed_person | 0.4415 | 0.3027 | 0.7550 | 0.2945 | k=10 (hit) |
| popquorn_politeness | train | crossed_person | 0.7073 | 0.1973 | 0.6939 | 0.4425 | k=6 (hit) |
| popquorn_politeness | all splits, in-sample | crossed_person | 0.7095 | 0.1970 | 0.6894 | 0.4446 | k=5 (hit) |
| dices990 | train | crossed_person | 0.1423 | 0.1475 | 0.4353 | 0.1962 | k=17 (hit) |
| dices990 | all splits, in-sample | crossed_person | 0.1420 | 0.1496 | 0.4422 | 0.1935 | k=17 (hit) |
| chaosnli_snli_mnli | train | nested_in_item | 0.2559 | unidentified | 0.2733 | 0.4835 | k=5 (hit) |
| chaosnli_snli_mnli | all splits, in-sample | nested_in_item | 0.2579 | unidentified | 0.2737 | 0.4851 | k=5 (hit) |
| veribench_fsc_native_main | train | crossed_person | 0.7758 | 0.1030 | 0.1413 | 0.7606 | k=2 (hit) |
| veribench_fsc_native_main | all splits, in-sample | crossed_person | 0.5754 | 0.0437 | 0.1196 | 0.7789 | k=2 (hit) |
