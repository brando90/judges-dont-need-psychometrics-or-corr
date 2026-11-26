# Experiment 31: metric-first companion for historical Table 5 (11-26-2026)

As of 11-26-2026. Finding g6-L2. The old 12-row correlation-only table remains preserved; this companion displays both methods in each panel with all three evaluation metrics first.

Held-out ranked probability score (RPS, normalized 0–1), cross-entropy (CE, nats) and total variation distance (TVD, 0–1 against empirical human ratings) are lower-is-better. Spearman and Pearson measure secondary association of predicted and human means. RPS and CE are proper scores; TVD is a distribution diagnostic. Values are saved point estimates: Exp 28 is the mean of five fixed frontier judges, exhaustive 27-transform head search, five held-out folds and 200 training draws; Exp 29 is raw LoRA training over ten nested draws, as described in its frozen protocol. `B` is the largest available total rating budget for that panel, and `d` is ratings per item. Exp 28 uniform CE is log(K) (alphaNLI RPS 0.25); Exp 29 uniform RPS ranges from 0.20388 to 0.26936. These are descriptive means with no newly computed confidence interval.

Sources: Exp 28 `results/scores_absolute.csv` and `results/correlations.csv`; Exp 29 `results/scores.csv`. Each method is joined by its exact panel and budget. Raw-label evidence limitations remain in the [dated QA note](QA_CORRECTIONS_11-26-2026.md).

| Source / panel | d | B | Method | RPS | CE (nats) | TVD | Spearman | Pearson |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Exp 28 / chaosnli_snli / five-judge mean | 1 | 256 | ce | 0.1153294000 | 0.7131430000 | 0.1689810000 | 0.9148800000 | 0.9175400000 |
| Exp 28 / chaosnli_snli / five-judge mean | 1 | 256 | pearson | 0.1335440000 | 0.8237604000 | 0.2353646000 | 0.9132200000 | 0.9127200000 |
| Exp 28 / chaosnli_mnli / five-judge mean | 1 | 256 | ce | 0.1627128000 | 0.9111004000 | 0.2075932000 | 0.8225400000 | 0.8231400000 |
| Exp 28 / chaosnli_mnli / five-judge mean | 1 | 256 | pearson | 0.1848362000 | 1.0372678000 | 0.2614132000 | 0.8267000000 | 0.8312000000 |
| Exp 28 / chaosnli_alphanli / five-judge mean | 1 | 256 | ce | 0.1202228000 | 0.3958186000 | 0.1253330000 | 0.8681000000 | 0.9021200000 |
| Exp 28 / chaosnli_alphanli / five-judge mean | 1 | 256 | pearson | 0.1251234000 | 0.4541162000 | 0.1312306000 | 0.8681000000 | 0.8991200000 |
| Exp 28 / dices350 / five-judge mean | 1 | 256 | ce | 0.2255486000 | 0.8513418000 | 0.1625260000 | 0.5257800000 | 0.5667400000 |
| Exp 28 / dices350 / five-judge mean | 1 | 256 | pearson | 0.2758482000 | 1.2719144000 | 0.3138674000 | 0.5257600000 | 0.5514200000 |
| Exp 28 / dices990 / five-judge mean | 1 | 256 | ce | 0.1983906000 | 0.7570920000 | 0.1376304000 | 0.7619200000 | 0.7069000000 |
| Exp 28 / dices990 / five-judge mean | 1 | 256 | pearson | 0.2607504000 | 1.0473436000 | 0.3145340000 | 0.7609000000 | 0.7452600000 |
| Exp 29 / chaos3_qwen | 1 | 2048 | ce_raw | 0.1490207298765558 | 0.8589277746496023 | 0.21937690036976254 | 0.8056736162002242 | 0.8032231823247903 |
| Exp 29 / chaos3_qwen | 1 | 2048 | pearson_raw | 0.162228486428876 | 1.002795472619873 | 0.26124629837085556 | 0.8022124587503772 | 0.7968292596410317 |
| Exp 29 / chaos3_qwen | 5 | 1280 | ce_raw | 0.1530494850241364 | 0.8814659715802069 | 0.23240273588920357 | 0.7950900272178366 | 0.7892839886133279 |
| Exp 29 / chaos3_qwen | 5 | 1280 | pearson_raw | 0.19099398428311495 | 1.7336005953473097 | 0.32749454895291985 | 0.7972008671086327 | 0.7643651466648638 |
| Exp 29 / alpha2_qwen | 1 | 1024 | ce_raw | 0.18624205358615123 | 0.5542208180037455 | 0.2569527938372537 | 0.6753491952052904 | 0.6241720853991624 |
| Exp 29 / alpha2_qwen | 1 | 1024 | pearson_raw | 0.2081954491208136 | 0.6499213637942338 | 0.2790872629521088 | 0.657667106460502 | 0.6019143904110906 |
| Exp 29 / dices3_qwen | 1 | 1024 | ce_raw | 0.2087624399230478 | 0.7746217773356927 | 0.14958926505305342 | 0.4672802378132846 | 0.4375518802398776 |
| Exp 29 / dices3_qwen | 1 | 1024 | pearson_raw | 0.2848570204225838 | 2.870265783439293 | 0.35122985248228306 | 0.44552396633837354 | 0.39058847225270615 |
| Exp 29 / hs2_qwen | 1 | 2048 | ce_raw | 0.15213335903629496 | 1.3379409659595372 | 0.4589239093486692 | 0.44239060192560753 | 0.5333362001222643 |
| Exp 29 / hs2_qwen | 1 | 2048 | pearson_raw | 0.1837557023906196 | 2.3441602931253422 | 0.5469019168988547 | 0.42474912355890915 | 0.5210259006311817 |
| Exp 29 / chaos3_llama | 1 | 2048 | ce_raw | 0.15546480012394504 | 0.8887276331340457 | 0.2471238828554017 | 0.7711470783174058 | 0.7804604034539244 |
| Exp 29 / chaos3_llama | 1 | 2048 | pearson_raw | 0.18981876181942828 | 1.701814507859065 | 0.3904271096273303 | 0.7621663757403381 | 0.7730639137464753 |
| Exp 29 / dices3_llama | 1 | 1024 | ce_raw | 0.2060293659900482 | 0.7703248844114712 | 0.14481563972985212 | 0.5477712324821791 | 0.5157654052284417 |
| Exp 29 / dices3_llama | 1 | 1024 | pearson_raw | 0.2505494728917481 | 1.433493519839009 | 0.32270785436240673 | 0.5413347360902863 | 0.49713727548160735 |
