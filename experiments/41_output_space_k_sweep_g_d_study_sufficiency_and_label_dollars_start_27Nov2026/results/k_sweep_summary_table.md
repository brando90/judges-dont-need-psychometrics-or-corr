Test normalized ranked probability score (RPS; RPS divided by K - 1; 0 to 1; lower is better), mean over five split seeds, Experiment 41 K-sweep PREREGISTERED run (ridge penalty fixed at 0.01 for every head; at most 35 iterations). gap = the first-named head minus the CE-trained head (positive = CE-trained head better). Columns 'matched' pass each head's scalar mean score through its own validation-fitted multinomial-logit calibration map (same family for every head). MSE = the head trained by mean squared error on the scalar rating (it was trained in the preregistered run). The sensitivity run with the penalty chosen on validation is in k_sweep_l2_sensitivity_tables.md.

| view | K | carries signal | constant nRPS | CE raw | CE matched | Pearson raw | Pearson matched | MSE raw | MSE matched | raw gap (Pearson minus CE) | matched gap (Pearson minus CE; range over seeds) | matched gap (MSE minus CE) | matched gap (Pearson minus CE) at cap 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| popquorn_offensiveness | 2 | yes | 0.1156 | 0.1153 | 0.1153 | 0.2430 | 0.1154 | 0.1146 | 0.1152 | +0.1277 | +0.0001 (+0.0000 to +0.0001) | -0.0001 | +0.0001 |
| popquorn_offensiveness | 3 | yes | 0.1529 | 0.1523 | 0.1521 | 0.2476 | 0.1522 | 0.1697 | 0.1518 | +0.0952 | +0.0001 (+0.0001 to +0.0002) | -0.0003 | +0.0001 |
| popquorn_offensiveness | 5 | yes | 0.1501 | 0.1494 | 0.1489 | 0.2538 | 0.1489 | 0.2134 | 0.1486 | +0.1044 | -0.0000 (-0.0001 to +0.0001) | -0.0004 | -0.0000 |
| popquorn_politeness | 2 | yes | 0.2489 | 0.2312 | 0.2243 | 0.2274 | 0.2267 | 0.2222 | 0.2221 | -0.0037 | +0.0024 (+0.0019 to +0.0031) | -0.0022 | +0.0024 |
| popquorn_politeness | 3 | yes | 0.2184 | 0.2068 | 0.2010 | 0.2369 | 0.2026 | 0.2526 | 0.1988 | +0.0301 | +0.0016 (+0.0012 to +0.0020) | -0.0022 | +0.0016 |
| popquorn_politeness | 5 | yes | 0.1758 | 0.1699 | 0.1632 | 0.2446 | 0.1641 | 0.1993 | 0.1613 | +0.0747 | +0.0009 (+0.0007 to +0.0011) | -0.0018 | +0.0009 |
| summeval_expert | 2 | yes | 0.2001 | 0.1975 | 0.1961 | 0.2445 | 0.1961 | 0.1952 | 0.1963 | +0.0470 | +0.0000 (-0.0005 to +0.0007) | +0.0003 | +0.0000 |
| summeval_expert | 3 | yes | 0.1517 | 0.1503 | 0.1495 | 0.2407 | 0.1496 | 0.1647 | 0.1496 | +0.0904 | +0.0002 (-0.0003 to +0.0006) | +0.0001 | +0.0002 |
| summeval_expert | 5 | yes | 0.1449 | 0.1432 | 0.1408 | 0.2420 | 0.1415 | 0.2052 | 0.1413 | +0.0988 | +0.0006 (-0.0011 to +0.0016) | +0.0005 | +0.0006 |
| summeval_crowd | 2 | no | 0.2233 | 0.2237 | 0.2232 | 0.2566 | 0.2232 | 0.2251 | 0.2232 | +0.0329 | -0.0000 (-0.0002 to +0.0000) | -0.0001 | -0.0000 |
| summeval_crowd | 3 | no | 0.1516 | 0.1518 | 0.1516 | 0.2517 | 0.1516 | 0.1605 | 0.1516 | +0.1000 | -0.0000 (-0.0002 to +0.0000) | -0.0000 | -0.0000 |
| summeval_crowd | 5 | no | 0.1277 | 0.1278 | 0.1280 | 0.2560 | 0.1279 | 0.1537 | 0.1280 | +0.1282 | -0.0000 (-0.0002 to +0.0001) | +0.0000 | -0.0000 |
| dices990 | 2 | yes | 0.1867 | 0.1843 | 0.1832 | 0.2598 | 0.1838 | 0.1829 | 0.1827 | +0.0756 | +0.0006 (+0.0004 to +0.0008) | -0.0005 | +0.0006 |
| dices990 | 3 | yes | 0.1980 | 0.1953 | 0.1941 | 0.2592 | 0.1948 | 0.2573 | 0.1936 | +0.0639 | +0.0006 (+0.0004 to +0.0008) | -0.0006 | +0.0006 |
| chaosnli_snli_mnli | 3 | yes | 0.1950 | 0.1937 | 0.1937 | 0.2512 | 0.1937 | 0.2357 | 0.1941 | +0.0575 | -0.0000 (-0.0002 to +0.0001) | +0.0004 | -0.0000 |
| chaosnli_alphanli | 2 | no | 0.2501 | 0.2515 | 0.2514 | 0.2535 | 0.2513 | 0.2563 | 0.2514 | +0.0020 | -0.0000 (-0.0008 to +0.0005) | +0.0000 | -0.0000 |
