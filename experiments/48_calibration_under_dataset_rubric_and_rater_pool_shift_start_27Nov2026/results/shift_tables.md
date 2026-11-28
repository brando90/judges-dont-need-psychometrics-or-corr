## Arm A: frozen LLM judges, primary score

| shift type | pairs (judge x direction) | gap reopens | calibration fails outright (worse than raw) |
|---|---|---|---|
| near | 20 | 9 | 6 |
| far | 40 | 33 | 21 |
| leave_one_set_out | 20 | 10 | 6 |
| rubric | 2 | 1 | 0 |

Per source -> target (over judges): primary score S, gain retained F = (S_raw - S_shift)/(S_raw - S_in), range over judges.

| shift | source -> target | score | judges | F range | D = S_shift - S_in range | reopens | fails |
|---|---|---|---|---|---|---|---|
| near | SNLI -> MNLI | brier | 5 | -0.59 to 0.51 | +0.0292 to +0.0554 | 4 | 2 |
| near | MNLI -> SNLI | brier | 5 | -5.10 to 0.16 | +0.0239 to +0.0499 | 5 | 4 |
| near | DICES-350 -> DICES-990 | nrps | 5 | 0.54 to 0.98 | +0.0010 to +0.0066 | 0 | 0 |
| near | DICES-990 -> DICES-350 | nrps | 5 | 0.89 to 1.00 | +0.0000 to +0.0024 | 0 | 0 |
| far | SNLI -> DICES-350 | nrps | 5 | 0.13 to 0.61 | +0.0072 to +0.0381 | 4 | 0 |
| far | SNLI -> DICES-990 | nrps | 5 | 0.11 to 0.59 | +0.0057 to +0.0317 | 4 | 0 |
| far | MNLI -> DICES-350 | nrps | 5 | -1.23 to 0.72 | +0.0099 to +0.0215 | 1 | 1 |
| far | MNLI -> DICES-990 | nrps | 5 | -3.30 to 0.49 | +0.0150 to +0.0275 | 5 | 1 |
| far | DICES-350 -> SNLI | brier | 5 | -64.86 to -0.76 | +0.0750 to +0.3018 | 5 | 5 |
| far | DICES-350 -> MNLI | brier | 5 | -7.01 to 0.51 | +0.0472 to +0.1716 | 4 | 4 |
| far | DICES-990 -> SNLI | brier | 5 | -71.04 to -3.85 | +0.2074 to +0.3560 | 5 | 5 |
| far | DICES-990 -> MNLI | brier | 5 | -9.32 to -0.57 | +0.1514 to +0.2213 | 5 | 5 |
| leave_one_set_out | other three -> SNLI | brier | 5 | -4.57 to 0.31 | +0.0204 to +0.0506 | 5 | 4 |
| leave_one_set_out | other three -> MNLI | brier | 5 | -0.45 to 0.52 | +0.0308 to +0.0599 | 4 | 2 |
| leave_one_set_out | other three -> DICES-350 | nrps | 5 | 0.78 to 0.92 | +0.0017 to +0.0093 | 0 | 0 |
| leave_one_set_out | other three -> DICES-990 | nrps | 5 | 0.18 to 0.83 | +0.0052 to +0.0105 | 1 | 0 |
| rubric | SummEval relevance -> SummEval consistency | nrps | 1 | 0.59 to 0.59 | +0.0674 to +0.0674 | 0 | 0 |
| rubric | SummEval consistency -> SummEval relevance | nrps | 1 | -0.06 to -0.06 | +0.0856 to +0.0856 | 1 | 0 |

## Arm B: Experiment 41 linear heads, G = nRPS(Pearson head + map) - nRPS(CE head, raw)

| arm | source -> target | in-domain G (seed mean) | shifted G (seed mean) | shifted G seed range | seeds with CI > 0 | reopens |
|---|---|---|---|---|---|---|
| rater_pool | summeval_expert|K2 -> summeval_crowd|K2 | -0.0005 | +0.0028 | -0.0004 to +0.0050 | 3/5 | True |
| rater_pool | summeval_crowd|K2 -> summeval_expert|K2 | -0.0015 | +0.0045 | +0.0015 to +0.0079 | 5/5 | True |
| rater_pool | summeval_expert|K3 -> summeval_crowd|K3 | -0.0002 | +0.0019 | +0.0015 to +0.0021 | 4/5 | False |
| rater_pool | summeval_crowd|K3 -> summeval_expert|K3 | -0.0006 | +0.0033 | +0.0025 to +0.0047 | 5/5 | True |
| rater_pool | summeval_expert|K5 -> summeval_crowd|K5 | +0.0001 | +0.0243 | +0.0202 to +0.0282 | 5/5 | True |
| rater_pool | summeval_crowd|K5 -> summeval_expert|K5 | -0.0017 | +0.0273 | +0.0235 to +0.0351 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> popquorn_offensiveness|K2 | +0.0001 | +0.2278 | +0.2006 to +0.2631 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> popquorn_politeness|K2 | -0.0044 | +0.0270 | +0.0166 to +0.0342 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> summeval_expert|K2 | -0.0015 | +0.0427 | +0.0298 to +0.0501 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> summeval_crowd|K2 | -0.0005 | +0.0250 | +0.0177 to +0.0341 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> dices990|K2 | -0.0005 | +0.1165 | +0.1098 to +0.1224 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> chaosnli_alphanli|K2 | -0.0001 | +0.0118 | +0.0058 to +0.0168 | 3/5 | True |
| leave_one_dataset_out | other same-K views -> popquorn_offensiveness|K3 | -0.0001 | +0.2454 | +0.2191 to +0.2782 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> popquorn_politeness|K3 | -0.0042 | +0.0085 | +0.0045 to +0.0111 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> summeval_expert|K3 | -0.0006 | +0.0469 | +0.0348 to +0.0532 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> summeval_crowd|K3 | -0.0002 | +0.0428 | +0.0356 to +0.0468 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> dices990|K3 | -0.0005 | +0.1655 | +0.1626 to +0.1713 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> chaosnli_snli_mnli|K3 | -0.0001 | +0.0846 | +0.0727 to +0.0985 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> popquorn_offensiveness|K5 | -0.0004 | +0.2576 | +0.2278 to +0.2922 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> popquorn_politeness|K5 | -0.0057 | +0.0165 | +0.0077 to +0.0240 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> summeval_expert|K5 | -0.0017 | +0.0419 | +0.0349 to +0.0490 | 5/5 | True |
| leave_one_dataset_out | other same-K views -> summeval_crowd|K5 | +0.0001 | +0.0207 | +0.0153 to +0.0242 | 5/5 | True |

rater_pool: 5 of 6 reopen; leave_one_dataset_out: 16 of 16 reopen
