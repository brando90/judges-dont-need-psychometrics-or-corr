# Experiment 48 protocol: does a calibration map fitted on one dataset, rubric or rater pool transfer to another? (frozen 11-27-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/48_calibration_under_dataset_rubric_and_rater_pool_shift_start_27Nov2026/PROTOCOL.md>

**TLDR:** "Calibration closes the gap" assumes in-domain held-out labels. We fit the calibrator on a source and score it on a different target (leave-one-dataset-out, rubric shift, rater-pool shift) and ask whether the proper-score gap reopens. Frozen before the script exists.

**As of:** 11-27-2026 (Friday, Pacific time). CPU, $0.

## Arm A (primary): frozen real LLM judges, same judge across sets

- Judges and data as in Experiment 46: the five frontier judges on ChaosNLI SNLI, MNLI, DICES-350, DICES-990 (all K = 3) and alphaNLI (K = 2); G-Eval GPT-4 on SummEval consistency and relevance (K = 5).
- Calibrator: multinomial logit on [log q, 1], ridge 1e-3, fitted by individual-rating CE (the Experiment 46 H3 map).
- **In-domain reference:** 5-fold group cross-fit on the target. **Shifted:** fitted on all items of one source set, applied to every target item. Pairs: every ordered (source, target) pair among the four K = 3 sets (12 pairs, of which SNLI<->MNLI and DICES-350<->DICES-990 are "near" shifts and NLI<->DICES are "far"); also leave-one-set-out (fit on the other three K = 3 sets pooled). Rubric shift: SummEval consistency <-> relevance (same items and experts, different rubric).
- Score: the target's Experiment 46 primary proper score S (Brier for nominal NLI, normalized RPS for DICES and SummEval); CE secondary.
- Quantities: reopened gap D = S(shifted) - S(in-domain); gain retained F = (S(raw) - S(shifted)) / (S(raw) - S(in-domain)). Paired item-cluster bootstrap, 4,000 draws, 95% intervals.
- **Decision rule.** The gap "reopens" for a pair if the 95% interval of D excludes 0 from above AND F < 0.5. "Calibration fails outright" if S(shifted) > S(raw) with an interval excluding 0.

## Arm B (secondary): Experiment 41 linear heads

- Rater-pool shift: SummEval expert vs crowd (same 6,400 item-aspects, different raters). A head trained on the target's training split; its map fitted on the source pool's validation ratings of the same validation items; scored on the target pool's test ratings. Both directions, K = 2, 3, 5 views, split seeds 0-4.
- Leave-one-dataset-out within K: for each target view, the Pearson-trained head of the target, with a map fitted on the pooled (standardized-score, rating) pairs of the other same-K views' Pearson-trained heads' validation items.
- Quantity: G_shift = nRPS(Pearson-trained head + shifted map) - nRPS(CE-trained head, raw), compared with the in-domain G from Experiment 41's design. Same bootstrap as Experiment 41 (2,000 draws, per seed).
- Decision rule: "reopens" if the seed-mean G_shift > 0.0024 (Exp 41's in-domain bound) and at least 3 of 5 seeds have intervals above 0.
