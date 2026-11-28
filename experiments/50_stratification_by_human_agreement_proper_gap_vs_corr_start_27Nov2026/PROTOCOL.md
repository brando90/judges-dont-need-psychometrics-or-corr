# Experiment 50 protocol: proper-score gaps and correlation stratified by human agreement (frozen 11-27-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/50_stratification_by_human_agreement_proper_gap_vs_corr_start_27Nov2026/PROTOCOL.md>

**TLDR:** The protocol in the paper promises scores split by human agreement. Bin test items into quintiles of human rating entropy, computed on a held-out half of each item's ratings, and report proper-score gaps, the skill score, correlation and majority agreement per bin. Frozen before the script exists.

**As of:** 11-27-2026 (Friday, Pacific time). CPU, $0.

## Design

- Binning without double-dipping: for every item with at least 4 ratings, its ratings are split at random (seed 20261127 + item position) into half A and half B. Half A gives the empirical Shannon entropy (nats); items are cut into five equal-count quintile bins of that entropy within each set or view (Q1 = most agreement). All scores in a bin use half B only. Sets with fewer than 4 ratings per item (SummEval experts: 3) are binned by the 3-rating entropy and scored on the same ratings; that arm is labelled "same-ratings binning, descriptive".
- **Arm A (Experiment 41 heads).** 16 views, split seeds 0-4, preregistered heads (L2 0.01, at most 35 iterations), matched map as Exp 41. Per bin: raw gap and matched gap (Pearson-trained minus CE-trained test normalized RPS; positive = CE-trained better), with paired item bootstrap 95% intervals (2,000 draws); Pearson correlation of each head's expected score with the half-B mean human score inside the bin.
- **Arm B (Experiment 46 real judges).** Frontier panel and G-Eval GPT-4, every set. Per bin: each judge's primary proper score and skill score (base = cross-fitted marginal), Pearson, top-1 agreement with the half-B modal label; and the Kendall tau between the judge rankings by Pearson and by the primary score inside the bin.
- **Predictions.** R1: the raw proper-score gap (Arm A) is larger in Q5 (least agreement) than in Q1, interval of the difference excluding 0, on most views with signal. R2: within-bin Pearson is lower than the pooled Pearson (range restriction) in every bin. R3: top-1 agreement falls from Q1 to Q5 for every judge while the proper-score ranking changes less. Each prediction is reported as supported or not.
