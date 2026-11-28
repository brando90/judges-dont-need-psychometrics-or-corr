# Experiment 56 protocol (frozen before the script was run): how many held-out human ratings does per-judge calibration of the frontier LLM judges need?

**Why.** Experiment 46 reports that per-judge 5-fold calibration removes the two Spearman-vs-skill reversals and makes every frontier judge's skill positive, and it cites "100–200 labels per task" for the cost of that remedy. That figure comes from Experiment 47, which measured Experiment 41's linear heads, not the frontier judges. This experiment measures the label budget on the frontier judges themselves, so the abstract can quote a real-judge number.

**Frozen inputs (no model calls, \$0).** The frozen verbalized forecasts of GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3 and DeepSeek V4.1 Flash (Experiment 28) on ChaosNLI SNLI (600 items), MNLI-m (600), alphaNLI (400), DICES-350 (350) and DICES-990 (400), loaded with Experiment 46's `load_all`. Scorers, probability floor, calibrator (`fit_cal`: multinomial logit on [log q, 1], ridge 1e-3, shrunk toward the identity map, fitted by rating cross-entropy) and primary rule per set (Brier for the NLI sets, normalized RPS for DICES) are Experiment 46's, unchanged.

**Design.** 25 cases (5 judges x 5 sets). For each case, 50 draws d. In each draw:
1. Split the items 50/50 at the group level into a calibration pool P and a test set T (seed SeedSequence([20261127, 56, set index, judge index, d])).
2. For each budget n in {10, 25, 50, 100, 200} (capped at |P|): draw n items from P without replacement and **one** rating per item (sampled from that item's observed ratings). Fit the calibrator on these n single ratings and apply it to T.
3. Reference: the calibrator fitted on all of P with all ratings ("full").
4. Base rate: the marginal rating histogram of all ratings in P.
5. Skill on T: SS = 1 − mean S(q, h) / mean S(base, h), with h the item's full human histogram and S the set's primary rule.

**Reported per case.** Mean over draws of SS_raw, SS_n and SS_full; the share of draws in which SS_n < SS_raw (calibration on n ratings hurts) and SS_n < 0.

**Decision quantity.** n\* = the smallest budget n with mean(SS_full − SS_n) ≤ 0.01 (within 0.01 skill of full calibration). Cases where the full calibration gain SS_full − SS_raw is under 0.01 are reported but excluded from the n\* summary (nothing to close).

**Reversal check.** For GPT-6 Astra vs GLM 5.3 on SNLI and MNLI (the two Experiment 46 reversals), the share of draws at each n in which calibrated Astra still has lower skill than calibrated GLM.

**Predictions (written before running).**
- H1: among cases with full gain ≥ 0.01, n\* ≤ 200 in at least half.
- H2: at n = 10, calibration hurts (SS_n < SS_raw) in more than 10% of draws in at least one case.

**Deviations** will be labelled in the README. Intervals resample items only; rater sampling enters through the single-rating draws.
