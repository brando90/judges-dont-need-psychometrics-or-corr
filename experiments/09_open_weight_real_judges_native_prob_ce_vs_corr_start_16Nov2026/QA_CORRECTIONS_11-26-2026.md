# Experiment 09 QA corrections (11-26-2026)

Date: 11-26-2026. CPU Python verification of retained records; no new model calls or scientific runs.

**g3-M1 — CONFIRMED.** The old label “individual-rating (`n = 195` ratings) correlations & signed bias” combines individual-rating correlations with artifact-weighted means and bias. Corrected 11-26-2026: the printed means and biases use equal weight per artifact (75 artifacts); the correlations use individual ratings (195 ratings). The old biases +0.48/+1.53 for Qwen3 and +0.14/+2.11 for Qwen2.5 remain in the original text. Individual-rating values recomputed by subtracting saved means are:

- `q3_4b_2507`, `ce_lora_val_rps_3`: old artifact-weighted bias +0.483058, human mean 2.626667; corrected individual-rating bias +0.562560, judge mean 3.116406, human mean 2.553846.
- `q3_4b_2507`, `pearson_lora_train_2`: old artifact-weighted bias +1.533555, human mean 2.626667; corrected individual-rating bias +1.585030, judge mean 4.138876, human mean 2.553846.
- `q25c_7b`, `ce_lora_val_rps_3`: old artifact-weighted bias +0.135848, human mean 2.626667; corrected individual-rating bias +0.203494, judge mean 2.757340, human mean 2.553846.
- `q25c_7b`, `pearson_lora_train_2`: old artifact-weighted bias +2.114081, human mean 2.626667; corrected individual-rating bias +2.179101, judge mean 4.732947, human mean 2.553846.

**g3-M2 — CONFIRMED evidence limitation, still open.** The old held-out RPS/CE scores, intervals, sign-flip values, repeat-stability claims, human ceiling, and “verified 11-16-2026 15:00 PST” / “All GPUs released” receipts remain historical claims. Saved aggregate arithmetic reproduces the verbal-pool RPS difference 0.0416708638. Private individual ratings, family loss vectors, repeated observations and timestamped GPU receipts are absent in this checkout, so this audit does not independently validate those losses, intervals, p-values or run chronology. No missing observations were fabricated.

**g3-L2 — CONFIRMED.** The Stage 3 link used an unavailable `/Users/brandomiranda/.worktrees/...` location. Its link is repaired to the report in this checkout, with the old target retained as inline text in that report.

**Test receipt (11-26-2026):** The existing verifier exits 0 with 693 checks, but its code skips private-rating reconstruction when the hard-coded dataset path is absent. This pass does not resolve g3-M2. Its generated verification receipt was copied into the QA folder and the original historical receipt retained byte-for-byte.
