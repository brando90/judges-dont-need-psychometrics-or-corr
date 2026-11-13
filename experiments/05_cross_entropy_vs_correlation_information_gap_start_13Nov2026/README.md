# 05_cross_entropy_vs_correlation_information_gap

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/05_cross_entropy_vs_correlation_information_gap_start_13Nov2026/README.md>

**TLDR:** Brando's Impromptu Call action item 1: derive the relationship between cross-entropy (CE) and correlation and quantify the information gap. Done on paper and checked numerically ($0, CPU, no model calls). Action item 2 (real VeriBench comparison) is a separate run because the human ratings are private on Brando's Mac: see [PLAN_06_veribench_comparison.md](PLAN_06_veribench_comparison.md).

**Objective:** show, with a proof and exact numbers, that correlation identifies only the mean of each forecast while CE identifies the whole human rating distribution, and give the gap in nats.

**Files:** [DERIVATION.md](DERIVATION.md) (results 1–4), [gap_check.py](gap_check.py) (checks), [results.md](results.md), [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md), [PLAN_06_veribench_comparison.md](PLAN_06_veribench_comparison.md).

**Run:** `python gap_check.py` (numpy, scipy).

**Harness and model:** Claude Code headless worker, Sonnet 5.5 (ultimate_brando9). No model judge ran. Label: no DT.
