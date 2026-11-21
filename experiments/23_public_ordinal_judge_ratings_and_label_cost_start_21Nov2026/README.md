# Public human ratings, judge training and annotation cost | 11-21-2026

As of 11-21-2026. On KonIQ-10k, a cross-entropy (CE) fit beat a Pearson fit and uniform on held-out ranked probability score (RPS) from 12 training ratings, or 24 when uncertainty also included which training sample was acquired. On expert-rated SummEval factual consistency, the original GPT-4 probability calibration never cleared those two baselines. On both, spending the same labels to calibrate a score into grade probabilities was a stronger comparison. [Research report](RESULTS_11-21-2026.md) · [Google Doc](https://docs.google.com/document/d/1lVju906fOGmBSyi7ya4j-rmvcy4U9GnRdqTt71qlqe8/edit) · [Tables for all budgets and cost scenarios](BUDGET_TABLES_11-21-2026.md)

| Question | Evidence |
|---|---|
| Which public benchmarks preserve human disagreement? Who really trains on correlation? | [Primary-paper and release inventory](PUBLIC_RATING_EVIDENCE_11-21-2026.md) |
| Which training objectives and held-out outcomes were decided in advance? | [Fixed benchmark design](PROTOCOL_11-21-2026.md) · [Exact fitting implementation before scoring](https://github.com/brando90/judges-dont-need-psychometrics-or-corr/commit/8d238e6) |
| What are the human and judge inputs? | [SummEval experts and archived GPT-4](summeval/DATA_11-21-2026.md) · [KonIQ crowd votes and image features](koniq/DATA_11-21-2026.md) |
| What do the test comparisons, uncertainty bands and metrics mean? | [Interpreting and reproducing the outputs](HOW_TO_READ_RESULTS.md) · [Learning curves as PDF](figures/public_rating_budget.pdf) · [Independent scoring verification](VERIFICATION_11-21-2026.md) |
| How sensitive was the result to judge design? | [Jointly trained multivariable models](continuous/RESULTS_11-21-2026.md) · [Same distribution calibration for all scores](matched_calibration/RESULTS_11-21-2026.md) · [Smoothing and training-sample sensitivity plan](SENSITIVITIES_11-21-2026.md) |

Experiment 22 owns the coworker's complementary HelpSteer2, ChaosNLI and DICES scope. Historical formal-specification coverage grades and frozen prospective VeriBench judge forecasts are available from their original experiments. [Experiment 22](../22_non_circular_evaluation_protocol_and_public_rating_sets_start_21Nov2026/README.md) · [VeriBench registration](../21_fresh_task_families_matched_open_closed_judges_start_21Nov2026/veribench/STATUS_11-21-2026.md)
