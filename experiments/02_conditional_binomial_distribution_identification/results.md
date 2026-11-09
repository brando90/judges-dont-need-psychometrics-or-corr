# Conditional-distribution study: live results

**Status:** implementation complete; parent multi-provider review and main publication pending.
**Last updated:** 11-08-2026 21:18 PST

| Phase | Status |
| --- | --- |
| Procedure/source hashes fixed before measurement | DONE: 9139470, test-only amendment 5c0e9fc |
| Eight analytic/scorer/split checks | DONE |
| Bounded CPU run | DONE: 768 conditions, 5,376 selections, 6,912 candidate evaluations; zero failures |
| Common-metric report and historical reproduction | DONE; old toy reproduced unchanged; exact public aggregate commits linked |
| Independent reconstruction | DONE: all 5,376 selections and 419,328 selected metric values, all summaries and hashes |
| Paper build and visual inspection | DONE: three pages, zero errors, all pages inspected; five underfull-box warnings |
| Requested multi-provider review / main publication | PENDING: parent owns sequencing and landing |

Cross-entropy (CE) selected the oracle in 249/256 one-rating replications; Brier and ranked probability score each selected it in 245/256. All three selected the oracle in 256/256 two-rating and three-rating replications. Pearson, Spearman, Kendall and normalized scalar mean squared error always retained all five shared-mean candidates. The tie includes the oracle; superiority is not independent of tie policy. Exact common-metric results and Monte Carlo intervals are in [the full report](RESULTS_conditional_identification.md) and `results/summary.json`.

Zero provider calls and $0 provider inference charges. The run took 13.51 seconds; synthetic evidence does not establish human generalization.

**TLDR-end:** [judge objectives: conditional identification] The full controlled study completed. Scalar criteria do not identify distribution shape in this constructed population; finite-sample proper-loss selection remains imperfect.
