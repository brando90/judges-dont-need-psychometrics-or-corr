# Experiment 07: results

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026/results.md>

**Status, 11-14-2026:** sweep complete (540 rows; the first launch stopped at 532 when the session ended and the runner resumed the last 8, scenario B at n = 1,024, seeds 22–29). Kept active because the one-pager and paper depend on it.

**TLDR:** The experiment 13 and 06 conclusions do not depend on the optimizer. Best-of-5 starts with a 5× budget find a better training objective in 57–90% of the Pearson, Spearman and ½ Pearson + ½ Spearman fits, yet held-out test RPS and cross-entropy (CE) barely change, and every crossover on held-out test RPS against the CE-trained judge is unchanged. The Pearson-type objectives are hard to optimize: their original fits almost never converged within 400 iterations, and most still hit the 2,000-iteration budget. Synthetic toy; not evidence about real ratings.

| Phase | Status |
| --- | --- |
| Protocol committed before measurement (`d9d1c1d`) | DONE |
| Reproduction check | DONE: experiment 06 exactly; experiment 13's CE and RPS fits to about 1e-17; experiment 13's Pearson fits differ slightly across machines (README) |
| Sweep: 2 scenarios × 9 training sizes × 30 seeds × 7 objectives × 3 conditions | DONE |
| Report | DONE |

## Main results (held-out test tasks, against the true distribution; 30 seeds)

- **Convergence of the original fits** (400 iterations): CE, RPS, Kendall and the calibrated head converge in 97–100% of fits. Pearson-trained converges in 0% of fits from 32 training tasks on (all hit the budget); ½ Pearson + ½ Spearman in 0–3%; Spearman in 7–52% from 16 tasks on. With 2,000 iterations, 91% of Pearson fits with n ≥ 32 still hit the budget.
- **Restarts improve the training objective, not the held-out result.** At 1,024 training tasks, test RPS excess of the Pearson-trained judge goes from 0.0183 (original) to 0.0173 (best of 5, 5× budget) in scenario A, against 0.0011 for the CE-trained judge; in scenario B from 0.0949 to 0.0918, against 0.0042.
- **Crossovers on held-out test RPS** (smallest number of training tasks from which the CE-trained judge is significantly better at every larger size; 3 ratings per task) are identical under the original and best-of-5 conditions: scenario A, 64 tasks (192 ratings) for Pearson, Spearman and ½ Pearson + ½ Spearman, 16 for Kendall, 4 for Pearson + calibrated head; scenario B, 4 tasks (12 ratings) for all. On held-out test CE, two scenario-A crossovers move by one grid step (Spearman 32 → 64 tasks, Kendall 32 → 16).
- **The calibrated head is sensitive to which Pearson judge it is built on**: choosing the Pearson judge with the best training correlation sometimes gives a worse head (scenario A, 32 tasks: test RPS excess 0.086 → 0.121).

The prediction in the README holds: no correlation-trained judge becomes competitive with the CE-trained judge, and crossovers move by at most one grid step.

Full tables: [results/tables.md](results/tables.md). Figure: [fig_restarts_vs_n.png](fig_restarts_vs_n.png). Reading guide: [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md). Regenerate with `python3 experiments/07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026/scripts/make_report.py`.

**TLDR-end:** [judge objectives: experiment 07] A 5× optimizer budget and best-of-5 restarts improve the correlation objectives' training fit but leave every held-out test RPS crossover unchanged, so experiments 13 and 06 do not hinge on the optimizer.
