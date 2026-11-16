# Experiment 08: results

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/results.md>

**Status, 11-14-2026:** sweep complete (4,400 rows: 4 scenarios × 11 training sizes × 100 seeds). Kept active because the one-pager and paper depend on it.

**TLDR:** By the decision rule fixed before measurement, neither hypothesis holds as stated. Correlation training beats cross-entropy (CE) training on held-out distribution accuracy among these four targets only in scenario A, where the target varies mainly in rater disagreement and its mean says little: up to 32 training tasks (96 ratings). Where the mean carries the information (B, C and D, including D, the informative linear-mean case), the CE-trained judge is better from 4 tasks (12 ratings). The CE-trained judge also has equal or higher held-out correlation until data are plentiful; the Pearson-trained judges overtake on their own metric only from 512–1,024 tasks. Synthetic toy; not evidence about real ratings.

| Phase | Status |
| --- | --- |
| Protocol and hypotheses committed before measurement (`4e2a8aa`) | DONE |
| Amendment: fallback when validation correlation is undefined (`094edde`) | DONE: fallback used in 1 of 4,400 rows (scenario C, 4 training tasks, seed 9); the 208 rows written before the amendment completed without needing it |
| Reproduction of experiment 13's data (scenarios A, B) | DONE: CE and ranked probability score (RPS) fits match to about 1e-17 |
| Sweep and report | DONE |

## Verdicts (paired by seed, 95% bootstrap intervals over 100 seeds)

| Pre-registered prediction | Result |
| --- | --- |
| H1: the Pearson-trained judge (②) has significantly lower held-out test ranked probability score (RPS) than the CE-trained judge (④) at the smallest sizes, most clearly in C and D | Fails: only in scenario A (4–32 tasks); never in B, C or D |
| H1: the crossover is larger in C and D than in B | Fails: 4 tasks (12 ratings) in B, C and D alike |
| H2: the CE-trained judge's held-out test correlation is never significantly lower | Fails at large n: lower than ② from 512 tasks in B and C and from 1,024 in D; never in A |
| H2: validation-tuned CE (③) is never significantly worse on test RPS than ① or ② | Fails at the smallest sizes: worse than ② at 4–16 tasks in A and at 4 tasks in D, and worse than ① at 4 tasks in A |

**QA check, 11-16-2026:** experiment 13's CE code clips and renormalizes probabilities at 1e-6 but returns an inconsistent gradient when clipping is active. Refitting every CE fit with an exact CE changed no reported cell mean, crossover or hypothesis check; one chosen-ridge share in results/tables.md would move from 69% to 68% (C, 256 tasks) (details in [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md); evidence in [results/ce_gradient_check.json](results/ce_gradient_check.json)). Saved measurements are unchanged.

## Main results

Ratings needed, as the smallest number of training tasks from which the first judge has significantly lower held-out test RPS at every larger size (3 ratings per task):

| Scenario | ④ CE-trained vs ② Pearson-trained | ③ tuned CE vs ① tuned Pearson |
| --- | --- | --- |
| A: mean nearly constant, disagreement varies | 64 tasks (192 ratings) | 32 tasks (96 ratings) |
| B: nonlinear mean, disagreement varies | 4 (12) | 4 (12) |
| C: nonlinear mean, constant concentration | 4 (12) | 4 (12) |
| D: linear mean, constant concentration | 4 (12) | 16 (48) |

Held-out test RPS excess over the true distribution (mean over 100 seeds), and held-out test Pearson correlation with the true mean:

| Scenario, n | constant | ④ CE | ③ tuned CE | ② Pearson | ① tuned Pearson | test Pearson ④ / ② / ① |
| --- | --- | --- | --- | --- | --- | --- |
| A, 4 | 0.0267 | 0.0106 | 0.0140 | 0.0096 | 0.0097 | 0.148 / −0.013 / 0.056 |
| A, 2048 | 0.0096 | 0.0007 | 0.0007 | 0.0253 | 0.0807 | 0.891 / 0.845 / 0.802 |
| B, 4 | 0.0923 | 0.0740 | 0.0829 | 0.0901 | 0.0898 | 0.378 / 0.019 / 0.093 |
| B, 2048 | 0.0670 | 0.0037 | 0.0036 | 0.0951 | 0.0326 | 0.970 / 0.985 / 0.992 |
| C, 4 | 0.0877 | 0.0682 | 0.0776 | 0.0830 | 0.0836 | 0.412 / 0.098 / 0.148 |
| C, 2048 | 0.0640 | 0.0032 | 0.0029 | 0.0893 | 0.0333 | 0.976 / 0.986 / 0.993 |
| D, 4 | 0.0654 | 0.0410 | 0.0481 | 0.0452 | 0.0474 | 0.364 / 0.125 / 0.083 |
| D, 2048 | 0.0449 | 0.0028 | 0.0027 | 0.0523 | 0.0359 | 0.970 / 0.983 / 0.989 |

- **Where the information lies matters more than complexity alone.** B has scenario A's varying disagreement plus an informative mean, and the CE-trained judge wins from 4 tasks there; only A, where the mean carries little signal, gives correlation training a low-data edge.
- **The Pearson-trained judge (②) becomes worse than a constant forecast.** Its held-out test RPS excess is significantly higher than predicting the training rating histogram for every task from 128 tasks in A, 8 in B, 16 in C and 256 in D (at 2,048 tasks the mean paired difference is +0.016, +0.028, +0.025 and +0.007, each 95% interval above zero; last table of [results/tables.md](results/tables.md), printed by `make_report.py` since 11-16-2026), while its test correlation reaches 0.85–0.99.
- **Validation tuning is noisy with few tasks.** Ridge chosen by 4-fold cross-validation over 4–16 training tasks varies widely across seeds (results/tables.md lists the chosen values), which is why ③ trails ② at the smallest sizes in A and D. Tuning Pearson's ridge by validation correlation (①) improves its test correlation but can worsen its distribution (scenario A at 2,048 tasks: RPS excess 0.081 vs 0.025 for ②).
- **The RPS-trained judge's test CE excess rises at large n in B, C and D** (figure, right column), consistent with RPS being less sensitive than CE to the probability given to rare ratings.

Full tables: [results/tables.md](results/tables.md). Figures: [fig_complexity_vs_n.png](fig_complexity_vs_n.png) (held-out test RPS and CE excess), [fig_complexity_correlation_vs_n.png](fig_complexity_correlation_vs_n.png) (held-out test Pearson). Reading guide: [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md). Regenerate with `python3 experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/make_report.py`.

**TLDR-end:** [judge objectives: experiment 08] Correlation training wins at low data only when the target varies mainly in disagreement and its mean says little; in the three tested targets with informative means, including the linear case, CE training wins from 12 ratings.
