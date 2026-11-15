# How to read the results of experiment 07

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Every judge is refitted three ways on experiment 13's exact training tasks and scored on the same 500 fresh test tasks per seed against the **true** rating distribution. The three ways differ only in the optimizer's starting point and budget; the choice among starts uses the **training** objective, never test data.

## What is compared

One judge family (softmax over the 6 ratings of a linear map of 13 cosine/sine features of x), trained with seven objectives: cross-entropy (CE, convex), ranked probability score (RPS), Pearson, Pearson + calibrated CE head, smooth Spearman, smooth Kendall, ½ Pearson + ½ smooth Spearman. Each objective under three conditions:

| Condition | Start | Budget (L-BFGS-B) |
| --- | --- | --- |
| orig | training rating histogram + 0.01 seeded perturbation (as experiment 13) | 400 iterations, 600 evaluations |
| long | same start | 2,000 iterations, 4,000 evaluations |
| best | the lowest training objective among 5 starts: the original plus 4 with N(0, 0.5²) added to every weight | 2,000 iterations each |

## Which data each number uses

| Data | Role |
| --- | --- |
| Training tasks | n ∈ {4, 8, 16, 32, 64, 128, 256, 512, 1024}, 3 sampled ratings each; used to fit and to choose among starts. |
| Test tasks | 500 fresh tasks per seed, scored against the true distribution; never used for fitting or choosing. |
| Repetition | 30 seeds (0–29), scenarios A and B. |

## Metrics

Held-out test RPS excess (judge's RPS minus the true distribution's; 0 is perfect; lower is better) and test CE excess, KL(true ‖ judge) in nats (lower is better); test Pearson and Spearman correlation between the judge's expected rating and the true mean (higher is better). Logged per fit: training objective, scipy success flag, iterations and whether the budget was hit.

## How to read a difference

"Best-of-5 minus original" is **the same test metric for the same objective under the two conditions, on the same seed's tasks**; negative means restarts and budget helped. Crossovers compare each objective's judge with the CE-trained judge under the same condition.

## Uncertainty

95% percentile bootstrap intervals over the 30 seeds (10,000 draws). No p-values were computed. Synthetic toy that works by construction; not evidence about real ratings.

**TLDR-end:** [judge objectives: experiment 07] Same tasks, three optimizer conditions per objective; starts are chosen on training data only.
