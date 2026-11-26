# How to read the results of experiment 07 (QA annotation 11-26-2026)

**QA correction (11-26-2026):** Current verified corrections, evidence limits and score-first companion panels are in [QA_CORRECTIONS_11-26-2026.md](QA_CORRECTIONS_11-26-2026.md). Original wording and scored tables below are retained as the historical record.

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Non-convex objectives are refitted three ways; the cross-entropy (CE) reference uses one start at two budgets, on experiment 13's exact training tasks and scored on the same 500 fresh test tasks per seed against the **true** rating distribution. The three ways differ only in the optimizer's starting point and budget; the choice among starts uses the **training** objective, never test data.

## What is compared

One judge family (softmax over the 6 ratings of a linear map of 13 cosine/sine features of x), trained with six objectives plus a calibrated-head variant: cross-entropy (CE; unclipped softmax objective is convex), ranked probability score (RPS), Pearson, Pearson + calibrated CE head, smooth Spearman, smooth Kendall, ½ Pearson + ½ smooth Spearman. The three conditions, with the CE exception below:

| Condition | Start | Budget (L-BFGS-B) |
| --- | --- | --- |
| orig | training rating histogram + 0.01 seeded perturbation (as experiment 13) | 400 iterations, 600 evaluations |
| long | same start | 2,000 iterations, 4,000 evaluations |
| best | the lowest training objective among 5 starts: the original plus 4 with N(0, 0.5²) added to every weight | 2,000 iterations each |

The CE reference uses only the original start at the longer budget; its “best” record equals “long”. Only the other non-convex objectives use five starts. The head is a single CE refit on each chosen Pearson score. Budgets are 400/2,000 iterations and 600/4,000 function evaluations; five starts increase total work further.

## Which data each number uses

| Data | Role |
| --- | --- |
| Training tasks | n ∈ {4, 8, 16, 32, 64, 128, 256, 512, 1024}, 3 sampled ratings each; used to fit and to choose among starts. |
| Test tasks | 500 fresh tasks per seed, scored against the true distribution; never used for fitting or choosing. |
| Repetition | 30 seeds (0–29), scenarios A and B. |

## Metrics

Held-out test RPS excess (judge's RPS minus the true distribution's; 0 is perfect; lower is better) and test CE excess, Kullback–Leibler divergence KL(true ‖ judge) in nats (lower is better); test Pearson and Spearman correlation between the judge's expected rating and the true mean (higher is better). Logged per fit: training objective, scipy success flag, iterations and whether the budget was hit.

## How to read a difference

"Best-of-5 minus original" is **the same test metric for the same objective under the two conditions, on the same seed's tasks**; negative means restarts and budget helped. Crossovers compare each objective's judge with the CE-trained judge under the same condition.

## Uncertainty

95% percentile bootstrap intervals over the 30 seeds (10,000 draws). No p-values were computed. Evidence is limited to the stated synthetic family; it does not establish performance on real ratings.

## Numerical QA note (11-16-2026)

The inherited `experiments/archive/13_*/scripts/toy.py::ce_loss_grad` clips and renormalizes probabilities before evaluating CE, but returns a gradient that is inconsistent when clipping is active. This is a real implementation defect, not a failure of CE propriety. A bounded check encountered it in experiment 08's validation search (C, n = 4, seed 9, c = 0.1); replacing it in memory with stable log-softmax CE left that row's selected metrics unchanged to numerical precision. Impact checked 11-16-2026: refitting every CE fit of experiments 07, 08 and 13 with an exact CE (log-sum-exp value, exact gradient, no floor) preserved the reported cell means at printed precision, crossovers and hypothesis checks, with the ridge-share exception below. Fixed-ridge CE metrics agree with the saved rows within 1e-9. Experiment 07's calibrated-head metrics also agree with saved rows within 1e-9; experiment 13's finite-sample heads agree within 1e-9 only between exact and frozen CE reruns on this machine. Its saved Pearson fits and heads came from a different machine and do not reproduce to that tolerance. In experiment 08, 338 of the 4,400 validation-tuned CE judges (337 of them at the weakest ridge, c = 0.01, where clipping is encountered) move by at most 0.00023 in test RPS excess, and one ridge choice changes (C, n = 256, seed 7: 0.01 to 0.1); no cell mean at the printed precision, crossover or hypothesis check changes, and one chosen-ridge share moves from 69% to 68%. All 95,291 exact-CE refits of experiment 08 converged (scipy's flag). Measurement sources and saved rows are unchanged. Evidence: [../08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/results/ce_gradient_check.json](../08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/results/ce_gradient_check.json), from [../08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/check_ce_gradient.py](../08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/check_ce_gradient.py). See the [stage-1 report](../../paper_latex_and_notes/notes/mega_qa_11-16-2026/qa1_codex.md).

The bootstrap intervals are pointwise and crossovers range only over the tested training sizes, not every possible sample size. They do not provide simultaneous 95% coverage across all comparisons. Scenarios A and B reuse earlier synthetic tasks; test tasks are excluded from each fit and validation selection.

**TLDR-end:** [judge objectives: experiment 07] Same tasks, three optimizer conditions per objective; starts are chosen on training data only.
