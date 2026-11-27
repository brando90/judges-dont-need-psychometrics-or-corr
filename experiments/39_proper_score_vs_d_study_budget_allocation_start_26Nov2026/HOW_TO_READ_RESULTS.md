# How to read experiment 39

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/39_proper_score_vs_d_study_budget_allocation_start_26Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** Study A numbers are held-out test scores on a frozen 30% item split of public ChaosNLI and DICES tables, or held-out mean-estimation error on that same split. Study B numbers are expected absolute error under a known synthetic two-facet law. Ranked probability score (RPS), cross-entropy (CE), and total variation distance (TVD) score a distributional forecast. Item-mean and grand-mean mean squared error (MSE) score a universe score. A difference always subtracts the same metric on the same items.

## Which data

| Setting | Items | What is random | What is held out |
| --- | --- | --- | --- |
| Study A, ChaosNLI SNLI and MNLI-m | 600 items (test 180, dev 420), Experiment 28 subset | The 80-item $\times$ 16-rating pilot, the chosen design, and the deployment training ratings, redrawn in each of 150 replications | Frozen test items (30%), from `SeedSequence([20261126, 1, k])`. Test labels and test features are not used to choose a design, a 27-transform, or a mixture weight. Prior exposure: the frozen Experiment 28 judge probabilities were collected earlier; the human rating tables are public |
| Study A, ChaosNLI alphaNLI and DICES-990 | 400 items (test 120, dev 280) | Same as above | Same rule |
| Study A, DICES-350 | 350 items (test 105, dev 245), fully crossed 350 $\times$ 123 | Same as above | Same rule |
| Study B, synthetic Phase B | New objects, raters, and occasions drawn from a known crossed Gaussian law $v(t)=(1-t)\mathrm{C1}+t\mathrm{C2}$ | The residual maximum likelihood (REML) pilot and the held-out evaluation panel, 200 replications (256 at the two reproduction cells) | No human rating is used. The target is the object universe score. Prior exposure: none. This is a synthetic check of D-study selection, not a conference empirical claim about human ratings |

ChaosNLI has per-item label counts and no rater identifiers. DICES-350 is a complete crossing. DICES-990 is incomplete (minimum 69 ratings per selected item).

## Metrics

RPS is the mean over held-out test items of $(1/(K-1))\sum_{k=0}^{K-2}(F(k)-G(k))^{2}$, the Brier score of the cumulative distribution, where $G$ is the empirical test histogram of that item and $K$ is the number of ordinal categories (3 on ChaosNLI and DICES). Range 0 to 1. Lower is better. Typical values in Study A are about 0.11 (ChaosNLI SNLI) to 0.23 (DICES-350).

CE is the mean held-out test cross-entropy $-\sum_{k} \hat{\pi}(k)\log q(k)$, in nats, after a small smoother recorded in the analysis script. Lower is better. TVD is the mean over test items of $0.5\sum_{k}\lvert q(k)-\hat{\pi}(k)\rvert$. Range 0 to 1. Lower is better. CE and TVD are secondary.

Item-mean MSE is the mean over held-out test items of $(\bar{y}_{\mathrm{hat}}-\bar{y}_{\mathrm{test}})^{2}$, the error in the item’s mean rating. Grand-mean MSE is $(\bar{y}_{\mathrm{hat}}-\bar{y}_{\mathrm{test}})^{2}$ for the average over test items. Lower is better. These are universe-score errors and are never mixed into an RPS comparison.

In Study B, the loss is the held-out absolute error of the object universe-score estimate under the chosen $(n_{a},n_{b})$ design. Lower is better. Population $v_{\Delta}$ is the D-study absolute-error variance from the true components.

## What is subtracted

“U1 minus proper-score (PS) test RPS” is the held-out test RPS of the design chosen by the per-item D-study, minus the held-out test RPS of the design chosen by the proper-score selector, on the same frozen test items and the same replication. Positive means the proper-score selector is better.

“U1 minus U2 item-mean MSE” uses that same order on item-mean MSE. Negative means U1 is better at estimating per-item means.

“REML-chosen minus equal $4\times 4$” in Study B is held-out absolute error of the REML-selected budget-16 design minus that of $4\times 4$ in the same replication. Negative means REML is better than equal allocation.

## How the forecasts and designs were chosen

Primary Study A predictor: the Experiment 28 27-transform CE-fit of a mixture of five frozen frontier-judge probabilities. The transform is chosen by cross-entropy on the deployment training ratings of the selected design, using only the remaining dev pool after the pilot is drawn. Search space: 27 discrete transforms, exhaustive. Amendment A1 is a secondary mixture $q=(1-\alpha)q_{\mathrm{transform}}+\alpha\widehat{\pi}$ with $\alpha\in\lbrace 0,0.25,0.5,0.75,1\rbrace$; primary claims do not use it.

U1: one-facet analysis of variance (ANOVA) on the pilot; minimize $\widehat{v}_{e}/n_{r}$. U2: minimize $\widehat{v}_{u}/n_{i}+\widehat{v}_{e}/B$. Proper-score selector: 5-fold $\times$ 8 inner draws on the pilot, minimize expected held-out RPS. Oracles are diagnostic only and are not used as selectors.

Study B selector: constrained REML on the pilot table, then the design among $\lbrace 8\times 2, 4\times 4, 2\times 8\rbrace$ with smallest estimated $v_{\Delta}$. This objective is not convex in the discrete design; the result reflects that estimator, that pilot size, and that three-design grid.

## Uncertainty

Student-$t$ 95% intervals over replications (150 in Study A, 200 in Study B, 256 at the two Experiment 04 reproduction cells), null mean difference 0, two-sided paired $t$ test. The replication redraws the pilot, the design choice, and the deployment or evaluation sample. It does not bootstrap a single frozen test split at the item level; the unit is the design-selection experiment. Proportions use Wilson 95% intervals. “Beats” requires a lower mean and a 95% interval that excludes 0. Intervals that include 0 are reported as inconclusive, not as wins. Exact ties (identical forecasts) have undefined or undefined-looking $p$ and are reported as exact ties.

**TLDR-end:** [judge objectives: experiment 39] RPS, CE, and TVD are held-out test scores of a distributional forecast on frozen items; item-mean and grand-mean MSE are universe-score errors on the same items; Study B is a synthetic REML selection check.
