# 08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/README.md>

**TLDR:** Brando's conjecture (11-13-2026): with little rating data, training a judge on correlation may work better because it only has to get the first moment (the mean) right, while cross-entropy (CE) tries to fit structure the data cannot yet support. If so, the number of ratings a CE-trained judge needs to win should grow with the complexity of the target rating distribution. This experiment tests that on the experiment 13 toy, adding targets whose information lies only in the mean, and gives every objective a validation-tuned regularizer. Synthetic toy that works by construction; not evidence about real ratings.

## Hypotheses and decision rule (written before measurement, 11-14-2026)

- **H1 (Brando's conjecture).** Correlation training is better with little data because it targets only the mean. Predictions:
  - the Pearson-trained judge has significantly lower held-out test RPS (ranked probability score) than the CE-trained judge at the smallest training sizes, most clearly in scenarios C and D, where all information is in the mean;
  - the crossover (the number of ratings from which the CE-trained judge is significantly better at every larger size) is larger in C and D than in B.
- **H2 (alternative).** Correlation training never learns even the mean faster: it sees one noisy mean per task, and its objective ignores scale. Any small-data advantage comes from staying close to a low-variance forecast, not from correlation. Predictions:
  - held-out test correlation with the true mean is at least as high for the CE-trained judge at every size;
  - a CE-trained judge whose regularization is chosen on validation data (formulation ③) is never significantly worse on held-out test RPS than the Pearson-trained judges (② fixed, ① validation-tuned), at any size, in any scenario.
- **Decision.** Comparisons are paired by seed, with 95% bootstrap intervals over 100 seeds. H1 is supported if its two predictions hold. H2 is supported if its two predictions hold. Both, neither or a mixture may happen, and all are reported.

## Setup

- **Target rating distributions.** R | x ~ BetaBin(5, μ(x)κ(x), (1 − μ(x))κ(x)) with x ~ Unif[−1, 1]; κ(x) = exp(1.5 + 2.2 cos(1.5πx)) where it varies.

| Scenario | Mean E[R \| x] = 5μ(x) | Disagreement κ(x) | Where the information is |
| --- | --- | --- | --- |
| A (exp 13) | μ = 0.5 + 0.08x (nearly constant) | varies | spread only |
| B (exp 13) | μ = 0.1 + 0.8((x + 1)/2)² | varies | mean and spread |
| C (new) | μ = 0.1 + 0.8((x + 1)/2)² | constant, κ = e^1.5 | mean only, nonlinear |
| D (new) | μ = 0.5 + 0.35x | constant, κ = e^1.5 | mean only, linear (correlation's most favorable case) |

- **Data.** n training tasks with 3 ratings each, n ∈ {4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048}; 500 fresh test tasks per seed, scored against the true distribution; seeds 0–99. Scenarios A and B regenerate experiment 13's exact data.
- **Judge family and optimizer.** Identical to experiment 13, imported from its frozen `scripts/toy.py`: softmax over the 6 ratings of a linear map of 13 cosine/sine features of x; L-BFGS-B, 400 iterations and 600 evaluations; start at the training rating histogram plus a 0.01 seeded perturbation; ridge (c/n)‖W‖². CE is convex in W; RPS and Pearson are not, so their results reflect this optimizer, start and budget (experiment 07 checks that).
- **Judges.**

| Name | Objective | Ridge c | Formulation |
| --- | --- | --- | --- |
| constant | none: the training rating histogram for every test task | none | baseline |
| ce_fixed | CE | 1 (experiment 13) | ④ |
| ce_valrps | CE | chosen by validation RPS | ③ |
| rps_fixed, rps_valrps | RPS | 1, or chosen by validation RPS | ④, ③ |
| pearson_fixed | Pearson | 1 (experiment 13) | ② |
| pearson_valcorr | Pearson | chosen by validation Pearson | ① |
| pearson_valrps | Pearson | chosen by validation RPS (common criterion) | ② with ③-style selection |

  Validation: 4-fold cross-validation over the training tasks, pooling out-of-fold predictions; c ∈ {0.01, 0.1, 1, 10, 100}; ties go to the larger c. Test data never enter any choice.
- **Metrics (held-out test tasks, against the true distribution).** RPS excess and CE excess, KL(true ‖ judge), both lower is better; test Pearson and Spearman between the judge's expected rating and the true mean, higher is better; total variation distance between each judge's test forecasts and the constant forecast.

## Run

```bash
python3 experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/run_sweep.py
```

CPU only, no model calls, $0. [results.md](results.md) is the live status.

**TLDR-end:** [judge objectives: experiment 08] Tests whether correlation training wins with little data and whether the ratings needed for CE to win grow with target complexity, using mean-only targets and validation-tuned regularization.
