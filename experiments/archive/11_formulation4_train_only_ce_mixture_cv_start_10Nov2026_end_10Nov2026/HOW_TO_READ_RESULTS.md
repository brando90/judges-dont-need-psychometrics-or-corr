# How to read the results of experiment 11

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/archive/11_formulation4_train_only_ce_mixture_cv_start_10Nov2026_end_10Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** All scores in [results.md](results.md) are out-of-fold: each is measured on a held-out test fold whose tasks were not used to fit that judge. "④ minus comparator" subtracts the **same** test metric for two judges. (This guide was added on 11-13-2026; `results.md` is a frozen copy and is unchanged.)

## What is compared

- **④ CE-trained mixture:** weights over the five cached judges chosen to minimize cross-entropy against the human ratings of the training folds.
- **Comparators:** the constant (training-fold human rating histogram for every artifact), the equal-weight mixture, and the single judge with the lowest training CE.

There is no correlation-trained judge in this experiment.

## Which data each number uses

| Data | Role |
| --- | --- |
| Training folds | 4 of 5 folds of tasks (20 tasks). Used only to fit the weights or pick the single judge. |
| Held-out test fold | The remaining 5 tasks. Every reported score is computed here. |
| Repetition | 20 repeats of 5-fold cross-validation (100 fits). Each artifact's test metric is averaged over its 20 out-of-fold scores, then over the 75 artifacts. |

"Mean fitted weights" and the single-judge counts describe choices made on **training** folds. Folds group tasks, not task families. **Prior exposure:** these ratings were used in experiments 06, 07 and 10.

## Metrics (all on held-out test folds)

| Metric | Definition | Range | Better |
| --- | --- | --- | --- |
| CE (primary here) | Mean over human ratings of −ln q(rating), probability floor 1e-6 | 0 to ∞, nats | Lower |
| RPS | Mean over each artifact's human ratings of (1/5) Σ_{k=0}^{4} (Q(k) − 1{rating ≤ k})² | 0 to 1 | Lower |
| TVD | 0.5 Σ_r abs(q(r) − p̂(r)), p̂ the artifact's human rating histogram | 0 to 1 | Lower |

## How to read a difference

"④ minus comparator" is **the column's test metric for the ④ judge minus the same test metric for the comparator.** All three metrics are lower-is-better, so a negative value means the ④ judge did better.

Example: test RPS of the ④ mixture minus test RPS of the equal-weight mixture = −0.0129 [−0.0232, −0.0024]; the ④ judge's test RPS (0.1367) is lower than the equal-weight mixture's (0.1496).

## Uncertainty

95% percentile intervals from 10,000 bootstrap draws over the 25 tasks. No p-values were computed.

**TLDR-end:** [judge objectives: experiment 11] Every score is out-of-fold on held-out tasks; differences subtract one test metric between the CE-trained mixture and a comparator.
