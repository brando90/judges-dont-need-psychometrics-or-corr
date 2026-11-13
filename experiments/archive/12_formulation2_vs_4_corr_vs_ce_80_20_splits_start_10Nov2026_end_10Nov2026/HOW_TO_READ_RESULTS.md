# How to read the results of experiment 12

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026/HOW_TO_READ_RESULTS.md>

**TLDR:** All scores in [results.md](results.md) are measured on held-out test tasks that were not used to choose or fit the judge. "②" and "④" name how a judge was **chosen**: ② by correlation, ④ by cross-entropy, both on training tasks only. Both are then scored with the **same** metric on the **same** test tasks. (This guide was added on 11-13-2026; `results.md` is a frozen copy and is unchanged.)

## What is compared

- **② judge (correlation-trained):** the single cached judge, or the mixture of the five cached judges, that maximizes Pearson (or Spearman) between the judge's expected rating and the human mean rating on the training tasks.
- **④ judge (loss-trained):** the single judge or mixture that minimizes cross-entropy against the human ratings of the training tasks.
- **Baselines:** the constant (training-task human rating histogram for every artifact) and the equal-weight mixture.

## Which data each number uses

| Data | Role |
| --- | --- |
| Training tasks | 20 of 25 tasks (60 artifacts) per split. Used only to choose or fit the judge. |
| Held-out test tasks | The other 5 tasks (15 artifacts). Every reported score is computed here. |
| Repetition | 1,000 random splits. Each artifact's test metric is averaged over the 167 to 226 splits in which it was held out, then averaged over the 75 artifacts. |
| Fixed split | One extra 20/5 split (seed 1210) where Claude Code agents chose judges from training data only. Five test tasks: an illustration, not evidence. |

"Mean fitted weights" and "best single judge chosen" describe what was chosen on **training** tasks. Splits group whole tasks, not task families, so related tasks can fall on both sides. **Prior exposure:** these 75 artifacts and ratings were used in earlier studies, so held out means held out from this fit, not never seen.

## Metrics (all on held-out test tasks)

| Metric | Definition | Range | Better |
| --- | --- | --- | --- |
| RPS (main) | Mean over each artifact's human ratings of (1/5) Σ_{k=0}^{4} (Q(k) − 1{rating ≤ k})², with Q the judge's cumulative probabilities | 0 to 1 | Lower |
| CE | Mean over human ratings of −ln q(rating), probability floor 1e-6 | 0 to ∞, nats | Lower |
| TVD | 0.5 Σ_r abs(q(r) − p̂(r)), p̂ the artifact's human rating histogram | 0 to 1 | Lower |
| Spearman / Pearson | Correlation between the judge's expected rating and the human mean rating across one split's 15 test artifacts | −1 to 1 | Higher |

For scale on test RPS: ④ mixture 0.136, ② Pearson mixture 0.142, equal-weight mixture 0.150, constant 0.184.

## How to read a difference

"② minus ④" means: **the column's test metric for the ② judge minus the same test metric for the ④ judge.** For RPS, CE and TVD (lower is better), a positive value means the ④ (loss-trained) judge did better. For Spearman and Pearson (higher is better), a positive value means the ② (correlation-trained) judge did better.

Example: test RPS of the ② Pearson mixture minus test RPS of the ④ CE mixture = +0.0056 [−0.0059, +0.0181]. The ② judge's test RPS is 0.0056 higher on average, but the interval includes 0: no difference is established.

## Uncertainty

95% percentile intervals from 10,000 bootstrap draws over the 25 tasks. Correlations are computed per split and attributed to that split's test tasks before resampling. No p-values were computed.

**TLDR-end:** [judge objectives: experiment 12] Every score is a held-out test score; "② minus ④" subtracts one test metric between the correlation-chosen and the loss-chosen judge.
