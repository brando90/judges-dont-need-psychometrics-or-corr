# Measurement contract — experiment 40, 11-27-2026

Written before `run_all.py` printed a held-out number.

## What is being compared

A classical decision study (D-study) recommends how many raters `k` an item needs so that the absolute coefficient Φ(k) reaches 0.80. That rule does not depend on the number of items `N`. The empirical allocation, at a fixed rating budget `B = N × k`, is the `(N, k)` that minimizes a proper score of a cross-entropy-trained linear head on held-out prompts.

## Estimator

For cross-entropy and ranked probability score (RPS), the test risk is the unbiased prompt average

`R = (1/|D_test|) sum_c (1/k_c) sum_j L(q(c), y_{c,j})`.

`sqrt(RPS)` is the square root of that prompt-average RPS, not the average of per-prompt square roots. Lower is better for both. A plug-in divergence against the test histogram is not this estimator.

## Split

Public benchmarks use the experiment 36 group split: seed 0, fractions 0.6 / 0.2 / 0.2. SummEval groups are source articles. ChaosNLI, DICES, and POPQUORN groups are item ids. Every rater of a group stays on one side.

VeriBench `native_main` uses the task-pure `exp27_split` published in trustworthy-theorem-judges experiment 28 at `efdc8c3` (`human_ratings_new_11-26-2026.csv`), not a fresh seed-0 shuffle. `R_v1` bytes at `veribench@d1c47355` are loaded only by the experiment 36 loader. If that fetch fails, the run does not invent those ratings.

## G-study

Train-split ratings choose the D-study `k`. An all-splits variance table is descriptive and is labeled in-sample. Person ids that are crossed (POPQUORN, DICES, VeriBench `native_main`) use Henderson Method III. A within-item vote index (SummEval, ChaosNLI) is not a person: those designs use the nested one-way estimator and leave the crossed rater component unidentified. Components may be negative; Φ uses the truncation at 0, and the raw value is kept. No normality is assumed. Homoscedastic uncorrelated random effects are assumed by the expected-mean-square algebra, and the reports say so.

## Head

Same family as experiment 36: 256-dimensional signed token hash, 24 principal components fit on the labeled train prompts of that cell only, plus an intercept, L2 penalty `1e-2` on non-intercept weights, L-BFGS-B from the zero vector, at most 35 iterations. The cross-entropy objective is convex in the linear logits. The reported head is that optimizer, start, and budget. Within each cell the `N` prompts are the prefix of a seed-0 permutation of train prompts that have at least `k` ratings, and each of those prompts contributes `k` ratings.

## Interval

Item-clustered percentile 95% interval, `B = 1000`, bootstrap seed 1. The unit is the held-out prompt. The interval is for the fixed trained head, not for retraining.

## Targets fixed before the run

- D-study reliability target: Φ(k) ≥ 0.80. If no feasible `k` reaches it, the rule still picks the largest feasible `k`, because Φ increases in `k`.
- Dollar hit for cross-entropy: smallest frontier budget whose held-out cross-entropy is at most 0.05 nats above the full-train head (every train rating).
- Dollar hit for `sqrt(RPS)`: smallest frontier budget whose held-out `sqrt(RPS)` is at most 0.02 above the full-train head.
- Crowd prices, stated regime, not a newly measured wage: `$0.05`, `$0.10`, `$0.20` per rating. The `$0.05` end is the POPQUORN crowd rate used in experiment 36.
- Domain-expert prices, stated regime: `$5`, `$10`, `$20` per rating. Experiment 36 already called `$5` an assumed expert rate.
- Researcher-plus-GPU anchor, stated and not measured here: `$15,000` to `$75,000` for 1–3 researcher-months.

## Oracle label

The `(N, k)` with the lowest test score uses the test labels to pick `k`. It is an oracle allocation. The validation-chosen `k` minimizes validation cross-entropy and is then scored on test. Both are reported. The D-study pick does not see validation or test labels.

## Output-space sweep

Real ordinal sets are rebinned only to `K` that the native scale supports (`K = 2`, `K = 3`, and the native `K`). A public WMT direct-assessment file that contains only a mean score and an annotator count is not treated as multi-rater microdata. The `K ∈ {2, 3, 5, 10, 20, 101}` curve, including the fine integer grid used as the continuous CRPS, is a labeled synthetic law.
