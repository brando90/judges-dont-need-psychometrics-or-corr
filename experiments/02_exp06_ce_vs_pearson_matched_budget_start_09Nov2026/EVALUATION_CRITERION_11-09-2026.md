# Evaluation criterion and repository ownership — 11-09-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/EVALUATION_CRITERION_11-09-2026.md>

**TLDR:** The prospective human-rating study uses normalized individual-rating ranked probability score (RPS) as its primary criterion, with cross-entropy (CE) and total variation distance (TVD) secondary. Completed experiment 03 keeps its frozen CE endpoint; neither its definition nor its result changes after this clarification.

## User clarification and ownership

Brando clarified on 11-09-2026 that “WE” was a typo for CE, cross-entropy. It is not a new metric.

This repository owns the general comparison of proper-loss training and correlation training. [trustworthy-theorem-judges](https://github.com/brando90/trustworthy-theorem-judges) owns the applied human-rating judge and human-rating collection. VeriBench owns formal-prover coverage and differential testing; inherited model-rating analyses are historical evidence, not its current intended coverage scorer.

## Criterion for the prospective human-rating comparison

For a probability forecast q over ratings 0 through 5, let Q(k) = sum_{r<=k} q(r). For an observed human rating y,

```math
S(q,y)=\frac{1}{5}\sum_{k=0}^{4}\left(Q(k)-\mathbf{1}\{y\le k\}\right)^2.
```

Average individual-rating losses within each artifact, then weight artifacts equally. Lower is better. The [current applied-study plan](https://github.com/brando90/trustworthy-theorem-judges/blob/e2626022447eb18c033713f5878fb4b13f638476/experiments/02_probabilistic_ordinal_judge/PRIORITY_2_VS_4_11-09-2026.md#L30-L56) uses the same validation RPS to select hyperparameters and stopping for all training objectives. Freeze the selected procedures before test inspection. Compare them on identical held-out artifacts and ratings with paired uncertainty and a prespecified practical margin.

Thus CE training beats correlation training **under the planned primary criterion** only if its held-out RPS improvement satisfies the prespecified decision rule. CE and TVD are explicit secondary metrics; six-category Brier, normalized score error and correlations remain in the common diagnostic panel. No numeric deployment margin is established by this audit.

## Prospective secondary metrics — user decision, 11-09-2026

Brando requested TVD explicitly in both repositories' prospective evaluation panels, keeping RPS primary and CE/TVD secondary. This changes no completed experiment's frozen protocol, scorer, forecast or result. The applied study owns the [prospective evaluator](https://github.com/brando90/trustworthy-theorem-judges/tree/main/experiments/02_design_a_judge_that_predicts_human_ratings_0_to_5_start_06Nov2026/native_probability); this repository uses the same definitions for prospective objective comparisons.

For artifact i, let q_i(r) be the forecast probability of rating r in {0,1,2,3,4,5}, and let p_hat_i(r) be the empirical frequency among that artifact's observed human ratings. All three metrics are lower-is-better.

| Role | Metric | Per-artifact definition |
| --- | --- | --- |
| Primary; common validation selection | Normalized individual-rating RPS | Average the individual-rating RPS defined above over the artifact's observed human ratings; range 0–1. |
| Secondary | CE | Average `-ln(q_i(y))` over the same human ratings; natural-log units. Fix any probability-floor policy before evaluation. |
| Secondary | TVD | `0.5 * sum_r abs(q_i(r) - p_hat_i(r))`; range 0–1. Compute once from the artifact's empirical human distribution, not by averaging distances to individual one-hot ratings. |

Average each metric equally across the same eligible artifacts; additional raters must not increase an artifact's weight. Keep the full scheduled denominator, missing forecasts, invalid forecasts and artifacts without human ratings visible. A mean over scored artifacts alone must be labeled as such. The prospective evaluator reports TVD full-denominator bounds using its range 0–1; those bounds are not confidence intervals or a replacement for the study's missingness policy. Use paired uncertainty on matched artifacts under the declared task/rater dependence.

TVD describes agreement with the empirical human distribution, which is noisy with sparse ratings; it does not provide an unbiased estimate of population conditional TVD in general and is not a proper individual-rating scoring rule. TVD also does not use the ordering of rating categories. TVD mutual information measures a different target and is not added by this decision. RPS remains the selection and primary evaluation criterion; CE or TVD cannot select a preferred test outcome after inspection.

## RPS, cumulative-distribution error and square roots

For empirical human cumulative distribution F_i(k), define D_i = (1/5) sum_k (Q_i(k)-F_i(k))^2. Mean individual-rating RPS for artifact i equals D_i + (1/5) sum_k F_i(k)(1-F_i(k)). The second term is shared by competing forecasts evaluated on the same ratings.

[Experiment 07's “sqrt RPS”](https://github.com/brando90/trustworthy-theorem-judges/blob/e2626022447eb18c033713f5878fb4b13f638476/experiments/07_exp1_metrics/results.md#L15-L16) is sqrt(mean_i D_i), a root mean squared cumulative-distribution distance. It is **not** sqrt(mean individual-rating RPS). Normalized root mean squared score error compares expected ratings and human means; it does not identify the full rating distribution. Sparse empirical human distributions are not known population distributions.

An earlier solver-transfer proposal used worst-solver cumulative discrepancy. That is a distinct estimand: human-disagreement terms can differ across solvers, so maximizing raw per-solver RPS is not equivalent to maximizing per-solver distribution error. Preserve that distinction when specifying transfer or worst-solver claims.

## Preserve completed experiments and competing proposals

[Experiment 03's protocol](PROTOCOL.md) froze CE fitting minus Pearson fitting on held-out CE as its primary contrast. Its [common metric table](RESULTS_comparison.md#full-held-out-metric-panel) also reports RPS. The reported CE and RPS contrasts both have intervals spanning zero; the audit establishes no new winner and performs no new fitting.

The [11-09 recommendations](../../paper_latex_and_notes/notes/correlation_and_loss_formulations_received_11-09-2026/RECOMMENDATIONS.md#L93) preserve the prospective human-study RPS endpoint and add CE/TVD as explicit secondary metrics. The historical AGENTS.md recommendation of CE-primary for an unrun formulation-3-versus-4 idea is superseded for this prospective panel. It does not amend a completed study. State the experiment and estimand beside every claim.

## Audit provenance

Inspected local working files and fetched main without merging or overwriting concurrent edits:

- This repository: main c306c8c52796cc2f1c6d6709e98c355092569801.
- Applied study: main e2626022447eb18c033713f5878fb4b13f638476.
- VeriBench's latest-main ownership policy and legacy consumers were also audited; its private path inventory remains private.

No model calls, human-rating collection, training or test-driven selection were performed by this audit. Existing protocols, prompts, results and checkpoints remain the evidence for their original conditions.

**TLDR-end:** [judges-corr: evaluation criterion] RPS is the prospective primary criterion, with CE and TVD secondary; completed experiment 03 retains its frozen CE endpoint. Training objectives and held-out evaluation criteria remain separate.
