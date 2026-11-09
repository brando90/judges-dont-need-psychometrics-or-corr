# Evaluation criterion and repository ownership — 11-09-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/03_exp06_ce_vs_pearson_matched_budget/EVALUATION_CRITERION_11-09-2026.md>

**TLDR:** The prospective human-rating study uses normalized individual-rating ranked probability score (RPS) as its primary criterion. Completed experiment 03 keeps its frozen cross-entropy (CE) endpoint; neither its definition nor its result changes after this clarification.

## User clarification and ownership

Brando clarified on 11-09-2026 that “WE” was a typo for CE, cross-entropy. It is not a new metric.

This repository owns the general comparison of proper-loss training and correlation training. [trustworthy-theorem-judges](https://github.com/brando90/trustworthy-theorem-judges) owns the applied human-rating judge and human-rating collection. VeriBench owns formal-prover coverage and differential testing; inherited model-rating analyses are historical evidence, not its current intended coverage scorer.

## Criterion for the prospective human-rating comparison

For a probability forecast q over ratings 0 through 5, let Q(k) = sum_{r<=k} q(r). For an observed human rating y,

```math
S(q,y)=\frac{1}{5}\sum_{k=0}^{4}\left(Q(k)-\mathbf{1}\{y\le k\}\right)^2.
```

Average individual-rating losses within each artifact, then weight artifacts equally. Lower is better. The [current applied-study plan](https://github.com/brando90/trustworthy-theorem-judges/blob/e2626022447eb18c033713f5878fb4b13f638476/experiments/02_probabilistic_ordinal_judge/PRIORITY_2_VS_4_11-09-2026.md#L30-L56) uses the same validation RPS to select hyperparameters and stopping for all training objectives. Freeze the selected procedures before test inspection. Compare them on identical held-out artifacts and ratings with paired uncertainty and a prespecified practical margin.

Thus CE training beats correlation training **under the planned primary criterion** only if its held-out RPS improvement satisfies the prespecified decision rule. CE, six-category Brier, normalized score error and correlations remain common secondary metrics. No numeric deployment margin is established by this audit.

## RPS, cumulative-distribution error and square roots

For empirical human cumulative distribution F_i(k), define D_i = (1/5) sum_k (Q_i(k)-F_i(k))^2. Mean individual-rating RPS for artifact i equals D_i + (1/5) sum_k F_i(k)(1-F_i(k)). The second term is shared by competing forecasts evaluated on the same ratings.

[Experiment 07's “sqrt RPS”](https://github.com/brando90/trustworthy-theorem-judges/blob/e2626022447eb18c033713f5878fb4b13f638476/experiments/07_exp1_metrics/results.md#L15-L16) is sqrt(mean_i D_i), a root mean squared cumulative-distribution distance. It is **not** sqrt(mean individual-rating RPS). Normalized root mean squared score error compares expected ratings and human means; it does not identify the full rating distribution. Sparse empirical human distributions are not known population distributions.

An earlier solver-transfer proposal used worst-solver cumulative discrepancy. That is a distinct estimand: human-disagreement terms can differ across solvers, so maximizing raw per-solver RPS is not equivalent to maximizing per-solver distribution error. Preserve that distinction when specifying transfer or worst-solver claims.

## Preserve completed experiments and competing proposals

[Experiment 03's protocol](PROTOCOL.md) froze CE fitting minus Pearson fitting on held-out CE as its primary contrast. Its [common metric table](RESULTS_comparison.md#full-held-out-metric-panel) also reports RPS. The reported CE and RPS contrasts both have intervals spanning zero; the audit establishes no new winner and performs no new fitting.

The [11-09 recommendations](../../paper_latex_and_notes/notes/correlation_and_loss_formulations_received_11-09-2026/RECOMMENDATIONS.md#L93) preserve the prospective human-study RPS endpoint. An additional main-branch AGENTS.md section recommends CE-primary for a proposed formulation-3-versus-4 comparison and explicitly calls it an unrun idea. That recommendation is not an amendment to a frozen study. State the experiment and estimand beside every claim.

## Audit provenance

Inspected local working files and fetched main without merging or overwriting concurrent edits:

- This repository: main c306c8c52796cc2f1c6d6709e98c355092569801.
- Applied study: main e2626022447eb18c033713f5878fb4b13f638476.
- VeriBench's latest-main ownership policy and legacy consumers were also audited; its private path inventory remains private.

No model calls, human-rating collection, training or test-driven selection were performed by this audit. Existing protocols, prompts, results and checkpoints remain the evidence for their original conditions.

**TLDR-end:** [judges-corr: evaluation criterion] RPS is the prospective applied study's primary criterion, while completed experiment 03 retains CE. Training objectives and held-out evaluation criteria must remain separate.
