# judges-dont-need-psychometrics-or-corr

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/README.md>

**Supported thesis:** correlation alone does not identify a judge’s desired conditional rating distribution. When that distribution is the target, fit and evaluate probability forecasts with proper probability losses, and retain correlations as useful diagnostics of scalar association.

The project’s original title expresses its research direction. The evidence does **not** establish that every psychometric method is unnecessary, or that cross-entropy always selects the best finite-sample system. Defining the human population, checking input correspondence, and studying reliability and construct validity remain separate questions.

**Status (11-08-2026):** the controlled synthetic conditional-distribution study is complete and awaiting the parent’s requested multi-provider review. The paper has been revised to distinguish distribution identification, mean prediction and historical evidence. No venue submission is authorized or performed.

**Current priority (11-09-2026):** [trustworthy-theorem-judges](https://github.com/brando90/trustworthy-theorem-judges) owns finding the best practical judge using method 2 (correlation training) and method 4 (proper-loss training). This repository tests the proposed inferiority of correlation-based optimization without assuming a winner. Follow the [current comparison and ownership instructions](AGENTS.md#current-priority--user-clarification-11-09-2026), including the completed experiment 03 and its checkpoint.

## What the new study establishes

[`02_conditional_binomial_distribution_identification`](experiments/02_conditional_binomial_distribution_identification/README.md) uses a known Binomial(5,t) population and five distributions with exactly the same conditional mean 5t. It includes the true distribution, an item-permutation control, and three constant controls. All seven selection criteria use the same nine candidates, disjoint train/validation/test families, and search budget.

- **Population identification:** cross-entropy (CE), Brier score and ranked probability score uniquely identify the true distribution in this candidate set. Pearson, Spearman, Kendall and scalar mean squared error tie across the five same-mean candidates on every possible rating realization.
- **Finite sampling:** with one synthetic rating per item, CE selected the true distribution in 249/256 rating replications; Brier and ranked probability score each did so in 245/256. All three did so in 256/256 replications with two or three ratings. Those are finite simulation counts, not a guarantee of zero selection error.
- **Tie accounting:** the scalar criteria retain five-way ties. Their exact expected test CE ranges from 1.328246 to 3.347873 nats across tied candidates; the uniform-tie mean is 1.994927. A tie policy can select the oracle, so the result does not establish superiority under every tie policy.
- **Conditional versus marginal:** the permutation control has the same aggregate histogram as the oracle but conditional Kullback–Leibler divergence 2.886690 nats. Matching marginal histograms cannot establish correct item-conditioned forecasts.

All 768 simulation conditions and 5,376 selections completed, with zero failed selections, zero provider calls and $0 provider inference charges. These are synthetic results only. The [full report](experiments/02_conditional_binomial_distribution_identification/RESULTS_conditional_identification.md) gives every common test metric, 95% Monte Carlo intervals over rating replications, complete tie ranges, and the scope of each uncertainty statement. No hypothesis test was declared (p-val=n/a).

## The target and the loss

Given candidate output y and reference y*, define x=(y,y*), a rating population, and its conditional distribution p_human(r|x). A probabilistic judge reports q_theta(r|x). The distributional target is

```math
\min_\theta \; \mathbb{E}_x D_{\mathrm{KL}}\big(p_{\mathrm{human}}(\cdot\mid x)\,\|\,q_\theta(\cdot\mid x)\big).
```

Population cross-entropy differs from this target by an entropy term independent of the judge. **One rating per item supplies a valid conditional log-loss observation**, -log q_theta(r|x). It does not require pooled marginal Kullback–Leibler divergence, and it does not reveal the full distribution of an unconstrained individual item. Shared structure and appropriate data are still needed to learn conditional distributions. Sparse empirical histograms are noisy summaries, not known human populations.

The [proper-scoring-rule literature](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) provides the population identification results underlying CE, Brier score and ranked probability score. Properness does not guarantee a finite-sample winner under every metric or model class.

**Mean squared error identifies a conditional mean, not a distribution.** Report normalized score error, mean absolute error and signed bias when scalar score values matter. Pearson measures linear association; Spearman and Kendall measure ordering with their specified tie conventions. A strictly increasing transform preserves ranks; compression to two bins generally introduces ties and need not preserve rank correlation.

## Prior evidence remains separate

[`01_synthetic_pearson_vs_kl_squashed_judge_0to5`](experiments/01_synthetic_pearson_vs_kl_squashed_judge_0to5/README.md) is the original marginal-histogram illustration. Its code and frozen outputs are preserved; a [new reproduction receipt](experiments/02_conditional_binomial_distribution_identification/results/old_toy_reproduction.json) confirms the original printed numbers. It is not evidence from conditional judge selection.

The sister repository’s historical `06_information_loss_vs_correlation_judge_selection` did **not** establish a held-out CE advantage for CE selection over Spearman selection under either five seeded five-fold family partitions (8.868 versus 8.098; +0.770 [-0.895, 1.317]) or the original 15-validation/30-test-artifact split (9.133 versus 8.710; +0.423), and those historical ratings, test outcomes and candidate prompts had prior exposure. Its fitted probability-head comparison also had widespread Pearson-optimizer convergence failures. The [public-evidence section](experiments/02_conditional_binomial_distribution_identification/RESULTS_conditional_identification.md#historical-evidence-preserved) links exact source revisions, preserves the negative/null findings, and records the separately owned completed weighted-probability/stable-Pearson follow-up. No raw human ratings, private candidate artifacts or sealed data enter this repository. Fresh human returns were not independently checked from cloud here and are not declared absent.

## Reproduce and inspect

```bash
python3 -m unittest discover -s experiments/02_conditional_binomial_distribution_identification -p 'test*.py' -v
python3 experiments/02_conditional_binomial_distribution_identification/verify_results.py
python3 experiments/02_conditional_binomial_distribution_identification/make_report.py
```

The [experiment index](experiments/README.md) links the complete procedure, hashes and results. The [paper source and current PDF](paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/README.md) live together. New measurement runs refuse to overwrite completed evidence; deterministic verification uses the saved rows without new rating samples.

## Origin and scope

The project began with the [original README formulation at commit `f2272f9`](https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/f2272f9628f1f31575ecbdaa304f33c9ad0743f1/README.md), the [11-07-2026 discussion notes](paper_latex_and_notes/notes/fathom_call_11-07-2026/notes.md), the [whiteboard](paper_latex_and_notes/notes/whiteboard_11-07-2026.png), and [handwritten selection/fitting notes received 11-08-2026](paper_latex_and_notes/notes/handwritten_corr_vs_loss_splits_received_11-08-2026/transcription.md). Those source records remain unchanged. The [research synthesis](paper_latex_and_notes/notes/correlation_vs_loss_judges_notes_11-07-2026.md) provides further historical context.

The [trustworthy-theorem-judges](https://github.com/brando90/trustworthy-theorem-judges) project studies the applied judge problem. This repository’s new experiment isolates a distribution-identification question; it does not duplicate historical probability fitting, validate a deployed human judge, or establish natural-task generalization.

**TLDR-end:** [judge objectives: conditional identification] Evaluate conditional probability forecasts when the full rating distribution is the target. Correlations and scalar errors remain informative diagnostics, but neither alone identifies distribution shape.
