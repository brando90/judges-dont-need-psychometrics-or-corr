# judges-dont-need-psychometrics-or-corr

**Working title (Brando, Nov 24 12:20): "No Free Lunch of AI Judges."** This is the correlation vs cross-entropy (CE) repo: CE-trained judges vs correlation-trained judges, scored with the proper scores RPS, CE and TVD. The framing: no need for G theory or D theory; really just CE and lots of data (scaling laws as always).

**How this differs from [correlation-is-not-trust](https://github.com/brando90/correlation-is-not-trust):** that repo asks whether a judge that correlates highly with humans still **fails trustworthy judge properties** (renaming, worse file, inserted bug, consistency, item order), i.e. bad properties despite high correlation. This repo asks **which objective to fit** (correlation vs CE) and when CE beats Pearson on held-out proper scores. Read both together.

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/README.md>

**Opinion piece (11-25-2026; the argument; the repo is the evidence):** https://cs.stanford.edu/people/brando9/2026/11/25/bitter-judges-lesson.html

**Current status and results in one page:** [STATUS_11-23-2026.md](STATUS_11-23-2026.md) (11-23-2026; historical snapshot: [STATUS_11-20-2026.md](STATUS_11-20-2026.md)). Quick links: [tech report PDF (arXiv edition)](paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/00_main_arxiv.pdf), [AAAI-27 submission PDF](paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/00_main_aaai_2027.pdf), [Version 29 meeting brief HTML](paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/judge-objectives-status.html), Stanford Digital Repository preprint (`DOI: 10.25740/bh388th6008`), and [Experiment 29 complete results](experiments/29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026/RESULTS_11-23-2026.md).

**Supported thesis:** correlation alone does not identify a judge’s desired conditional rating distribution. When that distribution is the target, fit and evaluate probability forecasts with proper probability losses, and retain correlations as useful diagnostics of scalar association.

The project’s original title expresses its research direction. The evidence does **not** establish that every psychometric method is unnecessary, or that cross-entropy always selects the best finite-sample system. Defining the human population, checking input correspondence, and studying reliability and construct validity remain separate questions.

**Status (11-23-2026):** Through Experiments 23–29 (`DOI: 10.25740/bh388th6008`, public `HelpSteer2`, `ChaosNLI`, `DICES`, `KonIQ`, and `SummEval` benchmarks plus all `1,401/1,401` matched four-objective `Qwen3-4B-Instruct-2507` [`921`] and `Llama-3.1-8B-Instruct` [`480`] LoRA runs), raw proper-loss training (`CE`/`RPS`) defeats `Pearson` once budgets pass `16–80` ratings for CE (`64–1,024` for RPS crossed; `16–256` conditional) across all `7/7` setups and clears raw `MSE` crossed on `5/6` `K >= 3` panels while remaining statistically indistinguishable from `MSE` on `K = 2` (`+0.00034`, 95% crossed CI `[-0.00209,+0.00278]`). Historical `STATUS_11-20-2026.md` and frozen completed experiments remain unchanged. No venue submission is performed from automated scripts without explicit approval.

**Current priority (11-09-2026):** [trustworthy-theorem-judges](https://github.com/brando90/trustworthy-theorem-judges) owns finding the best practical judge using method 2 (correlation training) and method 4 (proper-loss training). This repository tests the proposed inferiority of correlation-based optimization without assuming a winner. Follow the [current comparison and ownership instructions](AGENTS.md#current-priority--user-clarification-11-09-2026), including the experiment 02 comparison and its checkpoint.

**Prospective evaluation panel (user decision, 11-09-2026):** normalized ranked probability score (RPS) is primary, with cross-entropy (CE) and total variation distance (TVD) secondary. Every training objective uses the same validation RPS, and every selected judge receives the same held-out panel. TVD compares each forecast with its artifact's empirical human rating distribution; it is not TVD mutual information. The [definitions and weighting](experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/EVALUATION_CRITERION_11-09-2026.md#prospective-secondary-metrics--user-decision-11-09-2026) retain sparse-rating and missingness limitations. This prospective addition leaves completed experiments' frozen protocols and results unchanged.

## Consolidated experiment ownership — 11-11-2026

The general objective-comparison work now lives here: the earlier experiment 06 package remains within experiment 02, while completed experiments 11–14 are grouped in the [archive guide](experiments/archive/README.md). They examine different training rules, split procedures, synthetic constructions or judge pools; most empirical variants reuse the same human ratings. They are not independent new rating collections. The duplicate experiment 06 tree in trustworthy-theorem-judges is retired.

Every experiment folder has a `HOW_TO_READ_RESULTS.md` that says, for each reported number, which data it was computed on (train, validation or held-out test), which metric it is and what each difference subtracts from what.

Use the [experiment index](experiments/README.md) to distinguish the synthetic examples from historical human-rating comparisons. The applied repository continues to own packet preparation, collection, its practical judge and sample-size planning. The migration preserves scientific files and changes no outcome or acceptance criterion.

## What the new study establishes

[`01_conditional_binomial_distribution_identification_start_08Nov2026`](experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/README.md) uses a known Binomial(5,t) population and five distributions with exactly the same conditional mean 5t. It includes the true distribution, an item-permutation control, and three constant controls. All seven selection criteria use the same nine candidates, disjoint train/validation/test families, and search budget.

- **Population identification:** cross-entropy (CE), Brier score and ranked probability score uniquely identify the true distribution in this candidate set. Pearson, Spearman, Kendall and scalar mean squared error tie across the five same-mean candidates on every possible rating realization.
- **Finite sampling:** with one synthetic rating per item, CE selected the true distribution in 249/256 rating replications; Brier and ranked probability score each did so in 245/256. All three did so in 256/256 replications with two or three ratings. Those are finite simulation counts, not a guarantee of zero selection error.
- **Tie accounting:** the scalar criteria retain five-way ties. Their exact expected test CE ranges from 1.328246 to 3.347873 nats across tied candidates; the uniform-tie mean is 1.994927. A tie policy can select the oracle, so the result does not establish superiority under every tie policy.
- **Conditional versus marginal:** the permutation control has the same aggregate histogram as the oracle but conditional Kullback–Leibler divergence 2.886690 nats. Matching marginal histograms cannot establish correct item-conditioned forecasts.

All 768 simulation conditions and 5,376 selections completed, with zero failed selections, zero provider calls and $0 provider inference charges. These are synthetic results only. The [full report](experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/RESULTS_conditional_identification.md) gives every common test metric, 95% Monte Carlo intervals over rating replications, complete tie ranges, and the scope of each uncertainty statement. No hypothesis test was declared (p-val=n/a).

## The target and the loss

Given candidate output y and reference y*, define x=(y,y*), a rating population, and its conditional distribution p_human(r|x). A probabilistic judge reports q_theta(r|x). The distributional target is

```math
\min_\theta \; \mathbb{E}_x D_{\mathrm{KL}}\big(p_{\mathrm{human}}(\cdot\mid x)\,\|\,q_\theta(\cdot\mid x)\big).
```

Population cross-entropy differs from this target by an entropy term independent of the judge. **One rating per item supplies a valid conditional log-loss observation**, -log q_theta(r|x). It does not require pooled marginal Kullback–Leibler divergence, and it does not reveal the full distribution of an unconstrained individual item. Shared structure and appropriate data are still needed to learn conditional distributions. Sparse empirical histograms are noisy summaries, not known human populations.

The [proper-scoring-rule literature](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) provides the population identification results underlying CE, Brier score and ranked probability score. Properness does not guarantee a finite-sample winner under every metric or model class.

**Mean squared error identifies a conditional mean, not a distribution.** Report normalized score error, mean absolute error and signed bias when scalar score values matter. Pearson measures linear association; Spearman and Kendall measure ordering with their specified tie conventions. A strictly increasing transform preserves ranks; compression to two bins generally introduces ties and need not preserve rank correlation.

## Prior evidence remains separate

[`00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026`](experiments/00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026/README.md) is the original marginal-histogram illustration. Its code and frozen outputs are preserved; a [new reproduction receipt](experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/results/old_toy_reproduction.json) confirms the original printed numbers. It is not evidence from conditional judge selection.

The sister repository’s historical `06_information_loss_vs_correlation_judge_selection` did **not** establish a held-out CE advantage for CE selection over Spearman selection under either five seeded five-fold family partitions (8.868 versus 8.098; +0.770 [-0.895, 1.317]) or the original 15-validation/30-test-artifact split (9.133 versus 8.710; +0.423), and those historical ratings, test outcomes and candidate prompts had prior exposure. Its fitted probability-head comparison also had widespread Pearson-optimizer convergence failures. The [public-evidence section](experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/RESULTS_conditional_identification.md#historical-evidence-preserved) links exact source revisions, preserves the negative/null findings, and records the separately owned completed weighted-probability/stable-Pearson follow-up. No raw human ratings, private candidate artifacts or sealed data are added to public Git by these reports. The existing ignored experiment 02 store holds previously migrated historical reconstruction inputs. Fresh human returns remain the applied collection owner’s responsibility.

## Reproduce and inspect

```bash
python3 -m unittest discover -s experiments/01_conditional_binomial_distribution_identification_start_08Nov2026 -p 'test*.py' -v
python3 experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/verify_results.py
python3 experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/make_report.py
```

The [experiment index](experiments/README.md) links the complete procedure, hashes and results. The [paper source and current PDF](paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/README.md) live together. New measurement runs refuse to overwrite completed evidence; deterministic verification uses the saved rows without new rating samples.

## Origin and scope

The opinion piece [The Bitter Judges Lesson: Learn AI Judges the Way You Learn Any Other ML Task](https://cs.stanford.edu/people/brando9/2026/11/25/bitter-judges-lesson.html) (published 11-25-2026) states the position: define the target as the distribution of expert ratings, train with CE, score with proper scores, and scale the ratings. This repository holds the experiments behind it, including the [Exp 30 results](experiments/30_ce_vs_pearson_summary_readable_figures_start_25Nov2026/results_11-25-2026.md) and the [Exp 31 scaling-law fits](experiments/31_scaling_laws_for_judges_ce_vs_pearson_start_25Nov2026/results_11-25-2026.md) (commit [`9341110`](https://github.com/brando90/judges-dont-need-psychometrics-or-corr/commit/93411102e2148723ccf2d064af7279075da156eb)).

The project began with the [original README formulation at commit `f2272f9`](https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/f2272f9628f1f31575ecbdaa304f33c9ad0743f1/README.md), the [11-07-2026 discussion notes](paper_latex_and_notes/notes/fathom_call_11-07-2026/notes.md), the [whiteboard](paper_latex_and_notes/notes/whiteboard_11-07-2026.png), and [handwritten selection/fitting notes received 11-08-2026](paper_latex_and_notes/notes/handwritten_corr_vs_loss_splits_received_11-08-2026/transcription.md). Those source records remain unchanged. The [research synthesis](paper_latex_and_notes/notes/correlation_vs_loss_judges_notes_11-07-2026.md) provides further historical context.

The [trustworthy-theorem-judges](https://github.com/brando90/trustworthy-theorem-judges) project studies the applied judge problem. This repository’s new experiment isolates a distribution-identification question; it does not duplicate historical probability fitting, validate a deployed human judge, or establish natural-task generalization.

**TLDR-end:** [judge objectives: conditional identification] Evaluate conditional probability forecasts when the full rating distribution is the target. Correlations and scalar errors remain informative diagnostics, but neither alone identifies distribution shape.
