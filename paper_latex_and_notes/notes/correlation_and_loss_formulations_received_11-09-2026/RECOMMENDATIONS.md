# Recommendations for the four judge formulations

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/paper_latex_and_notes/notes/correlation_and_loss_formulations_received_11-09-2026/RECOMMENDATIONS.md>

**TLDR:** Keep correlation training as the comparison baseline, and separate training objectives from validation-selection criteria. The completed matched cross-entropy/Pearson comparison remains inconclusive; the next consequential step is a probability-output and human-rating pilot with explicit generalization tests across tasks, agents and raters.

Received 11-09-2026. This is an interpretation and recommendation, not a literal transcription or a change to any fixed experiment. See [the original-photo and transcription index](README.md). The instructions and conjectures written in the photos are source material, not newly executed tasks.

Goal: predict the distribution of ratings that qualified human raters would assign to the same artifact.
Confidence: high that proper scores target this distribution in expectation; empirical superiority of a particular fitted judge remains unresolved.
Importance: high.
Key uncertainty: whether a useful probability forecast can generalize beyond the previously inspected tasks, solver configurations and raters.

## The four numbers represent two stages

| Photo number | Operation | Data used for its objective |
| --- | --- | --- |
| 1 | Select among candidate judges by correlation | Validation |
| 2 | Fit or optimize a judge using correlation | Training |
| 3 | Select among candidate judges by a distribution loss | Validation |
| 4 | Fit or optimize a judge using a distribution loss | Training |

Methods 3 and 4 normally compose: fit on training data, then select settings on validation data. Methods 1 and 2 can compose in the same way. A fixed-candidate selection comparison, 1 versus 3, answers a different question from a matched fitting comparison, 2 versus 4.

The [earlier notes](../handwritten_corr_vs_loss_splits_received_11-08-2026/transcription.md) preserve the author's explicit clarification that 2 versus 4 is the primary comparison, with validation used for hyperparameters. The new photos do not justify dropping method 2: it remains the comparison needed to test the conjecture.

## Write the probability target and the selected model separately

Let z = (x, y_pred, y_gold) contain the Python source, candidate formal specifications and reference. Let r be a rating in {0,1,2,3,4,5}. Define q_theta(r | z) as the judge's probability forecast and p_H(r | z) as the probability that a person sampled from the specified qualified-rater population assigns r under the rubric.

The output is a six-category probability mass function:

q_theta(r | z) >= 0, and sum over r of q_theta(r | z) = 1.

Dividing ratings by five gives six support points {0,.2,.4,.6,.8,1}; it does not make the outcome continuous. A probability vector belongs to a simplex, not to the scalar real line. Use a sum for these categories, rather than the integral written on the whiteboard.

The optimizing argument is a configuration, not the distribution itself. For a training objective L, a validation criterion M and a setting lambda:

theta_hat(lambda,L) is a minimizer of training risk L for that setting.
lambda_hat(L,M) is a minimizer of validation risk M among those fitted candidates.
The final forecast is q_theta_hat(lambda_hat(L,M),L).

For correlation maximization, define the risk as negative correlation and specify what is correlated. The existing comparison uses Pearson correlation between expected judge ratings and empirical human mean ratings. Pearson and rank correlations are different objectives; a Pearson experiment does not test every correlation objective.

For probability fitting, score each individual human rating with cross-entropy (CE), Brier score or ranked probability score (RPS). Average those per-rating losses within an artifact first, then average artifacts using the declared population weights. Do not treat a two- or three-vote empirical histogram as the known population distribution, or automatically treat repeated scalar model outputs as human-rating probabilities.

Strictly proper scores have the true distribution as their unique unrestricted population optimum, under the applicable assumptions. This does not guarantee finite-sample superiority with a restricted model family, regularization or imperfect optimization. [Gneiting and Raftery, 2007](https://doi.org/10.1198/016214506000001437).

Correlation of expected ratings cannot identify the full distribution: two forecasts may have identical means on every artifact and different probabilities. Even Pearson correlation of flattened probability vectors is insufficient: mixing a nonconstant probability vector with a fixed fraction of the uniform vector preserves perfect correlation with the original vector while changing its probabilities.

## The literature does not support a universal correlation-training claim

| Primary source | Training or construction | Reported evaluation and implication |
| --- | --- | --- |
| [G-Eval, Microsoft, 2023](https://arxiv.org/html/2303.16634v3) | Prompted evaluator; score-token probabilities become a weighted scalar score | Reports human-rating correlations. This does not show a correlation training loss or an explicit validation-selection procedure. |
| [InstructGPT, OpenAI, 2022](https://arxiv.org/html/2203.02155v1) | Reward model uses pairwise logistic cross-entropy; the assistant policy then optimizes the learned reward | Human preference evaluation. Probability-loss training already exists, but its target is a binary preference, not the six-category rating distribution here. |
| [DeepSeek-GRM, DeepSeek-AI, 2025](https://arxiv.org/html/2504.02495v3) | Rejection-based fine-tuning followed by reinforcement learning rewarding correct best-response identification; a meta reward model uses binary cross-entropy | Evaluates reward-model accuracy and error detection. Its policy Kullback–Leibler penalty is not divergence against the human-rating distribution. |
| [CE-RM, Alibaba and collaborators, 2026](https://arxiv.org/html/2601.20327v1) | Reinforcement learning rewards criteria/evaluations that rank annotated chosen responses above rejected responses | Evaluates reward-model benchmarks and downstream use. Here CE means Criteria then Evaluation, not cross-entropy. |
| [How Proper Scoring Rules Shape LLM Forecasting, 2026](https://arxiv.org/abs/2608.28482v2) | Compares proper-score rewards while matching model, data and training setup | Binary event forecasting shows different log-score/Brier tradeoffs. It motivates a matched design but does not settle ordinal human-rating prediction. |

The defensible research question is whether probability-distribution fitting improves this specific judge task, compared with matched correlation fitting. It is not whether standard machine learning has never been used for judges.

## The newest completed comparison does not establish a winner

Source: [03_exp06_ce_vs_pearson_matched_budget_start_09Nov2026_end_09Nov2026 / RESULTS_comparison.md](../../../experiments/03_exp06_ce_vs_pearson_matched_budget_start_09Nov2026_end_09Nov2026/RESULTS_comparison.md), read on 11-09-2026.
This is a new simplex-mixture condition, not a relabeling of the failed historical affine-softmax fits.
Both methods fit the same five cached probability histograms with matched information, capacity and resource ceilings.
Scope: 75 artifacts, 25 tasks, 19 task families and 195 expert ratings; five historical partitions reuse that data.

| Held-out metric | CE fitting | Pearson fitting | CE minus Pearson, 95% refit interval |
| --- | ---: | ---: | --- |
| CE, natural logs, lower is better | 5.491441 | 5.751842 | -0.260401 [-1.457729, +0.198472] |
| Six-category Brier, lower is better | 0.889590 | 0.913693 | -0.024103 [-0.195676, +0.053535] |
| Normalized RPS, lower is better | 0.139080 | 0.143634 | -0.004555 [-0.035602, +0.009965] |

These are the same held-out artifacts and metrics. The intervals are paired percentile intervals from task-family resampling with refitting; 998/1,000 draws passed, with two failed numerical certificates retained and excluded from the intervals. p-val=n/a. Every listed paired interval contains zero.

Both probability mixtures have worse observed CE than the training-human-marginal baseline, 1.928001 [1.835097, 2.056740] with a fixed-prediction interval. This is a point-estimate comparison, not a refit paired significance claim. Sparse sampled-rating histograms constrain both models; further objective tuning alone may not resolve that limitation.

Mean normalized human rating is .525333; judge means are .591311 for CE and .588829 for Pearson. Signed biases, judge minus human, are +.065977 and +.063496. Their 95% refit intervals are [.005193,.137166] and [-.014875,.147779]. The full report contains correlations, absolute errors, repeated-score variability, numerical checks and all uncertainty limits.

All 50 primary numerical fit checks passed. No new model inference was needed. This is retrospective evidence; it does not test new solver populations or new human raters.

## Recommended next work

1. **[p 10/10] Preserve the completed 2-versus-4 result and its simple baseline.** Do not rerun it merely to obtain a CE win. Keep the synthetic distribution-identification result separate from finite-sample human performance.
2. **[p 10/10] Resolve the probability-output limitation in a small pilot.** Use explicit six-probability forecasts under the same human/AI inputs and fixed rubric. Compare a constant training-human distribution, the unchanged judge, and a low-capacity fitted judge. Separate this new forecast representation from the historical three-sample histogram condition. Record cost, invalid outputs and failures before broad collection.
3. **[p 10/10] Amend the split design before recruiting for the final study.** The user's stated requirement is disjoint train/validation/test identities across T, A and H. The current sister-repository experiment 02 plan shares its training/validation rater pool and uses leave-one-agent-out test procedures. It therefore does not implement strict three-way separation on every axis. Write the intended split explicitly, grouping related task variants and solver families. The decisive simultaneous-transfer block is T_test × A_test × H_test, with those identities absent from fitting and validation selection. Separate diagnostic blocks can measure generalization along one or two axes.
4. **[p 9/10] Compare training and selection as separate factors.** Start with CE versus Pearson training under the same candidate family. Where genuine hyperparameters/candidates exist, cross each with CE-based versus correlation-based validation selection. Add RPS/Brier training as declared comparisons, rather than selecting the most favorable metric afterward. Report every resulting judge on the same RPS, CE, total variation distance (TVD), Brier, expected-score correlations, absolute error, signed bias and repeatability panel.
5. **[p 9/10] Size the new tensor using the pilot and the intended claim.** Use paired loss differences and the actual task/rater dependence, missingness and workload. Fix a practically meaningful improvement before testing. Neither a dense T × A × H count nor the old correlation-based sizing guarantees adequate precision. Four chosen solver configurations provide limited information about a wider population of unseen agents.
6. **[p 6/10] Defer a scaling-law fit.** The whiteboard expression matches the Chinchilla form for language-model loss as a function of parameter count and training tokens. Its coefficients and exponents do not transfer automatically to judge training or to tasks, agents and raters. Collect judge-specific learning curves and test their predictive accuracy before using such a law for staffing or compute. [Hoffmann et al., 2022, section 3.3](https://arxiv.org/html/2203.15556v1#S3.SS3).

For the prospective human study, retain the existing planned primary RPS endpoint. **User decision, 11-09-2026:** explicitly report CE and total variation distance (TVD) as secondary metrics on the same held-out artifacts. TVD is half the sum of absolute differences between each forecast and its artifact's empirical human rating frequencies, then averaged equally across artifacts; it is a sparse-rating distribution diagnostic, not TVD mutual information. The [evaluation definitions](../../../experiments/03_exp06_ce_vs_pearson_matched_budget_start_09Nov2026_end_09Nov2026/EVALUATION_CRITERION_11-09-2026.md#prospective-secondary-metrics--user-decision-11-09-2026) preserve the common validation RPS, full denominator and missingness accounting. The training-by-selection factorial above is a proposed diagnostic, not a replacement of the primary endpoint. The completed experiment 03 keeps its CE primary contrast unchanged; its frozen protocols, executable sources and results are unchanged.

Training on “all the data” can mean refitting on training plus validation after choices are fixed; it must exclude the final test. If final performance is evaluated after such a refit, declare the refit beforehand and evaluate that exact procedure. Matching a human-rating distribution also remains distinct from proving formal-specification entailment or semantic correctness.

The active [sister-repository collection workflow](https://github.com/brando90/trustworthy-theorem-judges/blob/main/experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/README.md) preserves inherited set-N materials. That collection is distinct from the prospective distribution study. Preserve its delivered packets, assignment rules and evaluation-only scope; do not relabel those ratings as fresh training data or stop that collection because this note proposes a new study design.

No new measured run, new rater assignment or fixed-procedure change was launched by this transcription/recommendation task.

## Google companion synchronization

The intended destination is the existing [study methodology and sizing Q&A](https://docs.google.com/document/d/1zB-gYk-zU1wToS_E_carGSRGtUcF1NgZ-jWF5t1SJws/edit).
The connected Google Docs read returned HTTP 403 PERMISSION_DENIED on 11-09-2026; no update was applied.
The following exact text is retained for a supported authorized account once access is available:

> 11-09-2026 — Correlation and loss formulations. The four handwritten formulations distinguish training (2: correlation; 4: probability loss) from validation selection (1: correlation; 3: probability loss). Training and selection compose; retain 2 versus 4 as the main matched comparison and 1 versus 3 as a separate selection comparison. The completed historical-cache CE/Pearson mixture study does not establish superiority: paired CE difference -0.260401 [-1.457729, +0.198472], 95% task-family resampling with refitting, 998/1,000 valid draws, p-val=n/a. Both mixtures have worse observed CE than the training-human-marginal baseline. Next, pilot explicit six-category forecasts and revise the collection design to match the user's strict task × agent × human train/validation/test separation; the existing plan's shared training/validation rater pool does not yet implement that requirement. Proper scores identify the desired distribution in expectation, but do not guarantee finite-sample wins. The transcribed scaling formula is an analogy, not a validated judge sample-size law. Full original images, transcription, primary-source references and recommendations: https://github.com/brando90/judges-dont-need-psychometrics-or-corr/tree/main/paper_latex_and_notes/notes/correlation_and_loss_formulations_received_11-09-2026

**TLDR-end:** [judge objectives: formulation] Test correlation and proper-loss training fairly, using validation for selection and a genuinely separate three-axis test. Existing evidence supports the distribution-target distinction; it does not establish a universally best empirical judge.
