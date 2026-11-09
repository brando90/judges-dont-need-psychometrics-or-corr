# Conditional-distribution identification: results

**Status:** all declared measurements complete; independent deterministic verification complete. Parent multi-provider review and main publication remain pending.

Experiment `02_conditional_binomial_distribution_identification` uses a known synthetic binomial population. Correlation alone does not identify its conditional distribution: every shared-mean mixture remains tied under the scalar criteria. Proper losses identify the oracle in the population, but sampled validation ratings sometimes select a different mixture.

## Design and denominator

Nine candidates share 96 train, 48 validation and 48 test items from disjoint synthetic parameter-band families (6/3/3 families). The same 256 independent rating replications are compared with one, two or three ratings per item, paired by rating prefixes. Only the constant empirical distribution is fitted on train. Each criterion exhaustively searches the same nine candidates on validation. Test labels never enter fitting or selection.

All 768 simulation conditions, 5,376 selections and 6,912 candidate evaluations completed; 0 selections failed. CPU runtime 13.51 seconds, at most eight threads; zero provider calls and $0 provider inference charges. Source/input procedure was committed before measurement at `9139470`, with an assertion-shape-only correction at `5c0e9fc`. No study/scorer definition changed after measurement.

## Exact population test metrics

Cross-entropy (CE) and Kullback–Leibler divergence (KL) use natural logs. Brier sums six categories; ranked probability score (RPS) averages five ordinal thresholds. Normalized mean squared error (nMSE) and mean absolute error (MAE) compare the forecast mean to a future individual rating on [0,1]. These are exact finite-population expectations, so sampling intervals and p-values are not applicable (p-val=n/a).

| Candidate | CE | KL | Brier | RPS | nMSE | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mixture_0 | 1.328246 | 0.000000 | 0.699474 | 0.103693 | 0.037778 | 0.157269 |
| mixture_0.25 | 1.445526 | 0.117280 | 0.733285 | 0.109017 | 0.037778 | 0.157269 |
| mixture_0.5 | 1.688840 | 0.360594 | 0.834719 | 0.124992 | 0.037778 | 0.157269 |
| mixture_0.75 | 2.164147 | 0.835901 | 1.003777 | 0.151616 | 0.037778 | 0.157269 |
| mixture_0.95 | 3.347873 | 2.019627 | 1.187711 | 0.180584 | 0.037778 | 0.157269 |
| permuted | 4.214936 | 2.886690 | 1.112749 | 0.384040 | 0.282217 | 0.467547 |
| constant_binomial | 1.987848 | 0.659602 | 0.876649 | 0.188063 | 0.098888 | 0.273333 |
| constant_uniform | 1.791759 | 0.463513 | 0.833333 | 0.179629 | 0.098888 | 0.273333 |

The train-fitted constant varies across replications and is included in the full common candidate table below. For all five mixtures, Pearson/Spearman/Kendall against the exact conditional mean are 1, and Pearson against a future individual rating is 0.786111. Those are distinct targets. All five have zero conditional-mean error up to floating-point rounding; they have the same nonzero nMSE to individual ratings.

The permutation control preserves the aggregate histogram to 2.8e-17 maximum absolute difference but has conditional KL 2.886690. Its mean bias is zero by symmetry, despite wrong conditional forecasts. The fixed constants also have zero mean bias, but undefined correlations and larger conditional-mean error. Bias alone cannot detect these failures.

## Selection with sparse ratings

Ties are fully enumerated with absolute tolerance 1e-12. The reported tie mean is the expectation of choosing uniformly among minimizers, not the loss of an ensemble. A different fixed tie policy can choose the oracle for every scalar criterion; therefore the data do not imply superiority under every tie policy.

| Ratings/item | Selection objective | Oracle in selected set /256 | Mean tie size | Expected test CE [95% Monte Carlo CI] | Test CE tie minimum / maximum |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | ce | 249 | 1.0 | 1.331453 [1.329094, 1.333812] | 1.331453 / 1.331453 |
| 1 | brier | 245 | 1.0 | 1.333285 [1.330353, 1.336218] | 1.333285 / 1.333285 |
| 1 | rps | 245 | 1.0 | 1.333285 [1.330353, 1.336218] | 1.333285 / 1.333285 |
| 1 | pearson | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |
| 1 | spearman | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |
| 1 | kendall | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |
| 1 | nmse | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |
| 2 | ce | 256 | 1.0 | 1.328246 [1.328246, 1.328246] | 1.328246 / 1.328246 |
| 2 | brier | 256 | 1.0 | 1.328246 [1.328246, 1.328246] | 1.328246 / 1.328246 |
| 2 | rps | 256 | 1.0 | 1.328246 [1.328246, 1.328246] | 1.328246 / 1.328246 |
| 2 | pearson | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |
| 2 | spearman | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |
| 2 | kendall | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |
| 2 | nmse | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |
| 3 | ce | 256 | 1.0 | 1.328246 [1.328246, 1.328246] | 1.328246 / 1.328246 |
| 3 | brier | 256 | 1.0 | 1.328246 [1.328246, 1.328246] | 1.328246 / 1.328246 |
| 3 | rps | 256 | 1.0 | 1.328246 [1.328246, 1.328246] | 1.328246 / 1.328246 |
| 3 | pearson | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |
| 3 | spearman | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |
| 3 | kendall | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |
| 3 | nmse | 256 | 5.0 | 1.994927 [1.994927, 1.994927] | 1.328246 / 3.347873 |

Intervals use independent rating-replication seeds as the unit: Student-t Monte Carlo confidence intervals for the mean over repeated train/validation ratings on these fixed items. They do not quantify human/task-family transfer. All p-val=n/a; no confirmatory test was specified. The zero-width intervals when every replication makes the same choice describe observed repeat variability, not proof that population selection error is zero.

For a supplementary, post hoc description of finite simulation resolution, two-sided exact binomial 95% intervals for oracle selection are:

| Ratings/item | Criterion | Unique oracle selections | Exact binomial interval |
| ---: | --- | ---: | --- |
| 1 | ce | 249/256 | [0.944475, 0.988937] |
| 1 | brier | 245/256 | [0.924419, 0.978358] |
| 1 | rps | 245/256 | [0.924419, 0.978358] |
| 2 | ce | 256/256 | [0.985694, 1.000000] |
| 2 | brier | 256/256 | [0.985694, 1.000000] |
| 2 | rps | 256/256 | [0.985694, 1.000000] |
| 3 | ce | 256/256 | [0.985694, 1.000000] |
| 3 | brier | 256/256 | [0.985694, 1.000000] |
| 3 | rps | 256/256 | [0.985694, 1.000000] |

These supplementary intervals were added after observing the all-success counts; they do not replace the prespecified Monte Carlo analysis. Scalar objectives include the oracle in every tie but give it probability 1/5 under uniform tie-breaking. Inclusion is not identification.

## Common expected test metrics for every selection criterion

All values average each criterion’s uniform-tie expectation across the same replications. No criterion receives a preferred test metric. Full 95% Monte Carlo intervals, standard deviations and 2.5/97.5 percentiles for every metric and every tie extreme are in `results/summary.json`.

| Ratings/item | Criterion | CE | Brier | RPS | KL | nMSE | MAE | Bias | Judge mean | Population mean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | brier | 1.333285 | 0.700926 | 0.103921 | 0.005039 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 1 | ce | 1.331453 | 0.700398 | 0.103838 | 0.003207 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 1 | kendall | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 1 | nmse | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 1 | pearson | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 1 | rps | 1.333285 | 0.700926 | 0.103921 | 0.005039 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 1 | spearman | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 2 | brier | 1.328246 | 0.699474 | 0.103693 | 0.000000 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 2 | ce | 1.328246 | 0.699474 | 0.103693 | 0.000000 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 2 | kendall | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 2 | nmse | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 2 | pearson | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 2 | rps | 1.328246 | 0.699474 | 0.103693 | 0.000000 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 2 | spearman | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 3 | brier | 1.328246 | 0.699474 | 0.103693 | 0.000000 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 3 | ce | 1.328246 | 0.699474 | 0.103693 | 0.000000 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 3 | kendall | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 3 | nmse | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 3 | pearson | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 3 | rps | 1.328246 | 0.699474 | 0.103693 | 0.000000 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |
| 3 | spearman | 1.994927 | 0.891793 | 0.133980 | 0.666680 | 0.037778 | 0.157269 | 0.000000 | 0.500000 | 0.500000 |

## Sampled test-rating diagnostics

All seven selection criteria have identical sampled test correlations and scalar errors in this run, because every selected candidate belongs to the five mixtures with the same conditional mean. These diagnostic ties are retained rather than described as a correlation improvement.

Pearson, Spearman and Kendall tau-b compare forecast means with sampled item-rating means. Mean absolute error, bias and means use normalized scores. Bias is forecast minus rating, positive for higher forecasts. The same test outcomes serve every criterion; intervals are over the same 256 replication seeds.

| Ratings/item | Criterion | Pearson [95% CI] | Spearman [95% CI] | Kendall [95% CI] | MAE | Bias | Judge mean | Rating mean |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | brier | 0.791991 [0.785449, 0.798533] | 0.781760 [0.775100, 0.788419] | 0.623820 [0.616721, 0.630920] | 0.157773 | 0.001123 | 0.500000 | 0.498877 |
| 1 | ce | 0.791991 [0.785449, 0.798533] | 0.781760 [0.775100, 0.788419] | 0.623820 [0.616721, 0.630920] | 0.157773 | 0.001123 | 0.500000 | 0.498877 |
| 1 | kendall | 0.791991 [0.785449, 0.798533] | 0.781760 [0.775100, 0.788419] | 0.623820 [0.616721, 0.630920] | 0.157773 | 0.001123 | 0.500000 | 0.498877 |
| 1 | nmse | 0.791991 [0.785449, 0.798533] | 0.781760 [0.775100, 0.788419] | 0.623820 [0.616721, 0.630920] | 0.157773 | 0.001123 | 0.500000 | 0.498877 |
| 1 | pearson | 0.791991 [0.785449, 0.798533] | 0.781760 [0.775100, 0.788419] | 0.623820 [0.616721, 0.630920] | 0.157773 | 0.001123 | 0.500000 | 0.498877 |
| 1 | rps | 0.791991 [0.785449, 0.798533] | 0.781760 [0.775100, 0.788419] | 0.623820 [0.616721, 0.630920] | 0.157773 | 0.001123 | 0.500000 | 0.498877 |
| 1 | spearman | 0.791991 [0.785449, 0.798533] | 0.781760 [0.775100, 0.788419] | 0.623820 [0.616721, 0.630920] | 0.157773 | 0.001123 | 0.500000 | 0.498877 |
| 2 | brier | 0.878114 [0.874453, 0.881775] | 0.864227 [0.860184, 0.868269] | 0.702064 [0.696681, 0.707446] | 0.158097 | -0.000358 | 0.500000 | 0.500358 |
| 2 | ce | 0.878114 [0.874453, 0.881775] | 0.864227 [0.860184, 0.868269] | 0.702064 [0.696681, 0.707446] | 0.158097 | -0.000358 | 0.500000 | 0.500358 |
| 2 | kendall | 0.878114 [0.874453, 0.881775] | 0.864227 [0.860184, 0.868269] | 0.702064 [0.696681, 0.707446] | 0.158097 | -0.000358 | 0.500000 | 0.500358 |
| 2 | nmse | 0.878114 [0.874453, 0.881775] | 0.864227 [0.860184, 0.868269] | 0.702064 [0.696681, 0.707446] | 0.158097 | -0.000358 | 0.500000 | 0.500358 |
| 2 | pearson | 0.878114 [0.874453, 0.881775] | 0.864227 [0.860184, 0.868269] | 0.702064 [0.696681, 0.707446] | 0.158097 | -0.000358 | 0.500000 | 0.500358 |
| 2 | rps | 0.878114 [0.874453, 0.881775] | 0.864227 [0.860184, 0.868269] | 0.702064 [0.696681, 0.707446] | 0.158097 | -0.000358 | 0.500000 | 0.500358 |
| 2 | spearman | 0.878114 [0.874453, 0.881775] | 0.864227 [0.860184, 0.868269] | 0.702064 [0.696681, 0.707446] | 0.158097 | -0.000358 | 0.500000 | 0.500358 |
| 3 | brier | 0.913626 [0.911164, 0.916088] | 0.898857 [0.895993, 0.901721] | 0.742148 [0.737707, 0.746590] | 0.157570 | 0.000640 | 0.500000 | 0.499360 |
| 3 | ce | 0.913626 [0.911164, 0.916088] | 0.898857 [0.895993, 0.901721] | 0.742148 [0.737707, 0.746590] | 0.157570 | 0.000640 | 0.500000 | 0.499360 |
| 3 | kendall | 0.913626 [0.911164, 0.916088] | 0.898857 [0.895993, 0.901721] | 0.742148 [0.737707, 0.746590] | 0.157570 | 0.000640 | 0.500000 | 0.499360 |
| 3 | nmse | 0.913626 [0.911164, 0.916088] | 0.898857 [0.895993, 0.901721] | 0.742148 [0.737707, 0.746590] | 0.157570 | 0.000640 | 0.500000 | 0.499360 |
| 3 | pearson | 0.913626 [0.911164, 0.916088] | 0.898857 [0.895993, 0.901721] | 0.742148 [0.737707, 0.746590] | 0.157570 | 0.000640 | 0.500000 | 0.499360 |
| 3 | rps | 0.913626 [0.911164, 0.916088] | 0.898857 [0.895993, 0.901721] | 0.742148 [0.737707, 0.746590] | 0.157570 | 0.000640 | 0.500000 | 0.499360 |
| 3 | spearman | 0.913626 [0.911164, 0.916088] | 0.898857 [0.895993, 0.901721] | 0.742148 [0.737707, 0.746590] | 0.157570 | 0.000640 | 0.500000 | 0.499360 |

Sampling one/two/three independent binomial ratings gives per-item rating-mean standard deviations sqrt(5t(1-t)/m) before normalization. These are synthetic sampling variability, not language-model repeatability or human inter-rater reliability. A single model-call repeatability estimate is unavailable because no model is called.

## All candidate controls on common test metrics

Every candidate is evaluated regardless of selection. Rows below use one rating/item; population metrics for fixed candidates are unchanged at two/three ratings, and the complete train-constant/sample results for every condition are in `results/candidate_rows.csv`.

| Candidate | CE | Brier | RPS | nMSE | MAE | Mean MAE | Bias | Judge mean | Population mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixture_0 | 1.328246 | 0.699474 | 0.103693 | 0.037778 | 0.157269 | 0.000000 | 0.000000 | 0.500000 | 0.500000 |
| mixture_0.25 | 1.445526 | 0.733285 | 0.109017 | 0.037778 | 0.157269 | 0.000000 | 0.000000 | 0.500000 | 0.500000 |
| mixture_0.5 | 1.688840 | 0.834719 | 0.124992 | 0.037778 | 0.157269 | 0.000000 | 0.000000 | 0.500000 | 0.500000 |
| mixture_0.75 | 2.164147 | 1.003777 | 0.151616 | 0.037778 | 0.157269 | 0.000000 | 0.000000 | 0.500000 | 0.500000 |
| mixture_0.95 | 3.347873 | 1.187711 | 0.180584 | 0.037778 | 0.157269 | 0.000000 | 0.000000 | 0.500000 | 0.500000 |
| permuted | 4.214936 | 1.112749 | 0.384040 | 0.282217 | 0.467547 | 0.455556 | -0.000000 | 0.500000 | 0.500000 |
| constant_binomial | 1.987848 | 0.876649 | 0.188063 | 0.098888 | 0.273333 | 0.227778 | 0.000000 | 0.500000 | 0.500000 |
| constant_uniform | 1.791759 | 0.833333 | 0.179629 | 0.098888 | 0.273333 | 0.227778 | 0.000000 | 0.500000 | 0.500000 |
| constant_train | 1.793381 | 0.834069 | 0.179589 | 0.099187 | 0.273333 | 0.227778 | 0.000314 | 0.500314 | 0.500000 |

## Historical evidence preserved

The original `01_synthetic_pearson_vs_kl_squashed_judge_0to5` code and historical outputs were not edited. `results/old_toy_stdout.txt` and `results/old_toy_reproduction.json` record the new reproduction: the original printed values match exactly. It compares smoothed marginal histograms, not conditional judge selection. Compressing six scores into two bins introduces ties; it is not a strictly increasing transform.

The sister repository’s `06_information_loss_vs_correlation_judge_selection` has a different purpose: retrospective comparisons on 75 artifacts /25 tasks /19 families /195 expert ratings. Only public aggregate reports were read and cross-linked; no private data were copied.

- [Historical cached and fitted-head report](https://github.com/brando90/trustworthy-theorem-judges/blob/5aef572d30048e6ce574b39d91877c14ab9b3ad7/experiments/06_information_loss_vs_correlation_judge_selection/RESULTS.md): cached CE selection gave held-out CE 8.868 versus 8.098 for Spearman selection, difference +0.770 [-0.895, 1.317], selection-refit 95% family-bootstrap interval, p-val=n/a. This does not establish a CE advantage. The fitted-head CE difference was -0.370 [-0.690, -0.080], conditional on fitted heads; 215/225 Pearson fits and 24/25 final Pearson refits did not converge. The numerical optimizer failure limits interpretation.
- [Historical distance follow-up](https://github.com/brando90/trustworthy-theorem-judges/blob/5aef572d30048e6ce574b39d91877c14ab9b3ad7/experiments/06_information_loss_vs_correlation_judge_selection/followup_11-07-2026/RESULTS.md): Wasserstein difference +0.023822 [-0.036327, +0.053333] and squared maximum mean discrepancy difference +0.107807 [-0.128950, +0.171668] versus Spearman selection, selection-refit 95% family-bootstrap intervals, p-val=n/a. Neither establishes an advantage.
- [Completed weighted-probability/stable-Pearson follow-up](https://github.com/brando90/trustworthy-theorem-judges/blob/ea5d84238f6f27bf24ddc0359409921a2e55c459/experiments/06_information_loss_vs_correlation_judge_selection/training_followup_11-08-2026/RESULTS_training.md): weighted-mixture training gave weighted loss 0.191659 versus 0.202068 for stable Pearson training, difference -0.010409 [-0.061911, +0.005933], 95% family-bootstrap interval with both mixtures refitted, p-val=n/a. All 50 primary fits passed checks and all 900 secondary fits converged. The interval conditions on 998/1,000 valid bootstrap draws; two failed certificate checks were retained without replacement. It still does not resolve an advantage. Another owner completed this condition; this study did not duplicate or modify it. Fresh human returns were not independently verified from cloud here and are not declared absent.

Exact public source hashes and import assertions are in `results/public_ttj_evidence.json` and `import_public_evidence.py`.

## Interpretation and limits

The supported thesis is that correlation alone does not identify the desired conditional distribution. The scalar criteria receive identical conditional means by construction; the experiment isolates that information loss. It does not establish that correlation cannot help diagnose ranking, that every psychometric method is unnecessary, or that proper losses always select better systems with finite data. Mean squared error identifies a conditional mean, not an arbitrary distribution. Native probability forecasts, rater-population definitions, and matched inputs still matter.

One observed rating supplies a valid conditional log-loss observation. It does not identify an unconstrained item distribution by itself. Pooling marginal histograms is not required for single-rating evaluation and cannot repair wrong item conditioning. Generalization requires shared structure, data and assumptions beyond this synthetic design.

The oracle belongs to the candidate pool, the binomial family and independence assumptions are known, and task families are parameter bands. The study contains no natural task outputs, human ratings, rater heterogeneity, fitted neural judges, or new language-model calls. Test losses are exact for the fixed synthetic test grid; repeated validation outcomes are not a new independent natural-output test set.

## Reproduction and verification

`run_study.py` checks the procedure/source hashes and refuses to overwrite completed evidence. `test_study.py` passes eight analytic/scorer/sampler/split checks. `verify_results.py` independently reconstructs all 5,376 tie memberships, 419,328 selected metric entries, summary intervals and direct six-outcome population expectations without new rating samples. `make_report.py` produces this report from the stored tables. The input manifest, every compared candidate row, selection row and output hash are public.

```bash
python3 -m unittest discover -s experiments/02_conditional_binomial_distribution_identification -p test_study.py -v
python3 experiments/02_conditional_binomial_distribution_identification/verify_results.py
python3 experiments/02_conditional_binomial_distribution_identification/make_report.py
```

Copy-ready message: I isolated conditional rating-distribution identification in a synthetic binomial study. Proper probability losses identified the true distribution in the population, but finite-sample selection still made mistakes; correlations and scalar squared error tied across different distributions with the same mean, so they cannot identify distribution shape by themselves.

**TLDR-end:** [judge objectives: conditional identification] All declared synthetic comparisons completed with full tie accounting and exact expected test losses. The result supports conditional probability evaluation, while preserving null historical evidence and making no human-generalization or universal-superiority claim.
