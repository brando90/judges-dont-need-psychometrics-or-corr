#!/usr/bin/env python3
"""Render all common metrics from the fixed study's committed result rows."""
import csv,json,math
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='8'
import numpy as np
from scipy.stats import beta
root=Path(__file__).parent
out=root/'results'
summary=json.loads((out/'summary.json').read_text())
receipt=json.loads((out/'COMPLETE.json').read_text())
controls=json.loads((out/'analytic_controls.json').read_text())
historical=json.loads((out/'public_ttj_evidence.json').read_text())
pop=list(csv.DictReader((out/'population_metrics.csv').open()))
sel=list(csv.DictReader((out/'selection_rows.csv').open()))
cand=list(csv.DictReader((out/'candidate_rows.csv').open()))

def number(x):
    return 'undefined' if x is None or x=='' else f'{float(x):.6f}'

def interval(stats):
    if stats['mean'] is None:return 'undefined'
    return number(stats['mean'])+' ['+', '.join(number(v) for v in stats['mc_ci95'])+']'

def count_ci(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]

lines=['# Conditional-distribution identification: results','',
'**Status:** all declared measurements complete; independent deterministic verification complete. Parent multi-provider review and main publication remain pending.','',
'Experiment `02_conditional_binomial_distribution_identification` uses a known synthetic binomial population. Correlation alone does not identify its conditional distribution: every shared-mean mixture remains tied under the scalar criteria. Proper losses identify the oracle in the population, but sampled validation ratings sometimes select a different mixture.','',
'## Design and denominator','',
'Nine candidates share 96 train, 48 validation and 48 test items from disjoint synthetic parameter-band families (6/3/3 families). The same 256 independent rating replications are compared with one, two or three ratings per item, paired by rating prefixes. Only the constant empirical distribution is fitted on train. Each criterion exhaustively searches the same nine candidates on validation. Test labels never enter fitting or selection.','',
f'All {receipt["simulation_conditions"]} simulation conditions, {receipt["selection_rows"]:,} selections and {receipt["candidate_rows"]:,} candidate evaluations completed; {receipt["failures"]} selections failed. CPU runtime {receipt["wall_seconds"]:.2f} seconds, at most eight threads; zero provider calls and $0 provider inference charges. Source/input procedure was committed before measurement at `9139470`, with an assertion-shape-only correction at `5c0e9fc`. No study/scorer definition changed after measurement.','',
'## Exact population test metrics','',
'Cross-entropy (CE) and Kullback–Leibler divergence (KL) use natural logs. Brier sums six categories; ranked probability score (RPS) averages five ordinal thresholds. Normalized mean squared error (nMSE) and mean absolute error (MAE) compare the forecast mean to a future individual rating on [0,1]. These are exact finite-population expectations, so sampling intervals and p-values are not applicable (p-val=n/a).','',
'| Candidate | CE | KL | Brier | RPS | nMSE | MAE |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for row in pop:lines.append('| '+row['candidate']+' | '+' | '.join(number(row[k]) for k in ['ce','kl','brier','rps','nmse','mae'])+' |')
lines+=['','The train-fitted constant varies across replications and is included in the full common candidate table below. For all five mixtures, Pearson/Spearman/Kendall against the exact conditional mean are 1, and Pearson against a future individual rating is '+number(pop[0]['pearson_individual_rating'])+'. Those are distinct targets. All five have zero conditional-mean error up to floating-point rounding; they have the same nonzero nMSE to individual ratings.','',
f'The permutation control preserves the aggregate histogram to {controls["permutation_marginal_max_absolute_difference"]:.2g} maximum absolute difference but has conditional KL {controls["permutation_conditional_kl"]:.6f}. Its mean bias is zero by symmetry, despite wrong conditional forecasts. The fixed constants also have zero mean bias, but undefined correlations and larger conditional-mean error. Bias alone cannot detect these failures.','',
'## Selection with sparse ratings','',
'Ties are fully enumerated with absolute tolerance 1e-12. The reported tie mean is the expectation of choosing uniformly among minimizers, not the loss of an ensemble. A different fixed tie policy can choose the oracle for every scalar criterion; therefore the data do not imply superiority under every tie policy.','',
'| Ratings/item | Selection objective | Oracle in selected set /256 | Mean tie size | Expected test CE [95% Monte Carlo CI] | Test CE tie minimum / maximum |',
'| ---: | --- | ---: | ---: | --- | --- |']
for m,criteria in summary.items():
    for obj in ['ce','brier','rps','pearson','spearman','kendall','nmse']:
        data=criteria[obj]; met=data['metrics']
        lines.append(f'| {m} | {obj} | {data["oracle_in_tie_count"]} | {met["tie_size"]["mean"]:.1f} | {interval(met["test_expected_ce_tie_mean"])} | {number(met["test_expected_ce_tie_min"]["mean"])} / {number(met["test_expected_ce_tie_max"]["mean"])} |')
lines+=['',
'Intervals use independent rating-replication seeds as the unit: Student-t Monte Carlo confidence intervals for the mean over repeated train/validation ratings on these fixed items. They do not quantify human/task-family transfer. All p-val=n/a; no confirmatory test was specified. The zero-width intervals when every replication makes the same choice describe observed repeat variability, not proof that population selection error is zero.','',
'For a supplementary, post hoc description of finite simulation resolution, two-sided exact binomial 95% intervals for oracle selection are:','',
'| Ratings/item | Criterion | Unique oracle selections | Exact binomial interval |','| ---: | --- | ---: | --- |']
for m in ['1','2','3']:
    for obj in ['ce','brier','rps']:
        k=summary[m][obj]['oracle_in_tie_count']; lo,hi=count_ci(k,256)
        lines.append(f'| {m} | {obj} | {k}/256 | [{lo:.6f}, {hi:.6f}] |')
lines+=['','These supplementary intervals were added after observing the all-success counts; they do not replace the prespecified Monte Carlo analysis. Scalar objectives include the oracle in every tie but give it probability 1/5 under uniform tie-breaking. Inclusion is not identification.','',
'## Common expected test metrics for every selection criterion','',
'All values average each criterion’s uniform-tie expectation across the same replications. No criterion receives a preferred test metric. Full 95% Monte Carlo intervals, standard deviations and 2.5/97.5 percentiles for every metric and every tie extreme are in `results/summary.json`.','',
'| Ratings/item | Criterion | CE | Brier | RPS | KL | nMSE | MAE | Bias | Judge mean | Population mean |',
'| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for m,criteria in summary.items():
    for obj,data in criteria.items():
        lines.append('| '+m+' | '+obj+' | '+' | '.join(number(data['metrics']['test_expected_'+k+'_tie_mean']['mean']) for k in ['ce','brier','rps','kl','nmse','mae','bias','judge_mean','human_mean'])+' |')
lines+=['','## Sampled test-rating diagnostics','',
'All seven selection criteria have identical sampled test correlations and scalar errors in this run, because every selected candidate belongs to the five mixtures with the same conditional mean. These diagnostic ties are retained rather than described as a correlation improvement.','',
'Pearson, Spearman and Kendall tau-b compare forecast means with sampled item-rating means. Mean absolute error, bias and means use normalized scores. Bias is forecast minus rating, positive for higher forecasts. The same test outcomes serve every criterion; intervals are over the same 256 replication seeds.','',
'| Ratings/item | Criterion | Pearson [95% CI] | Spearman [95% CI] | Kendall [95% CI] | MAE | Bias | Judge mean | Rating mean |',
'| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
for m,criteria in summary.items():
    for obj,data in criteria.items():
        vals=[interval(data['metrics']['test_sample_'+k+'_tie_mean']) for k in ['pearson','spearman','kendall']]
        vals +=[number(data['metrics']['test_sample_'+k+'_tie_mean']['mean']) for k in ['mae','bias','judge_mean','human_mean']]
        lines.append('| '+m+' | '+obj+' | '+' | '.join(vals)+' |')
lines+=['','Sampling one/two/three independent binomial ratings gives per-item rating-mean standard deviations sqrt(5t(1-t)/m) before normalization. These are synthetic sampling variability, not language-model repeatability or human inter-rater reliability. A single model-call repeatability estimate is unavailable because no model is called.','',
'## All candidate controls on common test metrics','',
'Every candidate is evaluated regardless of selection. Rows below use one rating/item; population metrics for fixed candidates are unchanged at two/three ratings, and the complete train-constant/sample results for every condition are in `results/candidate_rows.csv`.','',
'| Candidate | CE | Brier | RPS | nMSE | MAE | Mean MAE | Bias | Judge mean | Population mean |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for name in [r['candidate'] for r in pop]+['constant_train']:
    rows=[r for r in cand if r['candidate']==name and r['ratings_per_item']=='1']
    lines.append('| '+name+' | '+' | '.join(number(np.mean([float(r['test_expected_'+k]) for r in rows])) for k in ['ce','brier','rps','nmse','mae','mean_mae','bias','judge_mean','human_mean'])+' |')
lines+=['','## Historical evidence preserved','',
'The original `01_synthetic_pearson_vs_kl_squashed_judge_0to5` code and historical outputs were not edited. `results/old_toy_stdout.txt` and `results/old_toy_reproduction.json` record the new reproduction: the original printed values match exactly. It compares smoothed marginal histograms, not conditional judge selection. Compressing six scores into two bins introduces ties; it is not a strictly increasing transform.','',
'The sister repository’s `06_information_loss_vs_correlation_judge_selection` has a different purpose: retrospective comparisons on 75 artifacts /25 tasks /19 families /195 expert ratings. Only public aggregate reports were read and cross-linked; no private data were copied.','',
f'- [Historical cached and fitted-head report]({historical["sources"][0]["url"]}): across five seeded five-fold task-family partitions (25 outer folds; 375 pooled out-of-fold rows reusing the 75 artifacts rather than 375 independent artifacts), cached CE selection gave held-out CE 8.868 versus 8.098 for Spearman selection, difference +0.770 [-0.895, 1.317], selection-refit 95% family-bootstrap interval, p-val=n/a. On the original historical split (15 validation artifacts selecting over 30 test artifacts / 10 tasks / 78 expert ratings), CE selection gave 9.133 versus 8.710 for Spearman selection (difference +0.423). Neither denominator establishes a CE advantage, and the historical ratings, test split and candidate prompts/demonstrations had prior exposure that new family folds do not undo. The fitted-head CE difference was -0.370 [-0.690, -0.080], conditional on fitted heads; 215/225 Pearson fits and 24/25 final Pearson refits did not converge. The numerical optimizer failure limits interpretation.',
f'- [Historical distance follow-up]({historical["sources"][1]["url"]}): Wasserstein difference +0.023822 [-0.036327, +0.053333] and squared maximum mean discrepancy difference +0.107807 [-0.128950, +0.171668] versus Spearman selection, selection-refit 95% family-bootstrap intervals, p-val=n/a. Neither establishes an advantage.',
f'- [Completed weighted-probability/stable-Pearson follow-up]({historical["sources"][2]["url"]}): weighted-mixture training gave weighted loss 0.191659 versus 0.202068 for stable Pearson training, difference -0.010409 [-0.061911, +0.005933], 95% family-bootstrap interval with both mixtures refitted, p-val=n/a. All 50 primary fits passed checks and all 900 secondary fits converged. The interval conditions on 998/1,000 valid bootstrap draws; two failed certificate checks were retained without replacement. It still does not resolve an advantage. Another owner completed this condition; this study did not duplicate or modify it. Fresh human returns were not independently verified from cloud here and are not declared absent.','',
'Exact public source hashes and import assertions are in `results/public_ttj_evidence.json` and `import_public_evidence.py`.','',
'## Interpretation and limits','',
'The supported thesis is that correlation alone does not identify the desired conditional distribution. The scalar criteria receive identical conditional means by construction; the experiment isolates that information loss. It does not establish that correlation cannot help diagnose ranking, that every psychometric method is unnecessary, or that proper losses always select better systems with finite data. Mean squared error identifies a conditional mean, not an arbitrary distribution. Native probability forecasts, rater-population definitions, and matched inputs still matter.','',
'One observed rating supplies a valid conditional log-loss observation. It does not identify an unconstrained item distribution by itself. Pooling marginal histograms is not required for single-rating evaluation and cannot repair wrong item conditioning. Generalization requires shared structure, data and assumptions beyond this synthetic design.','',
'The oracle belongs to the candidate pool, the binomial family and independence assumptions are known, and task families are parameter bands. The study contains no natural task outputs, human ratings, rater heterogeneity, fitted neural judges, or new language-model calls. Test losses are exact for the fixed synthetic test grid; repeated validation outcomes are not a new independent natural-output test set.','',
'## Reproduction and verification','',
'`run_study.py` checks the procedure/source hashes and refuses to overwrite completed evidence. `test_study.py` passes eight analytic/scorer/sampler/split checks, and `test_validated_scorer.py` exercises the prospective input-validation wrapper (`validated_scorer.py`). `verify_results.py` independently reconstructs all 5,376 tie memberships, 419,328 selected metric entries, summary intervals and direct six-outcome population expectations without new rating samples. `make_report.py` produces this report from the stored tables. The input manifest, every compared candidate row, selection row and output hash are public.','',
'```bash',
"python3 -m unittest discover -s experiments/02_conditional_binomial_distribution_identification -p 'test*.py' -v",
'python3 experiments/02_conditional_binomial_distribution_identification/verify_results.py',
'python3 experiments/02_conditional_binomial_distribution_identification/make_report.py',
'```','',
'Copy-ready message: I isolated conditional rating-distribution identification in a synthetic binomial study. Proper probability losses identified the true distribution in the population, but finite-sample selection still made mistakes; correlations and scalar squared error tied across different distributions with the same mean, so they cannot identify distribution shape by themselves.','',
'**TLDR-end:** [judge objectives: conditional identification] All declared synthetic comparisons completed with full tie accounting and exact expected test losses. The result supports conditional probability evaluation, while preserving null historical evidence and making no human-generalization or universal-superiority claim.','']
(root/'RESULTS_conditional_identification.md').write_text('\n'.join(lines))
print('\n'.join(lines[:25]))
