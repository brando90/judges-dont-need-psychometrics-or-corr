"""Generate the report from saved measured outputs; performs no new fitting."""
import gzip
import json
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results/results.json').read_text())
d=json.loads((ROOT/'results/diagnostic.json').read_text())
v=json.loads((ROOT/'results/verification.json').read_text())
b=json.loads(gzip.decompress((ROOT/'results/bootstrap_fit_receipts.json.gz').read_bytes()))
c=r['counts'];delta=r['ce_minus_pearson_refit']['ce']
doc='https://docs.google.com/document/d/1f6AtTPIoHx-B3jaOy4ot8TNH__t_Kbiut5KwO0kKuX0/edit'
url='https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/03_exp06_ce_vs_pearson_matched_budget/RESULTS_comparison.md'
fmt=lambda x:f'{x:.6f}'
ci=lambda x:f"[{x['lo']:.6f}, {x['hi']:.6f}]" if x['lo'] is not None else 'undefined'
lines=['# Matched cross-entropy versus Pearson: results',
       f'**Doc link:** [{url}]({url}) · [Google findings companion]({doc})',
       '**Status, 11-09-2026:** comparison complete; independent primary-prediction and metric reconstruction passed. Paper integration remains a live dependency.',
       'Experiment `03_exp06_ce_vs_pearson_matched_budget` continues historical experiment 06 in the sister repository. It diagnoses the old affine-softmax failure and reports a separate matched-simplex condition. It does not relabel the historical softmax fits as repaired or converged.',
       '## Main result',
       f"Cross-entropy (CE) fitting minus Pearson fitting in held-out CE was **{fmt(delta['estimate'])} {ci(delta)}, p-val=n/a**. The 95% percentile interval resamples task families and refits both methods; {delta['valid_draws']}/{c['bootstrap_draws']} draws passed all certificates. The interval includes zero: this cache does not establish a CE advantage.",
       f"CE fitting gave CE {fmt(r['matrix']['ce']['ce'])}, Pearson fitting {fmt(r['matrix']['pearson']['ce'])}, and the training-human-marginal baseline {fmt(r['matrix']['constant']['ce'])}. Both mixture methods are poor probability forecasts relative to that baseline, despite useful score correlations. Sparse sampled-rating histograms restrict the common mixture family; this comparison does not repair that representational limitation.",
       f"Scope: {c['artifacts']} artifacts, {c['tasks']} tasks, {c['families']} task families, {c['expert_ratings']} expert ratings. Five historical partitions reuse these same artifacts. All {c['primary_fits']} primary fits passed; {c['bootstrap_fits']} bootstrap fits were attempted, with {c['failed_bootstrap_fits']} failed certificates retained. There were no new model calls or provider inference charges. The recorded comparison elapsed time was {r['elapsed_seconds']:.2f} seconds on one local CPU thread.",
       '## Historical failure diagnosis',
       'The first historical failing receipt was selected in saved order, using only its training subset and the original initialization/settings. The reproduction hit the same iteration limit. The local SciPy version differs from the original, and function-call counts differ, so this is a configured-condition reproduction rather than a bitwise trajectory reproduction.',
       '| Training-only condition | Iterations | Evaluations | Converged | Gradient infinity norm | Expected-score SD | Training Pearson | L2 penalty |',
       '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
for a in d['attempts']:
 f=a['final'];lines.append(f"| {a['label']} | {a['nit']} | {a['nfev']} | {a['success']} | {f['gradient_inf']:.8g} | {f['prediction_sd']:.8g} | {fmt(f['true_rho'])} | {f['penalty']:.8g} |")
lines += ['L2 means a squared-coefficient penalty; SD means standard deviation. Expected scores are normalized to 0–1. These deterministic diagnostics have no sampling confidence interval or hypothesis test (p-val=n/a).',
          'The observed mechanism is ill-conditioned optimization near nearly constant expected scores, not a demonstrated analytic-gradient coding error. Pearson is insensitive to positive affine changes of expected scores; its gradient contains inverse prediction-spread terms. A coefficient penalty adds pressure toward smaller weights, while the softmax mean can collapse even with nonzero coefficients. The longer run decreased the objective but made score spread smaller and left a substantial gradient. Removing the penalty increased spread but still failed at the original budget: regularization is not established as the sole cause.',
          'Scaling the final coefficient matrix does not preserve correlation, because softmax is nonlinear; the scaling diagnostic explicitly shows this. Thus coefficient-scale invariance must not be confused with Pearson’s invariance in score space. Common shifts of all class logits leave probabilities unchanged, confirming redundant stored coefficients; removing that redundancy alone is not demonstrated to solve the failure.',
          '| Initial-coefficient scale | Best relative directional-gradient error across declared steps |',
          '| ---: | ---: |']
for scale in sorted({g['scale'] for g in d['directional_gradient_checks']},reverse=True):
 best=min(g['relative_error'] for g in d['directional_gradient_checks'] if g['scale']==scale)
 lines.append(f'| {scale:g} | {best:.8g} |')
lines += [f"The probability change under a common class-logit shift was {d['softmax_row_shift_max_probability_error']:.3g}. Directional checks used three declared finite-difference steps and show step-size sensitivity near constant predictions. Full traces, rather than only selected endpoints, are in `results/diagnostic.json`.",
          '## Corrected comparison and fair budgets',
          'Both objectives fit global nonnegative weights summing to one over the same five once-smoothed candidate histograms. Data, candidate order, capacity, folds and resource ceilings match. There is one configuration per method and no validation search or test-dependent output calibration. This is a new probability family, with four free mixture dimensions; comparing its results directly against historical softmax as an objective-only effect would be invalid.',
          'CE uses convex constrained optimization plus a bounded numerical refinement and an independent convex suboptimality bound. Pearson uses finite cone-projection face enumeration with fixed tie and constant-forecast rules. Matching resource ceilings does not mean that solver steps have equal computational cost. Objective-specific numerical checks establish optimization quality in each problem.',
          '| Objective | Primary fits | Failed | Max function evaluations | Max linear solves | Total primary-fit seconds |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
for loss in ['ce','pearson']:
 fs=[f for f in r['fits'] if f['objective']==loss]
 lines.append(f"| {loss} | {len(fs)} | {sum(not f['success'] for f in fs)} | {max(f['function_evaluations'] for f in fs)} | {max(f['linear_solves'] for f in fs)} | {sum(f['elapsed_seconds'] for f in fs):.6f} |")
lines += ['Common ceilings: 2,000 evaluations or linear solves and 1,000 iterative updates per fit; two hours for the complete comparison. Actual Newton refinement steps are separately recorded. Timing is descriptive single-run runtime, not a benchmark of solver speed.',
          '## Full held-out metric panel',
          'Every point estimate averages the pooled out-of-fold metric across the same five partitions. CE and Pearson intervals below include mixture refitting; the constant-baseline interval holds its original fold fits fixed. The paired contrast uses the same successful resamples for both methods. CE and forward Kullback–Leibler (KL) divergence use natural logs. Brier sums six categories; ranked probability score (RPS) averages five cumulative thresholds. Mean squared/absolute error compare predicted means to empirical human means, not to future individual ratings. Bias is judge minus human; positive means higher predicted ratings.',
          '| Metric | CE fitting [95% refit interval] | Pearson fitting [95% refit interval] | CE − Pearson [95% refit interval] | Constant [95% fixed interval] |',
          '| --- | --- | --- | --- | --- |']
for m in ['ce','kl','brier','rps','weighted_pmf','tvd','pearson','spearman','kendall','mse','mae','bias','human_mean','judge_mean','human_sd','judge_sd','entropy']:
 cells=[m]
 for loss in ['ce','pearson']:cells.append(fmt(r['matrix'][loss][m])+' '+ci(r['refit_intervals'][loss][m]))
 x=r['ce_minus_pearson_refit'][m];cells.append(fmt(x['estimate'])+' '+ci(x))
 cells.append(fmt(r['matrix']['constant'][m])+' '+ci(r['fixed_prediction_intervals']['constant'][m]))
 lines.append('| '+' | '.join(cells)+' |')
lines += ['Weighted PMF is the human-frequency-weighted squared probability-vector distance; PMF means probability mass function. TVD is total variation distance. SDs describe across-artifact expected-score variation, not uncertainty of the mean. Every table entry has p-val=n/a; no confirmatory test was specified. Human entropy is descriptive and cancels in the CE-versus-Pearson KL contrast.',
          '## Fold aggregation and uncertainty',
          '| Objective | Mean within-fold Pearson | Mean within-fold prediction SD | Mean pooled out-of-fold Pearson |',
          '| --- | ---: | ---: | ---: |']
for loss in ['ce','pearson']:
 fs=[f for f in r['fold_metrics'] if f['method']==loss]
 lines.append(f"| {loss} | {np.mean([f['pearson'] for f in fs]):.6f} | {np.mean([f['judge_sd'] for f in fs]):.6f} | {r['matrix'][loss]['pearson']:.6f} |")
lines += ['These are distinct descriptive summaries; use the pooled-metric intervals above for the prespecified aggregate. Per-partition and per-fold records are in `results/results.json`. The constant model varies across training folds, so pooled correlation need not be undefined even though each individual fold predicts a constant.',
          'Resampling unit: task family with all attached artifacts and multiplicities. Both mixture objectives are refitted within the fixed historical fold assignments for each draw. Empty test folds require no fit. The intervals include mixture fitting uncertainty, but not fresh human ratings, candidate-prompt selection or unseen solver/rater populations. The two excluded draws are retained and the reported intervals are conditional on passing numerical certificates; no replacement draws were used.',
          '| Failed draw | Partition index | Fold | Objective | Convex gap | Required bound |',
          '| ---: | ---: | ---: | --- | ---: | ---: |']
for f in b:
 if not f['success']:lines.append(f"| {f['draw']} | {f['seed_index']} | {f['fold']} | {f['objective']} | {f['simplex_frank_wolfe_gap']:.12g} | {f['certificate_tolerance']:.12g} |")
lines += ['Both failures were CE certificates just above the frozen tolerance despite optimizer success. They were not repaired or replaced after measurement. All Pearson bootstrap fits passed. The full compressed fit ledger and decompression/hash instructions are published alongside the results.',
          '## Existing repeated-score variability',
          '| Candidate | Cached ratings per artifact | Root mean square within-artifact sample SD (0–1) |',
          '| --- | ---: | ---: |']
for name,rr in r['repeat_variability'].items():lines.append(f"| {name} | {rr['repeats']} | {rr['raw_repeat_rms_sd']:.6f} |")
lines += ['These are historical selector aliases, not a claim about newly executed model identities. They describe raw-score repeat variation; repeated batches of aggregated forecasts and new-rater variability were not measured.',
          '## Verification, limits and continuation',
          f"Independent reconstruction checked {v['reconstructed_models']} primary models and {v['reconstructed_primary_forecasts']} forecasts; maximum probability discrepancy was {v['maximum_prediction_error']:.3g}. The independent implementation reproduced the full metric panel and paired interval endpoints. Bootstrap refits were not independently reexecuted. Five synthetic test groups passed, including 100 sparse-input certificate cases. The frozen protocol and sources are bound in `FREEZE.json`, committed before measurement at `0549cff`.",
          f"Measured environment: Python {r['python']}, NumPy {r['numpy']}, SciPy {r['scipy']}. Local execution followed Brando’s explicit fallback authorization; Rule 55 now records that standing remote-first/local-fallback preference. All three local Codex profiles have verified shared project/checkpoint pointers; native cross-account chat-history synchronization remains unverified. See `SESSION_ACCESS.md`.",
          'This is retrospective evidence from a small, previously exposed cache. Sparse human empirical histograms are not known population rating distributions. Different solver/rater populations, new human ratings, unrestricted language-model training, all correlation objectives, and universal CE superiority are outside its scope. Correlation’s failure to identify a full probability distribution is distinct from whether a particular finite-sample CE learner improves a particular held-out metric.',
          'The old softmax numerical failure remains unresolved as an optimization problem; the repaired scientific comparison avoids that confound by using a separately labeled family with numerical certificates. The paper remains unchanged. This experiment stays active because its measured outcome still needs a paper-integration decision.',
          'Reproduction: run the frozen scripts with the private hash-bound dataset, new output directories and recorded arguments; then run `scripts/verify.py`. For published fit receipts, run `gzip -dk results/bootstrap_fit_receipts.json.gz` before verification. Reports can be regenerated with `python build_report.py` using only committed aggregate evidence.',
          f"Copy-ready message: I reproduced the Pearson optimizer failure and reran cross-entropy versus Pearson within the same probability-mixture family. The log-loss difference was {delta['estimate']:.3f} [{delta['lo']:.3f}, {delta['hi']:.3f}] with family resampling and refitting, so the cache does not establish an advantage; both methods also lost to a simple marginal-distribution baseline on log loss.",
          '**TLDR-end:** [judge objectives: experiment 06] The historical failure was reproduced and a certified matched-family comparison completed. It does not establish CE superiority, and it exposes a substantial probability-modeling limitation in both mixture methods.']
report=re.sub(r'(?m)(^\|.*)\n\n(?=\|)',r'\1\n','\n\n'.join(lines))
(ROOT/'RESULTS_comparison.md').write_text(report+'\n')
print('\n'.join(lines[4:8]))
