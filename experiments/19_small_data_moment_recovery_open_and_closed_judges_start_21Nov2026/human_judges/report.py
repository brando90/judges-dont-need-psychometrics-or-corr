"""Generate the human-facing tables and figure from public aggregate results."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
URL="https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/19_small_data_moment_recovery_open_and_closed_judges_start_21Nov2026/human_judges/"
def read(name):
    with (HERE/name).open() as stream:return list(csv.DictReader(stream))
scores={(r['input'],int(r['n']),int(r['k']),r['method'],r['metric']):r for r in read('scores.csv')}
contrast={(r['input'],int(r['n']),int(r['k']),r['method'],r['subtract'],r['metric']):r for r in read('contrasts.csv')}
groups={'open_pair':'Open Qwen pair','closed_pair':'Closed GPT pair','qwen3_4b':'Qwen3-4B-Instruct-2507',
'qwen25_coder_14b':'Qwen2.5-Coder-14B-Instruct','gpt54_mini':'GPT-5.4 mini','gpt61_sol':'GPT-6.1 Sol'}
methods={'ce':'CE-fitted','pearson':'Pearson-fitted','identity':'Original forecast','constant':'Training constant','pearson_tie_rps':'Pearson with RPS tie-break'}
def score(g,n,k,a,m):
    s=scores[g,n,k,a,m]['mean']
    return f'{float(s):.4f}' if s else 'undefined'
def diff(g,n,k,a,b,m):
    r=contrast[g,n,k,a,b,m];return [float(r[key]) for key in ('difference','lower_95','upper_95')]
def ci(g,n,k,a,b,m):
    d,lo,hi=diff(g,n,k,a,b,m)
    return f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]'
def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'---|'*len(header)]+['| '+' | '.join(map(str,row))+' |' for row in rows])

pool=('open_pair','closed_pair')
budgets=[(n,k) for n in (3,7,15) for k in (1,2)]
metrics=('m1','m2','m3','m4','m5','variance')
metric_names=('First raw moment','Second raw moment','Third raw moment','Fourth raw moment','Fifth raw moment','Centered variance')

fig,axes=plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5.9))
for axis,key in zip(axes,pool):
    for i,(metric,label) in enumerate(zip(metrics,metric_names)):
        for offset,base,color,symbol in [(-.1,'constant','#176195','o'),(.1,'identity','#9A441B','s')]:
            d,lo,hi=diff(key,3,1,'ce',base,metric)
            axis.plot([lo,hi],[i+offset,i+offset],color=color,lw=2)
            axis.plot(d,i+offset,marker=symbol,color=color,label=('Training constant' if base=='constant' else 'Original forecast') if i==0 else None)
    axis.set_title(groups[key]);axis.set_xlabel("Held-out test squared moment loss\nCE-fitted minus reference")
    axis.axvline(0,lw=1,color='#555');axis.grid(alpha=.23);axis.set_axisbelow(True)
axes[0].set_yticks(range(len(metrics)),metric_names);axes[0].invert_yaxis();axes[0].legend(fontsize=8,title="Reference",loc='upper left')
fig.suptitle('What did calibration with one rating per artifact add?',fontsize=14)
fig.text(.5,.08,'Three training families, 12.1 ratings on average. Negative favors the cross-entropy (CE) fit. Raw ratings are divided by five.\n'
         'Raw-moment loss is on [0,1]; centered-variance loss uses noisy human sample variances. Bars: pointwise 95% Bayesian family intervals.\n'
         'Each fit minimizes training CE over the same 27 calibration heads. Original-forecast contrasts were added after initial inspection.',
         ha='center',fontsize=8)
fig.tight_layout(rect=(0,.17,1,.96));fig.savefig(HERE/'moment_skill.png',dpi=170);plt.close(fig)

parts=['# Which moments do four calibrated language-model judges predict? | 11-21-2026',
       f'**Doc link:** <{URL}RESULTS_11-21-2026.md>', 'Date: 11-21-2026',
       '**TLDR:** With roughly 12 training ratings, the cross-entropy (CE) fit on the two closed-weight GPT forecasts beat a training constant on all five raw-moment losses and centered variance. '
       'The untreated GPT forecasts performed better than the CE fit on every raw moment. The open-weight Qwen pair improved on its original distribution score, while its first- and second-moment comparisons with the constant were inconclusive. Neither pair established that the small training set taught the judge only the mean.',
       '## What the smallest training budget actually changed',
       'The 75 previously studied artifacts have 195 human ratings from 25 tasks in 19 families. Each of 250 matched splits holds out all artifacts of four families. '
       'The primary example fits on three families and one retained rating per training artifact (9–24 ratings, mean 12.108). '
       'It scores every available rating on 12–30 test artifacts per split (24–87 ratings). Neither raters nor models were held out. '
       'Each named pair averages the original six-category probabilities of its two backbones before calibration. '
       'Both objectives search the same 27 three-parameter calibration heads, without an optimization start or hyperparameter search. Fitting updates the output distribution while keeping the underlying language model fixed.',
       'Normalize an individual rating to z = rating/5. Raw-moment loss is the squared difference between the predicted moment and the realized power of z, range 0–1, lower is better. '
       'The first measures mean accuracy relative to a noisy future rating. The second and higher powers depend on level as well as shape. '
       'Centered-variance loss compares forecast variance with the unbiased within-artifact sample variance of the two or three test ratings. '
       'Unbiasedness for a population variance assumes independently sampled exchangeable raters. With two ratings the sample variance can reach 0.5 on this scale, so squared centered-variance loss ranges from 0 to 0.25. '
       'All differences subtract the same test metric for two procedures. Every bracket below is a pointwise 95% Bayesian bootstrap interval from 1,000 shared reweightings of 19 task families with refitting, including the constant. '
       'Positive Pearson-fitted minus CE-fitted favors CE. Negative fitted minus reference favors fitting. There is no practical margin or hypothesis test (p-val=n/a).',
       table(['Held-out test loss, 3 families × 1 rating','Open pair: CE-fitted minus training constant','Closed pair: CE-fitted minus training constant','Open pair: CE-fitted minus original','Closed pair: CE-fitted minus original'],
             [[label,ci('open_pair',3,1,'ce','constant',m),ci('closed_pair',3,1,'ce','constant',m),ci('open_pair',3,1,'ce','identity',m),ci('closed_pair',3,1,'ce','identity',m)] for m,label in zip(metrics,metric_names)]),
       'The identity and constant baselines were prespecified. The intervals comparing fitted with original probabilities are exploratory additions made to distinguish incremental learning from information already present in the base forecast. '
       'Improvement over the constant alone cannot establish what the few human labels taught the head. The empirical human rating law per artifact is too sparsely observed to certify recovery of its population probability vector.',
       '![Conditional moment losses for two reference forecasts](moment_skill.png)',
       '## Distribution scores across the training budgets',
       'The primary metric is normalized individual-rating ranked probability score (RPS), average squared cumulative error across five rating thresholds, range 0–1, lower is better. '
       'Each reported estimate averages over individual ratings within artifact, equally over held-out artifacts within split, and equally over the 250 overlapping splits. '
       'Correlation is computed across predicted and observed means within split, then averaged across defined splits.',
       table(['Input','Training families','Ratings per training artifact','Constant test RPS','Original test RPS','CE-fitted test RPS','Pearson-fitted test RPS','Pearson-fitted minus CE-fitted test RPS [95% interval]'],
             [[groups[g],n,k,*[score(g,n,k,a,'rps') for a in ('constant','identity','ce','pearson')],ci(g,n,k,'pearson','ce','rps')] for g in pool for n,k in budgets]),
       'At three training families × one rating, the CE fit improved on original open-pair test RPS by 0.0422, while the original closed pair had lower test RPS than its CE fit by 0.0166. The two corresponding fitted-minus-original intervals are '+ci('open_pair',3,1,'ce','identity','rps')+' and '+ci('closed_pair',3,1,'ce','identity','rps')+'.',
       '### Does Pearson fitting improve the mean or spread relative to CE?',
       'Both fits use the same training labels and head. At the smallest budget, neither pair resolves a first- or second-raw-moment loss advantage between fits. The centered-variance diagnostic favors the Pearson fit on both pairs. Because that diagnostic uses only two or three raters per artifact, the result concerns agreement with observed panel spread. It does not identify rater severity or population disagreement.',
       table(['Held-out loss, 3 families × 1 rating','Open pair: Pearson-fitted minus CE-fitted [95% interval]','Closed pair: Pearson-fitted minus CE-fitted [95% interval]'],[[label,ci('open_pair',3,1,'pearson','ce',m),ci('closed_pair',3,1,'pearson','ce',m)] for m,label in zip(metrics,metric_names)]),
       'On the original unweighted training fits, neither pooled input had a Pearson tie in any split or budget. Thus the training-RPS tie-break left their point estimates unchanged. Individual-backbone fit outcomes and all weighted contrasts are retained.',
       '## Individual backbones and the ranking diagnostics',
       'All four models use a human-mirror prompt and verbal probability outputs. Source paths, provider model identifiers, observed fallback counts and weight-access evidence are documented in the [model manifest](models.json) and [reading guide](HOW_TO_READ_RESULTS.md). '
       'The following compares each backbone at the smallest budget, then gives the absolute metric panel on both paired inputs. Total variation distance (TVD) is half the absolute probability difference summed against the empirical rating histogram, 0–1, lower is closer to that histogram. '
       'Test CE is negative log probability of individual ratings in nats, lower is better. Pearson measures linear association of artifact means and midrank Spearman measures rank association, both between −1 and 1, higher is better.',
       table(['Input, 3 families × 1 rating','Original test RPS','CE-fitted test RPS','Pearson-fitted test RPS','Pearson-fitted minus CE-fitted RPS [95% interval]','CE-fitted minus constant first-moment loss [95% interval]','CE-fitted minus constant second-moment loss [95% interval]'],
             [[groups[g],*[score(g,3,1,a,'rps') for a in ('identity','ce','pearson')],ci(g,3,1,'pearson','ce','rps'),ci(g,3,1,'ce','constant','m1'),ci(g,3,1,'ce','constant','m2')] for g in list(groups)[2:]]),
       table(['Input, 3 families × 1 rating','Method','Test CE (nats)','Test TVD','Test Pearson','Test Spearman','Mean judge rating','Mean human rating','Signed bias (judge minus human)','Mean absolute error (points)'],
             [[groups[g],methods[a],*[score(g,3,1,a,m) for m in ('ce','tvd','pearson','spearman','forecast_mean','human_mean','bias','mae')]] for g in pool for a in ('ce','pearson','identity','constant')]),
       '## Prespecified predictions and limits',
       '1. **CE improves the first moment relative to the constant for at least one pair at the smallest budget.** The closed GPT pair meets the interval criterion. The open Qwen pair does not.\n'
       '2. **At least one higher moment or centered variance remains unresolved in such a pair.** This failed for the closed GPT pair. All five raw-moment loss intervals and its centered-variance loss interval favor the CE-fitted forecast over the constant. '
       'This is skill versus a constant, not evidence that tiny samples recover all moments from scratch.\n'
       '3. **Train Pearson need not lead on held-out mean accuracy or correlation.** At three families × one rating, the open pair had higher held-out Pearson after CE fitting, 0.2257 versus 0.1852. The closed pair had higher held-out Pearson after Pearson fitting, 0.6411 versus 0.5932. These are descriptive correlations, without intervals. '
       'Within one class, different backbones and different baselines can support different conclusions. Pointwise intervals do not correct for the many budgets and metrics.',
       'Weight access, company, model size, prompting implementation, metadata, and original forecast quality are confounded. These four handpicked backbones cannot identify an effect of opening model weights. '
       'The human labels have prior exposure, raters may overlap across splits and final family weighting cannot create new raters. '
       'See the [dated protocol](PROTOCOL_11-21-2026.md), [absolute scores and valid counts](scores.csv), [paired contrasts](contrasts.csv), [budget and fallback counts](accounting.csv), and [fit parameter selection counts](choices.csv).']
(HERE/'RESULTS_11-21-2026.md').write_text('\n\n'.join(parts)+'\n')
