#!/usr/bin/env python3
"""Reconstruct stored comparisons independently, without new rating samples."""
import csv
import hashlib
import json
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='8'
import numpy as np
from scipy.stats import binom, t as student_t
import run_study as s

root=Path(__file__).parent
out=root/'results'
s.check_freeze()
complete=json.loads((out/'COMPLETE.json').read_text())
for name,digest in complete['artifact_hashes'].items():
    assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
candidates=list(csv.DictReader((out/'candidate_rows.csv').open()))
selections=list(csv.DictReader((out/'selection_rows.csv').open()))
assert len(candidates)==6912 and len(selections)==5376
assert len({(r['replication'],r['ratings_per_item'],r['candidate']) for r in candidates})==len(candidates)
assert len({(r['replication'],r['ratings_per_item'],r['objective']) for r in selections})==len(selections)
config=json.loads((root/'protocol.json').read_text())
lookup={(r['replication'],r['ratings_per_item'],r['candidate']):r for r in candidates}
checks=0
for row in selections:
    pool=[lookup[row['replication'],row['ratings_per_item'],n] for n in config['candidates']]
    key='validation_'+row['objective']
    sign=-1 if row['objective'] in ['pearson','spearman','kendall'] else 1
    values=[sign*float(r[key]) if r[key] else np.inf for r in pool]
    chosen=[config['candidates'][i] for i,v in enumerate(values) if abs(v-min(values))<=1e-12]
    assert row['selected_candidates'].split('|')==chosen
    for column,stored in row.items():
        if not column.startswith('test_'):
            continue
        original,stat=column.rsplit('_tie_',1)
        vals=[lookup[row['replication'],row['ratings_per_item'],name][original] for name in chosen]
        if any(v=='' for v in vals):
            assert stored==''
        else:
            numeric=np.array(vals,dtype=float)
            expected={'mean':np.mean,'min':np.min,'max':np.max}[stat](numeric)
            assert abs(float(stored)-expected)<1e-12,(column,stored,expected)
        checks+=1
summary=json.loads((out/'summary.json').read_text())
for m,criteria in summary.items():
    for objective,data in criteria.items():
        rows=[r for r in selections if r['ratings_per_item']==m and r['objective']==objective]
        assert len(rows)==256 and data['failed']==0
        for metric,stats in data['metrics'].items():
            values=np.array([float(r[metric]) for r in rows if r[metric]!=''])
            assert stats['n_defined']==len(values)
            if len(values):
                mean=float(np.mean(values))
                sd=float(np.std(values,ddof=1))
                delta=student_t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values))
                assert abs(stats['mean']-mean)<1e-12
                np.testing.assert_allclose(stats['mc_ci95'],[mean-delta,mean+delta],atol=1e-12)
                np.testing.assert_allclose(stats['repeat_q95'],np.quantile(values,[.025,.975]),atol=1e-12)
manifest=json.loads((root/'input_manifest.json').read_text())
t=np.array([r['t'] for r in manifest['items'] if r['split']=='test'])
p=binom.pmf(np.arange(6)[None,:],5,t[:,None])
for row in csv.DictReader((out/'population_metrics.csv').open()):
    name=row['candidate']
    if name.startswith('mixture_'):
        lam=float(name.removeprefix('mixture_'))
        endpoint=np.zeros_like(p)
        endpoint[:,0],endpoint[:,5]=1-t,t
        q=(1-lam)*p+lam*endpoint
    elif name=='permuted':q=binom.pmf(np.arange(6)[None,:],5,(1-t)[:,None])
    elif name=='constant_binomial':q=np.tile(binom.pmf(np.arange(6),5,.5),(len(t),1))
    else:q=np.full_like(p,1/6)
    # Direct sum over every possible future rating, independent of the scorer's expansion.
    losses={k:[] for k in ('ce','brier','rps','nmse','mae')}
    for i in range(len(t)):
        mu=sum(q[i,r]*r for r in range(6))
        for r in range(6):
            weight=p[i,r]/len(t)
            losses['ce'].append(weight*(-np.log(q[i,r])))
            losses['brier'].append(weight*sum((q[i,k]-(k==r))**2 for k in range(6)))
            losses['rps'].append(weight*sum((sum(q[i,:k+1])-(r<=k))**2 for k in range(5))/5)
            losses['nmse'].append(weight*((mu-r)/5)**2)
            losses['mae'].append(weight*abs(mu-r)/5)
    for metric,values in losses.items():
        assert abs(float(row[metric])-sum(values))<1e-12,(name,metric)
receipt={'status':'verified','checks':['source/input hashes','all result hashes','unique/full denominators',
         'all selection/tie memberships','all selected common-metric values','all Monte Carlo intervals and percentiles',
         'independent scipy binomial probabilities and direct six-outcome loss enumeration'],
         'selection_rows':len(selections),'candidate_rows':len(candidates),'selected_metric_checks':checks,
         'new_rating_samples':0,'new_provider_calls':0,'script_sha256':s.sha(__file__)}
s.write_json(out/'verification.json',receipt)
print(json.dumps(receipt,indent=2))
