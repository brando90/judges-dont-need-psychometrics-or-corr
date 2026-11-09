"""Execute a frozen CE-versus-Pearson simplex comparison and paired family refits."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'historical'))
from metrics import histogram, smooth, per_item, summarize
from run_cached import dump, partitions
from ce_mixture import fit, predict_mixture


def metrics(p,q,index):
    values = per_item(p,q)
    values['mse'] = (values['judge_mean']-values['human_mean'])**2
    result = summarize(values,index)
    result.update(judge_sd=float(values['judge_mean'][index].std()),
                  human_sd=float(values['human_mean'][index].std()))
    return result


def interval(values):
    values=np.asarray(values,dtype=float)
    valid=values[np.isfinite(values)]
    return {'lo':float(np.quantile(valid,.025)) if len(valid) else None,
            'hi':float(np.quantile(valid,.975)) if len(valid) else None,
            'valid_draws':len(valid),'invalid_draws':len(values)-len(valid),'p_value':None}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--private-out',required=True)
    args=ap.parse_args()
    root=HERE.parent
    cfg=json.loads((root/'protocol.json').read_text())
    freeze=json.loads((root/'FREEZE.json').read_text())
    required={'PROTOCOL.md','protocol.json'} | {str(p.relative_to(root)) for p in (root/'scripts').rglob('*.py')}
    assert required <= set(freeze['files']), 'Freeze must bind all sources and protocol files.'
    for name,digest in freeze['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    assert freeze['diagnostic_reviewed'], 'Review reproduction before freezing the new condition.'
    out,private=Path(args.out),Path(args.private_out)
    assert not (out/'results.json').exists() and not (private/'forecasts.json').exists()
    start=time.monotonic()
    raw=Path(args.data).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==cfg['dataset_sha256']
    data=json.loads(raw);rows=data['rows'];n=len(rows)
    p=np.array([histogram(r['ratings']) for r in rows])
    q=np.stack([smooth(np.array([histogram(v) for v in data['outputs'][name]]),cfg['epsilon'])
                for name in cfg['candidate_order']],axis=1)
    parts=partitions(rows,cfg['seeds'])
    families=np.array([r['family'] for r in rows]);fam=sorted(set(families))
    methods=['ce','pearson','constant']
    predictions={k:[] for k in methods};fits=[];models=[];fold_metrics=[]

    def progress(**kw):
        dump(out/'progress.json',dict(elapsed_seconds=time.monotonic()-start,**kw))
        print(json.dumps(kw),flush=True)
        if time.monotonic()-start > cfg['max_wall_seconds']:
            raise TimeoutError('Frozen total wall budget exhausted; retain partial files.')

    for si,(outer,_) in enumerate(parts):
        forecasts={k:np.full_like(p,np.nan) for k in methods}
        for fold in range(5):
            train=np.flatnonzero(outer!=fold);test=np.flatnonzero(outer==fold)
            assert not set(families[train]) & set(families[test])
            for loss in ['ce','pearson']:
                w,rec=fit(q[train],p[train],loss)
                fits.append(dict(seed_index=si,fold=fold,**rec))
                forecasts[loss][test]=predict_mixture(q[test],w)
                models.append({'seed_index':si,'fold':fold,'method':loss,'weights':w})
            const=smooth(p[train].mean(0),cfg['epsilon'])
            forecasts['constant'][test]=const
            models.append({'seed_index':si,'fold':fold,'method':'constant','pmf':const})
            for method in methods:
                fold_metrics.append({'seed_index':si,'fold':fold,'method':method,
                                     'artifacts':len(test),**metrics(p[test],forecasts[method][test],np.arange(len(test)))})
        for method in methods:predictions[method].append(forecasts[method])
        dump(private/'partial_forecasts.json',{'predictions':predictions,'models':models})
        dump(out/'fit_receipts.json',fits)
        progress(phase='fitting',completed_partitions=si+1)
    dump(private/'forecasts.json',{'predictions':predictions,'models':models})
    assert len(fits)==50 and len(models)==75

    def aggregate(index,forecasts=predictions,target=p):
        ans={}
        for method,qs in forecasts.items():
            mm=[metrics(target,qq,index) for qq in qs]
            ans[method]={k:float(np.mean([m[k] for m in mm])) for k in mm[0]}
        return ans

    matrix=aggregate(np.arange(n))
    fixed={k:{m:[] for m in matrix[k]} for k in methods}
    fixed_delta={m:[] for m in matrix['ce']}
    refit_delta={m:[] for m in matrix['ce']}
    refit_methods={k:{m:[] for m in matrix[k]} for k in ['ce','pearson']}
    rng=np.random.default_rng(cfg['bootstrap_seed']);ledger=[];bootstrap_fits=[]
    for b in range(cfg['bootstrap_draws']):
        draw=rng.choice(fam,size=len(fam),replace=True)
        ix=np.concatenate([np.flatnonzero(families==g) for g in draw])
        a=aggregate(ix)
        for method in methods:
            for metric,value in a[method].items():fixed[method][metric].append(value)
        for metric in fixed_delta:fixed_delta[metric].append(a['ce'][metric]-a['pearson'][metric])
        qs={k:[] for k in ['ce','pearson']};valid=True;reasons=[]
        for si,(outer,_) in enumerate(parts):
            forecasts={k:np.full((len(ix),6),np.nan) for k in qs}
            for fold in range(5):
                train=ix[outer[ix]!=fold];loc=np.flatnonzero(outer[ix]==fold)
                if not len(loc):continue
                if len(train)<2:
                    valid=False;reasons.append('insufficient_training_rows');continue
                for loss in qs:
                    w,rec=fit(q[train],p[train],loss)
                    bootstrap_fits.append({'draw':b,'seed_index':si,'fold':fold,**rec})
                    if not rec['success']:
                        valid=False;reasons.append(loss+'_certificate_failed')
                    forecasts[loss][loc]=predict_mixture(q[ix[loc]],w)
            for loss in qs:qs[loss].append(forecasts[loss])
        if valid:
            a=aggregate(np.arange(len(ix)),qs,p[ix])
        for metric in refit_delta:
            refit_delta[metric].append(a['ce'][metric]-a['pearson'][metric] if valid else np.nan)
            for loss in qs:refit_methods[loss][metric].append(a[loss][metric] if valid else np.nan)
        ledger.append({'draw':b,'valid':valid,'unique_families':len(set(draw)),
                       'reasons':sorted(set(reasons))})
        if (b+1)%25==0:
            dump(out/'bootstrap_ledger.json',ledger)
            dump(private/'bootstrap_partial.json',{'fixed_delta':fixed_delta,'refit_delta':refit_delta})
            progress(phase='family_bootstrap_with_refitting',completed=b+1,total=cfg['bootstrap_draws'])
    dump(private/'bootstrap.json',{'fixed':fixed,'fixed_delta':fixed_delta,'refit_delta':refit_delta,
                                  'refit_methods':refit_methods})
    dump(out/'bootstrap_fit_receipts.json',bootstrap_fits)
    result={'matrix':matrix,'fold_metrics':fold_metrics,'fits':fits,
            'per_seed':{method:[metrics(p,qq,np.arange(n)) for qq in qs] for method,qs in predictions.items()},
            'fixed_prediction_intervals':{k:{m:interval(v) for m,v in mm.items()} for k,mm in fixed.items()},
            'refit_intervals':{k:{m:interval(v) for m,v in mm.items()} for k,mm in refit_methods.items()},
            'ce_minus_pearson_fixed':{m:{'estimate':matrix['ce'][m]-matrix['pearson'][m],**interval(v)} for m,v in fixed_delta.items()},
            'ce_minus_pearson_refit':{m:{'estimate':matrix['ce'][m]-matrix['pearson'][m],**interval(v)} for m,v in refit_delta.items()},
            'counts':{'artifacts':n,'families':len(fam),'tasks':len({r['task'] for r in rows}),
                      'expert_ratings':sum(len(r['ratings']) for r in rows),'primary_fits':len(fits),
                      'failed_primary_fits':sum(not f['success'] for f in fits),
                      'bootstrap_draws':len(ledger),'invalid_bootstrap_draws':sum(not d['valid'] for d in ledger),
                      'bootstrap_fits':len(bootstrap_fits),'failed_bootstrap_fits':sum(not f['success'] for f in bootstrap_fits)},
            'repeat_variability':{name:{'repeats':3,'raw_repeat_rms_sd':float(np.sqrt(np.mean(np.var(np.array(data['outputs'][name])/5,axis=1,ddof=1))))} for name in cfg['candidate_order']},
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'dataset_sha256':cfg['dataset_sha256'],'freeze_sha256':hashlib.sha256((root/'FREEZE.json').read_bytes()).hexdigest(),
            'elapsed_seconds':time.monotonic()-start,'new_inference_calls':0,'provider_cost_usd':0}
    dump(out/'results.json',result)
    progress(phase='complete',counts=result['counts'])


if __name__=='__main__':main()
