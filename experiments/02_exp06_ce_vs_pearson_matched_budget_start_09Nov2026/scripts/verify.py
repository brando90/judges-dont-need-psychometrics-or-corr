"""Independently reconstruct primary predictions and the complete aggregate panel."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'historical'))
from run_cached import dump


def independent_metrics(p,q):
    a=np.round(p@np.linspace(0,1,6),12);b=np.round(q@np.linspace(0,1,6),12)
    fp=np.cumsum(p,axis=1)[:,:5];fq=np.cumsum(q,axis=1)[:,:5]
    ce=-np.sum(p*np.log(q),axis=1)
    ent=-np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0),axis=1)
    return {'ce':ce.mean(),'entropy':ent.mean(),'kl':(ce-ent).mean(),
            'brier':np.mean(1+np.sum(q*q-2*p*q,axis=1)),
            'rps':np.mean(fq*fq-2*fp*fq+fp),'weighted_pmf':np.mean(np.sum(p*(p-q)**2,axis=1)),
            'tvd':np.mean(np.sum(abs(p-q),axis=1)/2),'mae':np.mean(abs(b-a)),
            'bias':np.mean(b-a),'human_mean':a.mean(),'judge_mean':b.mean(),
            'mse':np.mean((b-a)**2),'judge_sd':b.std(),'human_sd':a.std(),
            'pearson':pearsonr(a,b).statistic,'spearman':spearmanr(a,b).statistic,
            'kendall':kendalltau(a,b).statistic}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--private-out',required=True)
    args=ap.parse_args();root=HERE.parent;out=Path(args.out);private=Path(args.private_out)
    cfg=json.loads((root/'protocol.json').read_text());raw=Path(args.data).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==cfg['dataset_sha256']
    data=json.loads(raw);rows=data['rows'];n=len(rows)
    hist=lambda r:np.bincount(r,minlength=6)/len(r)
    p=np.array([hist(r['ratings']) for r in rows])
    eps=cfg['epsilon']
    q=np.stack([np.array([hist(r) for r in data['outputs'][k]])*(1-6*eps)+eps for k in cfg['candidate_order']],axis=1)
    saved=json.loads((private/'forecasts.json').read_text());result=json.loads((out/'results.json').read_text())
    families=sorted({r['family'] for r in rows});seen=set();max_error=0.
    rebuilt={k:[np.full((n,6),np.nan) for _ in cfg['seeds']] for k in ['ce','pearson','constant']}
    for model in saved['models']:
        si,fold,method=model['seed_index'],model['fold'],model['method']
        assert (si,fold,method) not in seen;seen.add((si,fold,method))
        order=np.random.default_rng(cfg['seeds'][si]).permutation(families)
        testfamilies=set(order[fold::5])
        test=np.array([r['family'] in testfamilies for r in rows]);train=~test
        if method=='constant':
            pred=np.tile(p[train].mean(0)*(1-6*eps)+eps,(test.sum(),1))
            np.testing.assert_allclose(pred[0],model['pmf'],atol=1e-12,rtol=0)
        else:
            w=np.array(model['weights']);assert w.min()>=-1e-10 and abs(w.sum()-1)<1e-10
            pred=(q[test]*w[None,:,None]).sum(axis=1)
            if method=='ce':
                qt=(q[train]*w[None,:,None]).sum(axis=1)
                g=-np.mean(np.sum(p[train,None,:]*q[train]/qt[:,None,:],axis=2),axis=0)
                gap=g@w-g.min()
                receipt=next(r for r in result['fits'] if r['seed_index']==si and r['fold']==fold and r['objective']==method)
                assert abs(gap-receipt['simplex_frank_wolfe_gap'])<1e-10
        rebuilt[method][si][test]=pred
    assert len(seen)==75
    for method,forecasts in rebuilt.items():
        for si,forecast in enumerate(forecasts):
            assert np.isfinite(forecast).all() and forecast.min()>0
            np.testing.assert_allclose(forecast.sum(1),1,atol=1e-12,rtol=0)
            error=np.max(abs(forecast-np.array(saved['predictions'][method][si])))
            max_error=max(max_error,float(error));assert error<1e-12
        mm=[independent_metrics(p,f) for f in forecasts]
        for metric in result['matrix'][method]:
            value=np.mean([m[metric] for m in mm]);reported=result['matrix'][method][metric]
            np.testing.assert_allclose(value,reported,atol=1e-10,rtol=0)
    boot=json.loads((private/'bootstrap.json').read_text())
    for label,key in [('fixed_delta','ce_minus_pearson_fixed'),('refit_delta','ce_minus_pearson_refit')]:
        for metric,values in boot[label].items():
            valid=np.array([v for v in values if v is not None])
            ci=result[key][metric];assert len(values)==cfg['bootstrap_draws']
            assert len(valid)==ci['valid_draws']
            if len(valid):np.testing.assert_allclose(np.quantile(valid,[.025,.975]),[ci['lo'],ci['hi']],atol=1e-12,rtol=0)
    fits=json.loads((out/'bootstrap_fit_receipts.json').read_text())
    ledger=json.loads((out/'bootstrap_ledger.json').read_text())
    assert len(fits)==result['counts']['bootstrap_fits']
    assert sum(not f['success'] for f in fits)==result['counts']['failed_bootstrap_fits']
    assert sum(not d['valid'] for d in ledger)==result['counts']['invalid_bootstrap_draws']
    assert all(r['function_evaluations']<=2000 for r in fits+result['fits'])
    verification={'status':'PASS','reconstructed_models':len(seen),'reconstructed_primary_forecasts':75*5*3,
                  'maximum_prediction_error':max_error,'independent_full_metric_panel':True,
                  'bootstrap_intervals_recomputed':True,'private_refits_independently_reexecuted':False,
                  'results_sha256':hashlib.sha256((out/'results.json').read_bytes()).hexdigest()}
    dump(out/'verification.json',verification);print(json.dumps(verification))


if __name__=='__main__':main()
