"""Independent endpoint replay from movie-level event counts and an energy-score identity."""
import argparse
from collections import Counter
import csv
import os
from pathlib import Path
for key in ("OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np
from source import load_archive
from candidates import build

METHODS=("direct_ce","direct_pearson","uniform","train_histogram","calibrated_pearson",
         "calibrated_raw_ce","calibrated_loo_ce","fixed_reference","calibrated_reference")


def rows(file):
    with file.open() as stream:
        return list(csv.DictReader(stream))


def run(archive,root,panel):
    d,coverage=load_archive(archive,panel)
    forecast,names=build(d,"features")
    stars=np.arange(1,6)
    mu=(forecast*stars).sum(axis=-1)
    cuts=[np.unique(np.quantile(col,(.25,.50,.75))) for col in mu[d.train].T]
    bins=np.stack([np.digitize(mu[:,i],cut) for i,cut in enumerate(cuts)],axis=1)
    hist=d.counts[d.test]/d.counts[d.test].sum(axis=1,keepdims=True)
    distance=abs(stars[:,None]-stars[None,:])/4
    def evaluate(p):
        # Energy-score representation of normalized individual-ranked probability score.
        distances=p@distance
        return ((hist*distances).sum(axis=1).mean()-.5*(p*distances).sum(axis=1).mean(),
                -(hist*np.log(p)).sum(axis=1).mean(),1-np.minimum(p,hist).sum(axis=1).mean())
    saved=rows(root/panel/"scores.csv")
    headcounts=rows(root/panel/"head_counts.csv")
    for b in (6,768):
        values=np.zeros((9,3))
        selections=Counter()
        for rep in range(64):
            rng=np.random.default_rng(np.random.SeedSequence([20261121,23,rep]))
            order=rng.permutation(d.train)
            source_counts=d.counts[order]
            tickets=np.random.default_rng(np.random.SeedSequence([20261121,23,rep,1])).integers(
                source_counts.sum(axis=1))
            obs=np.array([np.searchsorted(counts.cumsum(),t,side="right")
                          for counts,t in zip(source_counts[:b],tickets[:b])])
            idx=order[:b]
            logs=-np.log(forecast[idx,:,obs]).mean(axis=0)
            ce=np.flatnonzero(logs<=min(logs)+1e-10)[0]
            x=mu[idx]-mu[idx].mean(axis=0)
            y=obs-obs.mean()
            valid=x.var(axis=0)>1e-10
            rr=np.full(len(names),-np.inf)
            if y.var()>1e-10 and valid.any():
                rr[valid]=(y@x[:,valid])/(np.linalg.norm(y)*np.linalg.norm(x[:,valid],axis=0))
                pear=np.flatnonzero(rr>=np.max(rr)-1e-10)[0]
            else:
                pear=0
            counts=np.bincount(obs,minlength=5)
            prior=(counts+.5)/(b+2.5)
            # Independent summation over actual held-out votes; consistent with full fold deletion.
            tables=np.zeros((len(names),4,5),int)
            head_index=np.arange(len(names))
            for i,g in zip(idx,obs):
                tables[head_index,bins[i],g]+=1
            subtotal=tables.sum(axis=2)
            per_obs=np.zeros(len(names))
            for outcome in range(5):
                cnt=tables[:,:,outcome]
                active=cnt>0
                own_prior=(counts[outcome]-.5)/(b+1.5)
                prob=np.divide(cnt-1+8*own_prior,subtotal+7,
                               out=np.ones_like(cnt,dtype=float),where=active)
                per_obs-=np.sum(cnt*np.log(prob),axis=1)/b
            loo=np.flatnonzero(per_obs<=min(per_obs)+1e-10)[0]
            def calibrated(head):
                qs=(tables[head]+8*prior)/(subtotal[head,:,None]+8)
                return qs[bins[d.test,head]]
            for method,head in (("direct_ce",ce),("direct_pearson",pear),("calibrated_loo_ce",loo)):
                selections[method,names[head]]+=1
            pp=(forecast[d.test,ce],forecast[d.test,pear],forecast[d.test,-1],
                np.broadcast_to(prior,hist.shape),calibrated(pear),calibrated(ce),calibrated(loo),
                forecast[d.test,0],calibrated(0))
            for mi,q in enumerate(pp):
                values[mi]+=np.array(evaluate(q))/64
        expected=Counter({(r["method"],r["head"]):int(r["draws"]) for r in headcounts if int(r["budget"])==b})
        assert selections==expected,(panel,b,selections-expected,expected-selections)
        for i,method in enumerate(METHODS):
            row=next(r for r in saved if int(r["budget"])==b and r["method"]==method)
            np.testing.assert_allclose(values[i],[float(row[k]) for k in ("rps","ce_nats","tvd")],
                                       atol=2e-12,rtol=0)
        print(panel,b,"64 matched draws, 3 selector counts and 27 endpoint scores reproduced")
    if panel=="disjoint_users":
        assert coverage["shared_users"]==0
    # Direct per-vote metric checks against the energy score; each movie still has equal weight.
    p=forecast[d.test,0]
    explicit=[]
    for counts,row in zip(d.counts[d.test],p):
        votes=np.repeat(stars,counts)
        events=votes[:,None] <= stars[:4]
        explicit.append([np.square(row.cumsum()[:4]-events).mean(), -np.log(row[votes-1]).mean(),
                         .5*np.abs(np.bincount(votes,minlength=6)[1:]/len(votes)-row).sum()])
    np.testing.assert_allclose(np.mean(explicit,axis=0),evaluate(p),atol=2e-12,rtol=0)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive",type=Path)
    ap.add_argument("results",type=Path)
    a=ap.parse_args()
    for panel_name in ("all_users","disjoint_users"):
        run(a.archive,a.results,panel_name)
