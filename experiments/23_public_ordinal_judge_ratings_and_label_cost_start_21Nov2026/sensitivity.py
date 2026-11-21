"""Explore finite model-vote smoothing and uncertainty across both crossed sampling units."""
import argparse
import csv
from pathlib import Path

import numpy as np

from analyze import BOOT, COMPARATORS, REPEATS, run, save
from candidates import R, bin_distribution, bin_indices, build
from design import draw, load


def smooth(path, out):
    data = load(path)
    with np.load(path) as f:
        totals = f["native_valid_votes"].reshape(-1,1)
    model_counts = np.rint(data.native*totals)
    assert np.array_equal(model_counts.sum(axis=1,keepdims=True),totals)
    zero_model = model_counts[data.test] == 0
    human = data.counts[data.test]
    misses = (zero_model*human).sum(axis=1)
    sources = [dict(test_items=len(human),human_ratings=int(human.sum()),
                    human_grade_missing_from_model_votes=int(misses.sum()),
                    affected_items=int((misses>0).sum()),
                    mean_item_missing_vote_rate=float((misses/human.sum(axis=1)).mean()))]
    data.native = (model_counts+.5)/(totals+2.5)
    run(data,"native",out)
    save(out/"previously_unseen_grades.csv",sources)


def rps(q, h):
    f=q.cumsum(axis=-1)[:,:-1]
    g=h.cumsum(axis=-1)[:,:-1]
    return np.mean(f*f + (1-2*f)*g,axis=-1)


def crossed(path, mode, results):
    data=load(path)
    q,names=build(data,mode)
    ids={v:i for i,v in enumerate(names)}
    with (results/"selections.csv").open() as f:
        fits={(int(r["draw"]),int(r["budget"])):r for r in csv.DictReader(f) if r["depth"]=="1"}
    budgets=sorted({b for _,b in fits})
    _, group, size=np.unique(data.group[data.test],return_inverse=True,return_counts=True)
    truth=data.counts[data.test]/data.counts[data.test].sum(axis=1,keepdims=True)
    sums=np.zeros((len(size),REPEATS,len(budgets)*len(COMPARATORS)))
    loss={i:rps(q[data.test,i],truth) for i in range(len(names))}
    bins_cache={}
    for rep in range(REPEATS):
        order,picked=draw(data,rep,1)
        for j,b in enumerate(budgets):
            fit=fits[rep,b]
            ce=ids[fit["ce"]]
            pearson=ids[fit["pearson"]]
            if pearson not in bins_cache:
                m=q[:,pearson]@R
                bins_cache[pearson]=bin_indices(m[data.train],m)
            bins=bins_cache[pearson]
            prior=(picked[:b].sum(axis=0)+.5)/(b+2.5)
            bins_q=bin_distribution(bins,order[:b],picked[:b],prior)
            comparators=[loss[pearson],loss[len(names)-1],rps(np.broadcast_to(prior,truth.shape),truth),
                         rps(bins_q[bins[data.test]],truth)]
            for ci,v in enumerate(comparators):
                sums[:,rep,4*j+ci]=np.bincount(group,weights=v-loss[ce],minlength=len(size))/size
    point=sums.mean(axis=(0,1))
    rng=np.random.default_rng(20261124)
    boots=np.empty((BOOT,len(point)))
    flat=np.ascontiguousarray(sums.reshape(len(size),-1))
    for i in range(0,BOOT,100):
        gw=rng.multinomial(len(size),np.full(len(size),1/len(size)),size=100)/len(size)
        dw=rng.multinomial(REPEATS,np.full(REPEATS,1/REPEATS),size=100)/REPEATS
        by_draw=(gw@flat).reshape(100,REPEATS,-1)
        boots[i:i+100]=np.einsum("ird,ir->id",by_draw,dw)
    se=boots.std(axis=0,ddof=1)
    active=se>1e-12
    tmax=np.max(abs(boots[:,active]-point[active])/se[active],axis=1)
    critical=np.quantile(tmax,.975,method="higher")
    out=[]
    draw_means=sums.mean(axis=0).reshape(REPEATS,len(budgets),4)
    for j,b in enumerate(budgets):
        wins=draw_means[:,j]>1e-12
        for ci,c in enumerate(COMPARATORS):
            v=4*j+ci
            out.append(dict(budget=b,comparator=c,rps_gain=point[v],crossed_lo=point[v]-critical*se[v],
                            crossed_hi=point[v]+critical*se[v],draws_beat_comparator=int(wins[:,ci].sum()),
                            draws_beat_pearson_and_uniform=int(wins[:,:2].all(axis=1).sum()),
                            draws_beat_all_four=int(wins.all(axis=1).sum()),draw_count=REPEATS))
    with (results/"contrasts.csv").open() as f:
        registered=[float(r["rps_gain"]) for r in csv.DictReader(f) if r["depth"]=="1"]
    np.testing.assert_allclose(point,registered,atol=1e-12,rtol=0)
    save(results/"crossed_sensitivity.csv",out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check",choices=["smooth","crossed"])
    parser.add_argument("npz",type=Path)
    parser.add_argument("mode",choices=["native","features"])
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    if args.check=="crossed":
        crossed(args.npz,args.mode,args.out)
    else:
        smooth(args.npz,args.out)
