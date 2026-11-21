"""Apply the archived KonIQ judge selections to a second, disjoint set of images."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from analyze import BUDGETS, METHODS, COMPARATORS, REPEATS, BOOT, bands, save
from candidates import R, TOL, best, bin_distribution, bin_indices, build, loss_by_grade, per_item
from design import draw, load


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify_training(data,q,names,source):
    """Re-select from training votes only via explicit centered Pearson and vote losses."""
    archived={(int(r["draw"]),int(r["budget"])):r for r in read(source/"selections.csv") if r["depth"]=="1"}
    assert len(archived)==len(BUDGETS)*REPEATS
    means=q[data.train]@R
    ce,rps=loss_by_grade(q[data.train])
    pos=np.full(len(q),-1)
    pos[data.train]=np.arange(len(data.train))
    for rep in range(REPEATS):
        order,obs=draw(data,rep,1)
        ids=pos[order[:BUDGETS[-1]]]
        assert (ids>=0).all() and len(set(ids))==len(ids)
        grades=obs[:len(ids)].argmax(axis=1)
        logsum=np.cumsum(ce[ids,grades],axis=0)
        rpsum=np.cumsum(rps[ids,grades],axis=0)
        for b in BUDGETS:
            m=means[ids[:b]]
            x=m-m.mean(axis=0)
            y=grades[:b]-grades[:b].mean()
            xv=(x*x).mean(axis=0)
            yv=np.mean(y*y)
            eligible=(xv>TOL)&(yv>TOL)
            rho=np.full(len(names),-np.inf)
            if eligible.any():
                rho[eligible]=(y@x[:,eligible])/(b*np.sqrt(yv*xv[eligible]))
                pearson=best(rho,maximize=True)
            else:
                pearson=0
            rec=archived[rep,b]
            assert [names[i] for i in (best(logsum[b-1]/b),pearson,best(rpsum[b-1]/b))]==[
                rec[k] for k in ("ce","pearson","rps")],(rep,b)
            assert int(rec["pearson_undefined"])==int(not eligible.any())
    return archived


def crossed_band(diffs):
    """Independent image and matched-acquisition weights, same seed and max-t as primary."""
    n,rep,*_=diffs.shape
    flat=np.ascontiguousarray(diffs.reshape(n,-1))
    point=diffs.mean(axis=(0,1))
    rng=np.random.default_rng(20261124)
    boots=[]
    for _ in range(BOOT//100):
        image_weights=rng.multinomial(n,np.full(n,1/n),size=100)/n
        rep_weights=rng.multinomial(rep,np.full(rep,1/rep),size=100)/rep
        by_rep=(image_weights@flat).reshape(100,rep,-1)
        boots.append(np.einsum("ird,ir->id",by_rep,rep_weights))
    bootstrap=np.concatenate(boots)
    se=bootstrap.std(axis=0,ddof=1)
    active=se>1e-12
    tmax=np.abs((bootstrap[:,active]-point.ravel()[active])/se[active]).max(axis=1)
    critical=np.quantile(tmax,.975,method="higher")
    half=critical*se.reshape(point.shape)
    return point,point-half,point+half


def thresholds(lower):
    def sustained(flags):
        return next((b for i,b in enumerate(BUDGETS) if flags[i:].all()),None)
    return dict(pearson_and_uniform=sustained((lower[:,:2]>0).all(axis=1)),
                all_four=sustained((lower>0).all(axis=1)))


def run(path,reference,out):
    data=load(path)
    with np.load(path,allow_pickle=False) as src:
        test=np.flatnonzero(src["split"]=="validation")
        assert np.array_equal(data.train,np.flatnonzero(src["split"]=="train"))
    sets=[set(data.id[i]) for i in (data.train,data.test,test)]
    assert list(map(len,sets))==[7058,2015,1000] and len(set.union(*sets))==10073
    assert np.array_equal(data.id,data.group)
    q,names=build(data,"features")
    assert len(names)==217
    archived=verify_training(data,q,names,reference)
    print("All 704 training winner triplets and undefined flags match before second-panel scoring.",flush=True)
    lookup={name:i for i,name in enumerate(names)}
    h=data.counts[test]/data.counts[test].sum(axis=1,keepdims=True)
    means=q@R
    cube=np.zeros((len(BUDGETS),len(METHODS),len(h),3))
    diffs=np.zeros((len(h),REPEATS,len(BUDGETS),len(COMPARATORS)))
    loss={i:per_item(q[test,i],h)[:,:3] for i in range(len(names))}
    partitions={}
    for rep in range(REPEATS):
        order,obs=draw(data,rep,1)
        for j,b in enumerate(BUDGETS):
            rec=archived[rep,b]
            ce,pearson,ranked=(lookup[rec[k]] for k in ("ce","pearson","rps"))
            if pearson not in partitions:
                partitions[pearson]=bin_indices(means[data.train,pearson],means[:,pearson])
            bins=partitions[pearson]
            prior=(obs[:b].sum(axis=0)+.5)/(b+2.5)
            posterior=bin_distribution(bins,order[:b],obs[:b],prior)
            outcomes=np.stack([loss[ce],loss[pearson],loss[len(names)-1],
                per_item(np.broadcast_to(prior,h.shape),h)[:,:3],per_item(posterior[bins[test]],h)[:,:3],
                loss[0],loss[ranked]])
            cube[j]+=outcomes/REPEATS
            diffs[:,rep,j]=(outcomes[1:5,:,0]-outcomes[:1,:,0]).T
    fixed,lo,hi,_,_,_=bands(cube,data.group[test],level=.975)
    crossed,clo,chi=crossed_band(diffs)
    np.testing.assert_allclose(fixed,crossed,atol=1e-12,rtol=0)
    individual=diffs.mean(axis=0)>1e-12
    scores=[]
    contrast=[]
    for j,b in enumerate(BUDGETS):
        for mi,method in enumerate(METHODS):
            val=cube[j,mi].mean(axis=0)
            scores.append(dict(budget=b,method=method,rps=val[0],ce_nats=val[1],tvd=val[2]))
        for c,method in enumerate(COMPARATORS):
            ws=individual[:,j]
            contrast.append(dict(budget=b,reference=method,subtract="ce",rps_gain=fixed[j,c],
                fixed_low=lo[j,c],fixed_high=hi[j,c],crossed_low=clo[j,c],crossed_high=chi[j,c],
                draws_win=int(ws[:,c].sum()),draws_beat_both=int(ws[:,:2].all(axis=1).sum()),
                draws_beat_all_four=int(ws.all(axis=1).sum()),total_draws=REPEATS))
    def original_low(filename,field,depth_filter):
        rows=[r for r in read(reference/filename) if not depth_filter or r["depth"]=="1"]
        by_key={(int(r["budget"]),r["comparator"]):float(r[field]) for r in rows}
        return np.asarray([[by_key[b,m] for m in COMPARATORS] for b in BUDGETS])
    original_fixed=original_low("contrasts.csv","simultaneous_lo",True)
    original_crossed=original_low("crossed_sensitivity.csv","crossed_lo",False)
    summary=dict(train_images=7058,original_test_images=2015,second_panel_images=len(test),
                 second_panel_votes=int(data.counts[test].sum()),group_overlap=0,
                 training_draws=REPEATS,training_winner_triplets_replayed=len(archived),
                 second_fixed=thresholds(lo),second_crossed=thresholds(clo),
                 original_fixed=thresholds(original_fixed),original_crossed=thresholds(original_crossed))
    out.mkdir(parents=True,exist_ok=True)
    save(out/"scores.csv",scores)
    save(out/"contrasts.csv",contrast)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))
    for row in contrast:
        if row["budget"] in (12,24) and row["reference"]=="calibrated_pearson":
            print(row)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz",type=Path)
    ap.add_argument("out",type=Path)
    ap.add_argument("--primary-results",type=Path,default=HERE.parent/"results"/"image_quality")
    args=ap.parse_args()
    run(args.npz,args.primary_results,args.out)
