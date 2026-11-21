"""Matched judge selection on historical MovieLens ratings. Write aggregate statistics only."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

for setting in ("OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[setting] = "1"
import numpy as np
from source import load_archive
from analyze import bands, save
from candidates import R, best, bin_distribution, bin_indices, build, pearson_from_totals, per_item
from design import draw
from matched_calibration.run import add_hist, loo_loss, bootstrap

BUDGETS=(6,12,24,48,96,192,384,768)
METHODS=("direct_ce","direct_pearson","uniform","train_histogram","calibrated_pearson",
         "calibrated_raw_ce","calibrated_loo_ce","fixed_reference","calibrated_reference")
METRICS=("rps","ce_nats","tvd")
PANELS=("all_users","disjoint_users")


def run(path, panel, out):
    data,info=load_archive(path,panel)
    q,names=build(data,"features")
    means=q@R
    assert len(names)==529
    log_table=-np.log(q)
    bins=np.stack([bin_indices(means[data.train,j],means[:,j]) for j in range(len(names))],
                  axis=1).astype(np.uint8)
    h=data.counts[data.test]/data.counts[data.test].sum(axis=1,keepdims=True)
    cube=np.zeros((len(BUDGETS),len(METHODS),len(h),len(METRICS)))
    draw_rps=np.zeros((64,len(BUDGETS),len(METHODS)))
    heads=Counter()
    n_undefined=Counter()
    for rep in range(64):
        order, labels=draw(data,rep,1)
        assert len(np.unique(order))==len(data.train) and set(order)==set(data.train)
        votes=labels.argmax(axis=1)
        frequency=np.zeros((len(names),4,5),dtype=int)
        nll,m1,m2,my=np.zeros((4,len(names)))
        y1,y2=0.,0.
        last=0
        for j,b in enumerate(BUDGETS):
            ix=order[last:b]
            grade=votes[last:b]
            y=R[grade]
            values=means[ix]
            nll+=log_table[ix,:,grade].sum(axis=0)
            m1+=values.sum(axis=0)
            m2+=np.einsum("ij,ij->j",values,values)
            my+=np.einsum("i,ij->j",y,values)
            y1+=y.sum()
            y2+=y@y
            ce=best(nll/b)
            pearson,undefined=pearson_from_totals(m1,m2,my,y1,y2,b)
            n_undefined[b]+=int(undefined)
            frequency=add_hist(bins[ix],grade,frequency)
            loo=best(loo_loss(frequency))
            for method,head in (("direct_ce",ce),("direct_pearson",pearson),("calibrated_loo_ce",loo)):
                heads[b,method,names[head]]+=1
            prior=(labels[:b].sum(axis=0)+.5)/(b+2.5)
            def calibrated(head):
                post=bin_distribution(bins[:,head],order[:b],labels[:b],prior)
                return post[bins[data.test,head]]
            preds=(q[data.test,ce],q[data.test,pearson],q[data.test,-1],
                   np.broadcast_to(prior,h.shape),calibrated(pearson),calibrated(ce),
                   calibrated(loo),q[data.test,0],calibrated(0))
            for mi,pred in enumerate(preds):
                losses=per_item(pred,h)[:,:3]
                cube[j,mi]+=losses/64
                draw_rps[rep,j,mi]=losses[:,0].mean()
            last=b
    coverage=len(h)/400
    points,lo,hi,plo,phi,_=bands(cube,data.group[data.test],level=.95)
    acceptable=(lo[:,0]>0)&(lo[:,1]>0)
    strong=(lo>0).all(axis=1)
    def sustained(gates):
        return next((b for j,b in enumerate(BUDGETS) if gates[j:].all()),None)
    info.update(weak_gate=sustained(acceptable) if panel=="all_users" else None,
                all_four_gate=sustained(strong) if panel=="all_users" else None,
                candidate_count=len(names),undefined_training_pearson_draws=dict(n_undefined))
    scores=[]
    contrasts=[]
    for j,b in enumerate(BUDGETS):
        for mi,method in enumerate(METHODS):
            scores.append(dict(budget=b,method=method,test_movies=len(h),draws=64,
                          **dict(zip(METRICS,cube[j,mi].mean(axis=0)))))
        for c,method in enumerate(METHODS[1:5]):
            v=points[j,c]
            contrasts.append(dict(budget=b,reference=method,subtract=METHODS[0],rps_diff=v,
                sim95_low=lo[j,c],sim95_high=hi[j,c],point95_low=plo[j,c],point95_high=phi[j,c],
                full400_identification_low=coverage*v-(1-coverage),
                full400_identification_high=coverage*v+(1-coverage),
                ce_lower_draws=int((draw_rps[:,j,c+1]-draw_rps[:,j,0]>1e-12).sum()),
                reference_lower_draws=int((draw_rps[:,j,c+1]-draw_rps[:,j,0]< -1e-12).sum())))
    comp=(5,6,8)
    delta=cube[:,4:5]-cube[:,comp]
    intervals=bootstrap(np.moveaxis(delta,2,0).reshape(len(h),-1),data.group[data.test]).reshape(
                       2,len(BUDGETS),len(comp),len(METRICS))
    matched=[]
    for j,b in enumerate(BUDGETS):
        for ci,mi in enumerate(comp):
            for k,metric in enumerate(METRICS):
                diff=delta[j,ci,:,k].mean()
                matched.append(dict(budget=b,reference=METHODS[4],subtract=METHODS[mi],metric=metric,
                    difference=diff,point95_low=intervals[0,j,ci,k],point95_high=intervals[1,j,ci,k],
                    full400_identification_low=coverage*diff-(1-coverage) if k==0 else "",
                    full400_identification_high=coverage*diff+(1-coverage) if k==0 else ""))
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("scores",scores),("contrasts",contrasts),("matched",matched),
                      ("head_counts",[dict(budget=b,method=m,head=n,draws=count)
                                      for (b,m,n),count in sorted(heads.items())])):
        save(out/(name+".csv"),rows)
    (out/"coverage.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    for panel_name in PANELS:
        run(args.archive,panel_name,args.output/panel_name)
