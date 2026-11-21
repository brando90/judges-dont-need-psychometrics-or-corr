"""Budgeted fits and evaluation on independently held-out public ordinal ratings."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from candidates import R, best, bin_distribution, bin_indices, build, loss_by_grade, pearson_for_pair, pearson_from_totals, per_item
from design import draw, load
from diagnostics import diagnostics, human_reference

BUDGETS = (6,12,24,48,96,192,384,768,1536,3072,6144)
METHODS = ("ce","pearson","uniform","histogram","calibrated_pearson","original","rps")
COMPARATORS = METHODS[1:5]
METRICS = ("rps","ce","tvd","brier")
REPEATS = 64
BOOT = 4000


def save(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k:("" if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v)
                             for k,v in row.items()})


class Evaluation:
    def __init__(self, data, q):
        self.test = data.test
        self.counts = data.counts[data.test]
        self.truth = self.counts/self.counts.sum(axis=1,keepdims=True)
        self.system = data.system[data.test] if data.system is not None else None
        self.q = q
        self.cached = {}

    def score(self, key, probabilities):
        if key not in self.cached:
            q = probabilities()
            self.cached[key] = (per_item(q,self.truth), diagnostics(q,self.counts,self.system))
        return self.cached[key]


def bands(cube,groups,level=.975):
    """Paired bands across every budget and the four prespecified comparators."""
    _, inverse, size = np.unique(groups,return_inverse=True,return_counts=True)
    bcount, _, nitems, _ = cube.shape
    assert nitems == len(groups) and np.unique(size).size == 1
    diff = (cube[:,1:5,:,0]-cube[:,0:1,:,0]).reshape(-1,nitems).T
    by_group = np.zeros((len(size),diff.shape[1]))
    np.add.at(by_group,inverse,diff)
    by_group /= size[:,None]
    point = by_group.mean(axis=0)
    rng = np.random.default_rng(20261123)
    boots = np.empty((BOOT,diff.shape[1]))
    for i in range(0,BOOT,100):
        w = rng.multinomial(len(size),np.ones(len(size))/len(size),size=100)
        boots[i:i+100] = w @ by_group/len(size)
    se = boots.std(axis=0,ddof=1)
    valid = se > 1e-12
    tmax = (abs((boots[:,valid]-point[valid])/se[valid]).max(axis=1)
            if valid.any() else np.zeros(BOOT))
    critical = np.quantile(tmax,level,method="higher")
    lo, hi = point-critical*se, point+critical*se
    point_lo,point_hi = np.quantile(boots,[.025,.975],axis=0)
    pvals=np.ones_like(point)
    for i in np.flatnonzero(valid):
        pvals[i]=(1+np.sum(tmax>=abs(point[i]/se[i])))/(BOOT+1)
    pvals[~valid & (abs(point)>1e-12)] = 1/(BOOT+1)
    return [a.reshape(bcount,4) for a in (point,lo,hi,point_lo,point_hi,pvals)]


def run(data,mode,out):
    out.mkdir(parents=True,exist_ok=True)
    q, names = build(data,mode)
    budgets = [b for b in BUDGETS if b <= len(data.train)]
    ev = Evaluation(data,q)
    # Per-head grade losses depend only on the predictor and realized grade index.
    # Precompute once for the complete training feature pool, then reuse across draws.
    means = q@R
    qlog, qrps = loss_by_grade(q[data.train])
    local = np.full(len(q),-1,int)
    local[data.train] = np.arange(len(data.train))
    bins_cache = {}
    rows,cons,draw_rows,diag_rows,selection_rows = [],[],[],[],[]
    for depth in (1,3):
        cube = np.zeros((len(budgets),len(METHODS),len(data.test),4))
        draw_scores = np.zeros((REPEATS,len(budgets),len(METHODS),4))
        diag_sum, diag_count = defaultdict(float), Counter()
        for rep in range(REPEATS):
            order, picked = draw(data,rep,depth)
            train_rows = local[order]
            sums = np.zeros((5,len(names)))
            sum_y = sq_y = 0.
            prev = 0
            for bidx,b in enumerate(budgets):
                n = b//depth
                ix = order[prev:n]
                loc = train_rows[prev:n]
                normalized = picked[prev:n]/depth
                y = normalized @ R
                m = means[ix]
                # Work on nonzero rating bins only and do not create model-sized subsets of unused grades.
                for k in range(5):
                    nz = np.where(normalized[:,k]>0)[0]
                    if len(nz):
                        weights = normalized[nz,k]
                        sums[0] += np.einsum("i,ih->h",weights,qlog[loc[nz],k])
                        sums[1] += np.einsum("i,ih->h",weights,qrps[loc[nz],k])
                sums[2] += m.sum(axis=0)
                sums[3] += np.einsum("ih,ih->h",m,m)
                sums[4] += np.einsum("i,ih->h",y,m)
                sum_y += y.sum()
                sq_y += np.dot(y,y)
                ce, rps = best(sums[0]/n), best(sums[1]/n)
                if n == 2:
                    pearson, undefined = pearson_for_pair(means[order[:n]], (picked[:n] / depth) @ R)
                else:
                    pearson, undefined = pearson_from_totals(*sums[2:],sum_y,sq_y,n)
                hist_count = picked[:n].sum(axis=0)
                prior = (hist_count+.5)/(b+2.5)
                if pearson not in bins_cache:
                    bins_cache[pearson]=bin_indices(means[data.train,pearson],means[:,pearson])
                bins = bins_cache[pearson]
                posterior = bin_distribution(bins,order[:n],picked[:n],prior)
                indices = {"ce":ce,"pearson":pearson,"uniform":len(names)-1,"original":0,"rps":rps}
                selection_rows.append(dict(depth=depth,budget=b,draw=rep,ce=names[ce],pearson=names[pearson],
                                           rps=names[rps],pearson_undefined=int(undefined)))
                for mi,method in enumerate(METHODS):
                    if method in indices:
                        head = indices[method]
                        losses,diags = ev.score(head,lambda: q[data.test,head])
                    else:
                        # These estimated probabilities generally change every draw, so score without caching.
                        pred = (np.broadcast_to(prior,ev.truth.shape) if method=="histogram"
                                else posterior[bins[data.test]])
                        losses = per_item(pred,ev.truth)
                        diags = diagnostics(pred,ev.counts,ev.system)
                    cube[bidx,mi] += losses
                    draw_scores[rep,bidx,mi]=losses.mean(axis=0)
                    for key,val in diags.items():
                        pos=(bidx,mi,key)
                        if np.isfinite(val):
                            diag_sum[pos]+=val
                            diag_count[pos]+=1
                        else:
                            diag_count[pos]+=0
                prev = n
        cube /= REPEATS
        point,lo,hi,point_lo,point_hi,pvals=bands(cube,data.group[data.test])
        for bi,b in enumerate(budgets):
            for mi,method in enumerate(METHODS):
                row=dict(depth=depth,budget=b,train_items=b//depth,method=method)
                for d,metric in enumerate(METRICS):
                    row[metric]=cube[bi,mi,:,d].mean()
                    row[metric+"_draw_sd"]=draw_scores[:,bi,mi,d].std(ddof=1)
                rows.append(row)
                dg=dict(depth=depth,budget=b,method=method)
                for ci,cm,key in diag_count:
                    if ci==bi and cm==mi:
                        pos=(bi,mi,key)
                        dg[key]=diag_sum[pos]/diag_count[pos] if diag_count[pos] else np.nan
                        dg[key+"_draws"]=diag_count[pos]
                diag_rows.append(dg)
            for ci,method in enumerate(COMPARATORS):
                col=METHODS.index(method)
                cons.append(dict(depth=depth,budget=b,comparator=method,subtract="ce",rps_gain=point[bi,ci],
                                 simultaneous_lo=lo[bi,ci],simultaneous_hi=hi[bi,ci],pointwise_lo=point_lo[bi,ci],
                                 pointwise_hi=point_hi[bi,ci],within_benchmark_max_t_p=pvals[bi,ci],
                                 paired_draw_sd=np.std(draw_scores[:,bi,col,0]-draw_scores[:,bi,0,0],ddof=1),
                                 test_items=len(data.test),test_groups=len(np.unique(data.group[data.test]))))
            for rep in range(REPEATS):
                draw_rows.append(dict(depth=depth,budget=b,draw=rep,**dict(zip(METHODS,draw_scores[rep,bi,:,0]))))
            print(data.name,mode,depth,b,*(round(v,5) for v in point[bi]),flush=True)
    save(out/"scores.csv",rows)
    save(out/"contrasts.csv",cons)
    save(out/"diagnostics.csv",diag_rows)
    save(out/"draw_rps.csv",draw_rows)
    save(out/"selections.csv",selection_rows)
    save(out/"human_reference.csv",[human_reference(ev.counts)])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("mode",choices=("native","features"))
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run(load(args.data),args.mode,args.out)
