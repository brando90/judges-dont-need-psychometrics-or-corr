"""Compare four score selectors under the identical, published four-bin calibration map."""
import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from candidates import R, best, bin_distribution, bin_indices, build
from design import draw, load

METHODS = ("fixed_reference", "raw_ce_selector", "pearson_selector", "loo_ce_selector")
METRICS = ("rps", "ce_nats", "tvd")
DATASETS = {"summeval": ("summeval_consistency.npz", "native", "gpt4_consistency"),
            "koniq": ("koniq.npz", "features", "image_quality")}


def original_rows(directory, name):
    with (directory / name).open() as source:
        return [r for r in csv.DictReader(source) if int(r["depth"]) == 1]


def dump(path, rows):
    with path.open("w", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def score(p, h):
    f, g = np.cumsum(p, axis=-1)[:, :-1], np.cumsum(h, axis=-1)[:, :-1]
    assert not np.any((p == 0) & (h > 0))
    logs = np.log(p, where=p > 0, out=np.zeros_like(p))
    return np.column_stack((np.mean(g*(1-f)**2 + (1-g)*f**2, axis=1),
                            -np.sum(h*logs, axis=1), np.sum(abs(h-p), axis=1)/2))


def add_hist(bins, y, count):
    # Indexing per (head, score bin, grade) permits one integer tally for every candidate.
    nhead = bins.shape[1]
    key = (4*np.arange(nhead)[None, :] + bins) * 5 + y[:, None]
    return np.bincount(key.ravel(), minlength=nhead*20).reshape(nhead, 4, 5) + count


def loo_loss(count):
    """Exact mean held-out-vote log loss, aggregated over occupied bin/grade pairs."""
    pool = count.sum(axis=1)
    budget = pool[0].sum()
    assert budget >= 2 and np.all(pool == pool[0])
    prior_own_grade = (pool-1+.5) / (budget-1+2.5)
    numerator = count-1 + 8*prior_own_grade[:, None, :]
    denominator = count.sum(axis=2, keepdims=True)-1+8
    probability = np.divide(numerator, denominator, out=np.ones_like(numerator), where=count>0)
    assert np.all(probability > 0) and np.all(probability <= 1+1e-12)
    return -(count*np.log(probability)).sum(axis=(1, 2))/budget


def explicit_loo(bins, y):
    """Slow independent leave-one-out retraining, used only to check small draws."""
    losses=[]
    for i, outcome in enumerate(y):
        keep = np.arange(len(y)) != i
        other = y[keep]
        prior = (np.bincount(other, minlength=5)+.5)/(len(other)+2.5)
        same = y[keep & (bins == bins[i])]
        probabilities = (np.bincount(same, minlength=5)+8*prior)/(len(same)+8)
        losses.append(-np.log(probabilities[outcome]))
    return np.mean(losses)


def oracle_rows(bins, h, names):
    # All fitted curves in this function are explicitly evaluated in-sample on test labels.
    single = score(np.broadcast_to(h.mean(axis=0), h.shape), h).mean(axis=0)
    item = score(h,h).mean(axis=0)
    opt=[]
    for head in range(len(names)):
        partition = bins[:,head]
        sums = np.zeros((4,5))
        np.add.at(sums,partition,h)
        sizes=np.bincount(partition,minlength=4)
        p=sums[partition]/sizes[partition,None]
        opt.append(score(p,h).mean(axis=0))
    opt = np.stack(opt)
    rows=[]
    for k,metric in enumerate(METRICS[:2]):
        head = int(np.argmin(opt[:,k]))
        assert item[k] <= opt[head,k]+1e-12 and opt[head,k] <= single[k]+1e-12
        rows.append(dict(metric=metric,best_partition=names[head],
            empirical_single_bin_loss=single[k],best_partition_loss=opt[head,k],
            max_gain_over_single_bin=single[k]-opt[head,k],fixed_reference_partition_loss=opt[0,k],
            item_specific_empirical_minimum=item[k],candidate_count=len(names),
            distinct_best_partition_bins=len(np.unique(bins[:,head]))))
    return rows


def bootstrap(deltas, groups):
    _, inv, size = np.unique(groups, return_counts=True, return_inverse=True)
    assert min(size)==max(size)
    grouped = np.zeros((len(size),deltas.shape[1]))
    np.add.at(grouped,inv,deltas)
    grouped/=size[:,None]
    rng=np.random.default_rng(20261124)
    samples=[]
    for _ in range(100):
        chosen = rng.multinomial(len(size),np.ones(len(size))/len(size),size=100)
        samples.append(chosen@grouped/len(size))
    return np.quantile(np.concatenate(samples),[.025,.975],axis=0)


def run(dataset, source, output):
    archive, mode, experiment = DATASETS[dataset]
    data = load(source/archive)
    reference = HERE.parent/"results"/experiment
    chosen = {(int(r["draw"]),int(r["budget"])):r for r in original_rows(reference,"selections.csv")}
    primary = {int(r["budget"]):r for r in original_rows(reference,"scores.csv")
               if r["method"]=="calibrated_pearson"}
    budgets = sorted(primary)
    forecasts,names = build(data,mode)
    means=forecasts@R
    idx={name:i for i,name in enumerate(names)}
    bins=np.stack([bin_indices(means[data.train,j],means[:,j]) for j in range(len(names))],
                  axis=1).astype(np.uint8)
    del forecasts,means
    test = data.test
    h=data.counts[test]/data.counts[test].sum(axis=1,keepdims=True)
    avg=np.zeros((len(budgets),len(METHODS),len(test),len(METRICS)))
    draws=np.zeros((64,len(budgets),len(METHODS),len(METRICS)))
    selected=Counter()
    training=[]
    for rep in range(64):
        order, labels=draw(data,rep,1)
        outcomes=np.argmax(labels,axis=1)
        freq=np.zeros((len(names),4,5),dtype=np.int64)
        last=0
        for j,b in enumerate(budgets):
            new=order[last:b]
            freq=add_hist(bins[new],outcomes[last:b],freq)
            cv=loo_loss(freq)
            loo=best(cv)
            published=chosen[rep,b]
            heads=[0,idx[published["ce"]],idx[published["pearson"]],loo]
            ids=order[:b]
            counts=labels[:b]
            prior=(counts.sum(axis=0)+.5)/(b+2.5)
            if rep in (0,1) and j in (0,1):
                for head in set([*heads,len(names)-1]):
                    np.testing.assert_allclose(explicit_loo(bins[ids,head],outcomes[:b]),cv[head],
                                               rtol=0,atol=1e-12)
            for mi,head in enumerate(heads):
                post=bin_distribution(bins[:,head],ids,counts,prior)
                loss=score(post[bins[test,head]],h)
                avg[j,mi] += loss/64
                draws[rep,j,mi] = loss.mean(axis=0)
                selected[b,METHODS[mi],names[head]]+=1
            training.append(dict(draw=rep,budget=b,loo_winner=names[loo],
                winning_training_loo_ce=cv[loo],raw_ce_head_training_loo_ce=cv[heads[1]],
                pearson_head_training_loo_ce=cv[heads[2]],
                tied_loo_minima=int((cv<=cv.min()+1e-10).sum())))
            last=b

    absolute, paired=[],[]
    comparators=(0,1,3)
    delta=avg[:,2:3]-avg[:,comparators]
    flat=np.moveaxis(delta,2,0).reshape(len(test),-1)
    intervals=bootstrap(flat,data.group[test]).reshape(2,len(budgets),len(comparators),len(METRICS))
    for j,b in enumerate(budgets):
        for mi,method in enumerate(METHODS):
            value=avg[j,mi].mean(axis=0)
            if mi==2:
                np.testing.assert_allclose(value,[float(primary[b][key]) for key in ("rps","ce","tvd")],
                                           rtol=0,atol=1e-12)
            absolute.append(dict(budget=b,method=method,test_items=len(test),
                test_groups=len(np.unique(data.group[test])),draws=64,**dict(zip(METRICS,value))))
        for ref,mi in enumerate(comparators):
            for k,metric in enumerate(METRICS):
                by_draw=draws[:,j,2,k]-draws[:,j,mi,k]
                paired.append(dict(budget=b,reference=METHODS[2],subtract=METHODS[mi],metric=metric,
                    difference=by_draw.mean(),lower95=intervals[0,j,ref,k],
                    upper95=intervals[1,j,ref,k],alternative_lower_draws=int((by_draw>1e-12).sum()),
                    pearson_lower_draws=int((by_draw < -1e-12).sum()),
                    tied_draws=int((abs(by_draw)<=1e-12).sum())))
    oracle=oracle_rows(bins[test],h,names)
    output.mkdir(parents=True,exist_ok=True)
    dump(output/"scores.csv",absolute)
    dump(output/"contrasts.csv",paired)
    dump(output/"selection_counts.csv",[dict(budget=b,method=m,head=n,draw_count=v)
                                      for (b,m,n),v in sorted(selected.items())])
    dump(output/"training_loo.csv",training)
    dump(output/"empirical_oracle.csv",oracle)
    print(dataset,len(names),"candidates. Baseline matches published calibrated Pearson for",len(budgets),
          "budgets and 3 metrics. Explicit deletion checks passed.")
    for r in paired:
        if r["metric"]=="rps" and r["subtract"]=="loo_ce_selector":
            print(r["budget"],r["difference"],r["lower95"],r["upper95"])
    print("Empirical RPS optimum within tested partitions",oracle[0])


def fixture():
    rng=np.random.default_rng(17)
    bins=rng.integers(0,4,(13,4)); bins[:,1]=bins[:,0]; bins[:,3]=0
    y=rng.integers(0,5,13)
    h=add_hist(bins,y,np.zeros((4,4,5),dtype=int))
    computed=loo_loss(h)
    np.testing.assert_allclose(computed,[explicit_loo(col,y) for col in bins.T],rtol=0,atol=1e-13)
    assert computed[0]==computed[1]
    prior=(np.bincount(y,minlength=5)+.5)/(len(y)+2.5)
    post=bin_distribution(bins[:,3],np.arange(len(y)),np.eye(5)[y],prior)
    np.testing.assert_array_equal(post[bins[:,3]],np.broadcast_to(post[0],(len(y),5)))
    print("Explicit vote deletion, duplicate partitions and constant-score predictions verified.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",choices=tuple(DATASETS))
    parser.add_argument("inputs",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    fixture()
    run(args.dataset,args.inputs,args.output)
