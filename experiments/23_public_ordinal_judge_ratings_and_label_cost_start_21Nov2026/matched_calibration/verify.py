"""Check stratification equality and calibration with independent individual-vote arithmetic."""
import argparse
import csv
from pathlib import Path
from run import DATASETS, HERE, R, best, bin_indices, build, draw, explicit_loo, load, np, original_rows


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def independently_calibrate(train_bins, test_bins, y):
    # Explicit five individual grade counts, followed by eight prior pseudocounts.
    marginal=np.array([(np.count_nonzero(y==r)+.5)/(len(y)+2.5) for r in range(5)])
    out=np.zeros((len(test_bins),5))
    for b in np.unique(test_bins):
        grades=y[train_bins==b]
        out[test_bins==b]=[(np.count_nonzero(grades==r)+8*marginal[r])/(len(grades)+8)
                          for r in range(5)]
    return out


def direct_score(p,h):
    dg=abs(np.arange(5)[:,None]-np.arange(5)[None,:])
    ep=p@dg
    rps=np.sum(ep*(h-p/2),axis=1)/4
    return np.stack((rps,-np.sum(h*np.log(p),axis=1),1-np.minimum(p,h).sum(axis=1)),axis=-1)


def replay(dataset, input_dir):
    archive,mode,primary=DATASETS[dataset]
    data=load(input_dir/archive)
    q,names=build(data,mode)
    means=q@R
    del q
    bins=np.array([bin_indices(means[data.train,i],means[:,i]) for i in range(len(names))])
    idx={name:i for i,name in enumerate(names)}
    frozen=original_rows(HERE.parent/"results"/primary,"selections.csv")
    orig={(int(r["draw"]),int(r["budget"])):r for r in frozen}
    out=HERE/dataset
    fit={(int(r["draw"]),int(r["budget"])):r for r in read(out/"training_loo.csv")}
    h=data.counts[data.test]/data.counts[data.test].sum(axis=1,keepdims=True)
    totals={}
    for rep in range(64):
        order,labels=draw(data,rep,1)
        for b in sorted({b for r,b in fit if r==rep}):
            r=fit[rep,b]
            y=np.argmax(labels[:b],axis=1)
            train=order[:b]
            if rep < 2 and b<=12:
                loo=np.array([explicit_loo(partition[train],y) for partition in bins])
                assert names[best(loo)]==r['loo_winner']
                np.testing.assert_allclose(float(r['winning_training_loo_ce']),loo.min(),atol=1e-12)
            heads=(("fixed_reference",0),("raw_ce_selector",idx[orig[rep,b]["ce"]]),
                   ("pearson_selector",idx[orig[rep,b]["pearson"]]),("loo_ce_selector",idx[r['loo_winner']]))
            pred={}
            for method,j in heads:
                bv=bins[j]
                pred[method]=independently_calibrate(bv[train],bv[data.test],y)
                val=direct_score(pred[method],h).mean(axis=0)
                totals[b,method]=totals.get((b,method),np.zeros(3))+val/64
            if dataset=="koniq" and b>=768:
                # Equality of every returned test probability, independent of mean metric ties.
                for m in ("raw_ce_selector","loo_ce_selector"):
                    np.testing.assert_array_equal(pred[m],pred["pearson_selector"])
    for r in read(out/"scores.csv"):
        np.testing.assert_allclose(totals[int(r['budget']),r['method']],
                                  [float(r[k]) for k in ('rps','ce_nats','tvd')],rtol=0,atol=1e-12)
    print(dataset,len(totals)*3,"absolute scores match independent calibration and scoring.")
    print("Exhaustive explicit LOO fits matched winners on first two draws at budgets 6 and 12.")
    if dataset=="koniq":
        print("At 768,1536,3072,6144, all three supervised selectors returned identical five-bin probabilities "
              "on every one of 2,015 images and 64 draws per budget.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",type=Path)
    args=parser.parse_args()
    for name in DATASETS:
        replay(name,args.inputs)
