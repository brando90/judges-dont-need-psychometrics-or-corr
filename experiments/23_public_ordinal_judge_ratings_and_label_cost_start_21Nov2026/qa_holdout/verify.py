"""Independently replay every 12/24 acquisition outcome and absolute second-panel loss."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from candidates import build
from design import draw,load
from verify_primary import predict_bins


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def check(path,out):
    d=load(path)
    with np.load(path) as archive:
        test=np.flatnonzero(archive["split"]=="validation")
    assert not set(d.id[test]) & (set(d.id[d.train])|set(d.id[d.test]))
    hist=d.counts[test]/d.counts[test].sum(axis=1,keepdims=True)
    stars=np.arange(1,6)
    distance=abs(stars[:,None]-stars)/4
    def scores(p):
        # Estimator based on the absolute-distance representation of the ordinal energy score.
        e=p@distance
        rps=((e*hist).sum(axis=1)-.5*(e*p).sum(axis=1)).mean()
        ce=-(np.log(p)*hist).sum(axis=1).mean()
        tvd=1-np.minimum(p,hist).sum(axis=1).mean()
        return np.array([rps,ce,tvd])
    p,names=build(d,"features")
    m=p@stars
    src=HERE.parent/"results"/"image_quality"/"selections.csv"
    fits={(int(r["draw"]),int(r["budget"])):r for r in read(src) if r["depth"]=="1"}
    written=read(out/"scores.csv")
    contrasts=read(out/"contrasts.csv")
    for b in (12,24):
        methods=("ce","pearson","uniform","histogram","calibrated_pearson","original","rps")
        all_scores=[]
        for rep in range(64):
            ids,picked=draw(d,rep,1)
            ids, picked=ids[:b],picked[:b]
            rec=fits[rep,b]
            ce,pear,rps=[names.index(rec[k]) for k in ("ce","pearson","rps")]
            prior=(picked.sum(axis=0)+.5)/(b+2.5)
            calibrated=predict_bins(m[:,pear],d.train,ids,picked,test,prior)
            preds=(p[test,ce],p[test,pear],p[test,-1],np.broadcast_to(prior,hist.shape),calibrated,
                   p[test,0],p[test,rps])
            all_scores.append([scores(x) for x in preds])
        all_scores=np.asarray(all_scores)
        for mi,method in enumerate(methods):
            ref=next(r for r in written if int(r["budget"])==b and r["method"]==method)
            np.testing.assert_allclose(all_scores[:,mi].mean(axis=0),[float(ref[k]) for k in
                                       ("rps","ce_nats","tvd")],rtol=0,atol=1e-12)
        d_rps=all_scores[:,1:5,0]-all_scores[:,:1,0]
        winners=d_rps>1e-12
        ref=next(r for r in contrasts if int(r["budget"])==b)
        assert int(ref["draws_beat_both"])==winners[:,:2].all(axis=1).sum()
        assert int(ref["draws_beat_all_four"])==winners.all(axis=1).sum()
        print(b,"ratings: all 21 absolute scores and both single-acquisition counts reproduced.")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz",type=Path)
    ap.add_argument("results",type=Path)
    a=ap.parse_args()
    check(a.npz,a.results)
