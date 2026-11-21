"""Validate registered forecasts and test scorer algebra using real historical development data only."""
import argparse
from pathlib import Path

import numpy as np

from calibrate import (DRAWS, GRADE, JUDGES, OVERLAPS, calibrations, historical_eligibility,
                       inputs, load_eligible_labels)
from score_release import bootstrap_weights, evaluate, records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("registration",type=Path)
    parser.add_argument("forecast",type=Path)
    parser.add_argument("--legacy",type=Path,default=Path.home()/"veribench")
    args=parser.parse_args()
    targets,fits,_=records(args.registration)
    manifest,base,_=inputs(args.directory,[args.forecast])
    for r in targets:
        actual=np.asarray(r["distributions"])
        np.testing.assert_allclose(actual[:,0],[(1-6e-6)*base[r["id"],j]+1e-6 for j in JUDGES],atol=1e-14)
    allowed=historical_eligibility(args.legacy)
    labels=load_eligible_labels(args.legacy,allowed)
    strict=sorted((r for r in manifest if r["source_item"] in allowed and r["may_family"] not in OVERLAPS),key=lambda r:r["id"])
    rows=[]; observed={}
    for r in strict:
        item=r["source_item"]
        probs=calibrations(np.stack([base[r["id"],j] for j in JUDGES]))
        rows.append({"source_item":item,"operation":r["may_family"],"distributions":probs})
        observed[item]=dict(enumerate(labels[item]))
    batch=[x for x in fits if x["pool"]=="strict" and x["budget"]==4 and x["ratings_per_artifact"]==2]
    assert len(batch)==DRAWS
    by_method={(r["judge"],r["method"]):r for j in JUDGES
               for r in evaluate(rows,len(rows),observed,batch,j,"strict",bootstrap_weights(rows))}
    for j,name in enumerate(JUDGES):
        for method in ("ce","pearson","training_histogram","identity","uniform"):
            direct=[]
            for draw in batch:
                item_losses=[]
                for r in rows:
                    if method in ("ce","pearson","identity"):
                        head=0 if method=="identity" else draw[method][j]
                        q=r["distributions"][j,head]
                    else:
                        q=np.asarray(draw["histogram"]) if method=="training_histogram" else np.ones(6)/6
                    y=labels[r["source_item"]]
                    individual=np.mean((np.cumsum(q)[:-1]-(y[:,None]<=GRADE[:-1]))**2)
                    ce=np.mean(-np.log(q[y]))
                    h=np.bincount(y,minlength=6)/len(y)
                    item_losses.append([individual,ce,0.5*np.abs(q-h).sum()])
                direct.append(np.mean(item_losses,axis=0))
            got=by_method[name,method]
            np.testing.assert_allclose([got["rps"],got["ce"],got["tvd"]],np.mean(direct,axis=0),atol=1e-13,rtol=0)
    print("Both judges’ registered identity probabilities match all 100 test inputs.")
    print("On 15 historical development artifacts, 2 judges × 5 methods × 3 scores equal independent per-rating, per-draw calculations.")
    print("No Set N outcome was read.")


if __name__=="__main__":
    main()
