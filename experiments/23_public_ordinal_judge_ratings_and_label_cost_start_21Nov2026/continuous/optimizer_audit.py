"""Exploratory, matched 10x iteration-cap replay; leaves all frozen functions on disk unchanged."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import sys
from unittest.mock import patch

import run as frozen
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

sys.path.insert(0,str(frozen.ROOT.parent))
from design import load, draw


def check_controls(dataset, audit_dir):
    def rows(directory, suffix):
        with (directory/f"{dataset}_{suffix}.csv").open() as source:
            return list(csv.DictReader(source))
    primary=frozen.ROOT/"results"
    def sorted_rows(rr, keys):
        return sorted(rr,key=lambda row: tuple(row[k] for k in keys))
    for suffix, keys in (("scores", ("budget","method")), ("ridge_baseline", ("budget","draw")),
                         ("starts", ("budget","objective","draw"))):
        original, repeated=rows(primary,suffix), rows(audit_dir,suffix)
        if suffix=="starts":
            original=[r for r in original if r["objective"]=="ce"]
            repeated=[r for r in repeated if r["objective"]=="ce"]
        elif suffix=="scores":
            original=[r for r in original if r["method"]!="pearson"]
            repeated=[r for r in repeated if r["method"]!="pearson"]
        assert sorted_rows(original,keys)==sorted_rows(repeated,keys)
    tally=Counter()
    for row in rows(audit_dir,"starts"):
        for which in ("selected","other"):
            n=int(row[which+"_iterations"])
            status=row[which+"_status"]
            assert n<=1000 and (status!="iteration_limit" or n==1000)
            tally[row["budget"],row["objective"],status]+=1
    for row in rows(audit_dir,"convergence"):
        if row["stage"]=="refit":
            assert tally.pop((row["budget"],row["objective"],row["status"]))==int(row["count"])
    assert not tally
    print(dataset,"matched original CE fits and score baseline, checked every final start and convergence count.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",choices=tuple(frozen.BUDGETS))
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--checks-only",action="store_true",help="Check completed output against frozen controls.")
    args=parser.parse_args()
    if args.checks_only:
        check_controls(args.dataset,args.output)
        return
    if args.output.resolve() == (frozen.ROOT/"results").resolve():
        raise ValueError("Choose a separate output directory for the exploratory cap.")
    diagnostics=defaultdict(list)

    def longer_minimize(*pos,**kwargs):
        opts=kwargs["options"]
        assert opts["maxiter"]==100
        kwargs["options"]={**opts,"maxiter":1000}
        fit=minimize(*pos,**kwargs)
        x,h,prior,penalty,objective=kwargs["args"]
        if len(x) in frozen.BUDGETS[args.dataset]:
            w=fit.x.reshape(x.shape[1],5)
            p=softmax(prior + x@w,axis=1)
            means=p@frozen.GRADE
            human=h@frozen.GRADE
            r=np.corrcoef(means,human)[0,1] if min(np.ptp(means),np.ptp(human))>1e-12 else np.nan
            diagnostics[(len(x),objective)].append((np.ptp(means),np.max(np.abs(fit.jac)),
                np.linalg.norm(w),float(r),penalty))
        return fit

    # Run the largest prespecified budget first. It uses exactly the same seed and fits.
    reversed_budgets=tuple(reversed(frozen.BUDGETS[args.dataset]))
    with patch.object(frozen,"minimize",longer_minimize), patch.dict(frozen.BUDGETS,{args.dataset:reversed_budgets}):
        frozen.run(args.dataset,load(args.archive),args.output,draw)
    summary=[]
    for (budget,method), observations in sorted(diagnostics.items()):
        span,grad,norm,corr,penalty=np.asarray(observations).T
        summary.append(dict(dataset=args.dataset,budget=budget,objective=method,
            evaluated_final_starts=len(span),min_train_mean_span=span.min(),median_train_mean_span=np.median(span),
            max_train_mean_span=span.max(),span_below_0p001=int((span<.001).sum()),
            span_below_0p01=int((span<.01).sum()),median_coefficient_norm=np.median(norm),
            median_gradient_max_norm=np.median(grad),median_defined_train_pearson=np.nanmedian(corr)))
    frozen.write_rows(args.output/(args.dataset+"_conditioning.csv"),summary)
    check_controls(args.dataset,args.output)


if __name__=="__main__":
    main()
