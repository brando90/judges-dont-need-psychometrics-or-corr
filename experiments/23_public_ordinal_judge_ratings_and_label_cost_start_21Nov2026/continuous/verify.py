"""Replay selected fits, check both gradients and score held-out ratings independently."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

from run import (BUDGETS, GRADE, LAMBDAS, METRICS, OBJECTIVES, ROOT, encode_features,
                 evaluate, fit_score_then_calibrate, fit_softmax, training_prior,
                 write_rows)
import numpy as np
from scipy.special import softmax
sys.path.insert(0,str(ROOT.parent))
from design import draw, load


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def energy_scores(p, h):
    """Independent ranked probability score via the discrete energy-distance identity."""
    dist = np.abs(GRADE[:,None] - GRADE[None,:])
    pdist = p@dist
    rps = (np.sum(pdist*h,axis=1)-.5*np.sum(pdist*p,axis=1))/4
    logloss = -(h*np.log(p)).sum(axis=1)
    tv = 1 - np.minimum(p,h).sum(axis=1)
    return np.column_stack((rps,logloss,tv))


def gradients():
    rng=np.random.default_rng(123)
    x=encode_features(rng.normal(size=(9,4)),np.arange(9))
    h=np.eye(5)[[0,3,0,4,2,1,1,2,4]]
    w=rng.normal(0,.2,x.shape[1]*5)
    for objective in OBJECTIVES:
        def fun(a):
            return evaluate(a,x,h,np.log(training_prior(h)),.1,objective)
        _, grad=fun(w)
        steps=np.eye(w.size)*1e-5
        numeric=np.array([(fun(w+d)[0]-fun(w-d)[0])/(2e-5) for d in steps])
        print(objective,"maximum numerical gradient discrepancy",max(abs(grad-numeric)))
        np.testing.assert_allclose(grad,numeric,rtol=1e-7,atol=1e-8)
    paired = np.eye(5)
    np.testing.assert_allclose(energy_scores(paired*.8+.04,paired)[:,0],
        np.mean((np.cumsum(paired*.8+.04,axis=1)[:,:4] - (GRADE[:,None]<=np.arange(4)))**2,axis=1))


def train_value(x, h, logprior, w, strength, objective):
    p = softmax(logprior+x@w,axis=1)
    grade_mean = p@GRADE
    if objective=="ce":
        value = -(h*np.log(p)).sum()/len(p)
    elif np.ptp(grade_mean)<=1e-12 or np.ptp(h@GRADE)<=1e-12:
        value = 2.
    else:
        value = 1-np.corrcoef(grade_mean,h@GRADE)[0,1]
    return value+strength*(w*w).sum()/2


def verify(name, data, result_dir):
    table = read(result_dir/(name+"_scores.csv"))
    starts = {(int(r["budget"]),int(r["draw"]),r["objective"]):r
              for r in read(result_dir/(name+"_starts.csv"))}
    scalar = {(int(r["budget"]),int(r["draw"])):float(r["penalty"])
              for r in read(result_dir/(name+"_ridge_baseline.csv"))}
    diag = Counter()
    x=encode_features(data.X,data.train)
    htest=data.counts[data.test]
    htest=htest/htest.sum(axis=1)[:,None]
    xtest=x[data.test]
    contrasts={(int(r["budget"]), r["reference"], r["metric"]):r
               for r in read(result_dir/(name+"_contrasts.csv"))}
    _, groups = np.unique(data.group[data.test],return_inverse=True)
    ngroups = 1+max(groups)
    checks=[]
    for b in BUDGETS[name]:
        draw_scores=[]
        per_group = np.zeros((3,ngroups,3))
        for rep in range(16):
            order,h=draw(data,rep,1)
            xb,h=x[order[:b]],h[:b]
            n=b-b//4
            per_method=[]
            for obj in OBJECTIVES:
                row=starts[b,rep,obj]
                val=[]
                for strength in LAMBDAS:
                    pr,attempts,ch=fit_softmax(xb[:n], h[:n],strength,obj)
                    for _,_,status,_ in attempts:
                        diag[obj,b,"validation",status]+=1
                    val.append(energy_scores(softmax(pr+xb[n:]@attempts[ch][1],axis=1),h[n:])[:,0].mean())
                lam=next(v for v,s in zip(LAMBDAS,val) if s<=min(val)+1e-12)
                assert lam==float(row["penalty"])
                pr,attempts,ch=fit_softmax(xb,h,lam,obj)
                assert int(row["selected_start"])==(1,-1)[ch]
                assert [row["selected_status"],row["other_status"]]==[attempts[ch][2],attempts[1-ch][2]]
                for value,w,status,_ in attempts:
                    diag[obj,b,"refit",status]+=1
                    np.testing.assert_allclose(train_value(xb,h,pr,w,lam,obj),value,atol=2e-9)
                np.testing.assert_allclose(float(row["selected_train_mean_span"]),
                    np.ptp(softmax(pr+xb@attempts[ch][1],axis=1)@GRADE),atol=1e-12)
                chosen_items = energy_scores(softmax(pr+xtest@attempts[ch][1],axis=1), htest)
                expected = [chosen_items.mean(0),
                            energy_scores(softmax(pr+xtest@attempts[1-ch][1],axis=1),htest).mean(0)]
                np.add.at(per_group[len(per_method)],groups,chosen_items)

                np.testing.assert_allclose(float(row["other_minus_selected_test_rps"]),
                    expected[1][0]-expected[0][0],atol=1e-11)
                per_method.append(expected[0])
            lam=scalar[b,rep]
            baseline=fit_score_then_calibrate(xb,h,lam)(xtest)
            base_items = energy_scores(baseline,htest)
            np.add.at(per_group[2],groups,base_items)
            per_method.append(base_items.mean(0))
            draw_scores.append(per_method)
        arr=np.array(draw_scores)
        for j,method in enumerate((*OBJECTIVES,"score_then_calibrate")):
            row, = [r for r in table if int(r["budget"])==b and r["method"]==method]
            np.testing.assert_allclose(arr[:,j].mean(0),[float(row[k]) for k in METRICS],atol=1e-11)
        per_group /= (16*np.bincount(groups))[None,:,None]
        for j,method in ((1,"pearson"),(2,"score_then_calibrate")):
            by_group = per_group[j]-per_group[0]
            rng=np.random.default_rng(20261123)
            draws=[]
            for _ in range(200):
                pick = rng.integers(ngroups,size=(50,ngroups))
                draws.extend(by_group[pick].mean(axis=1))
            ci=np.percentile(draws,(2.5,97.5),axis=0)
            for k,metric in enumerate(METRICS):
                reference=contrasts[b,method,metric]
                np.testing.assert_allclose([ci[0,k],ci[1,k]],
                    [float(reference["lower95"]),float(reference["upper95"])],atol=1e-11)
            delta=arr[:,j,0]-arr[:,0,0]
            checks.append(dict(dataset=name,budget=b,comparator=method,ce_lower=int((delta>1e-12).sum()),
                    comparator_lower=int((delta < -1e-12).sum()),tied=int((abs(delta)<=1e-12).sum()),
                    paired_training_draws=len(delta),average_difference=delta.mean()))
    for r in read(result_dir/(name+"_convergence.csv")):
        assert diag.pop((r["objective"],int(r["budget"]),r["stage"],r["status"]))==int(r["count"])
    assert not diag
    print(name,"all validation and final softmax start statuses match;",
          len(starts),"selected fits agree, all held-out scores and group-bootstrap bounds match independent scoring.")
    return checks


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",type=Path)
    parser.add_argument("results",type=Path)
    args=parser.parse_args()
    gradients()
    all_rows=[]
    for dataset,archive in (("summeval","summeval_consistency.npz"),("koniq","koniq.npz")):
        all_rows.extend(verify(dataset,load(args.inputs/archive),args.results))
    write_rows(args.results/"verified_draw_wins.csv",all_rows)
