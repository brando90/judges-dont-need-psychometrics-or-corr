"""Continuous softmax sensitivity on the shared, frozen experiment-23 design."""
import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

ROOT = Path(__file__).resolve().parent
GRADE = np.arange(5)
LAMBDAS = (0.001, 0.1, 1.0)
OBJECTIVES = ("ce", "pearson")
BUDGETS = {"summeval": (12, 192, 768), "koniq": (12, 192, 768, 6144)}
METRICS = ("rps", "ce_nats", "tvd")


def encode_features(raw, train):
    """Center/scale among all available training inputs, without reading grades."""
    finite = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(finite)):
        raise ValueError("The shared feature array must be finite")
    scale = finite[train].std(axis=0)
    z = (finite - finite[train].mean(axis=0)) / np.where(scale > 0, scale, 1.0)
    return np.column_stack((np.ones(len(z)), np.clip(z, -3, 3)))


def training_prior(y):
    return (y.sum(axis=0) + .5)/(y.sum() + 2.5)


def scores(p, h):
    """Per item, equal-rater expectation, with empirical grade probabilities h."""
    f = np.cumsum(p, axis=1)[:, :-1]
    g = np.cumsum(h, axis=1)[:, :-1]
    rps = np.mean(g*(1-f)**2 + (1-g)*f**2, axis=1)
    # softmax log probabilities remain representable on the declared feature scale.
    if np.any((p == 0) & (h > 0)):
        raise FloatingPointError("A scored outcome has zero forecast probability")
    ce = -(h*np.log(p, out=np.zeros_like(p), where=p>0)).sum(axis=1)
    return np.column_stack((rps, ce, np.abs(p-h).sum(axis=1)/2))


def evaluate(w_flat, x, h, log_prior, penalty, objective):
    weights = w_flat.reshape(x.shape[1], 5)
    logp = log_softmax(log_prior + x@weights, axis=1)
    p = np.exp(logp)
    if objective == "ce":
        value = -(h*logp).sum()/len(x)
        jacobian = x.T@(p-h)/len(x)
    else:
        mean = p@GRADE
        a = mean - mean.mean()
        observed = h@GRADE
        b = observed - observed.mean()
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if np.ptp(mean) <= 1e-12 or np.ptp(observed) <= 1e-12:
            value, jacobian = 2., np.zeros_like(weights)
        else:
            rho = (a/norm_a) @ (b/norm_b)
            value = 1 - rho
            dmean = (rho*a/norm_a - b/norm_b)/norm_a
            jacobian = x.T@(dmean[:,None]*p*(GRADE-mean[:,None]))
    return value + penalty*(w_flat@w_flat)/2, jacobian.ravel() + penalty*w_flat


def initial_weights(d, sign):
    # Both starting slopes have Frobenius norm 0.1, with opposite mean directions.
    direction = np.outer(np.arange(1, d), GRADE - GRADE.mean()).astype(float)
    return np.vstack((np.zeros(5), sign*.1*direction/np.linalg.norm(direction)))


def fit_softmax(x, h, penalty, objective):
    log_prior = np.log(training_prior(h))
    results = []
    for sign in (1, -1):
        start = initial_weights(x.shape[1], sign)
        if objective == "pearson" and (np.ptp(h@GRADE) <= 1e-12 or
                                       np.all(np.ptp(x,axis=0) <= 1e-12)):
            # A flat training target or identical features cannot supply Pearson information.
            w = np.zeros_like(start)
            value, _ = evaluate(w.ravel(), x, h, log_prior, penalty, objective)
            results.append((value, w, "undefined_prior", 0))
            continue
        opt = minimize(evaluate, start.ravel(), args=(x,h,log_prior,penalty,objective),
                       method="L-BFGS-B", jac=True, options={"maxiter":100, "ftol":1e-10,
                       "gtol":1e-7, "maxls":20})
        if not (np.isfinite(opt.fun) and np.isfinite(opt.x).all()):
            raise FloatingPointError("Optimizer returned a nonfinite objective or coefficient")
        w = opt.x.reshape(start.shape)
        final_value, _ = evaluate(w.ravel(), x, h, log_prior, penalty, objective)
        if objective == "pearson":
            z = np.exp(log_softmax(log_prior+x@w,axis=1)) @ GRADE
            if np.ptp(z) <= 1e-12:
                w = np.zeros_like(w)
                final_value = 2.
                results.append((final_value,w,"undefined_prior",opt.nit))
                continue
        results.append((final_value, w, "converged" if opt.success else
                        ("iteration_limit" if opt.status==1 else "solver_failure"), opt.nit))
    # The first listed start wins objective ties; success flags are reported separately.
    chosen = int(results[1][0] < results[0][0] - 1e-12)
    return log_prior, results, chosen


def predict_softmax(log_prior, weights, x):
    return np.exp(log_softmax(log_prior + x@weights, axis=1))


def fit_score_then_calibrate(x, h, penalty):
    z = x[:,1:]
    centered = z-z.mean(axis=0)
    y = h@GRADE
    beta = np.linalg.solve(centered.T@centered/len(z)+penalty*np.eye(z.shape[1]),
                           centered.T@(y-y.mean())/len(z))
    offset = y.mean() - z.mean(axis=0)@beta
    train_scores = z@beta+offset
    cuts = np.quantile(train_scores, (.25,.5,.75), method="linear")
    membership = np.searchsorted(cuts, train_scores, side="right")
    prior = training_prior(h)
    frequencies = np.stack([h[membership==k].sum(axis=0) + 8*prior for k in range(4)])
    probs = frequencies / frequencies.sum(axis=1,keepdims=True)
    return lambda test: probs[np.searchsorted(cuts,test[:,1:]@beta+offset,side="right")]


def winner(values):
    return next(i for i, value in enumerate(values) if value <= min(values)+1e-12)


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def track(diag, dataset, budget, objective, stage, tries):
    for _, _, status, _ in tries:
        diag[dataset,budget,objective,stage,status] += 1


def summarize_groups(data, metrics):
    """Equal test group weights and equal row weights within each group."""
    groups, index = np.unique(data.group[data.test], return_inverse=True)
    counts = np.bincount(index)
    if np.ptp(counts) != 0:
        raise ValueError("These protocols require balanced test groups for equal item weights")
    averaged = np.vstack([np.bincount(index,weights=metrics[:,i])/counts
                          for i in range(metrics.shape[1])]).T
    return groups, averaged


def bootstrap(group_scores, seed):
    # Resampling item groups only, conditional on these 16 fitted procedures.
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(100):
        ix = rng.integers(len(group_scores),size=(100,len(group_scores)))
        draws.extend(group_scores[ix].mean(axis=1))
    return np.quantile(draws,[.025,.975],axis=0)


def run(dataset, data, output, draw_plan):
    x = encode_features(data.X, data.train)
    counts = data.counts[data.test]
    truth = counts/counts.sum(axis=1,keepdims=True)
    test_x = x[data.test]
    summary, differences, starts, regs = [], [], [], []
    diag = Counter()
    for budget in BUDGETS[dataset]:
        by_method = {k:[] for k in (*OBJECTIVES, "score_then_calibrate")}
        for rep in range(16):
            order, depth_counts = draw_plan(data, rep, 1)
            chosen = order[:budget]
            h = np.asarray(depth_counts[:budget],dtype=float)
            assert h.shape == (budget,5) and np.all(h.sum(axis=1)==1)
            xb = x[chosen]
            tail = budget - budget//4
            train_x, train_y = xb[:tail],h[:tail]
            val_x, val_y = xb[tail:],h[tail:]
            for objective in OBJECTIVES:
                val = []
                for penalty in LAMBDAS:
                    prior, tries, selected = fit_softmax(train_x,train_y,penalty,objective)
                    track(diag,dataset,budget,objective,"validation",tries)
                    prob = predict_softmax(prior, tries[selected][1], val_x)
                    val.append(scores(prob,val_y)[:,0].mean())
                chosen_lambda = LAMBDAS[winner(val)]
                prior, tries, selected = fit_softmax(xb,h,chosen_lambda,objective)
                track(diag,dataset,budget,objective,"refit",tries)
                prob = predict_softmax(prior, tries[selected][1], test_x)
                per_item = scores(prob,truth)
                by_method[objective].append(per_item)
                alternate = scores(predict_softmax(prior,tries[1-selected][1],test_x),truth)
                groups, gap = summarize_groups(data, alternate-per_item)
                train_probs = predict_softmax(prior,tries[selected][1],xb)
                starts.append(dict(dataset=dataset,budget=budget,draw=rep,objective=objective,
                    penalty=chosen_lambda,selected_start=1 if selected==0 else -1,
                    selected_status=tries[selected][2],other_status=tries[1-selected][2],
                    selected_iterations=tries[selected][3],other_iterations=tries[1-selected][3],
                    penalized_train_objective=tries[selected][0],
                    other_minus_selected_train_objective=tries[1-selected][0]-tries[selected][0],
                    selected_train_mean_span=np.ptp(train_probs@GRADE),
                    other_minus_selected_test_rps=gap[:,0].mean()))
            models = [fit_score_then_calibrate(train_x,train_y,lam) for lam in LAMBDAS]
            val = [scores(model(val_x),val_y)[:,0].mean() for model in models]
            lam = LAMBDAS[winner(val)]
            regs.append(dict(dataset=dataset,budget=budget,draw=rep,penalty=lam))
            forecast = fit_score_then_calibrate(xb,h,lam)(test_x)
            by_method["score_then_calibrate"].append(scores(forecast,truth))
        group_scores = {}
        for method, values in by_method.items():
            _, group_scores[method] = summarize_groups(data,np.mean(values,axis=0))
            summary.append(dict(dataset=dataset,budget=budget,method=method,training_draws=16,
                test_groups=len(group_scores[method]),test_items=len(test_x),
                **dict(zip(METRICS,group_scores[method].mean(axis=0)))))
        for ref in ("pearson","score_then_calibrate"):
            delta = group_scores[ref]-group_scores["ce"]
            lo,hi = bootstrap(delta,20261123)
            for i,metric in enumerate(METRICS):
                differences.append(dict(dataset=dataset,budget=budget,reference=ref,
                    subtract="ce",metric=metric,difference=delta.mean(axis=0)[i],
                    lower95=lo[i],upper95=hi[i],test_groups=len(delta)))
        print(dataset,budget,"mean test RPS", {k:float(v[:,0].mean()) for k,v in group_scores.items()},flush=True)
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in (("scores",summary),("contrasts",differences),("starts",starts),("ridge_baseline",regs)):
        write_rows(output / (dataset+"_"+name+".csv"),rows)
    write_rows(output/(dataset+"_convergence.csv"), [dict(dataset=ds,budget=b,objective=o,stage=s,
              status=k,count=n) for (ds,b,o,s,k),n in sorted(diag.items())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",choices=tuple(BUDGETS))
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    sys.path.insert(0,str(ROOT.parent))
    from design import load, draw
    run(args.dataset,load(args.archive),args.output,draw)
