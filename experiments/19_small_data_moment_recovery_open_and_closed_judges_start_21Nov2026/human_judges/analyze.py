"""Retrospective calibration of observed native judge probabilities; publish aggregates only."""
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent.parent
HISTORICAL, = EXPERIMENTS.glob("18_when*/conditional_shape_and_rating_budget/historical_rating_budget")
# Import the frozen split generator under a distinct name from this file.
import importlib.util
spec = importlib.util.spec_from_file_location("exp18_human", HISTORICAL / "analyze.py")
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)

BUDGETS = history.BUDGETS
R = np.arange(6)
Z = R / 5
POWERS = Z[:, None] ** np.arange(1, 11)
RPS, CE, M1, M2, M3, M4, M5, VAR = range(8)
LOSS = ("rps", "ce", "m1", "m2", "m3", "m4", "m5", "variance")
METRICS = LOSS + ("tvd", "mae", "bias", "forecast_mean", "human_mean", "pearson", "spearman")
METHODS = ("ce", "pearson", "pearson_tie_rps", "identity", "constant")
COEFS = sorted(itertools.product((0.5, 1, 2), (-1, 0, 1), (-1, 0, 1)),
               key=lambda point: (np.log2(point[0])**2 + point[1]**2 + point[2]**2, point))
assert COEFS[0] == (1, 0, 0)
BOOT = 1000
EPS = 1e-6
TOL = 1e-12


def load():
    _, items, ratings, _ = history.load_data()
    ids = [row["item"] for row in items]
    manifest = json.loads((ROOT / "models.json").read_text())
    open_data = history.load("09*/results/open_weight_forecasts.json", manifest["open_weight_archive_sha256"])
    closed_data = history.load("10*/results/judge_forecasts.json", manifest["closed_weight_archive_sha256"])
    order = open_data["artifact_order"]
    assert len(set(ids)) == len(set(order)) == len(ids) == len(order) == 75 and set(ids) == set(order)
    keys, arrays, fallbacks = [], [], {}
    closed_specs = {row["judge_id"]: row for row in closed_data["arm_a_specs"]}
    for model in manifest["models"]:
        name, forecast = model["key"], model["forecast_key"]
        if model["weight_access"] == "open":
            raw = np.asarray(open_data["verbal_forecasts"][forecast])[[order.index(i) for i in ids]]
            stats = open_data["parse_stats"][forecast]
            fallbacks[name] = stats["n_scheduled"] - stats["verbal_ok"]
        else:
            assert closed_specs[forecast]["model"] == model["model"]
            records = {r["item"]: r for r in closed_data["native_prob_forecasts"][forecast]}
            assert set(records) == set(ids)
            raw = np.array([records[i]["probabilities"] for i in ids])
            fallbacks[name] = sum(records[i]["fallback_uniform_used"] for i in ids)
        assert raw.shape == (75, 6) and np.isfinite(raw).all() and raw.min() >= 0
        np.testing.assert_allclose(raw.sum(1), 1, rtol=0, atol=3.1e-8)
        arrays.append(raw / raw.sum(1, keepdims=True))
        keys.append(name)
    arrays.extend([np.mean(arrays[:2], axis=0), np.mean(arrays[2:4], axis=0)])
    keys.extend(["open_pair", "closed_pair"])
    base = (1 - 6*EPS) * np.stack(arrays, axis=1) + EPS
    return keys, base, items, ratings, fallbacks


def grid(p):
    u = (R-2.5)/2.5
    logp = np.log(p)
    logits = np.stack([t*logp + b*u + c*u**2 for t,b,c in COEFS], axis=2)
    q = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return q / q.sum(axis=-1, keepdims=True)


def target(ratings):
    h = history.histograms(ratings)
    moments = h @ POWERS
    variance = np.array([np.var(y/5, ddof=1) if len(y) > 1 else np.nan for y in ratings])
    return h, moments, variance


def loss_panel(q, h, target_moments, sample_variance):
    """Return artifact × input × coefficient × loss without pooling heterogeneous artifacts."""
    truth = h[:,None,None,:]
    cdf = q.cumsum(axis=-1)[..., :-1]
    real = truth.cumsum(axis=-1)[..., :-1]
    rps = (cdf**2 + (1-2*cdf)*real).mean(axis=-1)
    ce = -(np.log(q)*truth).sum(axis=-1)
    moments = q @ POWERS[:, :5]
    m = target_moments[:, None, None]
    losses = moments**2 - 2*moments*m[..., :5] + m[..., [1,3,5,7,9]]
    var = (moments[...,1] - moments[...,0]**2 - sample_variance[:,None,None])**2
    return np.concatenate([rps[...,None], ce[...,None], losses, var[...,None]], axis=-1)


def avg(w, array):
    """For each row of normalized family weights reduce the first (artifact) axis."""
    return (w @ array.reshape(len(array), -1)).reshape((len(w), *array.shape[1:]))


def select(w, mu, y, ce, rps):
    loss = avg(w, ce)
    c = loss <= loss.min(-1, keepdims=True) + TOL
    spread = np.ptp(mu, axis=0)
    good = (spread > TOL) & (np.ptp(y) > TOL)
    # Center and scale before raw weighted products. Nearly constant means can
    # otherwise split genuine Pearson ties through cancellation of E[m²]−E[m]².
    mu = (mu - mu.mean(axis=0)) / np.where(spread > TOL, spread, 1)
    y = y - y.mean()
    mbar = avg(w, mu)
    ybar = w @ y
    cov = avg(w, mu*y[:,None,None]) - mbar*ybar[:,None,None]
    var_m = np.maximum(avg(w, mu**2) - mbar**2, 0)
    var_y = np.maximum((w @ (y*y)) - ybar**2, 0)
    den = np.sqrt(var_m*var_y[:,None,None])
    if np.any(good) and not (den[:,good] > 0).all():
        raise ArithmeticError("Unresolved weighted Pearson variance")
    corr = np.divide(cov, den, out=np.full_like(cov, -np.inf), where=good[None] & (den>0))
    p = corr >= (corr.max(-1,keepdims=True) - TOL)
    fallback = ~good.any(-1)
    # Required fallback for completely undefined Pearson, for both tie policies.
    p[:,fallback,:] = False
    p[:,fallback,0] = True
    tr_rps = np.where(p, avg(w, rps), np.inf)
    p_rps = tr_rps <= tr_rps.min(-1,keepdims=True) + TOL
    return np.stack([c.argmax(-1), p.argmax(-1), p_rps.argmax(-1)], axis=-1), fallback, p[0].sum(-1)


def centered_corr(m, y):
    x, t = m-m.mean(axis=0), y-y.mean()
    den = np.linalg.norm(x, axis=0)*np.linalg.norm(t)
    return np.divide(np.sum(x*t[:,None,None],axis=0),den,out=np.full_like(den,np.nan),where=den>1e-12)


def rank_keys(m):
    # Merge adjacent levels within tolerance, including representation-level ties.
    order = np.argsort(m, axis=0)
    ordered = np.take_along_axis(m, order, axis=0)
    level = np.cumsum(np.diff(ordered,axis=0,prepend=ordered[:1]) > TOL, axis=0)
    keys = np.empty_like(level)
    np.put_along_axis(keys,order,level,axis=0)
    return rankdata(keys,axis=0)


def extra_panel(q, h, y):
    mu = q @ R
    delta = mu-y[:,None,None]
    tvd = np.abs(q-h[:,None,None,:]).sum(-1).mean(0)/2
    values = [tvd, np.abs(delta).mean(0), delta.mean(0), mu.mean(0),
              np.full_like(tvd,y.mean()), centered_corr(mu, y),
              centered_corr(rank_keys(mu), rankdata(y))]
    return np.stack(values,axis=-1)


def constant_losses(p, weights, h, m, var):
    # Each draw's constant forecast scores against the weighted test distribution.
    cdf = p.cumsum(-1)[:, :-1]
    cdf_h = (weights@h).cumsum(-1)[:, :-1]
    moment = p@POWERS[:,:5]
    truth = weights@m
    delta = moment**2 - 2*moment*truth[:,:5] + truth[:,[1,3,5,7,9]]
    v = moment[:,1] - moment[:,0]**2
    rows = [(cdf**2 + (1-2*cdf)*cdf_h).mean(-1), -(np.log(p)*(weights@h)).sum(-1),
            *delta.T, v*v-2*v*(weights@var) + weights@(var*var)]
    return np.stack(rows,axis=-1)


def write_csv(name, rows):
    with (ROOT / name).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run():
    keys, base, items, ys, fallbacks = load()
    families = np.array([i["family"] for i in items])
    _, fid = np.unique(families,return_inverse=True)
    q = grid(base)
    mu = q@R
    full, mom, var = target(ys)
    loss = loss_panel(q,full,mom,var)
    human = np.array([a.mean() for a in ys])
    rng = np.random.default_rng(20261123)
    w_all = np.vstack([np.ones(19),rng.exponential(size=(BOOT,19))])
    group = np.arange(len(keys))
    totals = {b:np.zeros((BOOT+1,len(keys),len(METHODS),len(LOSS))) for b in BUDGETS}
    values = {b:[] for b in BUDGETS}
    undefined = Counter()
    tied = Counter()
    choice_counts = Counter()
    counts = {b:[] for b in BUDGETS}
    for split,(ordered,heldout) in enumerate(history.splits(items)):
        te = np.flatnonzero(np.isin(families,list(heldout)))
        wt = w_all[:,fid[te]]
        wt = wt / wt.sum(1,keepdims=True)
        test = avg(wt,loss[te])
        extra = extra_panel(q[te],full[te],human[te])
        draw = np.random.default_rng(np.random.SeedSequence([20261121,190,split]))
        shuffled = [draw.permutation(y) for y in ys]
        for k in (1,2):
            trh, trm, trv = target([y[:k] for y in shuffled])
            train_loss = loss_panel(q,trh,trm,trv)
            for n in (3,7,15):
                b = n,k
                tr = np.flatnonzero(np.isin(families,ordered[:n]))
                assert not np.intersect1d(tr,te).size
                w = w_all[:,fid[tr]]
                w = w/w.sum(1,keepdims=True)
                chosen, fallback, ties = select(w,mu[tr],trh[tr]@R,
                                                train_loss[tr,...,CE],train_loss[tr,...,RPS])
                for g,key in enumerate(keys):
                    undefined[n,k,key] += int(fallback[g])
                    tied[n,k,key] += int(ties[g] > 1)
                    for j,name in enumerate(METHODS[:3]):
                        choice_counts[n,k,key,name,chosen[0,g,j]] += 1
                marginal = (w@trh[tr]) * (k*len(tr)) + 0.5
                marginal /= marginal.sum(1,keepdims=True)
                const = constant_losses(marginal,wt,full[te],mom[te],var[te])
                const_q = np.broadcast_to(marginal[0],(len(te),1,1,6))
                const_extra = extra_panel(const_q,full[te],human[te])[0,0]
                const_extra[-2:] = np.nan
                select_all = np.concatenate([chosen,np.zeros_like(chosen[:,:,:1])],axis=-1)
                fitted = test[np.arange(len(w))[:,None,None],group[None,:,None],select_all]
                const_mat = np.broadcast_to(const[:,None,None],(len(w),len(keys),1,len(LOSS)))
                result = np.concatenate([fitted,const_mat],axis=2)
                totals[b] += result
                extra_fit = extra[group[:,None],select_all[0]]
                extra_const = np.broadcast_to(const_extra[None,None],(len(keys),1,len(const_extra)))
                values[b].append(np.concatenate([result[0],np.concatenate([extra_fit,extra_const],axis=1)],axis=-1))
                counts[b].append([len(tr),len(tr)*k,len(te),sum(map(len,(ys[i] for i in te)))])
    scores, contrasts, accounting = [],[],[]
    for n,k in BUDGETS:
        cube=np.array(values[n,k])
        weighted=totals[n,k]/len(cube)
        np.testing.assert_allclose(cube[...,:len(LOSS)].mean(0),weighted[0],atol=1e-12)
        for g,key in enumerate(keys):
            for j,name in enumerate(METHODS):
                for m,metric in enumerate(METRICS):
                    x=cube[:,g,j,m]
                    valid=np.isfinite(x)
                    scores.append(dict(input=key,n=n,k=k,method=name,metric=metric,
                                       mean=float(x[valid].mean()) if valid.any() else "",valid_splits=int(valid.sum())))
            for a,c in ((1,0),(2,0),(0,4),(1,4),(0,3),(1,3)):
                for m,metric in enumerate(LOSS):
                    d = weighted[:,g,a,m]-weighted[:,g,c,m]
                    lo,hi=np.quantile(d[1:],[.025,.975])
                    contrasts.append(dict(input=key,n=n,k=k,method=METHODS[a],subtract=METHODS[c],
                                          metric=metric,difference=d[0],lower_95=lo,upper_95=hi,draws=BOOT))
            num=np.array(counts[n,k])
            accounting.append(dict(input=key,n=n,k=k,split_count=len(num),
                mean_train_artifacts=num[:,0].mean(),mean_train_ratings=num[:,1].mean(),
                mean_test_artifacts=num[:,2].mean(),mean_test_ratings=num[:,3].mean(),
                min_train_ratings=num[:,1].min(),max_train_ratings=num[:,1].max(),
                min_test_ratings=num[:,3].min(),max_test_ratings=num[:,3].max(),
                undefined_train_pearson=undefined[n,k,key],tied_train_pearson=tied[n,k,key]))
    choices=[dict(n=n,k=k,input=key,method=name,t=COEFS[i][0],b=COEFS[i][1],c=COEFS[i][2],times_selected=v)
             for (n,k,key,name,i),v in sorted(choice_counts.items())]
    for f,rows in (("scores.csv",scores),("contrasts.csv",contrasts),("accounting.csv",accounting),("choices.csv",choices)):
        write_csv(f,rows)
    print(json.dumps(dict(fallback_rows=fallbacks,groups=keys,splits=250,budgets=len(BUDGETS),
                          bootstrap_draws=BOOT,fit_grid=len(COEFS),score_rows=len(scores),contrast_rows=len(contrasts))))


if __name__=='__main__':
    run()
