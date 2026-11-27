"""QA check: reproduce POPQUORN cells, Pearson-head penalty effect, matched calibration control."""
import os, sys
EXP = sys.argv[1]
sys.path.insert(0, EXP)
import numpy as np
from scipy.optimize import minimize
import data_io, fit_eval
from run_all import prepare
from metrics import rps_one, ce_one, softmax

def per_item_scores(prob, ratings, idx):
    rps, ce = [], []
    for row, i in enumerate(idx):
        y = np.asarray(ratings[i], dtype=int)
        rps.append(np.mean([rps_one(prob[row], int(t)) for t in y]))
        ce.append(np.mean([ce_one(prob[row], int(t)) for t in y]))
    return np.array(rps), np.array(ce)

def pearson_sampled_train(pack, x, ratings, idx, values):
    prob, mu = fit_eval.predict_distributions(pack, x, pack['objective'])
    # mean-score vs individual ratings pooled correlation on train
    s, yv = [], []
    for row, i in enumerate(idx):
        for t in ratings[i]:
            s.append(mu[row]); yv.append(values[t])
    return float(np.corrcoef(s, yv)[0, 1]), float(np.mean(-np.sum(prob*np.log(np.clip(prob,1e-12,1)),axis=1)))

def calibrate(mu_val, ratings, val_idx, K):
    # multinomial logistic on [mu, 1], fit by per-rating CE on VALIDATION ratings only
    X = np.stack([mu_val, np.ones_like(mu_val)], axis=1)
    Y = [np.bincount(np.asarray(ratings[i], int), minlength=K) / len(ratings[i]) for i in val_idx]
    Y = np.array(Y)
    def loss(w):
        p = softmax(X @ w.reshape(2, K))
        return -np.mean(np.sum(Y * np.log(np.clip(p, 1e-12, 1)), axis=1))
    r = minimize(loss, np.zeros(2 * K), method='L-BFGS-B')
    return r.x.reshape(2, K)

def apply_cal(w, mu):
    return softmax(np.stack([mu, np.ones_like(mu)], axis=1) @ w)

for task in sys.argv[2:]:
    data = data_io.load_popquorn(task)
    prep = prepare(data)
    K = data.n_classes
    vals = data.class_values
    packs = {o: fit_eval.fit_objective(prep['x_train'], data.ratings, prep['train'], vals, o, None, 0) for o in ('ce', 'pearson')}
    print(f'== {task}: n_train={prep["train"].size} n_val={prep["val"].size} n_test={prep["test"].size}')
    res = {}
    for o, pack in packs.items():
        prob, mu = fit_eval.predict_distributions(pack, prep['x_test'], o)
        rps, ce = per_item_scores(prob, data.ratings, prep['test'])
        res[o] = (rps / (K - 1), ce)
        tr_r, tr_H = pearson_sampled_train(pack, prep['x_train'], data.ratings, prep['train'], vals)
        print(f'  {o:8s} test normRPS={np.mean(rps)/(K-1):.4f} CE={np.mean(ce):.4f} | train corr(mean score, rating)={tr_r:.4f} train mean entropy={tr_H:.3f} nit={pack["nit"]} success={pack["success"]}')
    # Pearson head without the L2 penalty
    orig = fit_eval._penalty
    fit_eval._penalty = lambda w, d, k, l2=0.0: 0.0
    try:
        p0 = fit_eval.fit_objective(prep['x_train'], data.ratings, prep['train'], vals, 'pearson', None, 0)
    finally:
        fit_eval._penalty = orig
    tr_r0, tr_H0 = pearson_sampled_train(p0, prep['x_train'], data.ratings, prep['train'], vals)
    prob0, mu0 = fit_eval.predict_distributions(p0, prep['x_test'], 'pearson')
    rps0, ce0 = per_item_scores(prob0, data.ratings, prep['test'])
    print(f'  pearson (L2=0) test normRPS={np.mean(rps0)/(K-1):.4f} CE={np.mean(ce0):.4f} | train corr={tr_r0:.4f} entropy={tr_H0:.3f} nit={p0["nit"]}')
    # matched calibration on VALIDATION: scalar mean -> distribution, same map family for both heads
    for o, pack in (('ce', packs['ce']), ('pearson', packs['pearson']), ('pearson_L2=0', p0)):
        _, mu_val = fit_eval.predict_distributions(pack, prep['x_val'], 'pearson' if o.startswith('pearson') else 'ce')
        _, mu_test = fit_eval.predict_distributions(pack, prep['x_test'], 'pearson' if o.startswith('pearson') else 'ce')
        w = calibrate(mu_val, data.ratings, prep['val'], K)
        q = apply_cal(w, mu_test)
        rps_c, ce_c = per_item_scores(q, data.ratings, prep['test'])
        res[o + '_cal'] = (rps_c / (K - 1), ce_c)
        print(f'  {o:14s} + val-calibrated scalar map: test normRPS={np.mean(rps_c)/(K-1):.4f} CE={np.mean(ce_c):.4f}')
    # paired item bootstrap of differences (normRPS), B=2000
    rng = np.random.default_rng(123)
    n = prep['test'].size
    draws = rng.integers(0, n, size=(2000, n))
    def paired(a, b):
        d = res[a][0] - res[b][0]
        bs = d[draws].mean(axis=1)
        return f'{d.mean():+.4f} [{np.quantile(bs,0.025):+.4f}, {np.quantile(bs,0.975):+.4f}]'
    print('  paired normRPS diff (first minus second; positive = second better), item bootstrap B=2000:')
    print('    pearson raw  - ce raw      :', paired('pearson', 'ce'))
    print('    pearson_cal  - ce raw      :', paired('pearson_cal', 'ce'))
    print('    pearson_cal  - ce_cal      :', paired('pearson_cal', 'ce_cal'))
    print('    pearson_L2=0_cal - ce raw  :', paired('pearson_L2=0_cal', 'ce'))
