"""Optimizer check for the experiment 13 / 06 toy (added 11-13-2026): convergence, larger budget, random restarts.

For each (scenario, n, seed) the exact experiment 13 data are regenerated and every objective is fitted under three
conditions, all with the same judge family, ridge (1/n)||W||^2 and objective code as experiments 13 and 06:

  orig  original start (training histogram + 0.01 seeded noise), maxiter 400 / maxfun 600 -- reproduces 13/06, now logged
  long  original start, maxiter 2000 / maxfun 4000
  best  best of 5 starts by the TRAINING objective value (original start + 4 random starts with N(0, 0.5^2) added to
        every weight), each with maxiter 2000 / maxfun 4000. Test data never enter the choice.

Objectives: ce (convex reference), rps, pearson, pearson_head (CE head on the chosen Pearson judge's score),
spearman, kendall, pearson_spearman. Logged per fit: training objective value, scipy success flag, iterations,
whether the iteration or evaluation budget was hit, and held-out test metrics.
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026", "scripts")))
import rank_objectives as ro  # noqa: E402  (also puts experiment 13's toy module on the path)

toy = ro.toy
OBJECTIVES = ["ce", "rps", "pearson", "spearman", "kendall", "pearson_spearman"]
SHORT, LONG = (400, 600), (2000, 4000)
N_STARTS, START_SD = 5, 0.5


def objective_and_grad(Phi, P, kind, n, c=toy.RIDGE_C):
    """Return f(w) -> (objective + ridge, gradient), identical to toy.fit / ro.fit_rank for each kind."""
    D = Phi.shape[1]
    lam = c / n
    h = P @ toy.GRID
    h_rank = ro.rankdata(h)

    def f(w):
        W = w.reshape(D, 6)
        Z = Phi @ W
        if kind == "ce":
            L, G = toy.ce_loss_grad(Z, P); obj = L.mean(); dZ = G / len(P)
        elif kind == "rps":
            L, G = toy.rps_loss_grad(Z, P); obj = L.mean(); dZ = G / len(P)
        elif kind == "pearson":
            Q = softmax(Z, 1); mu = Q @ toy.GRID
            a, b = mu - mu.mean(), h - h.mean(); na, nb = np.linalg.norm(a) + 1e-12, np.linalg.norm(b) + 1e-12
            r = a @ b / (na * nb); obj = -r
            dmu = -(b / nb - r * a / na) / na
            dZ = Q * (toy.GRID[None] - mu[:, None]) * dmu[:, None]
        else:
            Q = softmax(Z, 1); m = Q @ toy.GRID
            val, gm = ro.objective(kind, m, h, h_rank); obj = -val
            dZ = Q * (toy.GRID[None] - m[:, None]) * (-gm)[:, None]
        return obj + lam * (W ** 2).sum(), (Phi.T @ dZ + 2 * lam * W).ravel()
    return f


def fit_logged(Phi, P, kind, n, W0, budget):
    f = objective_and_grad(Phi, P, kind, n)
    res = minimize(f, W0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": budget[0], "maxfun": budget[1]})
    hit = res.nit >= budget[0] or res.nfev >= budget[1]
    return res.x.reshape(W0.shape), dict(train_obj=float(res.fun), success=bool(res.success), nit=int(res.nit),
                                         nfev=int(res.nfev), budget_hit=bool(hit))


def test_metrics(Q, Pt, xt):
    m = toy.metrics(Q, Pt, xt)
    return {k: float(m[k]) for k in ("rps_excess", "ce_excess", "pearson", "spearman")}


def head_metrics(Wp, Phi, P, Phit, Pt, xt, n, seed, budget):
    """Calibrated CE head on the Pearson-trained judge's score, exactly as in experiment 13."""
    s_tr = softmax(Phi @ Wp, 1) @ toy.GRID; s_te = softmax(Phit @ Wp, 1) @ toy.GRID
    mu, sd = s_tr.mean(), s_tr.std() + 1e-9
    H = np.stack([np.ones(n), (s_tr - mu) / sd], 1); Ht = np.stack([np.ones(len(xt)), (s_te - mu) / sd], 1)
    Wh, log = fit_logged(H, P, "ce", n, toy.marginal_W0(H, P, seed), budget)
    return dict(log, **test_metrics(softmax(Ht @ Wh, 1), Pt, xt))


def run_one(args):
    scenario, n, seed, k, ntest = args
    rng = np.random.default_rng(1000003 * seed + 7919 * n + (scenario == "B"))
    x = rng.uniform(-1, 1, n)
    cnt = np.array([rng.multinomial(k, p) for p in toy.truth(x, scenario)])
    P = cnt / k
    Phi = toy.phi(x)
    W0 = toy.marginal_W0(Phi, P, seed)
    xt = rng.uniform(-1, 1, ntest)
    Pt = toy.truth(xt, scenario)
    Phit = toy.phi(xt)
    starts = [W0] + [W0 + START_SD * np.random.default_rng([seed, n, r, scenario == "B"]).normal(size=W0.shape)
                     for r in range(1, N_STARTS)]
    out = {}
    for kind in OBJECTIVES:
        rec = {}
        W, log = fit_logged(Phi, P, kind, n, W0, SHORT)
        rec["orig"] = dict(log, **test_metrics(softmax(Phit @ W, 1), Pt, xt))
        fits = []
        for r, S in enumerate(starts if kind != "ce" else starts[:1]):
            Wr, logr = fit_logged(Phi, P, kind, n, S, LONG)
            fits.append((logr["train_obj"], r, Wr, dict(logr, **test_metrics(softmax(Phit @ Wr, 1), Pt, xt))))
        rec["long"] = fits[0][3]
        best = min(fits, key=lambda t: t[0])
        rec["best"] = dict(best[3], start=best[1])
        rec["restart_train_obj"] = [t[0] for t in fits]
        out[kind] = rec
        if kind == "pearson":
            out["pearson_head"] = {"orig": head_metrics(W, Phi, P, Phit, Pt, xt, n, seed, SHORT),
                                   "long": head_metrics(fits[0][2], Phi, P, Phit, Pt, xt, n, seed, LONG),
                                   "best": head_metrics(best[2], Phi, P, Phit, Pt, xt, n, seed, LONG)}
    return scenario, n, seed, out
