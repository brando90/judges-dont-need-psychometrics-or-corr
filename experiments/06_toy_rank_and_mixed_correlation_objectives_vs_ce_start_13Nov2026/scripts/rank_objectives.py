"""Rank-correlation and mixed-correlation training objectives for the experiment 13 toy (added 11-13-2026).

Reuses experiment 13's data generator, judge family (softmax of 13 cosine/sine features), ridge (c/n)||W||^2,
starting point and L-BFGS-B settings unchanged, imported from its frozen scripts/toy.py. Only the objective is new:

  spearman          maximize soft Spearman: Pearson correlation between soft ranks of the judge's expected ratings
                    u_i = sum_j sigmoid((m_i - m_j)/TAU) and the (average-tie) ranks of the training-task human means
  kendall           maximize soft Kendall tau-a: (2/(n(n-1))) sum_{i<j} tanh((m_i - m_j)/TAU) sign(h_i - h_j)
  pearson_spearman  maximize 0.5 * Pearson(m, h) + 0.5 * soft Spearman

Rank correlations are not differentiable, so training uses these standard smooth surrogates with TAU = 0.1 rating
points (fixed before measurement). Evaluation uses the same held-out metrics as experiment 13.
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax
from scipy.stats import kendalltau, rankdata

EXP13 = os.path.join(os.path.dirname(__file__), "..", "..", "archive",
                     "13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026", "scripts")
sys.path.insert(0, os.path.abspath(EXP13))
import toy  # noqa: E402  (frozen experiment 13 module)

TAU = 0.1
NEW_FITS = ["spearman", "kendall", "pearson_spearman"]


def pearson_grad(a_raw, b_raw):
    """corr(a, b) and d corr / d a."""
    a, b = a_raw - a_raw.mean(), b_raw - b_raw.mean()
    na, nb = np.linalg.norm(a) + 1e-12, np.linalg.norm(b) + 1e-12
    r = a @ b / (na * nb)
    return r, (b / nb - r * a / na) / na


def soft_spearman_grad(m, h_rank):
    """Soft Spearman between expected ratings m and fixed target ranks; returns value and gradient wrt m."""
    D = (m[:, None] - m[None, :]) / TAU
    sig = expit(D)
    u = sig.sum(1)
    r, gu = pearson_grad(u, h_rank)
    S = sig * (1 - sig) / TAU                      # symmetric; du_i/dm_k = delta_ik sum_j S_ij - S_ik
    return r, gu * S.sum(1) - S @ gu


def soft_kendall_grad(m, h):
    n = len(m)
    sgn = np.sign(h[:, None] - h[None, :])
    T = np.tanh((m[:, None] - m[None, :]) / TAU)
    c = 2.0 / (n * (n - 1))
    value = c * 0.5 * (T * sgn).sum()              # sum over i<j equals half the sum over i != j
    grad = c * ((1 - T ** 2) / TAU * sgn).sum(1)
    return value, grad


def objective(kind, m, h, h_rank):
    """Return (value to maximize, d value / d m)."""
    if kind == "spearman":
        return soft_spearman_grad(m, h_rank)
    if kind == "kendall":
        return soft_kendall_grad(m, h)
    if kind == "pearson_spearman":
        rp, gp = pearson_grad(m, h)
        rs, gs = soft_spearman_grad(m, h_rank)
        return 0.5 * rp + 0.5 * rs, 0.5 * gp + 0.5 * gs
    raise ValueError(kind)


def fit_rank(Phi, P, kind, n, W0, c=toy.RIDGE_C):
    """Same ridge, start and L-BFGS-B settings as toy.fit; only the objective differs."""
    D = Phi.shape[1]
    lam = c / n
    h = P @ toy.GRID
    h_rank = rankdata(h)

    def f(w):
        W = w.reshape(D, 6)
        Q = softmax(Phi @ W, 1)
        m = Q @ toy.GRID
        val, gm = objective(kind, m, h, h_rank)
        dZ = Q * (toy.GRID[None] - m[:, None]) * (-gm)[:, None]
        return -val + lam * (W ** 2).sum(), (Phi.T @ dZ + 2 * lam * W).ravel()

    res = minimize(f, W0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 400, "maxfun": 600})
    return res.x.reshape(D, 6)


def run_one(args):
    """Regenerate experiment 13's exact data for (scenario, n, seed), fit the new objectives, score on the same test tasks."""
    scenario, n, seed, k, ntest, check_ce = args
    rng = np.random.default_rng(1000003 * seed + 7919 * n + (scenario == "B"))
    x = rng.uniform(-1, 1, n)
    Ptr = toy.truth(x, scenario)
    cnt = np.array([rng.multinomial(k, p) for p in Ptr])
    P = cnt / k
    Phi = toy.phi(x)
    W0 = toy.marginal_W0(Phi, P, seed)
    xt = rng.uniform(-1, 1, ntest)
    Pt = toy.truth(xt, scenario)
    Phit = toy.phi(xt)
    out = {}
    for kind in NEW_FITS:
        W = fit_rank(Phi, P, kind, n, W0)
        Q = softmax(Phit @ W, 1)
        met = toy.metrics(Q, Pt, xt)
        mu, hm = Q @ toy.GRID, Pt @ toy.GRID
        met["kendall"] = float(kendalltau(mu, hm)[0]) if mu.std() > 1e-12 else float("nan")
        out[kind] = met
    if check_ce:                                   # reproduce experiment 13's CE fit to confirm identical data
        out["ce_check"] = toy.metrics(softmax(Phit @ toy.fit(Phi, P, "ce", n, W0), 1), Pt, xt)
    return scenario, n, seed, out
