"""Exp 18, step 3A: when does CE beat correlation? Factorial toy sweep over data amount n, ratings per item k
(rater noise), judge flexibility J (D = 2J+1 features), target scenario (A-D of experiments 08/11) and objective.
Reuses experiment 13's toy (basis, generator, Pearson/RPS fits, metrics) and experiment 11's exact CE / MSE fits.
Written 11-21-2026. CPU only, no model calls."""
import os, sys
import numpy as np
from scipy.special import softmax

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(EXP, "archive", "13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026", "scripts"))
sys.path.insert(0, os.path.join(EXP, "11_toy_mse_and_tuned_head_calibration_vs_ce_start_17Nov2026", "scripts"))
import toy                       # noqa: E402
import mse_and_tuned_head as e11  # noqa: E402

CODE = e11.CODE
C_GRID = [0.01, 0.3, 10.0, 1000.0]          # ridge multipliers, lam = c/n; 1000 is near the constant forecast
CB_GRID = [0.01, 1.0, 100.0, 1e8]           # Stage-2 slope ridge, 1e8 = constant
FOLDS = 3
ARMS = ["constant", "ce_tuned", "rps_tuned", "mse_tuned", "pearson_raw", "pearson_head_tuned"]


def fit(Phi, P, kind, n, seed, c):
    W0 = toy.marginal_W0(Phi, P, seed)
    if kind == "ce":
        return e11.fit_weights(Phi, P, "ce_exact", n, W0, c=c)
    if kind == "mse":
        return e11.fit_weights(Phi, P, "mse", n, W0, c=c)
    if kind == "rps":
        return toy.fit(Phi, P, "rps", n, W0, c=c)
    return toy.fit(Phi, P, "pearson", n, W0, c=c)


def cv_pick(Phi, P, kind, fold, seed, score):
    """Pick the ridge c on the training tasks only: FOLDS-fold out-of-fold score, lower is better."""
    scs = []
    for c in C_GRID:
        Q = np.zeros_like(P); Z = np.zeros(len(P))
        for f in range(FOLDS):
            tr, va = fold != f, fold == f
            W = fit(Phi[tr], P[tr], kind, int(tr.sum()), seed, c)
            Qf = softmax(Phi[va] @ W, 1); Q[va] = Qf
        scs.append(score(Q, P))
    return C_GRID[int(np.argmin(scs))], scs


def rps_pooled(Q, P):
    return e11.pooled_rps(Q, P)


def neg_pearson(Q, P):
    a, b = Q @ toy.GRID, P @ toy.GRID
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 1.0
    return -float(np.corrcoef(a, b)[0, 1])


def run_one(args):
    sc, n, seed, k, J, ntest = args
    rng = np.random.default_rng(1000003 * seed + 7919 * n + 131 * k + 17 * J + CODE[sc])
    x = rng.uniform(-1, 1, n)
    Ptr = e11.truth(x, sc)
    P = np.array([rng.multinomial(k, p) for p in Ptr]) / k
    Phi = toy.phi(x, J)
    xt = rng.uniform(-1, 1, ntest); Pt = e11.truth(xt, sc); Phit = toy.phi(xt, J)
    fold = np.empty(n, int); fold[np.random.default_rng([seed, n, CODE[sc], k, J, 99]).permutation(n)] = np.arange(n) % FOLDS
    out = {}
    Qc = np.tile(P.mean(0), (ntest, 1)); out["constant"] = toy.metrics(Qc, Pt, xt)
    chosen = {}
    for arm, kind in [("ce_tuned", "ce"), ("rps_tuned", "rps"), ("mse_tuned", "mse")]:
        c, _ = cv_pick(Phi, P, kind, fold, seed, rps_pooled)
        W = fit(Phi, P, kind, n, seed, c); out[arm] = toy.metrics(softmax(Phit @ W, 1), Pt, xt); chosen[arm] = c
    # Pearson raw (the paper's fixed c = 1, scale-free output as is)
    Wp = fit(Phi, P, "pearson", n, seed, 1.0); out["pearson_raw"] = toy.metrics(softmax(Phit @ Wp, 1), Pt, xt)
    # Pearson selected on correlation (ridge by out-of-fold -Pearson), then a calibrated affine head (Sanmi's score-then-calibrate)
    cp, _ = cv_pick(Phi, P, "pearson", fold, seed, neg_pearson); chosen["pearson_stage1"] = cp
    z = np.zeros(n)
    for f in range(FOLDS):
        tr, va = fold != f, fold == f
        W = fit(Phi[tr], P[tr], "pearson", int(tr.sum()), seed, cp)
        s_tr = softmax(Phi[tr] @ W, 1) @ toy.GRID; s_va = softmax(Phi[va] @ W, 1) @ toy.GRID
        z[va] = (s_va - s_tr.mean()) / (s_tr.std() + 1e-9)
    # pick the head ridge by out-of-fold pooled RPS on the OOF scores
    best, bsc = None, 9
    for cb in CB_GRID:
        Q = np.zeros_like(P)
        for f in range(FOLDS):
            tr, va = fold != f, fold == f
            a, b = e11.fit_stage2_head(z[tr], P[tr], cb); Q[va] = e11.predict_head(z[va], a, b)
        s = rps_pooled(Q, P)
        if s < bsc: bsc, best = s, cb
    chosen["head_cb"] = best
    Wf = fit(Phi, P, "pearson", n, seed, cp)
    s_tr = softmax(Phi @ Wf, 1) @ toy.GRID; s_te = softmax(Phit @ Wf, 1) @ toy.GRID
    sd = s_tr.std() + 1e-9
    a, b = e11.fit_stage2_head(z, P, best)          # head fitted on the out-of-fold scores
    out["pearson_head_tuned"] = toy.metrics(e11.predict_head((s_te - s_tr.mean()) / sd, a, b), Pt, xt)
    pc = np.cumsum(Pt, 1)[:, :5]
    out["true_rps"] = float((pc * (1 - pc)).mean()); out["chosen"] = chosen
    return sc, n, seed, k, J, out
