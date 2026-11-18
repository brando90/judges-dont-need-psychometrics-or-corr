"""Experiment 11 core routines: MSE-trained and slope-regularized / cross-fitted Stage-2 calibrated heads.

Reuses Experiment 13's frozen cosine/sine basis, Beta-Binomial data generator, Pearson fit,
and test metrics without altering any earlier experiment file. Uses exact log-sum-exp CE
loss/gradient (no floor clipping during optimization) so Stage-1 CE and Stage-2 heads are strictly convex.
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

EXP13 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "archive",
    "13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026",
    "scripts",
)
sys.path.insert(0, os.path.abspath(EXP13))
import toy  # noqa: E402

CODE = {"A": 0, "B": 1, "C": 2, "D": 3}
KAPPA_CONST = float(np.exp(1.5))
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
C_HEAD_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1e4, 1e8]
FOLDS = 4


def truth(x, sc):
    """True 6-category conditional rating distribution p(r | x) for scenarios A-D."""
    if sc in "AB":
        return toy.truth(x, sc)
    mu = 0.1 + 0.8 * ((x + 1) / 2) ** 2 if sc == "C" else 0.5 + 0.35 * x
    return toy.betabinom(mu, np.full_like(x, KAPPA_CONST))


def ce_exact_obj_grad(w, Phi, P, lam):
    """Exact log-sum-exp mean cross-entropy + lam * ||W||^2 and analytic gradient."""
    D = Phi.shape[1]
    W = w.reshape(D, 6)
    Z = Phi @ W
    logQ = log_softmax(Z, axis=1)
    Q = np.exp(logQ)
    val = float(-(P * logQ).sum(1).mean() + lam * (W ** 2).sum())
    dZ = (Q - P) / len(P)
    grad = (Phi.T @ dZ + 2.0 * lam * W).ravel()
    return val, grad


def mse_obj_grad(w, Phi, P, lam):
    """Mean squared error between predicted expected rating and sample mean + lam * ||W||^2."""
    D = Phi.shape[1]
    W = w.reshape(D, 6)
    Z = Phi @ W
    Q = softmax(Z, axis=1)
    mu = Q @ toy.GRID
    h = P @ toy.GRID
    err = mu - h
    val = float((err ** 2).mean() + lam * (W ** 2).sum())
    dmu = 2.0 * err / len(P)
    dZ = Q * (toy.GRID[None, :] - mu[:, None]) * dmu[:, None]
    grad = (Phi.T @ dZ + 2.0 * lam * W).ravel()
    return val, grad


def fit_weights(Phi, P, kind, n, W0, c=1.0):
    """Fit D x 6 softmax weights with objective in {'ce_exact', 'mse', 'pearson'}."""
    if kind == "pearson":
        return toy.fit(Phi, P, "pearson", n, W0, c=c)
    D = Phi.shape[1]
    lam = float(c) / float(n)
    fn = ce_exact_obj_grad if kind == "ce_exact" else mse_obj_grad
    res = minimize(
        fn,
        W0.ravel(),
        args=(Phi, P, lam),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 400, "maxfun": 600},
    )
    return res.x.reshape(D, 6)


def stage2_obj_grad(params, z, P, log_p_bar, lam_b, lam_a=1e-9):
    """Convex 12-param Stage-2 CE objective: a_r + b_r * z_i with slope penalty lam_b * ||b||^2.

    The tiny gauge penalty lam_a * ||a - log(p_bar)||^2 removes softmax shift indeterminacy
    and guarantees that as lam_b -> inf (b -> 0), Q -> P.mean(0) (exact constant baseline).
    """
    a = params[:6]
    b = params[6:]
    Z = a[None, :] + z[:, None] * b[None, :]
    logQ = log_softmax(Z, axis=1)
    Q = np.exp(logQ)
    da_gauge = a - log_p_bar
    val = float(-(P * logQ).sum(1).mean() + lam_b * (b ** 2).sum() + lam_a * (da_gauge ** 2).sum())
    dZ = (Q - P) / len(P)
    ga = dZ.sum(0) + 2.0 * lam_a * da_gauge
    gb = (dZ * z[:, None]).sum(0) + 2.0 * lam_b * b
    return val, np.concatenate([ga, gb])


def fit_stage2_head(z, P, c_b, n_eff=None):
    """Fit convex 12-parameter (a, b) head by exact CE on scalar features z and targets P."""
    n = len(P) if n_eff is None else n_eff
    p_bar = np.clip(P.mean(0), 1e-6, None)
    p_bar = p_bar / p_bar.sum()
    log_p_bar = np.log(p_bar)
    lam_b = float(c_b) / float(n)
    p0 = np.concatenate([log_p_bar, np.zeros(6)])
    res = minimize(
        stage2_obj_grad,
        p0,
        args=(z, P, log_p_bar, lam_b),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 400, "maxfun": 600, "ftol": 1e-13, "gtol": 1e-9},
    )
    a, b = res.x[:6], res.x[6:]
    return a, b


def predict_head(z, a, b):
    """Evaluate softmax(a_r + b_r * z) safely for scalar array z."""
    return softmax(a[None, :] + z[:, None] * b[None, :], axis=1)


def standardize_pair(s_tr, s_eval):
    """Standardize using training mean/sd; zero out if degenerate."""
    m = float(s_tr.mean())
    sd = float(s_tr.std())
    if sd <= 1e-12:
        return np.zeros_like(s_tr), np.zeros_like(s_eval), True
    return (s_tr - m) / sd, (s_eval - m) / sd, False


def pooled_rps(Q, P):
    """Mean RPS over empirical histograms P (for training-fold validation selection)."""
    pc = np.cumsum(P, axis=1)[:, :5]
    qc = np.cumsum(Q, axis=1)[:, :5]
    return float(((qc - pc) ** 2 + pc * (1.0 - pc)).mean())


def val_rps_stage1(Phi, P, kind, c, fold, seed):
    """4-fold pooled out-of-fold validation RPS for Stage-1 ridge c."""
    Q = np.zeros_like(P)
    for f in range(FOLDS):
        tr, va = fold != f, fold == f
        n_tr = int(tr.sum())
        W = fit_weights(Phi[tr], P[tr], kind, n_tr, toy.marginal_W0(Phi[tr], P[tr], seed), c=c)
        Q[va] = softmax(Phi[va] @ W, axis=1)
    return pooled_rps(Q, P)


def pick_best(grid, scores, lower_better=True):
    """Select best hyperparameter from finite grid; ties within 1e-12 go to largest c."""
    finite = [(c, s) for c, s in zip(grid, scores) if np.isfinite(s)]
    if not finite:
        return grid[-1]
    best = min(s for _, s in finite) if lower_better else max(s for _, s in finite)
    return max(c for c, s in finite if abs(s - best) <= 1e-12)


def test_metrics(Q, Pt, xt, Qconst):
    """Compute held-out test metrics relative to truth plus TVD from constant baseline."""
    m = toy.metrics(Q, Pt, xt)
    out = {k: float(m[k]) for k in ("rps", "rps_excess", "ce_excess", "pearson", "spearman")}
    out["tvd_from_constant"] = float(0.5 * np.abs(Q - Qconst).sum(1).mean())
    return out


def run_one(args):
    """Execute one (scenario, n, seed) cell across all compared judges."""
    sc, n, seed, k, ntest = args
    rng = np.random.default_rng(1000003 * seed + 7919 * n + CODE[sc])
    x = rng.uniform(-1, 1, n)
    P = np.array([rng.multinomial(k, p) for p in truth(x, sc)]) / k
    Phi = toy.phi(x)
    W0 = toy.marginal_W0(Phi, P, seed)

    xt = rng.uniform(-1, 1, ntest)
    Pt = truth(xt, sc)
    Phit = toy.phi(xt)

    fold = np.empty(n, int)
    fold[np.random.default_rng([seed, n, CODE[sc], 99]).permutation(n)] = np.arange(n) % FOLDS

    Qconst = np.tile(P.mean(0), (ntest, 1))
    pc_t = np.cumsum(Pt, axis=1)[:, :5]
    true_rps = float((pc_t * (1.0 - pc_t)).mean())
    out = {
        "constant": test_metrics(Qconst, Pt, xt, Qconst),
        "true_rps": true_rps,
    }

    # 1. Exact CE fixed (4) & valrps (3)
    W_ce_1 = fit_weights(Phi, P, "ce_exact", n, W0, c=1.0)
    out["ce_fixed"] = test_metrics(softmax(Phit @ W_ce_1, 1), Pt, xt, Qconst)
    ce_scores = [val_rps_stage1(Phi, P, "ce_exact", c, fold, seed) for c in C_GRID]
    c_ce = pick_best(C_GRID, ce_scores, lower_better=True)
    W_ce_v = W_ce_1 if c_ce == 1.0 else fit_weights(Phi, P, "ce_exact", n, W0, c=c_ce)
    out["ce_valrps"] = dict(test_metrics(softmax(Phit @ W_ce_v, 1), Pt, xt, Qconst), c=float(c_ce))

    # 2. Pearson fixed (2) and Naive Stage-2 calibrated head (exact toy.py protocol)
    W_p = fit_weights(Phi, P, "pearson", n, W0, c=1.0)
    out["pearson_fixed"] = test_metrics(softmax(Phit @ W_p, 1), Pt, xt, Qconst)

    s_p_tr = softmax(Phi @ W_p, 1) @ toy.GRID
    s_p_te = softmax(Phit @ W_p, 1) @ toy.GRID
    m_p, sd_p = float(s_p_tr.mean()), float(s_p_tr.std()) + 1e-9
    H_tr = np.stack([np.ones(n), (s_p_tr - m_p) / sd_p], 1)
    H_te = np.stack([np.ones(ntest), (s_p_te - m_p) / sd_p], 1)
    Wh_naive = fit_weights(H_tr, P, "ce_exact", n, toy.marginal_W0(H_tr, P, seed), c=1.0)
    out["pearson_head_naive"] = test_metrics(softmax(H_te @ Wh_naive, 1), Pt, xt, Qconst)

    # 3 & 4. Cache 4-fold Pearson & MSE c=1 fits once for both in-sample tuned head and cross-fitted heads
    p_fold_z = {}
    z_p_oof = np.zeros(n)
    p_oof_degen = False
    W_mse_fold = {}
    z_mse_oof = np.zeros(n)
    mse_oof_degen = False
    for f in range(FOLDS):
        tr, va = fold != f, fold == f
        n_tr = int(tr.sum())
        W0_f = toy.marginal_W0(Phi[tr], P[tr], seed)
        Wp_f = fit_weights(Phi[tr], P[tr], "pearson", n_tr, W0_f, c=1.0)
        sf_tr = softmax(Phi[tr] @ Wp_f, 1) @ toy.GRID
        sf_va = softmax(Phi[va] @ Wp_f, 1) @ toy.GRID
        zf_tr, zf_va, degen_p = standardize_pair(sf_tr, sf_va)
        p_oof_degen = p_oof_degen or degen_p
        z_p_oof[va] = zf_va
        p_fold_z[f] = (zf_tr, zf_va)
        Wm_f = fit_weights(Phi[tr], P[tr], "mse", n_tr, W0_f, c=1.0)
        mf_tr = softmax(Phi[tr] @ Wm_f, 1) @ toy.GRID
        mf_va = softmax(Phi[va] @ Wm_f, 1) @ toy.GRID
        _, zm_va, degen_m = standardize_pair(mf_tr, mf_va)
        mse_oof_degen = mse_oof_degen or degen_m
        z_mse_oof[va] = zm_va
        W_mse_fold[f] = Wm_f

    z_in_tr, z_in_te, degen_in = standardize_pair(s_p_tr, s_p_te)
    if degen_in:
        a0, b0 = fit_stage2_head(np.zeros_like(s_p_tr), P, 1e8)
        out["pearson_head_in_tuned"] = dict(test_metrics(predict_head(np.zeros_like(s_p_te), a0, b0), Pt, xt, Qconst), c_b=1e8)
    else:
        scores_in = []
        for c_b in C_HEAD_GRID:
            Q_cv = np.zeros_like(P)
            for f in range(FOLDS):
                tr, va = fold != f, fold == f
                zf_tr, zf_va = p_fold_z[f]
                af, bf = fit_stage2_head(zf_tr, P[tr], c_b)
                Q_cv[va] = predict_head(zf_va, af, bf)
            scores_in.append(pooled_rps(Q_cv, P))
        c_b_in = pick_best(C_HEAD_GRID, scores_in, lower_better=True)
        a_in, b_in = fit_stage2_head(z_in_tr, P, c_b_in)
        out["pearson_head_in_tuned"] = dict(test_metrics(predict_head(z_in_te, a_in, b_in), Pt, xt, Qconst), c_b=float(c_b_in))

    def run_crossfit_head(z_oof, degen_oof, s_full_tr, s_full_te):
        _, z_te, degen_tr = standardize_pair(s_full_tr, s_full_te)
        if degen_oof or degen_tr or float(z_oof.std()) <= 1e-12:
            a0, b0 = fit_stage2_head(np.zeros_like(z_oof), P, 1e8)
            return dict(test_metrics(predict_head(np.zeros_like(s_full_te), a0, b0), Pt, xt, Qconst), c_b=1e8)
        cv_scores = []
        for c_b in C_HEAD_GRID:
            Q_cv = np.zeros_like(P)
            for f in range(FOLDS):
                tr, va = fold != f, fold == f
                af, bf = fit_stage2_head(z_oof[tr], P[tr], c_b)
                Q_cv[va] = predict_head(z_oof[va], af, bf)
            cv_scores.append(pooled_rps(Q_cv, P))
        c_b_best = pick_best(C_HEAD_GRID, cv_scores, lower_better=True)
        a_f, b_f = fit_stage2_head(z_oof, P, c_b_best)
        return dict(test_metrics(predict_head(z_te, a_f, b_f), Pt, xt, Qconst), c_b=float(c_b_best))

    out["pearson_head_tuned"] = run_crossfit_head(z_p_oof, p_oof_degen, s_p_tr, s_p_te)

    # 5. Direct MSE-trained fixed (c=1), valrps, and cross-fitted tuned head
    W_mse_1 = fit_weights(Phi, P, "mse", n, W0, c=1.0)
    out["mse_fixed"] = test_metrics(softmax(Phit @ W_mse_1, 1), Pt, xt, Qconst)

    # Reuse fold c=1 fits from above in val_rps check for mse!
    mse_scores = []
    for c in C_GRID:
        Q_cv = np.zeros_like(P)
        for f in range(FOLDS):
            tr, va = fold != f, fold == f
            if c == 1.0:
                W_f = W_mse_fold[f]
            else:
                W_f = fit_weights(Phi[tr], P[tr], "mse", int(tr.sum()), toy.marginal_W0(Phi[tr], P[tr], seed), c=c)
            Q_cv[va] = softmax(Phi[va] @ W_f, 1)
        mse_scores.append(pooled_rps(Q_cv, P))
    c_mse = pick_best(C_GRID, mse_scores, lower_better=True)
    W_mse_v = W_mse_1 if c_mse == 1.0 else fit_weights(Phi, P, "mse", n, W0, c=c_mse)
    out["mse_valrps"] = dict(test_metrics(softmax(Phit @ W_mse_v, 1), Pt, xt, Qconst), c=float(c_mse))

    s_mse_tr = softmax(Phi @ W_mse_1, 1) @ toy.GRID
    s_mse_te = softmax(Phit @ W_mse_1, 1) @ toy.GRID
    out["mse_head_tuned"] = run_crossfit_head(z_mse_oof, mse_oof_degen, s_mse_tr, s_mse_te)

    return sc, n, seed, out
