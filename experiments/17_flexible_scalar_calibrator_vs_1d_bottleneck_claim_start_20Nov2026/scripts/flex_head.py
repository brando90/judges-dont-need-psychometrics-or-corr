"""Experiment 17 core: does a flexible scalar-to-distribution calibrator remove the 'one-dimensional bottleneck' of experiment 11?

Reuses experiment 11's generator, Stage-1 fits and test metrics unchanged (imports its module), and adds Stage-2 heads that are nonlinear in the scalar score:
softmax(W^T [1, z, RBF_1(z), ..., RBF_12(z)]), exact CE, ridge on every weight except the intercept row (so a very large ridge gives the constant baseline), ridge tuned by 4-fold
cross-fitting on training tasks only. Scores z: Pearson-trained OOF score, MSE-trained OOF score, and (oracle) the exact true mean rating of each task, standardized on train.
"""
import os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "11_toy_mse_and_tuned_head_calibration_vs_ce_start_17Nov2026", "scripts")))
import mse_and_tuned_head as ex11  # noqa: E402
toy = ex11.toy
CENTERS = np.linspace(-2.5, 2.5, 12); WIDTH = float(CENTERS[1] - CENTERS[0])
C_FLEX_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1e4, 1e8]
FOLDS = ex11.FOLDS

def feats(z):
    z = np.clip(z, -4.0, 4.0)
    return np.concatenate([np.ones((len(z), 1)), z[:, None], np.exp(-0.5 * ((z[:, None] - CENTERS[None, :]) / WIDTH) ** 2)], axis=1)

def _obj(w, H, P, lam, logpbar, lam_a=1e-9):
    D = H.shape[1]; W = w.reshape(D, 6); Z = H @ W; logQ = log_softmax(Z, axis=1); Q = np.exp(logQ)
    g0 = W[0] - logpbar
    val = float(-(P * logQ).sum(1).mean() + lam * (W[1:] ** 2).sum() + lam_a * (g0 ** 2).sum())
    dZ = (Q - P) / len(P); G = H.T @ dZ; G[1:] += 2.0 * lam * W[1:]; G[0] += 2.0 * lam_a * g0
    return val, G.ravel()

def fit_flex(z, P, c):
    n = len(P); H = feats(z); pbar = np.clip(P.mean(0), 1e-6, None); pbar = pbar / pbar.sum(); lp = np.log(pbar)
    w0 = np.zeros((H.shape[1], 6)); w0[0] = lp
    res = minimize(_obj, w0.ravel(), args=(H, P, float(c) / n, lp), jac=True, method="L-BFGS-B", options={"maxiter": 600, "maxfun": 800, "ftol": 1e-13, "gtol": 1e-9})
    return res.x.reshape(H.shape[1], 6)

def predict_flex(z, W):
    return softmax(feats(z) @ W, axis=1)

def tune_and_fit(z_fit, P, fold, z_eval):
    """Pick the ridge by 4-fold cross-fitted pooled RPS on training data only, refit on all training tasks, predict z_eval."""
    scores = []
    for c in C_FLEX_GRID:
        Q = np.zeros_like(P)
        for f in range(FOLDS):
            tr, va = fold != f, fold == f
            Q[va] = predict_flex(z_fit[va], fit_flex(z_fit[tr], P[tr], c))
        scores.append(ex11.pooled_rps(Q, P))
    c = ex11.pick_best(C_FLEX_GRID, scores, lower_better=True)
    return predict_flex(z_eval, fit_flex(z_fit, P, c)), float(c)

def run_one(args):
    sc, n, seed, k, ntest = args
    rng = np.random.default_rng(1000003 * seed + 7919 * n + ex11.CODE[sc])        # identical stream to experiment 11
    x = rng.uniform(-1, 1, n); P = np.array([rng.multinomial(k, p) for p in ex11.truth(x, sc)]) / k
    Phi = toy.phi(x); W0 = toy.marginal_W0(Phi, P, seed)
    xt = rng.uniform(-1, 1, ntest); Pt = ex11.truth(xt, sc); Phit = toy.phi(xt)
    fold = np.empty(n, int); fold[np.random.default_rng([seed, n, ex11.CODE[sc], 99]).permutation(n)] = np.arange(n) % FOLDS
    Qconst = np.tile(P.mean(0), (ntest, 1))
    out = {"constant": ex11.test_metrics(Qconst, Pt, xt, Qconst)}
    # reference arms from experiment 11 (recomputed, same data): end-to-end CE with validation-tuned ridge, and the 12-parameter tuned Pearson head
    sc_, n_, seed_, ref = ex11.run_one(args)
    for key in ("ce_fixed", "ce_valrps", "pearson_head_tuned", "mse_head_tuned", "pearson_fixed"):
        out[key] = ref[key]
    # Stage-1 OOF scores (Pearson and MSE) exactly as in experiment 11
    s_p_tr = softmax(Phi @ ex11.fit_weights(Phi, P, "pearson", n, W0, c=1.0), 1) @ toy.GRID
    W_p = ex11.fit_weights(Phi, P, "pearson", n, W0, c=1.0); s_p_te = softmax(Phit @ W_p, 1) @ toy.GRID
    W_m = ex11.fit_weights(Phi, P, "mse", n, W0, c=1.0); s_m_tr = softmax(Phi @ W_m, 1) @ toy.GRID; s_m_te = softmax(Phit @ W_m, 1) @ toy.GRID
    z_p_oof = np.zeros(n); z_m_oof = np.zeros(n); deg = False
    for f in range(FOLDS):
        tr, va = fold != f, fold == f; n_tr = int(tr.sum()); W0f = toy.marginal_W0(Phi[tr], P[tr], seed)
        Wp = ex11.fit_weights(Phi[tr], P[tr], "pearson", n_tr, W0f, c=1.0)
        _, zv, d1 = ex11.standardize_pair(softmax(Phi[tr] @ Wp, 1) @ toy.GRID, softmax(Phi[va] @ Wp, 1) @ toy.GRID); z_p_oof[va] = zv
        Wm = ex11.fit_weights(Phi[tr], P[tr], "mse", n_tr, W0f, c=1.0)
        _, zm, d2 = ex11.standardize_pair(softmax(Phi[tr] @ Wm, 1) @ toy.GRID, softmax(Phi[va] @ Wm, 1) @ toy.GRID); z_m_oof[va] = zm
        deg = deg or d1 or d2
    for name, z_oof, s_tr, s_te in (("pearson_head_flex", z_p_oof, s_p_tr, s_p_te), ("mse_head_flex", z_m_oof, s_m_tr, s_m_te)):
        _, z_te, d = ex11.standardize_pair(s_tr, s_te)
        if deg or d or float(z_oof.std()) <= 1e-12:
            Q, c = predict_flex(np.zeros(ntest), fit_flex(np.zeros(n), P, 1e8)), 1e8
        else:
            Q, c = tune_and_fit(z_oof, P, fold, z_te)
        out[name] = dict(ex11.test_metrics(Q, Pt, xt, Qconst), c=c)
    # oracle score: exact true mean rating of each task, standardized on the training tasks
    mu_tr = ex11.truth(x, sc) @ toy.GRID; mu_te = Pt @ toy.GRID
    z_tr, z_te, d = ex11.standardize_pair(mu_tr, mu_te)
    Qf, cf = tune_and_fit(z_tr, P, fold, z_te)
    out["truemean_head_flex"] = dict(ex11.test_metrics(Qf, Pt, xt, Qconst), c=cf)
    a, b = ex11.fit_stage2_head(z_tr, P, 1.0)                                      # the 12-parameter affine-logit head on the same oracle score, ridge as experiment 11's default
    scores = []
    for cb in ex11.C_HEAD_GRID:
        Q = np.zeros_like(P)
        for f in range(FOLDS):
            tr, va = fold != f, fold == f; af, bf = ex11.fit_stage2_head(z_tr[tr], P[tr], cb); Q[va] = ex11.predict_head(z_tr[va], af, bf)
        scores.append(ex11.pooled_rps(Q, P))
    cb = ex11.pick_best(ex11.C_HEAD_GRID, scores, lower_better=True); a, b = ex11.fit_stage2_head(z_tr, P, cb)
    out["truemean_head_affine"] = dict(ex11.test_metrics(ex11.predict_head(z_te, a, b), Pt, xt, Qconst), c_b=float(cb))
    return sc, n, seed, out
