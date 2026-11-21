"""Experiment 20 (11-21-2026): shared code. Two trained judges, identical except for the objective.
TLDR: a judge is a softmax-linear head q(x) = softmax([x,1] @ Theta) over the six ratings 0..5. 'ce' fits Theta by cross-entropy on individual ratings;
'corr' fits Theta by maximising Pearson(mean of q, item mean rating). Same init (Theta ~ N(0, 0.05^2) for the weights, bias 0, same draw for both objectives; Pearson has zero gradient at exactly Theta=0), same L2 on the weights, same optimiser (L-BFGS), same data.
'corr_cal' = the corr judge followed by a 3-parameter calibration head (temperature, level, spread) fit by CE on the same training data.
CPU only, numpy+scipy, no model calls. Written by ultimate_brando9 with Claude Code, Claude Sonnet 5.5."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

R = np.arange(6.0)
U = (R - 2.5) / 2.5

def softmax(L):
    L = L - L.max(-1, keepdims=True); E = np.exp(L); return E / E.sum(-1, keepdims=True)

def design(X): return np.hstack([X, np.ones((len(X), 1))])

def fit(Xd, C, objective, lam=0.01, maxiter=300, init_seed=0, init_sd=0.05):
    """Xd: m x (d+1) design (last col 1). C: m x 6 rating counts. Returns Theta (d+1) x 6."""
    m, D = Xd.shape; n = C.sum(); ybar = (C @ R) / C.sum(1); Wmask = np.ones((D, 6)); Wmask[-1] = 0
    def f_ce(th):
        T = th.reshape(D, 6); L = Xd @ T; Lm = L - L.max(1, keepdims=True); lse = np.log(np.exp(Lm).sum(1, keepdims=True)); logq = Lm - lse
        q = np.exp(logq); loss = -(C * logq).sum() / n + lam * ((T * Wmask) ** 2).sum()
        G = Xd.T @ ((q * C.sum(1, keepdims=True) - C) / n) + 2 * lam * T * Wmask
        return loss, G.ravel()
    def f_corr(th):
        T = th.reshape(D, 6); q = softmax(Xd @ T); mu = q @ R; mc = mu - mu.mean(); yc = ybar - ybar.mean()
        sm, sy = np.sqrt((mc ** 2).mean()), np.sqrt((yc ** 2).mean())
        if sm < 1e-12 or sy < 1e-12: c = 0.0; dmu = np.zeros(m)
        else:
            c = (mc * yc).mean() / (sm * sy); dmu = ((yc / (sm * sy)) - c * mc / sm ** 2) / m
        loss = -c + lam * ((T * Wmask) ** 2).sum()
        G = Xd.T @ (q * (R[None, :] - mu[:, None]) * dmu[:, None]) * -1 + 2 * lam * T * Wmask
        return loss, G.ravel()
    f = f_ce if objective == 'ce' else f_corr
    th0 = np.random.default_rng(init_seed).normal(0, init_sd, size=(D, 6)); th0[-1] = 0
    res = minimize(f, th0.ravel(), jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    return res.x.reshape(D, 6)

def calibrate(Lt, C, maxiter=100):
    """3-parameter head on logits Lt (m x 6): q = softmax(t*Lt + b*u + c*u^2); fit by CE on the training counts. Returns (t,b,c)."""
    n = C.sum(); B = np.stack([U, U ** 2])
    def f(p):
        t, b, c = p; L = t * Lt + b * U + c * U ** 2; Lm = L - L.max(1, keepdims=True); logq = Lm - np.log(np.exp(Lm).sum(1, keepdims=True)); q = np.exp(logq)
        G = (q * C.sum(1, keepdims=True) - C) / n
        return -(C * logq).sum() / n, np.array([(G * Lt).sum(), (G * U).sum(), (G * U ** 2).sum()])
    res = minimize(f, np.array([1.0, 0.0, 0.0]), jac=True, method='L-BFGS-B', bounds=[(0, 50), (-20, 20), (-20, 20)], options={'maxiter': maxiter})
    return res.x

def predict(X, Theta, cal=None):
    L = design(X) @ Theta
    if cal is not None: t, b, c = cal; L = t * L + b * U + c * U ** 2
    return softmax(L)

def fit_arms(Xtr, Ctr, lam, init_seed=0):
    """Returns dict arm -> function X -> Q."""
    Xd = design(Xtr); T_ce = fit(Xd, Ctr, 'ce', lam, init_seed=init_seed); T_co = fit(Xd, Ctr, 'corr', lam, init_seed=init_seed)
    cal = calibrate(Xd @ T_co, Ctr)
    const = (Ctr.sum(0) + 0.5) / (Ctr.sum() + 3.0)
    return {'ce': lambda X: predict(X, T_ce), 'corr': lambda X: predict(X, T_co), 'corr_cal': lambda X: predict(X, T_co, cal),
            'constant': lambda X: np.tile(const, (len(X), 1))}

# ---------- metrics against a KNOWN distribution P (synthetic)
def moments(P):
    mu = P @ R; var = P @ (R ** 2) - mu ** 2; k3 = (P * (R[None, :] - mu[:, None]) ** 3).sum(1); sk = k3 / np.maximum(var, 1e-12) ** 1.5
    return mu, var, sk

def spearman(a, b):
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12: return np.nan
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]

def pearson(a, b):
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12: return np.nan
    return np.corrcoef(a, b)[0, 1]

def metrics_known(Q, P, mu_bins):
    mq, vq, sq = moments(Q); mp, vp, sp = moments(P); cq, cp = Q.cumsum(1)[:, :-1], P.cumsum(1)[:, :-1]
    inb = [spearman(vq[mu_bins == b], vp[mu_bins == b]) for b in np.unique(mu_bins) if (mu_bins == b).sum() > 5]
    return {'ce': float(-(P * np.log(Q)).sum(1).mean()), 'kl': float((P * (np.log(P + 1e-300) - np.log(Q))).sum(1).mean()),
            'tvd': float(0.5 * np.abs(P - Q).sum(1).mean()), 'rps_excess': float(((cq - cp) ** 2).sum(1).mean() / 5),
            'mean_rmse': float(np.sqrt(((mq - mp) ** 2).mean())), 'var_rmse': float(np.sqrt(((vq - vp) ** 2).mean())), 'skew_rmse': float(np.sqrt(((sq - sp) ** 2).mean())),
            'pearson': pearson(mq, mp), 'spearman': spearman(mq, mp), 'var_spearman': spearman(vq, vp), 'var_spearman_in_mean_bin': float(np.nanmean(inb)) if inb else np.nan}
