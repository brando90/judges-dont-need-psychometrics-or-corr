"""Fitting and scoring for experiment 28 (11-23-2026). Vectorized over training draws.

Everything here is specified in PROTOCOL_11-23-2026.md. verify.py re-implements the fits and scores with plain loops.
"""
import itertools
import numpy as np
from scipy.stats import rankdata

TOL = 1e-12
COEFFICIENTS = sorted(itertools.product((0.5, 1.0, 2.0), (-1, 0, 1), (-1, 0, 1)),
                      key=lambda v: (np.log2(v[0]) ** 2 + v[1] ** 2 + v[2] ** 2, v))
METHODS = ("ce", "pearson", "rps", "uniform", "hist", "pcal")
METRICS = ("rps", "ce", "tvd", "top1")
DATASETS = {"chaosnli_snli": 0, "chaosnli_mnli": 1, "chaosnli_alphanli": 2, "dices350": 3, "dices990": 4}
BUDGETS = {1: (4, 8, 16, 32, 64, 128, 256), 5: (10, 20, 40, 80, 160, 320)}
DRAWS = 200
FOLDS = 5


def transforms(p):
    """p [n,K] -> q [n,27,K]; transform 0 is the identity (up to the floor)."""
    K = p.shape[-1]
    p = (1 - K * 1e-6) * p + 1e-6
    t, b, c = np.asarray(COEFFICIENTS).T
    u = (np.arange(K) - (K - 1) / 2) / ((K - 1) / 2)
    logits = np.log(p)[:, None, :] * t[None, :, None] + b[None, :, None] * u + c[None, :, None] * u ** 2
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def rps_of(q, h):
    f = q.cumsum(-1)[..., :-1]
    g = h.cumsum(-1)[..., :-1]
    return (f * f + g * (1 - 2 * f)).mean(-1)


def score(q, h):
    """q [..., K], h [..., K] broadcast; returns [..., 4] = rps, ce, tvd, top1."""
    K = q.shape[-1]
    rps = rps_of(q, h)
    ce = -(h * np.log(q)).sum(-1)
    tvd = np.abs(h - q).sum(-1) / 2
    qs, hs = np.sort(q, -1), np.sort(h, -1)
    uq, uh = qs[..., -1] > qs[..., -2] + TOL, hs[..., -1] > hs[..., -2] + TOL
    top1 = (uq & uh & (q.argmax(-1) == h.argmax(-1))).astype(float)
    return np.stack((rps, ce, tvd, top1), -1)


def folds_of(groups, k):
    uniq = list(dict.fromkeys(groups))
    rng = np.random.default_rng(np.random.SeedSequence([20261123, 100 + k]))
    order = rng.permutation(len(uniq))
    fold_of = {uniq[g]: pos % FOLDS for pos, g in enumerate(order)}
    return np.array([fold_of[g] for g in groups])


def draw_training(counts, pool, d, k, fold, draws=DRAWS):
    """counts [n,K] integer vote counts; pool = training item indices.
    Returns order [R,P] (item index) and c [R,P,K]: the d ratings sampled without replacement per item."""
    counts = np.asarray(counts).astype(int)
    rng = np.random.default_rng(np.random.SeedSequence([20261123, 200 + k, fold, d]))
    P, K = len(pool), counts.shape[1]
    order = np.empty((draws, P), dtype=int)
    c = np.zeros((draws, P, K))
    for r in range(draws):
        order[r] = pool[rng.permutation(P)]
        for j, i in enumerate(order[r]):
            if d == 1:
                c[r, j, rng.choice(K, p=counts[i] / counts[i].sum())] = 1
            else:
                c[r, j] = rng.multivariate_hypergeometric(counts[i], d)
    return order, c


def first_min(loss):
    """Earliest transform within TOL of the minimum along the last axis (ties go to closest to identity)."""
    return (loss <= loss.min(-1, keepdims=True) + TOL).argmax(-1)


def fit_all(Q, S, order, c, d, n, scores):
    """Q [N,27,K], S [N,27] expected score, first n items of each draw. Returns idx for ce, pearson, rps and data for pcal/hist."""
    it = order[:, :n]
    cc = c[:, :n]                                   # [R,n,K]
    logq = np.log(Q[it])                            # [R,n,27,K]
    ce_loss = -(cc[:, :, None, :] * logq).sum(-1).sum(1) / (n * d)          # [R,27]
    ce = first_min(ce_loss)
    hist = cc / d
    qq = Q[it]
    f = qq.cumsum(-1)[..., :-1]; g = hist.cumsum(-1)[:, :, None, :-1]
    rps_loss = ((f * f + g * (1 - 2 * f)).mean(-1)).mean(1)                 # [R,27]
    rps = first_min(rps_loss)
    m = S[it]                                       # [R,n,27]
    y = (cc * scores).sum(-1) / d                   # [R,n] training item mean rating
    mc = m - m.mean(1, keepdims=True); yc = y - y.mean(1, keepdims=True)
    norm = np.linalg.norm(mc, axis=1) * np.linalg.norm(yc, axis=1)[:, None]
    valid = (np.ptp(m, axis=1) > TOL) & (np.ptp(y, axis=1)[:, None] > TOL) & (norm > 0)
    rho = np.divide((mc * yc[:, :, None]).sum(1), norm, out=np.full(norm.shape, -np.inf), where=valid)
    pearson = (rho >= rho.max(-1, keepdims=True) - TOL).argmax(-1)
    pearson[~valid.any(-1)] = 0
    return ce, pearson, rps, it, cc


def forecasts_for(Q, S, test, idx, it, cc, d, scores):
    """Return dict method -> q [R,E,K] for the held-out items."""
    R, n, K = cc.shape
    qs = {}
    for name, ix in idx.items():
        qs[name] = Q[test][:, :, :][:, ix, :].transpose(1, 0, 2)            # [R,E,K]
    qs["uniform"] = np.full((R, len(test), K), 1 / K)
    tot = cc.sum(1)                                  # [R,K]
    h = (tot + 0.5) / (tot.sum(-1, keepdims=True) + 0.5 * K)
    qs["hist"] = np.broadcast_to(h[:, None, :], (R, len(test), K)).copy()
    # Pearson fit, then train-only distribution calibration in four quantile bins of its expected score
    ix = idx["pearson"]
    mtr = S[it][np.arange(R)[:, None], np.arange(n)[None, :], ix[:, None]]   # [R,n]
    mte = S[test][:, ix].T                                                   # [R,E]
    edges = np.quantile(mtr, [0.25, 0.5, 0.75], axis=1).T                   # [R,3]
    btr = (mtr[:, :, None] > edges[:, None, :]).sum(-1)
    bte = (mte[:, :, None] > edges[:, None, :]).sum(-1)
    onehot = (btr[:, :, None] == np.arange(4)).astype(float)                # [R,n,4]
    cnt = np.einsum("rnb,rnk->rbk", onehot, cc)
    tot_b = cnt.sum(-1, keepdims=True)
    pb = np.where(tot_b > 0, (cnt + 0.5) / (tot_b + 0.5 * K), h[:, None, :])
    qs["pcal"] = pb[np.arange(R)[:, None], bte]                             # [R,E,K]
    return qs


def corr_rows(a, b):
    """Mean over rows of the Pearson correlation of each row of a [R,E] with b [E]; constant rows are skipped."""
    ac = a - a.mean(1, keepdims=True); bc = b - b.mean()
    den = np.linalg.norm(ac, axis=1) * np.linalg.norm(bc)
    ok = (np.ptp(a, axis=1) > TOL) & (den > 0)
    return float(np.mean((ac @ bc)[ok] / den[ok])) if ok.any() else float("nan")


_DRAWS = {}


def cached_draws(counts, pool, d, k, fold):
    key = (k, d, fold)
    if key not in _DRAWS:
        _DRAWS[key] = draw_training(counts, pool, d, k, fold, DRAWS)
    return _DRAWS[key]


def run_set(p, counts, groups, k, d, scores):
    """p [N,K] frozen judge forecasts. Returns per-item scores [N, B, methods, 4], raw judge scores [N,4], and fold-level
    correlations [FOLDS, B, methods, 2] (Pearson, Spearman of expected score vs human mean, test fold)."""
    N, K = p.shape
    Q = transforms(p)
    S = Q @ scores
    h = counts / counts.sum(1, keepdims=True)
    hm = h @ scores
    fold = folds_of(groups, k)
    budgets = BUDGETS[d]
    out = np.zeros((N, len(budgets), len(METHODS), 4))
    cor = np.zeros((FOLDS, len(budgets), len(METHODS), 2))
    for f in range(FOLDS):
        test = np.flatnonzero(fold == f)
        pool = np.flatnonzero(fold != f)
        order, c = cached_draws(counts, pool, d, k, f)
        for bi, B in enumerate(budgets):
            n = B // d
            ce, pe, rp, it, cc = fit_all(Q, S, order, c, d, n, scores)
            qs = forecasts_for(Q, S, test, {"ce": ce, "pearson": pe, "rps": rp}, it, cc, d, scores)
            for mi, mname in enumerate(METHODS):
                q = qs[mname]
                out[test, bi, mi, :] = score(q, h[test][None]).mean(0)
                es = q @ scores                              # [R,E]
                cor[f, bi, mi] = (corr_rows(es, hm[test]), corr_rows(rankdata(es, axis=1), rankdata(hm[test])))
    raw = score(Q[:, 0, :], h)
    return out, raw, cor, fold


def simultaneous_lower(delta, boots, rng, alpha=0.025):
    """delta [N,B] per-item paired differences (comparator minus CE, positive = CE better). Returns mean, lower [B]."""
    N, B = delta.shape
    w = rng.multinomial(N, np.full(N, 1 / N), size=boots) / N                 # [boots,N]
    bm = w @ delta                                                            # [boots,B]
    mean = delta.mean(0)
    se = bm.std(0, ddof=1)
    ok = se > 0
    z = np.where(ok, (bm - mean) / np.where(ok, se, 1), 0).max(1)
    crit = np.quantile(z, 1 - alpha)
    return mean, np.where(ok, mean - crit * se, np.where(mean > 0, np.inf, -np.inf)), se


def sustained(lower, budgets):
    """Smallest budget from which every larger-or-equal budget has lower > 0; None if the largest fails."""
    best = None
    for b in range(len(budgets) - 1, -1, -1):
        if lower[b] > 0:
            best = budgets[b]
        else:
            break
    return best
