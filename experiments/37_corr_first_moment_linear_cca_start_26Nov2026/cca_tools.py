"""Canonical correlations, proper scores, and the item-clustered bootstrap.

Canonical correlation is the cosine of the principal angles between the centered
column spaces. An invertible linear mix of one view does not change those angles.
Proper scores below are the unbiased prompt average: within a prompt, average the
ratings with equal weight, then average prompts with equal weight.
"""

from __future__ import annotations

import numpy as np

TOL_CCA = 1e-8


def canonical_correlations(x: np.ndarray, y: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Descending canonical correlations of two real views. Empty if either view has no variance."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        return np.zeros(0)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    def basis(a: np.ndarray) -> np.ndarray:
        # Economy SVD. Columns of U spanning singular values above tol * largest.
        _u, s, _vt = np.linalg.svd(a, full_matrices=False)
        if s.size == 0 or not np.isfinite(s[0]) or s[0] <= tol:
            return np.zeros((a.shape[0], 0))
        keep = s > tol * s[0]
        return _u[:, keep]

    xb, yb = basis(x), basis(y)
    if xb.shape[1] == 0 or yb.shape[1] == 0:
        return np.zeros(0)
    singular = np.linalg.svd(xb.T @ yb, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    return np.sort(singular)[::-1]


def correlations_close(a: np.ndarray, b: np.ndarray, tol: float = TOL_CCA) -> bool:
    """Pad the shorter vector with zeros and compare. Both are already sorted descending."""
    n = max(a.size, b.size)
    aa = np.zeros(n)
    bb = np.zeros(n)
    aa[: a.size] = a
    bb[: b.size] = b
    if not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
        return False
    return bool(np.max(np.abs(aa - bb)) <= tol)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def brier_one(q: np.ndarray, y: int) -> float:
    e = np.zeros(q.shape[0], dtype=np.float64)
    e[y] = 1.0
    d = q - e
    return float(np.dot(d, d))


def rps_one(q: np.ndarray, y: int) -> float:
    """Ranked probability score on classes ordered by index. Range [0, K-1]. Lower is better."""
    q = np.asarray(q, dtype=np.float64)
    k = q.shape[0]
    if k <= 1:
        return 0.0
    cdf = np.cumsum(q)[:-1]
    indicator = (np.arange(k - 1) >= y).astype(np.float64)
    d = cdf - indicator
    return float(np.dot(d, d))


def ce_one(q: np.ndarray, y: int) -> float:
    p = float(q[y])
    if p <= 0.0:
        return float("inf")
    return float(-np.log(p))


def tvd_one(q: np.ndarray, p_hat: np.ndarray) -> float:
    """Plug-in total variation: 0.5 * sum |q - p_hat|. Range [0, 1]. Lower is better."""
    d = np.asarray(q, dtype=np.float64) - np.asarray(p_hat, dtype=np.float64)
    return float(0.5 * np.sum(np.abs(d)))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    vx = float(np.dot(x, x))
    vy = float(np.dot(y, y))
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / np.sqrt(vx * vy))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    # average ranks for ties
    def midrank(v):
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(v.size, dtype=np.float64)
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            ranks[order[i : j + 1]] = avg
            i = j + 1
        return ranks
    return pearson(midrank(x), midrank(y))


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float, int]:
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan"), float("nan"), 0
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(s, [alpha, 1.0 - alpha])
    return float(lo), float(hi), int(s.size)


def item_cluster_bootstrap(stat_on_indices, n_items: int, b: int = 1000, seed: int = 37) -> np.ndarray:
    """Resample prompt indices with replacement. `stat_on_indices` maps an index array to a float."""
    rng = np.random.default_rng(seed)
    out = np.empty(b, dtype=np.float64)
    for i in range(b):
        draw = rng.integers(0, n_items, size=n_items)
        out[i] = stat_on_indices(draw)
    return out


def fit_ce_softmax(x: np.ndarray, targets: np.ndarray, l2: float = 1e-2, maxiter: int = 200):
    """L2-regularized multinomial logistic regression. Convex in the logits.

    The last column of `x` is an unpenalized intercept. Returns probabilities on the rows of `x`,
    the success flag, and the iteration count.
    """
    from scipy.optimize import minimize

    x = np.asarray(x, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n, d = x.shape
    k = targets.shape[1]

    def unpack(w):
        return w.reshape(d, k)

    def loss_grad(w):
        weight = unpack(w)
        prob = softmax(x @ weight)
        prob = np.clip(prob, 1e-12, 1.0)
        prob = prob / prob.sum(axis=1, keepdims=True)
        nll = float(-(targets * np.log(prob)).sum(axis=1).mean())
        penalty_w = weight.copy()
        penalty_w[-1, :] = 0.0
        pen = 0.5 * l2 * float(np.sum(penalty_w ** 2))
        resid = (prob - targets) / n
        grad = x.T @ resid + l2 * penalty_w
        return nll + pen, grad.reshape(-1)

    res = minimize(
        loss_grad,
        np.zeros(d * k),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12},
    )
    weight = unpack(res.x)
    prob = softmax(x @ weight)
    return prob, weight, bool(res.success), int(res.nit), float(res.fun)


def project_train_basis(x_train: np.ndarray, others: list[np.ndarray], n_comp: int = 8):
    mu = x_train.mean(axis=0)
    centered = x_train - mu
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(s > 1e-8))
    k = max(1, min(n_comp, rank))
    basis = vt[:k].T

    def project(x):
        z = (x - mu) @ basis
        return np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)

    return project(x_train), [project(x) for x in others], k
