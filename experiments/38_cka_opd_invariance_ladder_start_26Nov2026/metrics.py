"""Alignment metrics and proper scores for the invariance-group ladder.

Linear CKA is the squared cosine of centered Gram matrices,

    CKA(X, Y) = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F).

In one column this equals the squared Pearson correlation of the centered
columns. Normalized orthogonal Procrustes distance is

    OPD(X, Y) = sqrt(max(0, 2 - 2 * ||Xhat^T Yhat||_*)),

with Xhat, Yhat the column-centered matrices scaled to Frobenius norm 1, and
||.||_* the nuclear norm. Lower OPD is a closer orthogonal match.

Proper scores are the unbiased prompt average of the within-prompt mean loss.
Lower cross-entropy, Brier, and ranked probability score (RPS) are better.
"""

from __future__ import annotations

import numpy as np


def center_columns(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a - a.mean(axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA. Returns NaN if either centered matrix is identically zero."""
    xc = center_columns(x)
    yc = center_columns(y)
    cross = xc.T @ yc
    num = float(np.sum(cross * cross))
    sx = float(np.sum((xc.T @ xc) ** 2))
    sy = float(np.sum((yc.T @ yc) ** 2))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return num / np.sqrt(sx * sy)


def cka_sq_rational(x: np.ndarray, y: np.ndarray) -> float:
    """CKA squared, N^2 / (Sx * Sy), which stays a ratio of rational Gram squares."""
    xc = center_columns(x)
    yc = center_columns(y)
    num = float(np.sum((xc.T @ yc) ** 2))
    sx = float(np.sum((xc.T @ xc) ** 2))
    sy = float(np.sum((yc.T @ yc) ** 2))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return (num * num) / (sx * sy)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    xc = x - x.mean()
    yc = y - y.mean()
    vx = float(np.dot(xc, xc))
    vy = float(np.dot(yc, yc))
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return float(np.dot(xc, yc) / np.sqrt(vx * vy))


def normalized_opd(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized orthogonal Procrustes distance. Lower means a closer match."""
    xc = center_columns(x)
    yc = center_columns(y)
    nx = np.linalg.norm(xc)
    ny = np.linalg.norm(yc)
    if nx <= 0.0 or ny <= 0.0:
        return float("nan")
    singular = np.linalg.svd((xc / nx).T @ (yc / ny), compute_uv=False)
    cosine = float(np.sum(singular))
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * cosine)))


def cca_sq(x: np.ndarray, y: np.ndarray, rank_tol: float = 1e-10) -> float:
    """Sum of squared canonical correlations. Invariant to invertible column maps."""
    xc = center_columns(x)
    yc = center_columns(y)

    def orthonormal_columns(a: np.ndarray) -> np.ndarray:
        q, r = np.linalg.qr(a, mode="reduced")
        if r.size == 0:
            return q[:, :0]
        keep = np.abs(np.diag(r)) > rank_tol
        return q[:, keep]

    qx = orthonormal_columns(xc)
    qy = orthonormal_columns(yc)
    if qx.shape[1] == 0 or qy.shape[1] == 0:
        return float("nan")
    singular = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(np.sum(singular ** 2))


def rbf_cka(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """RBF-kernel CKA with a fixed gamma. Distances are invariant to isometries."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    def kernel(a: np.ndarray) -> np.ndarray:
        sq = np.sum(a * a, axis=1, keepdims=True)
        dist2 = np.maximum(sq + sq.T - 2.0 * (a @ a.T), 0.0)
        return np.exp(-gamma * dist2)

    def hsic(k: np.ndarray, ell: np.ndarray) -> float:
        n = k.shape[0]
        h = np.eye(n) - np.ones((n, n)) / n
        kc = h @ k @ h
        lc = h @ ell @ h
        return float(np.sum(kc * lc))

    kx, ky = kernel(x), kernel(y)
    denom = np.sqrt(hsic(kx, kx) * hsic(ky, ky))
    if denom <= 0.0:
        return float("nan")
    return hsic(kx, ky) / denom


def median_gamma(x: np.ndarray) -> float:
    """Gamma = 1 / (2 * median squared distance), frozen on the untransformed matrix."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    sq = np.sum(x * x, axis=1, keepdims=True)
    dist2 = np.maximum(sq + sq.T - 2.0 * (x @ x.T), 0.0)
    tri = dist2[np.triu_indices(dist2.shape[0], k=1)]
    med = float(np.median(tri)) if tri.size else 1.0
    if med <= 0.0:
        med = 1.0
    return 1.0 / (2.0 * med)


def brier_one(q: np.ndarray, y: int) -> float:
    e = np.zeros(q.shape[0], dtype=float)
    e[y] = 1.0
    d = q - e
    return float(np.dot(d, d))


def rps_one(q: np.ndarray, y: int) -> float:
    """Ranked probability score on classes ordered by index. Lower is better. Range [0, K-1]."""
    q = np.asarray(q, dtype=float)
    k = q.shape[0]
    if k <= 1:
        return 0.0
    cdf = np.cumsum(q)[:-1]
    indicator = (np.arange(k - 1) >= y).astype(float)
    d = cdf - indicator
    return float(np.dot(d, d))


def ce_one(q: np.ndarray, y: int) -> float:
    p = float(q[y])
    if p <= 0.0:
        return float("inf")
    return float(-np.log(p))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def item_mean_losses(q: np.ndarray, ratings: np.ndarray) -> dict[str, float]:
    ratings = np.asarray(ratings, dtype=int)
    ce = br = rp = 0.0
    n_inf = 0
    for y in ratings:
        c = ce_one(q, int(y))
        if not np.isfinite(c):
            n_inf += 1
        ce += c
        br += brier_one(q, int(y))
        rp += rps_one(q, int(y))
    k = float(ratings.size)
    return {"ce": ce / k, "brier": br / k, "rps": rp / k, "n_infinite_ce": n_inf}


def unbiased_risk(item_means: np.ndarray) -> float:
    item_means = np.asarray(item_means, dtype=float)
    if item_means.size == 0:
        return float("nan")
    if not np.isfinite(item_means).all():
        return float("inf")
    return float(np.mean(item_means))


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(s, [alpha, 1.0 - alpha])
    return float(lo), float(hi)
