"""Proper scores, correlations, the unbiased test risk, and item-clustered bootstrap.

Notation matches the experiment note. ``c`` is a prompt. ``y`` is one human rating.
``q_hat(c)`` is a distribution on the ordered classes. The test risk of a proper
loss averages within each prompt first, then across prompts:

    R_hat(q) = (1/|D_test|) * sum_c (1/k_c) * sum_j L(q(c), y_{c,j})

A plug-in divergence against the test histogram is not this estimator.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, rankdata


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def one_hot(k: int, index: int, floor: float = 0.0) -> np.ndarray:
    """Point mass, or a floored point mass when ``floor > 0`` so cross-entropy is finite."""
    q = np.full(k, floor, dtype=float)
    q[index] = 1.0 - floor * (k - 1)
    if q[index] <= 0:
        raise ValueError("floor is too large for this number of classes")
    return q


def brier_one(q: np.ndarray, y: int) -> float:
    """One-hot Brier score ‖q - e_y‖₂². Lower is better. Range [0, 2]."""
    e = np.zeros(q.shape[0], dtype=float)
    e[y] = 1.0
    d = q - e
    return float(np.dot(d, d))


def rps_one(q: np.ndarray, y: int) -> float:
    """Ordinal ranked probability score. Classes are ordered by index.

    RPS(q, y) = sum_{t=0}^{K-2} (F_q(t) - 1_{y ≤ t})².
    Lower is better. Range [0, K-1].
    """
    q = np.asarray(q, dtype=float)
    k = q.shape[0]
    if k <= 1:
        return 0.0
    cdf = np.cumsum(q)[:-1]
    indicator = (np.arange(k - 1) >= y).astype(float)
    d = cdf - indicator
    return float(np.dot(d, d))


def ce_one(q: np.ndarray, y: int) -> float:
    """Cross-entropy -log q(y), in nats. A nonpositive probability returns +infinity."""
    p = float(q[y])
    if p <= 0.0:
        return float("inf")
    return float(-np.log(p))


def normalized_rps_one(q: np.ndarray, y: int) -> float:
    """RPS divided by K-1, so the score lies in [0, 1]."""
    k = int(q.shape[0])
    if k <= 1:
        return 0.0
    return rps_one(q, y) / (k - 1)


def item_mean_losses(q: np.ndarray, ratings: np.ndarray) -> dict[str, float]:
    """Mean proper loss of one prompt, averaging its ``k_c`` ratings with equal weight."""
    ratings = np.asarray(ratings, dtype=int)
    if ratings.size == 0:
        raise ValueError("a prompt with no ratings has no within-prompt mean")
    ce = br = rp = 0.0
    for y in ratings:
        ce += ce_one(q, int(y))
        br += brier_one(q, int(y))
        rp += rps_one(q, int(y))
    k = float(ratings.size)
    return {"ce": ce / k, "brier": br / k, "rps": rp / k}


def unbiased_risk(item_means: np.ndarray) -> float:
    """Unbiased prompt-average of within-prompt mean losses. ``item_means`` has one entry per prompt."""
    item_means = np.asarray(item_means, dtype=float)
    if item_means.size == 0:
        raise ValueError("empty test set")
    return float(np.mean(item_means))


def plugin_l2(p_hat: np.ndarray, q: np.ndarray) -> float:
    """Plug-in D_B(p_hat ‖ q) = ‖p_hat - q‖₂². Not an unbiased test risk."""
    d = np.asarray(p_hat, dtype=float) - np.asarray(q, dtype=float)
    return float(np.dot(d, d))


def expected_plugin_bias(p_star: np.ndarray, n: int) -> float:
    """E[‖q_N - p‖₂²] = (1/N) (1 - ‖p‖₂²), also the bias of plug-in D_B(q_N ‖ q) above D_B(p ‖ q)."""
    p = np.asarray(p_star, dtype=float)
    return (1.0 - float(np.dot(p, p))) / float(n)


def pearson_from_sums(n: float, sum_y: float, sum_y2: float, sum_s: float, sum_s2: float, sum_sy: float) -> float:
    """Pearson correlation from pooled sums. Undefined (NaN) if either variable has zero sample variance."""
    if n < 2:
        return float("nan")
    my = sum_y / n
    ms = sum_s / n
    vy = sum_y2 / n - my * my
    vs = sum_s2 / n - ms * ms
    cov = sum_sy / n - my * ms
    if vy <= 1e-15 or vs <= 1e-15:
        return float("nan")
    return float(cov / np.sqrt(vy * vs))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho is the Pearson correlation of the midranks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    tau, _ = kendalltau(x, y)
    return float(tau)


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float, int]:
    """Percentile interval. NaNs are dropped. Returns (lo, hi, n_finite)."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan"), float("nan"), 0
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(s, [alpha, 1.0 - alpha])
    return float(lo), float(hi), int(s.size)


def two_point_on_grid(mu: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Distribution on adjacent grid points whose mean equals ``mu`` clipped to the grid."""
    values = np.asarray(values, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), values[0], values[-1])
    k = values.shape[0]
    out = np.zeros((mu.shape[0], k), dtype=float)
    idx = np.searchsorted(values, mu, side="right") - 1
    idx = np.clip(idx, 0, k - 2)
    v0 = values[idx]
    v1 = values[idx + 1]
    span = v1 - v0
    w1 = np.divide(mu - v0, span, out=np.zeros_like(mu), where=span > 0)
    w1 = np.clip(w1, 0.0, 1.0)
    rows = np.arange(mu.shape[0])
    out[rows, idx] = 1.0 - w1
    out[rows, idx + 1] += w1
    at_top = mu >= values[-1] - 1e-12
    out[at_top] = 0.0
    out[at_top, -1] = 1.0
    return out


def binary_brier_equals_two_l2(q: float, y: float) -> float:
    """(q-y)² + ((1-q)-(1-y))² - 2(q-y)², which is identically 0."""
    return (q - y) ** 2 + ((1 - q) - (1 - y)) ** 2 - 2 * (q - y) ** 2
