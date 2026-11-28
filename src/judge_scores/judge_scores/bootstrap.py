"""Item-clustered bootstrap: items are resampled with replacement; every statistic is recomputed on the resample."""
from __future__ import annotations

import numpy as np


def boot_weights(n_items: int, B: int = 4000, seed: int = 0):
    """[B, N] multiplicity of each item in each resample. Reuse one matrix for every statistic that should be paired."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_items, size=(B, n_items))
    W = np.zeros((B, n_items))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=n_items)
    return W


def boot_weights_grouped(groups, B: int = 4000, seed=0):
    """[B, N] item multiplicities when whole GROUPS are resampled with replacement (items that share a group move together)."""
    groups = [str(g) for g in groups]
    ug, ginv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), size=(B, len(ug)))
    Wg = np.zeros((B, len(ug)))
    for b in range(B):
        Wg[b] = np.bincount(idx[b], minlength=len(ug))
    return Wg[:, ginv]


def pct_ci(draws, level: float = 0.95):
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    a = (1.0 - level) / 2.0 * 100.0
    return float(np.percentile(d, a)), float(np.percentile(d, 100.0 - a))


def boot_p(draws):
    """Two-sided percentile bootstrap p-value for 'the statistic is 0' (the +1 keeps it above 0)."""
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    lo = (1 + np.sum(d <= 0)) / (len(d) + 1)
    hi = (1 + np.sum(d >= 0)) / (len(d) + 1)
    return float(min(1.0, 2.0 * min(lo, hi)))


def boot_mean(values, W):
    """Draws of the item-weighted mean of per-item ``values`` [N] under weights W [B, N]."""
    v = np.asarray(values, dtype=float)
    return (W @ v) / W.sum(1)


def boot_ratio_skill(item_risk, item_base_risk, W):
    """Draws of 1 - mean(item_risk)/mean(item_base_risk) with the base risk resampled together with the items."""
    return 1.0 - boot_mean(item_risk, W) / boot_mean(item_base_risk, W)


def summarize(point, draws, level: float = 0.95):
    lo, hi = pct_ci(draws, level)
    return {"estimate": float(point), "ci_lo": lo, "ci_hi": hi, "p": boot_p(draws)}
