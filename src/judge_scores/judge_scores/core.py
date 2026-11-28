"""Item-level strictly proper scores, base-rate forecasts, skill and cross-fitting folds.

Conventions (identical to experiments/44_*/src/lib44.py and experiments/46_*/scripts/analyze_rank_reversal.py):
  * labels are integers 0..K-1; ordered labels are ordered by that integer.
  * a forecast is a probability vector q over the K labels; it is floored at 1e-6 by ``floor_probs`` before scoring.
  * an item's ratings are given as counts [N, K]; the item's score is the EXPECTED score over its own ratings, so the mean over
    items is the unbiased per-sample proper risk (one rating per item is enough for an unbiased risk).
  * all scores are losses: lower is better.
        brier = sum_k (q_k - 1{y=k})^2                         range [0, 2]
        ce    = -log q_y                                       range [0, -log 1e-6]
        nrps  = (K-1)^-1 sum_{m<K-1} (Q_m - 1{y<=m})^2         range [0, 1]  (normalized ranked probability score)
  * skill = 1 - risk(judge) / risk(base-rate forecast); 0 means no better than the human base rate, 1 is a perfect forecast.
"""
from __future__ import annotations

import numpy as np

FLOOR = 1e-6
RULES = ("brier", "ce", "nrps")


def floor_probs(q, floor: float = FLOOR):
    """Mix a forecast with a tiny uniform component so every label has probability >= floor (rows still sum to 1)."""
    q = np.asarray(q, dtype=float)
    k = q.shape[-1]
    return (1.0 - k * floor) * q + floor


def ratings_to_counts(ratings, K: int):
    """Per-item rating lists (each a sequence of integer labels in 0..K-1) -> counts [N, K]."""
    out = np.zeros((len(ratings), K))
    for i, r in enumerate(ratings):
        r = np.asarray(r, dtype=int)
        if r.size == 0:
            raise ValueError(f"item {i} has no ratings")
        if r.min() < 0 or r.max() >= K:
            raise ValueError(f"item {i} has a rating outside 0..{K - 1}")
        out[i] = np.bincount(r, minlength=K)
    return out


def human_dist(counts):
    """Counts [N, K] -> each item's empirical rating distribution."""
    c = np.asarray(counts, dtype=float)
    s = c.sum(-1, keepdims=True)
    if np.any(s <= 0):
        raise ValueError("an item has no ratings")
    return c / s


def _check_forecast(q):
    q = np.asarray(q, dtype=float)
    if q.ndim != 2:
        raise ValueError("forecasts must be a 2-D array [items, labels]")
    if np.any(q < 0) or not np.allclose(q.sum(1), 1.0, atol=1e-8):
        raise ValueError("every forecast row must be nonnegative and sum to 1")
    return q


def scores_vs_distribution(q, h, rule: str):
    """Expected score of forecasts q [N, K] when the rating is drawn from the distributions h [N, K] (rows sum to 1)."""
    q = _check_forecast(q)
    h = np.asarray(h, dtype=float)
    if q.shape != h.shape:
        raise ValueError("forecasts and rating distributions must have the same shape")
    if rule == "brier":
        return (q * q).sum(-1) - 2.0 * (q * h).sum(-1) + 1.0
    if rule == "ce":
        with np.errstate(divide="ignore"):
            lq = np.log(q)
        return -np.where(h > 0, h * lq, 0.0).sum(-1)
    if rule == "nrps":
        if q.shape[1] < 2:
            raise ValueError("RPS needs at least two labels")
        f = np.cumsum(q, -1)[:, :-1]
        g = np.cumsum(h, -1)[:, :-1]
        return (f * f - 2.0 * f * g + g).mean(-1)
    raise ValueError(f"unknown rule {rule!r}; use one of {RULES}")


def item_scores(q, counts, rule: str):
    """Per-item expected score of the forecasts q against each item's own ratings (counts [N, K])."""
    return scores_vs_distribution(q, human_dist(counts), rule)


def risk(q, counts, rule: str) -> float:
    """Unbiased per-sample proper risk: mean over items of the item's expected score."""
    return float(item_scores(q, counts, rule).mean())


def matched_rule(K: int, ordered: bool) -> str:
    """The strictly proper rule that matches the label encoding. Ordinal RPS is only meaningful for ordered labels."""
    return "nrps" if (ordered and K > 2) else "brier"


def make_folds(n_items: int, n_folds: int = 5, seed: int = 0):
    """Random item folds 0..n_folds-1 (items, not raters, are the unit; no item appears in two folds)."""
    perm = np.random.default_rng(seed).permutation(n_items)
    fold = np.empty(n_items, dtype=int)
    fold[perm] = np.arange(n_items) % n_folds
    return fold


def make_group_folds(groups, n_folds: int = 5, seed=0):
    """Folds over GROUPS: every item of a group (for example all SNLI items that share an image) lands in the same fold.
    ``seed`` may be an int or a numpy SeedSequence."""
    groups = [str(g) for g in groups]
    ug = sorted(set(groups))
    perm = np.random.default_rng(seed).permutation(len(ug))
    fold_of = {ug[j]: pos % n_folds for pos, j in enumerate(perm)}
    return np.array([fold_of[g] for g in groups])


def base_rate_forecast(counts, floor: float = FLOOR):
    """The pooled human rating histogram as a floored probability vector [K]."""
    h = np.asarray(counts, dtype=float).sum(0)
    return floor_probs(h / h.sum(), floor)


def crossfit_base_rate(counts, folds, floor: float = FLOOR):
    """[N, K] base-rate forecasts: each item's base rate is the histogram of the ratings of the items in the OTHER folds."""
    counts = np.asarray(counts, dtype=float)
    out = np.zeros_like(counts)
    for f in np.unique(folds):
        te = folds == f
        out[te] = base_rate_forecast(counts[~te], floor)
    return out


def skill(risk_judge, risk_base):
    """1 - risk(judge)/risk(base rate). Works on scalars or arrays of bootstrap draws."""
    if np.any(np.asarray(risk_base) <= 0):
        raise ValueError("the base-rate risk must be positive")
    return 1.0 - np.asarray(risk_judge) / np.asarray(risk_base)
