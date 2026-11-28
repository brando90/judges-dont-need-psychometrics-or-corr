"""One-call report: agreement and correlation beside the matched proper score, as skill over the human base rate."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .bootstrap import boot_mean, boot_ratio_skill, boot_weights, summarize
from .calibration import calibration_split
from .core import (RULES, crossfit_base_rate, floor_probs, human_dist, item_scores, make_folds, matched_rule)


def mode_weights(rows):
    """Each row's argmax set with ties shared equally (the repo's top-1 agreement convention)."""
    t = rows >= rows.max(1, keepdims=True) - 1e-12
    return t / t.sum(1, keepdims=True)


def _weighted_corr(a, b, w):
    W = w.sum()
    ca, cb = a - (w * a).sum() / W, b - (w * b).sum() / W
    den = np.sqrt((w * ca * ca).sum() * (w * cb * cb).sum())
    return (w * ca * cb).sum() / den if den > 1e-15 else np.nan


def agreement_vs_skill_report(q, counts, ordered: bool, rule: str | None = None, B: int = 4000, seed: int = 0,
                              n_folds: int = 5, calibrate: bool = False):
    """Top-1 agreement with the human majority, the always-majority baseline, Pearson and Spearman of the forecast mean against
    the human item mean (ordered labels only), and skill under the matched strictly proper rule, with item-clustered bootstrap
    95% intervals and the base rate cross-fitted over item folds.

    Raises ValueError for the ordinal RPS on unordered labels: RPS depends on the label order, so that score is meaningless there."""
    q = floor_probs(np.asarray(q, dtype=float))
    counts = np.asarray(counts, dtype=float)
    n, K = counts.shape
    rule = rule or matched_rule(K, ordered)
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    if rule == "nrps" and not ordered and K > 2:
        raise ValueError("the normalized ranked probability score needs ordered labels; use 'brier' or 'ce' for unordered labels")
    h = human_dist(counts)
    folds = make_folds(n, n_folds, seed)
    base = crossfit_base_rate(counts, folds)
    s_j, s_b = item_scores(q, counts, rule), item_scores(base, counts, rule)
    W = boot_weights(n, B, seed + 1)
    out = {"n_items": n, "rule": rule, "ordered": bool(ordered), "B": B, "seed": seed}
    top1_items = (mode_weights(q) * mode_weights(h)).sum(1)
    maj = int(np.argmax(counts.sum(0)))
    base_items = mode_weights(h)[:, maj]
    out["top1_agreement"] = summarize(top1_items.mean(), boot_mean(top1_items, W))
    out["always_majority_agreement"] = summarize(base_items.mean(), boot_mean(base_items, W))
    out["risk"] = float(s_j.mean())
    out["base_risk"] = float(s_b.mean())
    out["skill"] = summarize(1.0 - s_j.mean() / s_b.mean(), boot_ratio_skill(s_j, s_b, W))
    if ordered:
        v = np.arange(K, dtype=float)
        x, y = q @ v, h @ v
        out["pearson"] = summarize(_weighted_corr(x, y, np.ones(n)), [_weighted_corr(x, y, W[b]) for b in range(B)])
        rx, ry = rankdata(x), rankdata(y)
        out["spearman"] = summarize(_weighted_corr(rx, ry, np.ones(n)), [_weighted_corr(rx, ry, W[b]) for b in range(B)])
    if calibrate:
        sp = calibration_split(q, counts, rule, folds)
        out["calibration_split"] = {k: sp[k] for k in ("unc", "risk_raw", "risk_cal", "calibration_error", "resolution",
                                                       "skill_raw", "skill_cal")}
    return out
