"""Shared setup for the Experiment 40 scripts.

Reuses Experiment 36's loaders, group split, features, head fitting and per-item loss code UNCHANGED
(the experiment folder 36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026 is imported, never edited).
Everything added here is new: re-binning to a different number of classes K, the shared one-dimensional calibration map,
paired item-clustered bootstrap, and the human-entropy strata.

Notation. c is a prompt (an item). y is one human rating. q(c) is a forecast distribution on K ordered classes.
RPS is the ranked probability score; "normalized RPS" is RPS/(K-1), range 0-1, lower is better.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
EXP40 = HERE.parent
E36 = EXP40.parent / "36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026"
CACHE = Path(os.environ.get("E40_DATA_CACHE", "/tmp/e40_data_cache"))
if not CACHE.exists():
    shutil.copytree(E36 / "data_cache", CACHE)  # POPQUORN and VeriBench R_v1 are committed there; the rest download once.
os.environ["SANMI36_DATA"] = str(CACHE)
sys.path.insert(0, str(E36))
sys.dont_write_bytecode = True

import data_io as d  # noqa: E402
import fit_eval as fe  # noqa: E402
from metrics import brier_one, ce_one, rps_one, softmax  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------------
# Re-binning a rating set to a different number of classes K. Same prompts, same raters, same groups; only the label map changes.
# ----------------------------------------------------------------------------------------------------------------------
def rebin(data, value_to_new: dict, new_values, name: str, kind: str):
    """Map each original class VALUE to a new class index. A threshold or grouping of the same ratings, not a new collection."""
    old_values = [float(v) for v in data.class_values]
    mapping = np.array([value_to_new[v] for v in old_values], dtype=int)
    ratings = [mapping[np.asarray(r, dtype=int)] for r in data.ratings]
    note = f"{name}: re-binned from {data.name} by value map {value_to_new}; " + data.notes
    return d.JudgingSet(name, kind, list(new_values), data.texts, data.groups, ratings, data.rater_ids, note)


def three_level(data, name: str):
    vals = sorted(float(v) for v in data.class_values)
    if len(vals) == 5:  # 1..5 -> {1,2}|{3}|{4,5}
        m = {vals[0]: 0, vals[1]: 0, vals[2]: 1, vals[3]: 2, vals[4]: 2}
    elif len(vals) == 6:  # 0..5 -> {0,1}|{2,3}|{4,5}
        m = {vals[0]: 0, vals[1]: 0, vals[2]: 1, vals[3]: 1, vals[4]: 2, vals[5]: 2}
    else:
        raise ValueError("three_level is defined for 5- and 6-level scales only")
    return rebin(data, m, [0, 1, 2], name, "ordinal")


# ----------------------------------------------------------------------------------------------------------------------
# Shared one-dimensional calibration map (the matched-calibration control).
# ----------------------------------------------------------------------------------------------------------------------
def fit_logit_map(z, targets, ridge=1e-4):
    """Multinomial logit on [z, 1] fitted by cross-entropy to the VALIDATION items' empirical rating histograms."""
    x = np.column_stack([z, np.ones_like(z)])
    n, k = targets.shape

    def f(w):
        wm = w.reshape(2, k)
        p = softmax(x @ wm)
        ll = -(targets * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1).mean()
        g = x.T @ (p - targets) / n
        g[0] += ridge * wm[0]
        return float(ll + 0.5 * ridge * (wm[0] ** 2).sum()), g.ravel()

    res = minimize(f, np.zeros(2 * k), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x.reshape(2, k)


def histograms(ratings, idx, k):
    return np.stack([np.bincount(np.asarray(ratings[i], int), minlength=k) / len(ratings[i]) for i in idx])


def calibrate(mu_val, mu_test, ratings, val_idx, k):
    """Standardize the scalar mean score with VALIDATION moments, fit the map on validation ratings, apply to test."""
    m, s = float(mu_val.mean()), float(mu_val.std())
    s = s if s > 1e-12 else 1.0
    w = fit_logit_map((mu_val - m) / s, histograms(ratings, val_idx, k))
    x = np.column_stack([(mu_test - m) / s, np.ones_like(mu_test)])
    return softmax(x @ w)


# ----------------------------------------------------------------------------------------------------------------------
# Paired item-clustered bootstrap.
# ----------------------------------------------------------------------------------------------------------------------
def paired_ci(a, b, boot=2000, seed=1):
    """Mean of (a - b) over items, with a 95% item-clustered percentile interval. a and b are per-item arrays on the SAME items."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = a.shape[0]
    idx = np.random.default_rng(seed).integers(0, n, size=(boot, n))
    dd = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))


def per_item_on(prob_rows, ratings, idx, y_sets, k):
    """Per-item mean cross-entropy and normalized RPS over a chosen subset of each item's ratings."""
    ce, nr = [], []
    for row, i, ys in zip(prob_rows, idx, y_sets):
        ce.append(np.mean([ce_one(row, int(y)) for y in ys]))
        nr.append(np.mean([rps_one(row, int(y)) for y in ys]) / (k - 1))
    return np.asarray(ce), np.asarray(nr)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
    return obj
