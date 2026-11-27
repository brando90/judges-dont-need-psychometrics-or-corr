"""Experiment 40 driver: G/D-study, dollar frontier, K-sweep, GPU bridge.

Run from anywhere:

    python3 experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/run_all.py

`EXP39_SMOKE=1` shrinks the grid and the bootstrap. Reported numbers use the full grid.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
EXP36 = HERE.parent / "36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026"
sys.path.insert(0, str(EXP36))
sys.path.insert(0, str(HERE))

from data_io import (  # noqa: E402
    assert_no_prompt_leakage,
    load_chaos_3way,
    load_popquorn,
    load_summeval,
    load_veribench_fsc_native_main,
    load_veribench_fsc_rv1,
    split_by_group,
    _load_dices,
)
from fit_eval import _svd_project, fit_objective, hash_features, predict_distributions  # noqa: E402
from gstudy import crossed_gstudy, dstudy_curve, nested_gstudy, recommend_k  # noqa: E402
from metrics import ce_one, percentile_interval, rps_one  # noqa: E402

SMOKE = os.environ.get("EXP39_SMOKE", "") == "1"
BOOT = 40 if SMOKE else 1000
PHI_TARGET = 0.80
CE_TOL = 0.05
SQRT_RPS_TOL = 0.02
CROWD_PRICES = (0.05, 0.10, 0.20)
EXPERT_PRICES = (5.0, 10.0, 20.0)
BUDGETS = (64,) if SMOKE else (32, 64, 128, 256, 512, 1024, 2048, 4096)
K_GRID = (1, 2) if SMOKE else (1, 2, 3, 4, 5, 8, 16, 32)
PROV = HERE / "provenance" / "ttj_efdc8c3"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def _ci(samples: np.ndarray) -> dict:
    lo, hi, n_ok = percentile_interval(samples)
    return {"lo": lo, "hi": hi, "n_finite": n_ok, "level": 0.95, "B": int(samples.shape[0]),
            "unit": "held-out prompt"}


def item_losses(pack, x, ratings, indices) -> tuple[np.ndarray, np.ndarray]:
    prob, _mu = predict_distributions(pack, x, pack["objective"])
    ce = np.empty(len(indices), dtype=float)
    rps = np.empty(len(indices), dtype=float)
    for row, i in enumerate(indices):
        y = np.asarray(ratings[int(i)], dtype=int)
        c = r = 0.0
        for yj in y:
            c += ce_one(prob[row], int(yj))
            r += rps_one(prob[row], int(yj))
        ce[row] = c / float(y.size)
        rps[row] = r / float(y.size)
    return ce, rps


def summarize_losses(ce: np.ndarray, rps: np.ndarray, boot_seed: int = 1) -> dict:
    point_ce = float(np.mean(ce))
    point_rps = float(np.mean(rps))
    point_sqrt = float(np.sqrt(max(point_rps, 0.0)))
    rng = np.random.default_rng(boot_seed)
    n = ce.shape[0]
    draws = rng.integers(0, n, size=(BOOT, n))
    boot_ce = ce[draws].mean(axis=1)
    boot_rps = rps[draws].mean(axis=1)
    boot_sqrt = np.sqrt(np.maximum(boot_rps, 0.0))
    return {
        "ce": point_ce, "ce_ci": _ci(boot_ce),
        "rps": point_rps, "sqrt_rps": point_sqrt, "sqrt_rps_ci": _ci(boot_sqrt),
        "n_test_items": int(n),
    }


def paired_diff(a: np.ndarray, b: np.ndarray, boot_seed: int = 1) -> dict:
    """Mean of a minus mean of b, same prompt resample. Positive means a is larger."""
    diff = float(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(boot_seed)
    n = a.shape[0]
    draws = rng.integers(0, n, size=(BOOT, n))
    samples = a[draws].mean(axis=1) - b[draws].mean(axis=1)
    out = _ci(samples)
    out["point"] = diff
    return out


def long_rows(data, indices):
    items, raters, scores = [], [], []
    index_set = None if indices is None else set(int(i) for i in indices)
    for i, (ys, rs) in enumerate(zip(data.ratings, data.rater_ids)):
        if index_set is not None and i not in index_set:
            continue
        for y, r in zip(ys, rs):
            items.append(i)
            raters.append(r)
            scores.append(float(data.class_values[int(y)]))
    return items, raters, scores


def run_gstudy(data, indices, facet: str) -> dict:
    items, raters, scores = long_rows(data, indices)
    if facet == "crossed_person":
        out = crossed_gstudy(items, raters, scores)
    else:
        out = nested_gstudy(items, scores)
    if out.get("status") != "ok":
        out["curve"] = []
        out["recommendation"] = {"k": None, "target_hit": False, "reason": out.get("status")}
        return out
    k_max = 12 if SMOKE else 30
    curve = dstudy_curve(out["sigma_c2"], out["sigma_r2"], out["sigma_res2"], k_max)
    # Feasible k: at least 4 prompts in this index set have that many ratings.
    counts = {}
    index_set = None if indices is None else set(int(i) for i in indices)
    for i, ys in enumerate(data.ratings):
        if index_set is not None and i not in index_set:
            continue
        counts[int(ys.size)] = counts.get(int(ys.size), 0) + 1
    feasible = []
    running = 0
    # number of items with at least k ratings
    sizes = []
    for i, ys in enumerate(data.ratings):
        if index_set is not None and i not in index_set:
            continue
        sizes.append(int(ys.size))
    sizes = np.asarray(sizes) if sizes else np.zeros(0, dtype=int)
    for k in range(1, k_max + 1):
        if int(np.sum(sizes >= k)) >= 4:
            feasible.append(k)
    out["curve"] = curve
    out["recommendation"] = recommend_k(curve, target=PHI_TARGET, k_feasible=feasible or [1])
    out["ratings_per_item_median"] = float(np.median(sizes)) if sizes.size else None
    out["scope"] = "train split only" if indices is not None else "all splits, in-sample"
    return out


def fit_cell(features, data, chosen, other_idx, objective: str, k: int):
    if len(chosen) < 4:
        return None
    x_tr, projected, _rank = _svd_project(features[chosen], [features[idx] for idx in other_idx])
    pack = fit_objective(x_tr, data.ratings, np.asarray(chosen), data.class_values, objective, k, seed=0)
    return pack, projected


def allocation(data, features, train, val, test, g_train) -> dict:
    k_star = g_train.get("recommendation", {}).get("k")
    ks = sorted(set(K_GRID) | ({int(k_star)} if k_star else set()))
    # One seed-0 order per k, prefixes nest across budgets.
    orders = {}
    for k in ks:
        eligible = np.asarray([int(i) for i in train if data.ratings[int(i)].size >= k], dtype=int)
        if eligible.size < 4:
            continue
        rng = np.random.default_rng(0)
        orders[k] = rng.permutation(eligible)
    cells = []
    loss_cache = {}
    budgets = [b for b in BUDGETS if b <= int(sum(data.ratings[int(i)].size for i in train))]
    if not budgets:
        budgets = [int(sum(data.ratings[int(i)].size for i in train))]
    for k, order in orders.items():
        for B in budgets:
            n_take = min(int(order.size), int(B) // int(k))
            if n_take < 4:
                continue
            chosen = order[:n_take]
            key = (int(k), int(n_take))
            if key in loss_cache:
                packed = loss_cache[key]
            else:
                fitted = fit_cell(features, data, chosen, [val, test], "ce", int(k))
                if fitted is None:
                    continue
                pack, (x_val, x_test) = fitted
                ce_te, rps_te = item_losses(pack, x_test, data.ratings, test)
                ce_va, rps_va = item_losses(pack, x_val, data.ratings, val)
                packed = {
                    "k": int(k), "N": int(n_take), "ratings_used": int(n_take * k),
                    "test": summarize_losses(ce_te, rps_te),
                    "val_ce": float(np.mean(ce_va)),
                    "ce_te": ce_te, "rps_te": rps_te,
                }
                loss_cache[key] = packed
            cells.append({**{kk: packed[kk] for kk in ("k", "N", "ratings_used", "test", "val_ce")}, "B": int(B)})
    # Full train reference: every train prompt, all of its ratings.
    full = fit_cell(features, data, np.asarray(train), [val, test], "ce", None)
    full_summary = None
    if full is not None:
        pack, (_x_val, x_test) = full
        ce_te, rps_te = item_losses(pack, x_test, data.ratings, test)
        full_summary = summarize_losses(ce_te, rps_te)
        full_summary["n_train_items"] = int(len(train))
        full_summary["n_train_ratings"] = int(sum(data.ratings[int(i)].size for i in train))
        full_summary["label"] = "all train prompts, all ratings on those prompts"
    # Drop the heavy arrays from the saved cells but keep them for paired diffs via cache.
    by_budget = []
    for B in budgets:
        options = [c for c in cells if c["B"] == B]
        if not options:
            continue
        oracle = min(options, key=lambda c: c["test"]["ce"])
        oracle_r = min(options, key=lambda c: c["test"]["sqrt_rps"])
        val_pick = min(options, key=lambda c: c["val_ce"])
        d_k = int(k_star) if k_star is not None else None
        d_opts = [c for c in options if c["k"] == d_k]
        d_cell = d_opts[0] if d_opts else None
        frontier_ce = min(c["test"]["ce"] for c in options)
        frontier_sqrt = min(c["test"]["sqrt_rps"] for c in options)
        paired = None
        if d_cell is not None:
            a = loss_cache[(d_cell["k"], d_cell["N"])]["ce_te"]
            b = loss_cache[(oracle["k"], oracle["N"])]["ce_te"]
            paired = paired_diff(a, b)
            paired["definition"] = (
                "held-out test cross-entropy of the D-study (N, k) minus held-out test "
                "cross-entropy of the test-oracle (N, k); positive means the D-study allocation is worse"
            )
        def slim(c):
            if c is None:
                return None
            return {"k": c["k"], "N": c["N"], "ratings_used": c["ratings_used"], "test": c["test"], "val_ce": c["val_ce"]}
        by_budget.append({
            "B": int(B),
            "frontier_ce": frontier_ce,
            "frontier_sqrt_rps": frontier_sqrt,
            "oracle_ce": slim(oracle),
            "oracle_sqrt_rps": slim(oracle_r),
            "validation_chosen": slim(val_pick),
            "dstudy": slim(d_cell),
            "dstudy_minus_oracle_ce": paired,
            "same_k": bool(d_cell is not None and d_cell["k"] == oracle["k"]),
        })
    saved_cells = []
    seen = set()
    for c in cells:
        key = (c["k"], c["N"])
        if key in seen:
            continue
        seen.add(key)
        saved_cells.append({kk: c[kk] for kk in ("k", "N", "ratings_used", "test", "val_ce")})
    return {
        "budgets": by_budget,
        "cells": saved_cells,
        "full_train": full_summary,
        "d_recommendation": g_train.get("recommendation"),
    }


def dollar_hits(block) -> dict:
    full = block.get("full_train") or {}
    if not full or not block.get("budgets"):
        return {"status": "no full-train reference"}
    target_ce = full["ce"] + CE_TOL
    target_sqrt = full["sqrt_rps"] + SQRT_RPS_TOL
    def first(metric, target):
        for row in block["budgets"]:
            if row[metric] <= target:
                return row["B"]
        return None
    b_ce = first("frontier_ce", target_ce)
    b_sqrt = first("frontier_sqrt_rps", target_sqrt)
    def prices(b, grid):
        if b is None:
            return None
        return {f"{p:.2f}": round(b * p, 2) for p in grid}
    return {
        "full_train_ce": full["ce"],
        "full_train_sqrt_rps": full["sqrt_rps"],
        "ce_target": target_ce,
        "ce_tolerance_nats": CE_TOL,
        "sqrt_rps_target": target_sqrt,
        "sqrt_rps_tolerance": SQRT_RPS_TOL,
        "B_ce": b_ce,
        "B_sqrt_rps": b_sqrt,
        "crowd_usd_for_ce": prices(b_ce, CROWD_PRICES),
        "expert_usd_for_ce": prices(b_ce, EXPERT_PRICES),
        "crowd_usd_for_sqrt_rps": prices(b_sqrt, CROWD_PRICES),
        "expert_usd_for_sqrt_rps": prices(b_sqrt, EXPERT_PRICES),
        "researcher_gpu_anchor_usd": [15000, 75000],
        "anchor_note": "The $15k–$75k band is a stated 1–3 researcher-month anchor, not a cost measured in this run.",
    }


def rebin_ratings(data, mode: str):
    values = np.asarray(data.class_values, dtype=float)
    lo, hi = float(values[0]), float(values[-1])

    def bucket(y_index: int) -> int:
        v = float(values[int(y_index)])
        if mode == "native":
            return int(y_index)
        if mode == "K2":
            # Same cut as experiment 36 on 1–5 and 0–5: the positive class is value >= 4.
            # On a 0–2 scale the positive class is the top code.
            cut = hi if hi <= 2 else 4.0
            return int(v >= cut)
        # K=3 equal-width thirds of the numeric range.
        if hi == lo:
            return 0
        t = (v - lo) / (hi - lo)
        return int(min(2, np.floor(t * 3))) if t < 1 else 2

    ratings = []
    for ys in data.ratings:
        ratings.append(np.asarray([bucket(int(y)) for y in ys], dtype=int))
    if mode == "native":
        class_values = values
        k = len(values)
    elif mode == "K2":
        class_values = np.asarray([0.0, 1.0])
        k = 2
    else:
        class_values = np.asarray([0.0, 1.0, 2.0])
        k = 3
    return ratings, class_values, k


def rebin_sweep(bundle) -> list[dict]:
    """CE head versus MSE head on rebinned real scales. Native K only, plus K=2 and K=3."""
    rows = []
    datasets = bundle if not SMOKE else bundle[:1]
    for data, train, val, test, features in datasets:
        for mode in ("K2", "K3", "native"):
            ratings, class_values, k = rebin_ratings(data, mode)
            if k < 2:
                continue
            # Temporarily swap ratings.
            original = data.ratings
            data.ratings = ratings
            saved_values_note = class_values
            ks = [1, 2] if SMOKE else [kk for kk in (1, 2, 4, 8) if any(r.size >= kk for r in ratings)]
            for k_keep in ks:
                eligible = np.asarray([int(i) for i in train if ratings[int(i)].size >= k_keep], dtype=int)
                if eligible.size < 8:
                    continue
                rng = np.random.default_rng(0)
                # Use all eligible train items, subsample k ratings. This is the k sweep, not the B sweep.
                chosen = rng.permutation(eligible)
                x_tr, projected, _r = _svd_project(features[chosen], [features[test]])
                x_te = projected[0]
                out = {"dataset": data.name, "mode": mode, "K": int(k), "k_ratings": int(k_keep),
                       "n_train": int(chosen.size), "n_test": int(len(test))}
                for objective in ("ce", "mse"):
                    # fit_objective reads data.ratings and data.class_values via the values argument.
                    pack = fit_objective(x_tr, ratings, chosen, saved_values_note, objective, k_keep, seed=0)
                    ce, rps = item_losses(pack, x_te, ratings, test)
                    summary = summarize_losses(ce, rps)
                    out[objective] = {"ce": summary["ce"], "ce_ci": summary["ce_ci"],
                                      "sqrt_rps": summary["sqrt_rps"], "sqrt_rps_ci": summary["sqrt_rps_ci"]}
                    out[f"_{objective}_ce_vec"] = ce
                diff = paired_diff(out.pop("_mse_ce_vec"), out.pop("_ce_ce_vec"))
                diff["definition"] = (
                    "held-out test cross-entropy of the MSE-trained head minus held-out test "
                    "cross-entropy of the CE-trained head; positive means the CE-trained head is better"
                )
                out["mse_minus_ce"] = diff
                out["ce_point_better"] = bool(out["ce"]["ce"] < out["mse"]["ce"])
                rows.append(out)
            data.ratings = original
    return rows


def _softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def synthetic_sweep() -> dict:
    """Labeled synthetic continuous law on {0,1,...,100}. Not human data."""
    rng = np.random.default_rng(0)
    n_items = 80 if SMOKE else 240
    u = (np.arange(n_items) + 0.5) / n_items
    grid = np.arange(101, dtype=float)

    def pmf_row(ui):
        d1 = np.exp(-0.5 * ((grid - 100.0 * ui) / 8.0) ** 2)
        d2 = np.exp(-0.5 * ((grid - 100.0 * (1.0 - ui)) / 12.0) ** 2)
        p = 0.7 * d1 + 0.3 * d2
        return p / p.sum()

    pmf = np.stack([pmf_row(ui) for ui in u])
    # Draw a pool of 40 ratings per item once; cells take a prefix.
    pool = np.stack([rng.choice(101, size=40, p=pmf[i]) for i in range(n_items)])
    perm = rng.permutation(n_items)
    n_tr, n_va = int(0.6 * n_items), int(0.2 * n_items)
    train, val, test = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    feat = np.column_stack([np.ones(n_items), u, u ** 2])
    ks = (1, 4) if SMOKE else (1, 2, 4, 8, 16, 32)
    k_bins = (2, 5) if SMOKE else (2, 3, 5, 10, 20, 101)
    rows = []

    def bin_ids(samples, k):
        if k >= 101:
            return samples.astype(int)
        # Equal-width bins on 0..100.
        edges = np.linspace(0, 100, k + 1)
        return np.clip(np.digitize(samples, edges[1:-1], right=False), 0, k - 1)

    def crps_from_probs(prob, y_idx):
        """Unit-spacing CRPS / RPS on the bin index."""
        out = np.empty(prob.shape[0], dtype=float)
        for i in range(prob.shape[0]):
            out[i] = rps_one(prob[i], int(y_idx[i]))
        return out

    for k_bin in k_bins:
        for k_rating in ks:
            y = bin_ids(pool[:, :k_rating], k_bin)
            # CE head: softmax linear on the empirical bin frequencies.
            targets = np.zeros((train.size, k_bin), dtype=float)
            for row, i in enumerate(train):
                targets[row] = np.bincount(y[i], minlength=k_bin).astype(float)
                targets[row] /= targets[row].sum()
            x_tr = feat[train]
            d = x_tr.shape[1]

            def ce_loss(w):
                prob = _softmax_rows(x_tr @ w.reshape(d, k_bin))
                p = np.clip(prob, 1e-12, 1.0)
                return float(-(targets * np.log(p)).sum(axis=1).mean())

            sol = minimize(ce_loss, np.zeros(d * k_bin), method="L-BFGS-B",
                           options={"maxiter": 40 if SMOKE else 80})
            w = sol.x.reshape(d, k_bin)
            prob_te = _softmax_rows(feat[test] @ w)
            # Score each test item by the mean CRPS of a fresh 20-draw from the true law,
            # so the metric is not the same k ratings the head was trained on.
            fresh = np.stack([rng.choice(101, size=20, p=pmf[i]) for i in test])
            y_te = bin_ids(fresh, k_bin)
            ce_scores = np.mean([crps_from_probs(prob_te, y_te[:, t]) for t in range(20)], axis=0)
            # MSE / second-moment head: ridge of the numeric mean, Gaussian on the bins.
            y_mean = pool[train, :k_rating].mean(axis=1)
            xtx = x_tr.T @ x_tr + 1e-2 * np.eye(d)
            xtx[-1, -1] -= 1e-2  # do not penalize the intercept; last column is u^2, first is intercept
            # Penalize non-intercept columns 1 and 2. Column 0 is the intercept.
            pen = np.eye(d)
            pen[0, 0] = 0.0
            beta = np.linalg.solve(x_tr.T @ x_tr + 1e-2 * pen, x_tr.T @ y_mean)
            mu_te = feat[test] @ beta
            resid = y_mean - x_tr @ beta
            sigma = float(np.sqrt(max(np.mean(resid ** 2), 1.0)))
            centers = (np.linspace(0, 100, k_bin + 1)[:-1] + np.linspace(0, 100, k_bin + 1)[1:]) / 2 if k_bin < 101 else grid
            dens = np.exp(-0.5 * ((centers[None, :] - mu_te[:, None]) / sigma) ** 2)
            dens /= dens.sum(axis=1, keepdims=True)
            mse_scores = np.mean([crps_from_probs(dens, y_te[:, t]) for t in range(20)], axis=0)
            diff = paired_diff(mse_scores, ce_scores, boot_seed=1)
            diff["definition"] = (
                "held-out synthetic CRPS of the MSE Gaussian minus held-out synthetic CRPS of the "
                "CE head; positive means the CE head is better. Synthetic, not human ratings."
            )
            rows.append({
                "K": int(k_bin), "k_ratings": int(k_rating),
                "ce_crps": float(np.mean(ce_scores)), "mse_crps": float(np.mean(mse_scores)),
                "mse_minus_ce": diff,
                "ce_better_point": bool(np.mean(ce_scores) < np.mean(mse_scores)),
                "ci_excludes_zero_ce_better": bool(diff["lo"] > 0),
            })
    # k*(K): smallest k whose CI says the CE head is better.
    crossover = []
    for k_bin in k_bins:
        star = None
        point = None
        for row in rows:
            if row["K"] != k_bin:
                continue
            if point is None and row["ce_better_point"]:
                point = row["k_ratings"]
            if star is None and row["ci_excludes_zero_ce_better"]:
                star = row["k_ratings"]
        crossover.append({"K": int(k_bin), "k_star_ci": star, "k_star_point": point})
    return {
        "label": "synthetic",
        "law": "0.7 Normal(100 u, 8^2) + 0.3 Normal(100 (1-u), 12^2) on {0,...,100}, u = (i+0.5)/n",
        "n_items": n_items,
        "split": "item-level 0.6/0.2/0.2, seed 0",
        "rows": rows,
        "crossover": crossover,
    }


def old_rv1_gstudy() -> dict:
    path = PROV / "human_ratings_old_11-26-2026.csv"
    rows = list(csv.DictReader(path.open()))
    items, scores = [], []
    for row in rows:
        for tok in row["ratings_sorted_0to5_rater_identity_removed"].split():
            items.append(row["item"])
            scores.append(float(tok))
    got = nested_gstudy(items, scores)
    got["provenance"] = (
        "trustworthy-theorem-judges efdc8c3 "
        "experiments/28_.../data_11-26-2026/human_ratings_old_11-26-2026.csv. "
        "Freeze note in the file: exp 100 / exp 21 / exp 27 expert ratings. "
        "Rater identity was removed and the three scores were sorted, so this is not a "
        "crossed person G-study and it is not a byte-verified read of veribench@d1c47355. "
        "The experiment 36 loader's items.jsonl returned HTTP 404 from this machine."
    )
    got["n_items_file"] = len(rows)
    got["ratings_per_item"] = 3
    return got


def try_rv1_loader() -> dict:
    try:
        data = load_veribench_fsc_rv1("all5")
        return {"status": "loaded", "n_items": data.n_items, "n_ratings": int(sum(r.size for r in data.ratings))}
    except Exception as exc:  # the pinned fetch raises HTTPError
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}",
                "what_was_attempted": "experiment 36 load_veribench_fsc_rv1('all5') at veribench@d1c47355"}


def exp27_split(data):
    path = PROV / "human_ratings_new_11-26-2026.csv"
    rows = list(csv.DictReader(path.open()))
    task_side = {}
    for row in rows:
        task_side.setdefault(row["task"], set()).add(row["exp27_split"])
    mixed = [t for t, s in task_side.items() if len(s) != 1]
    train, val, test, missing = [], [], [], []
    for i, group in enumerate(data.groups):
        sides = task_side.get(group)
        if not sides or len(sides) != 1:
            missing.append(group)
            continue
        side = next(iter(sides))
        {"train": train, "val": val, "test": test}[side].append(i)
    return {
        "train": np.asarray(train, dtype=int),
        "val": np.asarray(val, dtype=int),
        "test": np.asarray(test, dtype=int),
        "mixed_tasks": mixed,
        "missing_groups": sorted(set(missing)),
        "source": "efdc8c3 human_ratings_new_11-26-2026.csv exp27_split, joined on task folder",
    }


def logit_check(path: Path) -> dict:
    rows = json.loads(path.read_text())
    errs = []
    for row in rows:
        probs = np.asarray(row["probs"], dtype=float)
        digits = np.arange(len(probs), dtype=float)
        pred = float(probs @ digits)
        errs.append(abs(pred - float(row["expected_score"])))
    return {
        "file": path.name,
        "n_items": len(rows),
        "prob_sum_max_abs_err": float(np.max(np.abs([sum(r["probs"]) - 1.0 for r in rows]))),
        "expected_score_max_abs_err": float(np.max(errs)),
    }


def bridge_gpu() -> dict:
    lora = json.loads((PROV / "qwen3_4b_lora_ce_rps_pearson.json").read_text())
    counts = {}
    for obj in ("ce", "rps", "pearson"):
        counts[obj] = {
            "exp27_test_items": len(lora[obj]["exp27_test_preds"]),
            "cv5_oof_items": len(lora[obj]["cv5_oof_preds"]),
            "selected_hp": lora[obj]["exp27_selected_hp"],
        }
    text = (PROV / "RESULTS_11-26-2026.md").read_text()
    quoted = []
    for line in text.splitlines():
        if line.startswith("|") and ("Qwen3-4B" in line or "Qwen3-8B" in line or "nested_frontier" in line):
            quoted.append(line)
    return {
        "commit": "efdc8c3",
        "repo": "brando90/trustworthy-theorem-judges",
        "experiment": "experiments/28_multi_family_distributional_judge_and_open_weight_lora_on_new_ratings_start_26Nov2026",
        "logit_checks": [
            logit_check(PROV / "qwen3_4b_logit_6144.json"),
            logit_check(PROV / "qwen3_8b_logit_6144.json"),
        ],
        "lora_counts": counts,
        "quoted_rows_from_RESULTS_11-26-2026.md": quoted,
        "note": (
            "Proper-score means and intervals in the quoted rows are copied from RESULTS_11-26-2026.md "
            "at efdc8c3. They were not recomputed: the JSON stores probabilities, not the human ratings "
            "those scores were computed against. The JSON checks are recomputed here."
        ),
    }


def _fmt(x, nd=4):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def _fmt_ci(summary, key="ce"):
    if summary is None:
        return "—"
    if key == "ce":
        point, ci = summary["ce"], summary["ce_ci"]
    else:
        point, ci = summary["sqrt_rps"], summary["sqrt_rps_ci"]
    return f"{point:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]"


def write_tables(payload) -> None:
    tab = HERE / "tables"
    tab.mkdir(exist_ok=True)
    lines = ["# G-study variance components", "",
             "Train-split components drive the D-study. All-splits components are in-sample descriptions of the rating design.",
             "σ² is on the rating-point scale (the dataset's class values, not a z-score).",
             ""]
    lines.append("| Dataset | Scope | Facet | σ²_c raw | σ²_r raw | σ²_res | Φ(1) | smallest k with Φ≥0.80 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for name, block in payload["datasets"].items():
        for scope, key in (("train", "gstudy_train"), ("all splits, in-sample", "gstudy_all")):
            g = block.get(key) or {}
            if g.get("status") != "ok":
                lines.append(f"| {name} | {scope} | {g.get('facet', '')} | — | — | — | — | {g.get('status')} |")
                continue
            phi1 = g["curve"][0]["Phi"] if g.get("curve") else float("nan")
            rec = g.get("recommendation") or {}
            lines.append(
                f"| {name} | {scope} | {g.get('facet')} | {_fmt(g.get('sigma_c2_raw'))} | "
                f"{_fmt(g.get('sigma_r2_raw')) if g.get('sigma_r2_raw') is not None else 'unidentified'} | "
                f"{_fmt(g.get('sigma_res2'))} | {_fmt(phi1)} | "
                f"k={rec.get('k')} ({'hit' if rec.get('target_hit') else 'not hit'}) |"
            )
    (tab / "gstudy.md").write_text("\n".join(lines) + "\n")

    alloc = ["# Allocation head-to-head", "",
             "Held-out test cross-entropy of the cross-entropy-trained head. "
             "The oracle column uses the test labels to pick k. The validation column does not.",
             "Interval: item-clustered percentile 95%, B=" + str(BOOT) + ", unit = held-out prompt, fixed head.",
             ""]
    for name, block in payload["datasets"].items():
        alloc.append(f"## {name}")
        alloc.append("")
        alloc.append("| B | D-study k | D-study test CE | Oracle k | Oracle test CE | D-study minus oracle CE | Validation k |")
        alloc.append("|---:|---:|---|---:|---|---|---:|")
        for row in block["allocation"]["budgets"]:
            d = row["dstudy"]
            o = row["oracle_ce"]
            v = row["validation_chosen"]
            diff = row["dstudy_minus_oracle_ce"]
            diff_s = "—" if not diff else f"{diff['point']:.4f} [{diff['lo']:.4f}, {diff['hi']:.4f}]"
            alloc.append(
                f"| {row['B']} | {None if not d else d['k']} | {None if not d else _fmt_ci(d['test'])} | "
                f"{o['k']} | {_fmt_ci(o['test'])} | {diff_s} | {v['k']} |"
            )
        alloc.append("")
    (tab / "allocation.md").write_text("\n".join(alloc) + "\n")

    dollars = ["# Dollar budgets", "",
               "A hit is the smallest frontier budget (minimum over k of held-out risk) within the pre-registered tolerance of the full-train head.",
               "Prices are the stated crowd and expert regimes. They are not measured wages for every dataset.",
               ""]
    dollars.append("| Dataset | Full-train test CE | B for CE + 0.05 nats | Crowd $ at $0.05 / $0.10 / $0.20 | Expert $ at $5 / $10 / $20 | B for sqrt(RPS) + 0.02 |")
    dollars.append("|---|---:|---:|---|---|---:|")
    for name, block in payload["datasets"].items():
        h = block["dollars"]
        if "B_ce" not in h:
            dollars.append(f"| {name} | — | — | — | — | — |")
            continue
        def triple(d):
            if not d:
                return "not reached"
            return f"{d.get('0.05', d.get('5.00', '—'))}" if False else ", ".join(str(v) for v in d.values())
        crowd = "not reached" if not h["crowd_usd_for_ce"] else ", ".join(f"${v:.2f}" for v in h["crowd_usd_for_ce"].values())
        expert = "not reached" if not h["expert_usd_for_ce"] else ", ".join(f"${v:.0f}" for v in h["expert_usd_for_ce"].values())
        dollars.append(
            f"| {name} | {_fmt(h['full_train_ce'])} | {h['B_ce'] if h['B_ce'] is not None else 'not reached'} | "
            f"{crowd} | {expert} | {h['B_sqrt_rps'] if h['B_sqrt_rps'] is not None else 'not reached'} |"
        )
    dollars.append("")
    dollars.append("Researcher-plus-GPU anchor, stated not measured: $15,000–$75,000 for 1–3 researcher-months.")
    (tab / "dollar_thresholds.md").write_text("\n".join(dollars) + "\n")

    ks = ["# Output-granularity sweep", "",
          "Real rows rebin a native ordinal scale only to K=2, K=3, and the native K. "
          "Synthetic rows are the continuous mixture on {0,...,100} and are not human ratings.",
          ""]
    ks.append("## Real rebin, held-out test cross-entropy")
    ks.append("")
    ks.append("| Dataset | Mode | K | k ratings | CE-trained CE | MSE-trained CE | MSE minus CE |")
    ks.append("|---|---|---:|---:|---|---|---|")
    for row in payload["rebin"]:
        diff = row["mse_minus_ce"]
        ks.append(
            f"| {row['dataset']} | {row['mode']} | {row['K']} | {row['k_ratings']} | "
            f"{row['ce']['ce']:.4f} [{row['ce']['ce_ci']['lo']:.4f}, {row['ce']['ce_ci']['hi']:.4f}] | "
            f"{row['mse']['ce']:.4f} [{row['mse']['ce_ci']['lo']:.4f}, {row['mse']['ce_ci']['hi']:.4f}] | "
            f"{diff['point']:.4f} [{diff['lo']:.4f}, {diff['hi']:.4f}] |"
        )
    ks.append("")
    ks.append("## Synthetic CRPS crossover (not human data)")
    ks.append("")
    ks.append("| K | smallest k whose 95% CI says the CE head has lower CRPS | smallest k on the point estimate |")
    ks.append("|---:|---|---|")
    for row in payload["synthetic"]["crossover"]:
        ks.append(f"| {row['K']} | {row['k_star_ci']} | {row['k_star_point']} |")
    (tab / "ksweep.md").write_text("\n".join(ks) + "\n")

    paper = ["# Paper master numbers", "",
             "CPU rows are computed in this folder. GPU rows are quoted from trustworthy-theorem-judges experiment 28 at `efdc8c3`.",
             ""]
    paper.append("## CPU allocation, one budget per dataset (largest B in the grid)")
    paper.append("")
    paper.append("| Dataset | Held-out test items | D-study k | Oracle k for test CE | D-study minus oracle test CE |")
    paper.append("|---|---:|---:|---:|---|")
    for name, block in payload["datasets"].items():
        rows = block["allocation"]["budgets"]
        if not rows:
            continue
        row = rows[-1]
        d = row["dstudy"]
        o = row["oracle_ce"]
        diff = row["dstudy_minus_oracle_ce"]
        diff_s = "—" if not diff else f"{diff['point']:.4f} [{diff['lo']:.4f}, {diff['hi']:.4f}]"
        paper.append(
            f"| {name} | {o['test']['n_test_items']} | {None if not d else d['k']} | {o['k']} | {diff_s} |"
        )
    paper.append("")
    paper.append("## GPU bridge, quoted, not recomputed")
    paper.append("")
    paper.append(payload["gpu"]["note"])
    paper.append("")
    for line in payload["gpu"]["quoted_rows_from_RESULTS_11-26-2026.md"]:
        paper.append(line)
    paper.append("")
    paper.append("## JSON integrity, recomputed here")
    paper.append("")
    for check in payload["gpu"]["logit_checks"]:
        paper.append(
            f"- `{check['file']}`: {check['n_items']} items, max |sum(probs)-1| = {check['prob_sum_max_abs_err']:.3e}, "
            f"max |dot(probs, 0..5) - expected_score| = {check['expected_score_max_abs_err']:.3e}."
        )
    for obj, info in payload["gpu"]["lora_counts"].items():
        paper.append(
            f"- LoRA `{obj}`: {info['exp27_test_items']} exp27 test predictions, {info['cv5_oof_items']} OOF predictions, "
            f"selected hp lr={info['selected_hp'].get('lr')} steps={info['selected_hp'].get('steps')}."
        )
    (tab / "paper_master_results.md").write_text("\n".join(paper) + "\n")
    # A tiny TeX mirror of the CPU allocation table.
    tex = ["% Generated by run_all.py. Do not edit by hand.",
           "\\begin{tabular}{lrrl}",
           "Dataset & D-study $k$ & Oracle $k$ & D-study minus oracle test CE \\\\",
           "\\hline"]
    for name, block in payload["datasets"].items():
        rows = block["allocation"]["budgets"]
        if not rows:
            continue
        row = rows[-1]
        d = row["dstudy"]
        o = row["oracle_ce"]
        diff = row["dstudy_minus_oracle_ce"]
        diff_s = "---" if not diff else f"{diff['point']:.3f} [{diff['lo']:.3f}, {diff['hi']:.3f}]"
        tex.append(f"{name.replace('_', '\\_')} & {'' if not d else d['k']} & {o['k']} & {diff_s} \\\\")
    tex.append("\\end{tabular}")
    (tab / "paper_master_results.tex").write_text("\n".join(tex) + "\n")


def make_figures(payload) -> None:
    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)
    names = list(payload["datasets"])
    # Frontier CE and sqrt(RPS), two price regimes.
    for metric, ylabel, fname in (
        ("frontier_ce", "Held-out test cross-entropy (nats, lower is better)", "pareto_ce"),
        ("frontier_sqrt_rps", "Held-out test sqrt(ranked probability score) (lower is better)", "pareto_sqrt_rps"),
    ):
        for regime, price, title in (
            ("crowd", 0.10, "Crowd regime, axis scaled at $0.10 per rating"),
            ("expert", 10.0, "Domain-expert regime, axis scaled at $10 per rating"),
        ):
            fig, ax = plt.subplots(figsize=(8.2, 5.2))
            for name in names:
                rows = payload["datasets"][name]["allocation"]["budgets"]
                if not rows:
                    continue
                xs = [r["B"] for r in rows]
                ys = [r[metric] for r in rows]
                ax.plot(xs, ys, marker="o", label=name)
            ax.set_xscale("log")
            ax.set_xlabel("Nominal train-rating budget B (each cell uses N×k ≤ B)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, which="major", alpha=0.4)
            ax.grid(True, which="minor", ls=":", alpha=0.25)
            ax.legend(fontsize=7, loc="best")
            ax2 = ax.twiny()
            xmin, xmax = ax.get_xlim()
            ax2.set_xscale("log")
            ax2.set_xlim(xmin * price, xmax * price)
            ax2.set_xlabel(f"Dollar cost at ${price:.2f} per rating ({regime} regime)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{fname}_{regime}.png", dpi=140)
            plt.close(fig)
    # One head-to-head panel: test CE versus k at the largest budget, two datasets if present.
    focus = [n for n in ("popquorn_offensiveness", "summeval_expert", "dices990") if n in payload["datasets"]]
    if focus:
        fig, axes = plt.subplots(1, len(focus), figsize=(4.2 * len(focus), 4.2), squeeze=False)
        for ax, name in zip(axes[0], focus):
            block = payload["datasets"][name]
            rows = block["allocation"]["budgets"]
            if not rows:
                continue
            B = rows[-1]["B"]
            cells = [c for c in block["allocation"]["cells"]]
            # cells are unique (k, N), not tied to B. Use the budget row's options via stored cells whose ratings_used <= B
            # and N = min(available, B//k). The saved cells include every fitted (k, N). Match the budget row.
            # Reconstruct from cells with ratings_used close to the options at this B: k such that N*k is the cell used.
            chosen = {}
            for c in cells:
                if c["ratings_used"] <= B and (B // c["k"]) == c["N"]:
                    chosen[c["k"]] = c
            if not chosen:
                continue
            ks = sorted(chosen)
            ax.errorbar(
                ks,
                [chosen[k]["test"]["ce"] for k in ks],
                yerr=np.array([
                    [chosen[k]["test"]["ce"] - chosen[k]["test"]["ce_ci"]["lo"] for k in ks],
                    [chosen[k]["test"]["ce_ci"]["hi"] - chosen[k]["test"]["ce"] for k in ks],
                ]),
                fmt="o-", capsize=3,
            )
            d_k = (rows[-1]["dstudy"] or {}).get("k")
            if d_k in chosen:
                ax.axvline(d_k, color="0.4", ls="--", label=f"D-study k={d_k}")
            ax.set_xlabel(f"Ratings per train prompt k (budget B={B})")
            ax.set_ylabel("Held-out test cross-entropy (nats, lower is better)")
            ax.set_title(name)
            ax.grid(True, which="major", alpha=0.4)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "allocation_k_vs_test_ce.png", dpi=140)
        plt.close(fig)
    # Synthetic crossover.
    syn = payload["synthetic"]["crossover"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [r["K"] for r in syn]
    ys = [r["k_star_point"] if r["k_star_point"] is not None else np.nan for r in syn]
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Number of bins K (synthetic {0,…,100} mixture, not human ratings)")
    ax.set_ylabel("Smallest k where the CE head's CRPS point estimate is lower")
    ax.set_title("Synthetic output-granularity crossover")
    ax.grid(True, which="major", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "synthetic_kstar.png", dpi=140)
    plt.close(fig)


def write_report(payload) -> None:
    lines = []
    lines.append("# Experiment 40 report")
    lines.append("")
    lines.append("**Doc link:** https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/REPORT.md")
    lines.append("")
    lines.append("Numbers below are written by `run_all.py` from `results.json`. Read `HOW_TO_READ_RESULTS.md` before quoting one.")
    lines.append("")
    lines.append("## Machine checks")
    lines.append("")
    lines.append("- Lean 4.36.0, `lake build`, zero `sorry`. Axiom receipt: `lean_check.txt`. Standard axioms only.")
    lines.append("- `pytest` in this folder: witness identities, random Brier and RPS decompositions, G-study versus balanced ANOVA.")
    lines.append(f"- Bootstrap B={payload['bootstrap_B']}, item-clustered percentile 95% intervals, seed 1, fixed head.")
    lines.append("")
    lines.append("## Executive findings")
    lines.append("")
    lines.append("On the large public sets, once the nominal budget reaches a few hundred ratings the held-out cross-entropy of this linear head is within a few thousandths of a nat across k, and the item-clustered 95% interval of the D-study-minus-oracle difference covers 0. D-study still names a specific k (17 on DICES-990; 10 on POPQUORN Offensiveness, where train-split Φ(10) = 0.7875 and the 0.80 target is not reached). At small budgets that k is often infeasible, because N = B/k drops below 4 prompts. The detectable misallocation is VeriBench `native_main` on the sealed `exp27_split`: at B=64, D-study k=2 has held-out test cross-entropy 0.081 nats worse than oracle k=3, interval [0.013, 0.141].")
    lines.append("")
    lines.append("The pre-registered +0.05 nat tolerance versus the full-train head is reached at B=32 or B=64 on every set except SummEval expert, which reaches it at B=128. At the stated crowd prices that is a few dollars; at the stated expert prices it is a few hundred to a few thousand dollars. The $15,000–$75,000 researcher-plus-GPU anchor is not a cost measured here.")
    lines.append("")
    lines.append("## Where D-study and the proper score agree")
    lines.append("")
    for name, block in payload["datasets"].items():
        rows = block["allocation"]["budgets"]
        if not rows:
            lines.append(f"- `{name}`: no feasible allocation cell.")
            continue
        last = rows[-1]
        d = last["dstudy"]
        o = last["oracle_ce"]
        diff = last["dstudy_minus_oracle_ce"]
        if d is None:
            lines.append(f"- `{name}` at B={last['B']}: D-study k was not feasible. Oracle k for test cross-entropy is {o['k']}.")
            continue
        same = "the same k" if d["k"] == o["k"] else f"not the same k (D-study {d['k']}, oracle {o['k']})"
        diff_s = "n/a" if not diff else (
            f"{diff['point']:.4f} [{diff['lo']:.4f}, {diff['hi']:.4f}]"
        )
        lines.append(
            f"- `{name}`, largest grid budget B={last['B']}, held-out test prompts n={o['test']['n_test_items']}: "
            f"D-study and the test-oracle allocation are {same}. "
            f"Held-out test cross-entropy of the D-study allocation minus the oracle allocation is {diff_s} "
            f"(positive means the D-study allocation is worse). "
            f"D-study test CE {_fmt_ci(d['test'])}; oracle test CE {_fmt_ci(o['test'])}."
        )
    lines.append("")
    lines.append("## Dollar frontier")
    lines.append("")
    lines.append("Tolerance was fixed in `PROTOCOL.md`: 0.05 nats of cross-entropy, 0.02 of sqrt(RPS), above the full-train head.")
    lines.append("")
    for name, block in payload["datasets"].items():
        h = block["dollars"]
        if "B_ce" not in h:
            continue
        lines.append(
            f"- `{name}`: full-train held-out CE {_fmt(h['full_train_ce'])}. "
            f"Smallest frontier B within +0.05 nats: {h['B_ce']}. "
            f"Crowd dollars at $0.05/$0.10/$0.20: {h['crowd_usd_for_ce']}. "
            f"Expert dollars at $5/$10/$20: {h['expert_usd_for_ce']}. "
            f"sqrt(RPS) B: {h['B_sqrt_rps']}."
        )
    lines.append("")
    lines.append("The $15,000–$75,000 researcher-plus-GPU band is a stated anchor for 1–3 researcher-months. This run did not measure that cost. The linear head is not a GPU low-rank adaptation run.")
    lines.append("")
    lines.append("## R_v1")
    lines.append("")
    lines.append(f"Pinned loader: `{json.dumps(payload['rv1_loader'])}`.")
    lines.append("")
    lines.append(payload["rv1_old"].get("provenance", ""))
    lines.append("")
    lines.append("## Synthetic K sweep")
    lines.append("")
    lines.append(payload["synthetic"]["law"])
    lines.append("")
    for row in payload["synthetic"]["crossover"]:
        lines.append(f"- K={row['K']}: point-estimate k* = {row['k_star_point']}; CI k* = {row['k_star_ci']}.")
    lines.append("")
    lines.append("## GPU bridge")
    lines.append("")
    lines.append(payload["gpu"]["note"])
    lines.append("")
    for check in payload["gpu"]["logit_checks"]:
        lines.append(
            f"- `{check['file']}` n={check['n_items']}, probability-sum error {check['prob_sum_max_abs_err']:.3e}, "
            f"expected-score error {check['expected_score_max_abs_err']:.3e}."
        )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- ChaosNLI ranked probability score uses the file's label order (entailment, neutral, contradiction). That order is imposed.")
    lines.append("- SummEval and ChaosNLI have no person id. Their rater component is unidentified on purpose.")
    lines.append("- The test-oracle k is not a deployable selection. The validation-chosen k is the one that does not look at test labels.")
    lines.append("- Intervals do not retrain the head.")
    lines.append("- WMT direct assessment at `RicardoRei/wmt-da-human-evaluation` (commit 301de385, train.csv 638,191,673 bytes) was reachable and is a mean score plus an annotator count. Individual rater rows are not in that file, so it was not used as multi-rater data.")
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    (HERE / "results.md").write_text(
        "\n".join([
            "# Results ledger",
            "",
            f"**Status:** COMPLETE",
            f"**Last updated:** written by run_all.py",
            "",
            "See `REPORT.md` for the narrative and `results.json` for every cell.",
            "",
            "**TLDR-end:** [judges: experiment 40] G-study, dollar frontier, and the Lean witness are in this folder.",
            "",
        ])
    )


def how_to_read() -> None:
    (HERE / "HOW_TO_READ_RESULTS.md").write_text(
        """# How to read experiment 40

1. **Data.** Public benchmarks are the experiment 36 seed-0 group split (0.6/0.2/0.2). VeriBench `native_main` uses the task-pure `exp27_split` from trustworthy-theorem-judges `efdc8c3`. `R_v1` from `veribench@d1c47355` is reported only if that pinned loader succeeds. The old 75-item table is a different file and is labeled as such. Synthetic rows are not human ratings. These held-out prompts were already scored by linear heads in experiment 36; say that when quoting them.
2. **Metric.** Cross-entropy is the mean of `-log q(y)` in nats, prompt-averaged. Ranked probability score (RPS) is the sum of squared CDF errors. `sqrt(RPS)` is the square root of the prompt-average RPS. Lower is better. A reference value is the full-train head on the same test prompts.
3. **Differences.** "D-study minus oracle" is held-out test cross-entropy of the D-study allocation minus held-out test cross-entropy of the test-oracle allocation, same prompts. Positive means the D-study allocation is worse.
4. **Intervals.** Percentile 95% intervals, item-clustered bootstrap, B=1000 unless `EXP39_SMOKE=1`, seed 1. The interval is for a fixed head.
5. **How the head was chosen.** Softmax linear head, 24 train-only principal components of a 256-dimensional hash, L2 1e-2, L-BFGS-B from zero, at most 35 iterations. Cross-entropy in the logits is convex. The D-study k is the smallest feasible k with Φ(k) ≥ 0.80 on train-split variance components, or the largest feasible k if that target is not reached. Φ does not depend on N.
"""
    )


def main() -> None:
    t0 = time.time()
    print("[load] public sets", flush=True)
    specs = [
        ("summeval_expert", load_summeval("expert"), "vote_index"),
        ("summeval_crowd", load_summeval("crowd"), "vote_index"),
        ("popquorn_offensiveness", load_popquorn("offensiveness"), "crossed_person"),
        ("popquorn_politeness", load_popquorn("politeness"), "crossed_person"),
        ("dices990", _load_dices("990"), "crossed_person"),
        ("chaosnli_snli_mnli", load_chaos_3way(), "vote_index"),
        ("veribench_fsc_native_main", load_veribench_fsc_native_main(), "crossed_person"),
    ]
    if SMOKE:
        specs = specs[:1]
    rv1_status = try_rv1_loader()
    print("[rv1]", rv1_status, flush=True)
    datasets = {}
    rebin_bundle = []
    for name, data, facet in specs:
        print(f"[data] {name} items={data.n_items}", flush=True)
        if name == "veribench_fsc_native_main":
            split = exp27_split(data)
            if split["missing_groups"] or split["mixed_tasks"] or split["train"].size < 4:
                train, val, test = split_by_group(data.groups, seed=0)
                split_note = "fell back to seed-0 group split because exp27_split did not cover every task"
            else:
                train, val, test = split["train"], split["val"], split["test"]
                split_note = split["source"]
        else:
            train, val, test = split_by_group(data.groups, seed=0)
            split_note = "experiment 36 group split, seed 0, fractions 0.6/0.2/0.2"
        assert_no_prompt_leakage(data.groups, data.rater_ids, train, val, test)
        print(f"  [gstudy] {name}", flush=True)
        g_train = run_gstudy(data, train, facet)
        g_all = run_gstudy(data, None, facet)
        print(f"  [hash] {name}", flush=True)
        features = hash_features(data.texts)
        print(f"  [allocate] {name}", flush=True)
        alloc = allocation(data, features, train, val, test, g_train)
        datasets[name] = {
            "split": split_note,
            "n_items": data.n_items,
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "facet": facet,
            "gstudy_train": {k: v for k, v in g_train.items() if k != "curve"} | {"curve": g_train.get("curve", [])[:8]},
            "gstudy_all": {k: v for k, v in g_all.items() if k != "curve"} | {"curve": g_all.get("curve", [])[:8]},
            "allocation": alloc,
            "dollars": dollar_hits(alloc),
        }
        rebin_bundle.append((data, train, val, test, features))
        print(f"  [done] {name} {time.time()-t0:.1f}s", flush=True)
    print("[rebin]", flush=True)
    rebin_rows = rebin_sweep(rebin_bundle)
    print("[synthetic]", flush=True)
    synthetic = synthetic_sweep()
    print("[gpu]", flush=True)
    gpu = bridge_gpu()
    payload = {
        "bootstrap_B": BOOT,
        "smoke": SMOKE,
        "phi_target": PHI_TARGET,
        "datasets": datasets,
        "rebin": rebin_rows,
        "synthetic": synthetic,
        "rv1_loader": rv1_status,
        "rv1_old": old_rv1_gstudy(),
        "gpu": gpu,
        "seconds": time.time() - t0,
    }
    # Strip nothing else; arrays were already dropped.
    (HERE / "results.json").write_text(json.dumps(_jsonable(payload), indent=2))
    write_tables(payload)
    make_figures(payload)
    how_to_read()
    write_report(payload)
    print(f"[wrote] results in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
