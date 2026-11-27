"""Classical G-study variance components and D-study projections.

The crossed estimator is Henderson's Method III for the two-way additive
model (item + rater, one averaged observation per filled cell). Residual
variance is the interaction-plus-error term, which is all a one-observation
cell can identify. Expected-mean-square coefficients use

    kappa_c = sum_i (n_i - sum_{j on i} 1/n_j) = tr(Z_c^T (I - P_rater) Z_c).

On a balanced grid this reduces to the textbook formula
sigma_c^2 = (MS_c - MS_res) / n_raters. The algebra uses second moments only.
It does not assume normality. It does assume uncorrelated homoscedastic random
effects; that assumption is stated wherever a coefficient is reported.

A vote-index facet is not a person. Those designs use the nested one-way
estimator and leave the crossed rater component unidentified.
"""

from __future__ import annotations

import numpy as np


def _as_codes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    uniq, codes = np.unique(labels, return_inverse=True)
    return uniq, codes.astype(int)


def collapse_cells(item, rater, score) -> dict:
    """Average duplicate (item, rater) cells. Sorting ratings is not done here."""
    item = np.asarray(item)
    rater = np.asarray(rater)
    score = np.asarray(score, dtype=float)
    if item.shape[0] != rater.shape[0] or item.shape[0] != score.shape[0]:
        raise ValueError("item, rater, and score must have the same length")
    buckets: dict[tuple, list[float]] = {}
    for i, r, y in zip(item.tolist(), rater.tolist(), score.tolist()):
        buckets.setdefault((i, r), []).append(float(y))
    n_dup = sum(1 for v in buckets.values() if len(v) > 1)
    rows = [(i, r, float(np.mean(v)), len(v)) for (i, r), v in buckets.items()]
    return {
        "item": np.array([t[0] for t in rows], dtype=object),
        "rater": np.array([t[1] for t in rows], dtype=object),
        "score": np.array([t[2] for t in rows], dtype=float),
        "n_cells": len(rows),
        "n_duplicate_cells": int(n_dup),
        "n_raw": int(score.shape[0]),
    }


def _components(item_codes: np.ndarray, rater_codes: np.ndarray) -> int:
    """Connected components of the bipartite item–rater graph."""
    a = int(item_codes.max()) + 1 if item_codes.size else 0
    b = int(rater_codes.max()) + 1 if rater_codes.size else 0
    parent = np.arange(a + b)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, j in zip(item_codes.tolist(), rater_codes.tolist()):
        union(int(i), a + int(j))
    roots = {find(i) for i in range(a)} | {find(a + j) for j in range(b)}
    return len(roots)


def _group_mean(codes: np.ndarray, values: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(n_groups, dtype=float)
    counts = np.zeros(n_groups, dtype=float)
    np.add.at(sums, codes, values)
    np.add.at(counts, codes, 1.0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return means, counts


def _kappa(obs_item: np.ndarray, obs_rater: np.ndarray, n_item: np.ndarray, n_rater: np.ndarray, which: str) -> float:
    """kappa_c or kappa_r from the projection trace."""
    if which == "c":
        # sum over items of (n_i - sum_{j on i} 1/n_j)
        acc = n_item.copy()
        np.add.at(acc, obs_item, -1.0 / n_rater[obs_rater])
        return float(acc.sum())
    acc = n_rater.copy()
    np.add.at(acc, obs_rater, -1.0 / n_item[obs_item])
    return float(acc.sum())


def _fit_crossed(item: np.ndarray, rater: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares fitted values of y = alpha_item + beta_rater.

    One rater effect is fixed at 0 inside each connected component so the
    normal matrix is nonsingular. Fitted values do not depend on that choice.
    """
    a = int(item.max()) + 1
    b = int(rater.max()) + 1
    # Component label on items; raters inherit it.
    parent_i = np.arange(a)
    parent_r = np.full(b, -1)

    def find(x: int) -> int:
        while parent_i[x] != x:
            parent_i[x] = parent_i[parent_i[x]]
            x = int(parent_i[x])
        return x

    item_of_comp: dict[int, list[int]] = {}
    # Union items that share a rater.
    first_item_of_rater = np.full(b, -1, dtype=int)
    for i, j in zip(item.tolist(), rater.tolist()):
        i = int(i)
        j = int(j)
        if first_item_of_rater[j] < 0:
            first_item_of_rater[j] = i
        else:
            ra, rb = find(first_item_of_rater[j]), find(i)
            if ra != rb:
                parent_i[rb] = ra
    comp_of_item = np.array([find(i) for i in range(a)], dtype=int)
    for j in range(b):
        if first_item_of_rater[j] >= 0:
            parent_r[j] = comp_of_item[first_item_of_rater[j]]
    fitted = np.zeros(y.shape[0], dtype=float)
    # Solve each component on its own index set.
    for comp in sorted(set(comp_of_item.tolist())):
        items = np.flatnonzero(comp_of_item == comp)
        raters = np.flatnonzero(parent_r == comp)
        if items.size == 0 or raters.size == 0:
            continue
        item_map = {int(g): t for t, g in enumerate(items.tolist())}
        rater_map = {int(g): t for t, g in enumerate(raters.tolist())}
        mask = np.array([int(i) in item_map and int(j) in rater_map for i, j in zip(item.tolist(), rater.tolist())])
        # mask is all obs whose item is in the component; rater should match.
        ii = np.array([item_map[int(i)] for i, m in zip(item.tolist(), mask.tolist()) if m], dtype=int)
        jj = np.array([rater_map[int(j)] for j, m in zip(rater.tolist(), mask.tolist()) if m], dtype=int)
        yy = y[mask]
        fitted[mask] = _fit_one_component(ii, jj, yy, items.size, raters.size)
    return fitted


def _fit_one_component(item: np.ndarray, rater: np.ndarray, y: np.ndarray, a: int, b: int) -> np.ndarray:
    """Fit one connected component. Rater 0's effect is fixed at 0."""
    if b == 1 or a == 1 and b == 1:
        # Item means (rater effect absorbed). With one rater this is exact.
        means, _ = _group_mean(item, y, a)
        return means[item]
    ybar, n_item = _group_mean(item, y, a)
    _, n_rater = _group_mean(rater, y, b)
    m = b - 1
    A = np.zeros((m, m), dtype=float)
    rhs = np.zeros(m, dtype=float)
    # Accumulate the Schur complement of the item effects.
    lists: list[list[int]] = [[] for _ in range(a)]
    y_at: list[list[float]] = [[] for _ in range(a)]
    for i, j, yi in zip(item.tolist(), rater.tolist(), y.tolist()):
        lists[i].append(int(j))
        y_at[i].append(float(yi))
    for i in range(a):
        js = lists[i]
        ni = float(len(js))
        inv = 1.0 / ni
        free = [j for j in js if j != 0]
        for k in free:
            for m_ix in free:
                A[k - 1, m_ix - 1] += inv
        # rhs_k += ybar_i - Y_ik
        for j, yi in zip(js, y_at[i]):
            if j == 0:
                continue
            rhs[j - 1] += float(ybar[i] - yi)
    for k in range(1, b):
        A[k - 1, k - 1] -= float(n_rater[k])
    try:
        beta_free = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        beta_free = np.linalg.lstsq(A, rhs, rcond=None)[0]
    beta = np.zeros(b, dtype=float)
    beta[1:] = beta_free
    alpha = np.zeros(a, dtype=float)
    for i in range(a):
        js = lists[i]
        alpha[i] = float(ybar[i] - np.mean(beta[js]))
    return alpha[item] + beta[rater]


def crossed_gstudy(item, rater, score) -> dict:
    """Crossed item × person G-study on (possibly unbalanced) cells."""
    cells = collapse_cells(item, rater, score)
    y = cells["score"]
    if y.size < 3:
        return {"status": "too_few_cells", "n_cells": int(y.size), **{k: cells[k] for k in ("n_duplicate_cells", "n_raw")}}
    _, item_c = _as_codes(cells["item"])
    _, rater_c = _as_codes(cells["rater"])
    a = int(item_c.max()) + 1
    b = int(rater_c.max()) + 1
    n_comp = _components(item_c, rater_c)
    ybar_i, n_item = _group_mean(item_c, y, a)
    ybar_j, n_rater = _group_mean(rater_c, y, b)
    rss_raters = float(np.sum((y - ybar_j[rater_c]) ** 2))
    rss_items = float(np.sum((y - ybar_i[item_c]) ** 2))
    fitted = _fit_crossed(item_c, rater_c, y)
    rss_full = float(np.sum((y - fitted) ** 2))
    # Numerical floor. A negative extra sum of squares is a solver failure.
    ss_c = rss_raters - rss_full
    ss_r = rss_items - rss_full
    if ss_c < -1e-6 or ss_r < -1e-6:
        raise RuntimeError(f"negative extra sum of squares ss_c={ss_c} ss_r={ss_r}")
    ss_c = max(ss_c, 0.0)
    ss_r = max(ss_r, 0.0)
    df_c = a - n_comp
    df_r = b - n_comp
    df_res = int(y.size) - a - b + n_comp
    if df_res <= 0 or df_c <= 0 or df_r <= 0:
        return {
            "status": "rank_deficient",
            "facet": "crossed_person",
            "n_items": a,
            "n_raters": b,
            "n_cells": int(y.size),
            "n_components": n_comp,
            "df_res": df_res,
            "n_duplicate_cells": cells["n_duplicate_cells"],
            "n_raw": cells["n_raw"],
        }
    sigma_res = rss_full / df_res
    kappa_c = _kappa(item_c, rater_c, n_item, n_rater, "c")
    kappa_r = _kappa(item_c, rater_c, n_item, n_rater, "r")
    sigma_c = (ss_c - df_c * sigma_res) / kappa_c if abs(kappa_c) > 1e-12 else float("nan")
    sigma_r = (ss_r - df_r * sigma_res) / kappa_r if abs(kappa_r) > 1e-12 else float("nan")
    return {
        "status": "ok",
        "facet": "crossed_person",
        "n_items": a,
        "n_raters": b,
        "n_cells": int(y.size),
        "n_raw": cells["n_raw"],
        "n_duplicate_cells": cells["n_duplicate_cells"],
        "n_components": n_comp,
        "df_c": df_c,
        "df_r": df_r,
        "df_res": df_res,
        "sigma_c2_raw": float(sigma_c),
        "sigma_r2_raw": float(sigma_r),
        "sigma_res2": float(sigma_res),
        "sigma_c2": float(max(sigma_c, 0.0)),
        "sigma_r2": float(max(sigma_r, 0.0)),
        "kappa_c": float(kappa_c),
        "kappa_r": float(kappa_r),
        "truncated_at_zero": bool(sigma_c < 0 or sigma_r < 0),
        "assumptions": "uncorrelated homoscedastic random effects; no normality used; one cell mean per item×rater",
    }


def nested_gstudy(item, score) -> dict:
    """One-way random effects, ratings nested in items. No crossed person facet."""
    item = np.asarray(item)
    score = np.asarray(score, dtype=float)
    _, codes = _as_codes(item)
    a = int(codes.max()) + 1 if codes.size else 0
    n = int(score.size)
    if a < 2 or n <= a:
        return {"status": "rank_deficient", "facet": "nested_in_item", "n_items": a, "n_ratings": n}
    means, counts = _group_mean(codes, score, a)
    grand = float(score.mean())
    ss_w = float(np.sum((score - means[codes]) ** 2))
    ss_b = float(np.sum(counts * (means - grand) ** 2))
    df_b = a - 1
    df_w = n - a
    ms_b = ss_b / df_b
    ms_w = ss_w / df_w
    n0 = (n - float(np.sum(counts ** 2)) / n) / df_b
    sigma_res = ms_w
    sigma_c = (ms_b - ms_w) / n0
    return {
        "status": "ok",
        "facet": "nested_in_item",
        "n_items": a,
        "n_ratings": n,
        "df_c": df_b,
        "df_res": df_w,
        "n0": float(n0),
        "sigma_c2_raw": float(sigma_c),
        "sigma_r2_raw": None,
        "sigma_res2": float(sigma_res),
        "sigma_c2": float(max(sigma_c, 0.0)),
        "sigma_r2": 0.0,
        "rater_component": "unidentified; there is no crossed person id",
        "truncated_at_zero": bool(sigma_c < 0),
        "assumptions": "one-way random effects; no normality used; vote index is not treated as a person",
    }


def balanced_anova(Y: np.ndarray) -> dict:
    """Textbook balanced item × rater ANOVA. Reference for the crossed estimator."""
    Y = np.asarray(Y, dtype=float)
    n_c, n_r = Y.shape
    grand = Y.mean()
    item_means = Y.mean(axis=1)
    rater_means = Y.mean(axis=0)
    ss_c = n_r * np.sum((item_means - grand) ** 2)
    ss_r = n_c * np.sum((rater_means - grand) ** 2)
    resid = Y - item_means[:, None] - rater_means[None, :] + grand
    ss_e = float(np.sum(resid ** 2))
    df_c, df_r = n_c - 1, n_r - 1
    df_e = df_c * df_r
    ms_c, ms_r, ms_e = ss_c / df_c, ss_r / df_r, ss_e / df_e
    return {
        "sigma_c2": float((ms_c - ms_e) / n_r),
        "sigma_r2": float((ms_r - ms_e) / n_c),
        "sigma_res2": float(ms_e),
    }


def dstudy_curve(sigma_c2: float, sigma_r2: float, sigma_res2: float, k_max: int) -> list[dict]:
    """Relative Eρ²(k) and absolute Φ(k) for k = 1..k_max raters on a new item.

    Eρ²(k) = σ_c² / (σ_c² + σ_res² / k)
    Φ(k) = σ_c² / (σ_c² + (σ_r² + σ_res²) / k)
    Coefficients use components truncated at 0. They do not depend on N.
    """
    sc = max(float(sigma_c2), 0.0)
    sr = max(float(sigma_r2), 0.0)
    se = max(float(sigma_res2), 0.0)
    rows = []
    for k in range(1, int(k_max) + 1):
        den_rel = sc + se / k
        den_abs = sc + (sr + se) / k
        rows.append({
            "k": k,
            "E_rho2": float(sc / den_rel) if den_rel > 0 else float("nan"),
            "Phi": float(sc / den_abs) if den_abs > 0 else float("nan"),
        })
    return rows


def recommend_k(curve: list[dict], target: float = 0.80, k_feasible: list[int] | None = None) -> dict:
    """Smallest feasible k with Φ(k) ≥ target. If none, the largest feasible k.

    Φ increases in k, so this rule spends ratings on k and is silent about N.
    """
    allowed = set(k_feasible) if k_feasible is not None else {row["k"] for row in curve}
    rows = [row for row in curve if row["k"] in allowed and np.isfinite(row["Phi"])]
    if not rows:
        return {"k": None, "target": target, "target_hit": False, "reason": "no feasible k"}
    hits = [row for row in rows if row["Phi"] >= target]
    if hits:
        chosen = min(hits, key=lambda row: row["k"])
        return {"k": int(chosen["k"]), "Phi": chosen["Phi"], "E_rho2": chosen["E_rho2"],
                "target": target, "target_hit": True, "reason": "smallest k with Phi >= target"}
    chosen = max(rows, key=lambda row: row["k"])
    return {"k": int(chosen["k"]), "Phi": chosen["Phi"], "E_rho2": chosen["E_rho2"],
            "target": target, "target_hit": False,
            "reason": "target infeasible on the allowed k grid; D-study still picks the largest k because Phi increases in k"}
