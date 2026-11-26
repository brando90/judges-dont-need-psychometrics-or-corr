#!/usr/bin/env python3
"""Where generalizability theory agrees with cross-entropy, and where it does not.

Synthetic ordinal ratings on {0,1,2,3,4,5}. No model calls and no human ratings.
The population block is closed form or exact vertex enumeration. The Monte Carlo
block refits every learner on fresh training ratings. Criteria live in
preregistered_criteria.json and are not updated from the Monte Carlo output.

Run from the repository root or from this folder:

    python3 run_experiment.py
    python3 run_experiment.py --population-only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from fractions import Fraction as Fr
from itertools import combinations
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
CRITERIA = json.loads((HERE / "preregistered_criteria.json").read_text())
Y = np.arange(6, dtype=float)
K = 6
EPS = float(CRITERIA["ce_smoothing_epsilon"])
THETA_LO, THETA_HI = CRITERIA["binomial_theta_clip"]
SEED = int(CRITERIA["seed"])
N_REP = int(CRITERIA["n_replications"])
N_GRID = [int(x) for x in CRITERIA["n_train_grid"]]
N_VAL = int(CRITERIA["n_validation_for_selection"])
N_TEST = int(CRITERIA["n_test_ratings_per_item_for_evaluator_check"])
PRIMARY_TIE = list(CRITERIA["tie_break_primary"])
REVERSE_TIE = list(CRITERIA["tie_break_sensitivity"])


def py(x):
    if isinstance(x, dict):
        return {str(k): py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [py(v) for v in x]
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, (np.integer,)):
        x = int(x)
    if isinstance(x, float):
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if math.isnan(x):
            return None
        return x
    if x is None or isinstance(x, (str, int, bool)):
        return x
    return x


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(py(obj), indent=2, sort_keys=True) + "\n")


def binom_pmf(theta: float) -> np.ndarray:
    th = float(theta)
    return np.array([comb(5, y) * th**y * (1 - th) ** (5 - y) for y in range(6)], dtype=float)


def shift_pmf(p: np.ndarray, k: int) -> np.ndarray:
    q = np.zeros(6, dtype=float)
    for y, pr in enumerate(np.asarray(p, float)):
        q[int(np.clip(y + k, 0, 5))] += pr
    return q


def onehot(loc: int) -> np.ndarray:
    q = np.zeros(6, dtype=float)
    q[int(loc)] = 1.0
    return q


def round_half_up(x) -> np.ndarray:
    return np.floor(np.asarray(x, float) + 0.5).astype(int)


def row_means(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, float) @ Y


def row_vars(p: np.ndarray) -> np.ndarray:
    m = row_means(p)
    return (np.asarray(p, float) * (Y - m[:, None]) ** 2).sum(axis=1)


def cdf_rows(p: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(p, float), axis=1)[:, :-1]


def expected_rps_rows(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    fq, fp = cdf_rows(q), cdf_rows(p)
    return ((fq - fp) ** 2).sum(axis=1) / 5.0 + (fp * (1.0 - fp)).sum(axis=1) / 5.0


def expected_ce_rows(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logq = np.log(np.where(q > 0, q, 1.0))
    terms = np.where(p > 0, -p * logq, 0.0)
    out = terms.sum(axis=1)
    out[(p > 0).any(axis=1) & ((p > 0) & (q <= 0)).any(axis=1)] = np.inf
    return out


def tvd_rows(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(np.asarray(q, float) - np.asarray(p, float)).sum(axis=1)


def smooth_rows(q: np.ndarray, eps: float = EPS) -> np.ndarray:
    return (1.0 - K * eps) * np.asarray(q, float) + eps


def panel_coeffs(pmfs: np.ndarray) -> dict:
    """Population generalizability coefficient of one source.

    v_u is the variance of the item means with divisor N (the number of items),
    matching experiment 04. There is no second rater, so the absolute
    dependability coefficient equals the relative generalizability coefficient.
    A zero denominator is undefined, not perfect reliability.
    """
    m = row_means(pmfs)
    vu = float(np.mean((m - m.mean()) ** 2))
    ve = float(np.mean(row_vars(pmfs)))
    # A pooled forecast has one distribution for every item. Float noise can leave
    # a 1e-16 variance of the means; that is not a generalizability coefficient.
    if vu < 1e-12:
        vu = 0.0
    if ve < 1e-12:
        ve = 0.0
    den = vu + ve
    coef = None if den == 0.0 else vu / den
    return {"vu": vu, "ve": ve, "erho2": coef, "phi_own": coef}


def joint_components(mu_h, var_h, mu_j, var_j) -> dict:
    """Population components of a two-source crossed panel.

    Each item has a human source and a judge source. The universe score is the
    average of the two source means. Relative error excludes the source main
    effect. Absolute error includes it. Divisors are population divisors
    (N items, 2 sources).
    """
    mu_h = np.asarray(mu_h, float)
    mu_j = np.asarray(mu_j, float)
    var_h = np.asarray(var_h, float)
    var_j = np.asarray(var_j, float)
    grand = 0.5 * (mu_h.mean() + mu_j.mean())
    uni = 0.5 * (mu_h + mu_j)
    v_u = float(np.mean((uni - uni.mean()) ** 2))
    alpha_h = float(mu_h.mean() - grand)
    alpha_j = float(mu_j.mean() - grand)
    v_a = float(0.5 * (alpha_h**2 + alpha_j**2))
    delta_h = 0.5 * ((mu_h - mu_h.mean()) - (mu_j - mu_j.mean()))
    v_ia = float(np.mean(np.concatenate([delta_h**2, delta_h**2])))
    v_e = float(0.5 * (var_h.mean() + var_j.mean()))
    v_delta = v_ia + v_e
    v_Delta = v_delta + v_a
    erho = None if v_u + v_delta == 0 else v_u / (v_u + v_delta)
    phi = None if v_u + v_Delta == 0 else v_u / (v_u + v_Delta)
    return {
        "v_u": v_u,
        "v_source": v_a,
        "v_interaction": v_ia,
        "v_e": v_e,
        "erho2_one_random_source": erho,
        "phi_one_random_source": phi,
        "source_effect_human": alpha_h,
        "source_effect_judge": alpha_j,
    }


def corrs(m_hat: np.ndarray, m_true: np.ndarray) -> dict:
    if np.ptp(m_hat) == 0 or np.ptp(m_true) == 0:
        return {"pearson": None, "spearman": None, "kendall_tau_b": None}
    return {
        "pearson": float(stats.pearsonr(m_hat, m_true)[0]),
        "spearman": float(stats.spearmanr(m_hat, m_true)[0]),
        "kendall_tau_b": float(stats.kendalltau(m_hat, m_true)[0]),
    }


def mean_metrics(q: np.ndarray, p: np.ndarray) -> dict:
    q = np.asarray(q, float)
    p = np.asarray(p, float)
    mh, mt = row_means(q), row_means(p)
    coef = panel_coeffs(q)
    raw_ce = expected_ce_rows(q, p)
    ce_raw = float(np.mean(raw_ce)) if np.isfinite(raw_ce).all() else math.inf
    out = {
        "rps": float(np.mean(expected_rps_rows(q, p))),
        "ce_raw": ce_raw,
        "ce_smooth": float(np.mean(expected_ce_rows(smooth_rows(q), p))),
        "tvd": float(np.mean(tvd_rows(q, p))),
        "bias_forecast_minus_true": float(np.mean(mh - mt)),
        "mae_of_means": float(np.mean(np.abs(mh - mt))),
        "true_mean": float(np.mean(mt)),
        "forecast_mean": float(np.mean(mh)),
    }
    out.update(corrs(mh, mt))
    out.update(coef)
    return out


def t_interval(x: np.ndarray) -> dict:
    """Student-t 95% interval for the mean. A zero sample standard deviation has no t p-value."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "se": None, "n": 0, "p_value": None}
    mean = float(x.mean())
    if n < 2:
        return {"mean": mean, "lo": mean, "hi": mean, "se": 0.0, "n": n, "p_value": None}
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        return {"mean": mean, "lo": mean, "hi": mean, "se": 0.0, "n": n, "p_value": None}
    se = sd / math.sqrt(n)
    crit = float(stats.t.ppf(0.975, n - 1))
    p = float(stats.ttest_1samp(x, 0.0).pvalue)
    return {"mean": mean, "lo": mean - crit * se, "hi": mean + crit * se, "se": se, "n": n, "p_value": p}


def wilson(k: int, n: int) -> dict:
    z = 1.959963984540054
    phat = k / n
    den = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / den
    return {"k": int(k), "n": int(n), "proportion": phat, "lo": center - half, "hi": center + half}


def solve3(rows, b):
    a = [r[:] + [bi] for r, bi in zip(rows, b)]
    for col in range(3):
        piv = next((r for r in range(col, 3) if a[r][col] != 0), None)
        if piv is None:
            raise ZeroDivisionError
        a[col], a[piv] = a[piv], a[col]
        div = a[col][col]
        a[col] = [v / div for v in a[col]]
        for r in range(3):
            if r == col:
                continue
            fac = a[r][col]
            a[r] = [a[r][c] - fac * a[col][c] for c in range(4)]
    return [a[i][3] for i in range(3)]


def same_moment_vertex(p_frac, floor: Fr):
    """Largest ranked-probability-score gap at a vertex of the same-moment polytope.

    Probability floor, sum 1, and the first two power moments are linear
    constraints. The ranked probability score gap is convex in the forecast,
    so a maximum on this compact polytope is attained at a vertex. Each vertex
    sets three categories to the floor. All 20 choices are solved in exact
    rational arithmetic.
    """
    m = sum(Fr(y) * p_frac[y] for y in range(6))
    second = sum(Fr(y * y) * p_frac[y] for y in range(6))
    target_sum = 1 - 6 * floor
    target_y = m - floor * sum(Fr(y) for y in range(6))
    target_y2 = second - floor * sum(Fr(y * y) for y in range(6))
    best = None
    n_ok = 0
    for free in combinations(range(6), 3):
        try:
            z = solve3(
                [
                    [Fr(1) for _ in free],
                    [Fr(i) for i in free],
                    [Fr(i * i) for i in free],
                ],
                [target_sum, target_y, target_y2],
            )
        except ZeroDivisionError:
            continue
        if any(zi < 0 for zi in z):
            continue
        q = [floor] * 6
        for i, zi in zip(free, z):
            q[i] = floor + zi
        if any(qi < 0 for qi in q) or sum(q) != 1:
            continue
        qm = sum(Fr(y) * q[y] for y in range(6))
        qv = sum((Fr(y) - qm) ** 2 * q[y] for y in range(6))
        pv = sum((Fr(y) - m) ** 2 * p_frac[y] for y in range(6))
        if qm != m or qv != pv:
            continue
        qf = np.array([float(v) for v in q], dtype=float)
        pf = np.array([float(v) for v in p_frac], dtype=float)
        gap = float(expected_rps_rows(qf[None, :], pf[None, :])[0] - expected_rps_rows(pf[None, :], pf[None, :])[0])
        n_ok += 1
        if best is None or gap > best["rps_excess"]:
            best = {"q": qf, "q_exact": [str(v) for v in q], "rps_excess": gap, "free": list(free)}
    if best is None:
        raise RuntimeError("no same-moment vertex")
    best["n_vertices"] = n_ok
    return best


def binomial_items(index) -> np.ndarray:
    thetas = [(j + 1) / 14 for j in index]
    return np.stack([binom_pmf(th) for th in thetas])


def binomial_projection(means: np.ndarray) -> np.ndarray:
    theta = np.clip(np.asarray(means, float) / 5.0, THETA_LO, THETA_HI)
    return np.stack([binom_pmf(th) for th in theta])


def add_one(counts: np.ndarray) -> np.ndarray:
    return (np.asarray(counts, float) + 1.0) / (counts.sum(axis=1, keepdims=True) + K)


def pooled_add_one(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=0)
    q = (total + 1.0) / (total.sum() + K)
    return np.repeat(q[None, :], counts.shape[0], axis=0)


def point_mass_at_lenient_mean(means: np.ndarray, offset: int = 1) -> np.ndarray:
    loc = np.clip(round_half_up(means) + offset, 0, 5)
    return np.stack([onehot(int(v)) for v in loc])


def counts_from_pmfs(rng: np.random.Generator, pmfs: np.ndarray, n: int) -> np.ndarray:
    return np.stack([rng.multinomial(n, np.asarray(p, float) / np.asarray(p, float).sum()) for p in pmfs])


def anova_erho(X: np.ndarray):
    """Balanced items by replicates. Negative components are kept and also clipped.

    Returns the clipped coefficient used for selection, and the raw components.
    """
    X = np.asarray(X, float)
    n_u, n_a = X.shape
    if n_u < 2 or n_a < 2:
        return {"erho2_clipped": None, "vu_raw": None, "ve_raw": None}
    grand = X.mean()
    mu = X.mean(axis=1)
    ma = X.mean(axis=0)
    ss_u = n_a * np.sum((mu - grand) ** 2)
    ss_e = np.sum((X - mu[:, None] - ma[None, :] + grand) ** 2)
    df_u, df_e = n_u - 1, (n_u - 1) * (n_a - 1)
    ms_u = ss_u / df_u
    ms_e = ss_e / df_e if df_e else 0.0
    vu = (ms_u - ms_e) / n_a
    ve = ms_e
    vu_c, ve_c = max(vu, 0.0), max(ve, 0.0)
    coef = None if vu_c + ve_c == 0 else vu_c / (vu_c + ve_c)
    return {"erho2_clipped": coef, "vu_raw": float(vu), "ve_raw": float(ve)}


def matrix_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, int)
    n = int(counts[0].sum())
    X = np.empty((counts.shape[0], n), dtype=float)
    for i, row in enumerate(counts):
        X[i] = np.repeat(np.arange(6), row)
    return X


def rater_pmfs(neutral: np.ndarray) -> dict:
    lenient = np.stack([shift_pmf(p, 2) for p in neutral])
    harsh = np.stack([shift_pmf(p, -2) for p in neutral])
    marginal = (4.0 * neutral + lenient + harsh) / 6.0
    return {"neutral": neutral, "lenient": lenient, "harsh": harsh, "marginal": marginal}


def build_worlds():
    wide_index = list(range(12))
    wide = binomial_items(wide_index)
    shape_index = list(range(2, 10))
    shape_binom = binomial_items(shape_index)
    floor = Fr(1, 100)
    shape_rows = []
    q_rows = []
    for j, p in zip(shape_index, shape_binom):
        theta = Fr(j + 1, 14)
        p_frac = [Fr(comb(5, y)) * theta**y * (1 - theta) ** (5 - y) for y in range(6)]
        # The float binomial and the rational binomial agree to well under 1e-12.
        vertex = same_moment_vertex(p_frac, floor)
        q_rows.append(vertex["q"])
        shape_rows.append({"item_index": j, "theta": str(theta), **{k: vertex[k] for k in ("q_exact", "rps_excess", "free", "n_vertices")}})
    shaped = np.stack(q_rows)
    return {
        "wide_index": wide_index,
        "wide": wide,
        "shape_index": shape_index,
        "shape_binom": shape_binom,
        "shaped": shaped,
        "shape_rows": shape_rows,
        "raters": rater_pmfs(wide),
    }


def population(world) -> dict:
    wide, shaped, base = world["wide"], world["shaped"], world["shape_binom"]
    raters = world["raters"]
    const_wide = np.repeat(wide.mean(axis=0)[None, :], len(wide), axis=0)
    lenient = point_mass_at_lenient_mean(row_means(wide), 1)
    moment_wide = binomial_projection(row_means(wide))
    out = {
        "s0_binomial_agreement": {
            "oracle": mean_metrics(wide, wide),
            "binomial_projection_at_true_mean": mean_metrics(moment_wide, wide),
            "pooled_frequency": mean_metrics(const_wide, wide),
            "projection_matches_oracle": bool(np.allclose(moment_wide, wide)),
        },
        "s1_leniency": {},
        "s2_shape": {},
        "s3_multifacet": {},
    }
    out["s1_leniency"] = {
        "oracle": mean_metrics(wide, wide),
        "lenient_point": mean_metrics(lenient, wide),
        "pooled_frequency": mean_metrics(const_wide, wide),
        "binomial_projection": mean_metrics(moment_wide, wide),
        "locations": np.clip(round_half_up(row_means(wide)) + 1, 0, 5).tolist(),
        "joint_human_and_lenient_point": joint_components(
            row_means(wide), row_vars(wide), row_means(lenient), row_vars(lenient)
        ),
    }
    # Pure integer shift of a distribution that does not hit the ceiling, as an identity.
    interior = wide[3:9]
    shifted = np.stack([shift_pmf(p, 1) for p in interior])
    out["s1_leniency"]["interior_shift_plus_1_identity"] = {
        "human": mean_metrics(interior, interior),
        "shifted_forecast_scored_on_human": mean_metrics(shifted, interior),
        "own_coefficients_of_shifted_forecast": panel_coeffs(shifted),
        "joint": joint_components(row_means(interior), row_vars(interior), row_means(shifted), row_vars(shifted)),
        "means_equal_after_removing_the_shift": bool(
            np.allclose(row_means(shifted) - row_means(interior), 1.0, atol=1e-8)
        ),
    }
    const_shape = np.repeat(shaped.mean(axis=0)[None, :], len(shaped), axis=0)
    moment_shape = binomial_projection(row_means(shaped))
    out["s2_shape"] = {
        "oracle_shaped": mean_metrics(shaped, shaped),
        "binomial_projection": mean_metrics(moment_shape, shaped),
        "pooled_frequency": mean_metrics(const_shape, shaped),
        "moment_match_max_abs_mean": float(np.max(np.abs(row_means(moment_shape) - row_means(shaped)))),
        "moment_match_max_abs_var": float(np.max(np.abs(row_vars(moment_shape) - row_vars(shaped)))),
        "coefficient_abs_diff": abs(panel_coeffs(moment_shape)["erho2"] - panel_coeffs(shaped)["erho2"]),
        "vertices": world["shape_rows"],
        "mean_vertex_rps_excess": float(np.mean([r["rps_excess"] for r in world["shape_rows"]])),
    }
    marg = raters["marginal"]
    const_m = np.repeat(marg.mean(axis=0)[None, :], len(marg), axis=0)
    lenient_point = point_mass_at_lenient_mean(row_means(raters["lenient"]), 0)
    out["s3_multifacet"] = {
        "oracle_marginal": mean_metrics(marg, marg),
        "neutral_true": mean_metrics(raters["neutral"], marg),
        "lenient_true": mean_metrics(raters["lenient"], marg),
        "harsh_true": mean_metrics(raters["harsh"], marg),
        "lenient_point": mean_metrics(lenient_point, marg),
        "pooled_frequency": mean_metrics(const_m, marg),
        "binomial_projection_of_marginal": mean_metrics(binomial_projection(row_means(marg)), marg),
        "own_coefficients": {
            "neutral": panel_coeffs(raters["neutral"]),
            "lenient": panel_coeffs(raters["lenient"]),
            "harsh": panel_coeffs(raters["harsh"]),
            "marginal": panel_coeffs(marg),
            "lenient_point": panel_coeffs(lenient_point),
        },
    }
    return out


def exp04_oracle_rps() -> float:
    rows = []
    uniform = np.full(6, 1.0 / 6.0)
    for j in range(1, 12):
        p = 0.9 * binom_pmf(j / 12) + 0.1 * uniform
        rows.append(p)
    p = np.stack(rows)
    return float(np.mean(expected_rps_rows(p, p)))


def independent_rps(q, p) -> float:
    total = 0.0
    for pr, qr in zip(p, q):
        acc = 0.0
        mass = 0.0
        for y in range(6):
            mass += qr[y]
            if y < 5:
                acc += (mass - np.sum(pr[: y + 1])) ** 2
        irr = 0.0
        c = 0.0
        for y in range(5):
            c += pr[y]
            irr += c * (1 - c)
        total += acc / 5.0 + irr / 5.0
    return total / len(p)


def hard_checks(world, pop) -> list:
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    rps = exp04_oracle_rps()
    add(
        "experiment_04_oracle_rps_reproduced",
        abs(rps - 0.11670795387394607) < 1e-9,
        {"recomputed": rps, "published": 0.11670795387394607},
    )
    wide = world["wide"]
    add(
        "independent_rps_matches",
        abs(independent_rps(wide, wide) - mean_metrics(wide, wide)["rps"]) < 1e-12,
        independent_rps(wide, wide),
    )
    # Strict propriety on one item: every one-hot and the uniform lose to the truth.
    p0 = wide[6]
    oracle = float(expected_rps_rows(p0[None, :], p0[None, :])[0])
    others = [float(expected_rps_rows(onehot(y)[None, :], p0[None, :])[0]) for y in range(6)]
    others.append(float(expected_rps_rows(np.full((1, 6), 1 / 6), p0[None, :])[0]))
    add("proper_rps_oracle_is_best_on_traced_item", oracle <= min(others) + 1e-12, {"oracle": oracle, "worst_other": max(others)})
    s2 = pop["s2_shape"]
    add("s2_means_match", s2["moment_match_max_abs_mean"] < 1e-8, s2["moment_match_max_abs_mean"])
    add("s2_variances_match", s2["moment_match_max_abs_var"] < 1e-8, s2["moment_match_max_abs_var"])
    add("s2_coefficients_match", s2["coefficient_abs_diff"] < 1e-8, s2["coefficient_abs_diff"])
    add("s2_every_item_gap_above_0.01", all(r["rps_excess"] > 0.01 for r in s2["vertices"]), [r["rps_excess"] for r in s2["vertices"]])
    add("s0_projection_is_the_truth", pop["s0_binomial_agreement"]["projection_matches_oracle"], True)
    add(
        "pmfs_sum_to_one",
        all(np.allclose(block.sum(axis=1), 1.0) for block in (wide, world["shaped"], world["raters"]["marginal"])),
        True,
    )
    # Identical sources: joint relative and absolute coefficients match the one-source coefficient.
    joint_same = joint_components(row_means(wide), row_vars(wide), row_means(wide), row_vars(wide))
    one = panel_coeffs(wide)["erho2"]
    add(
        "joint_identity_when_sources_match",
        abs(joint_same["erho2_one_random_source"] - one) < 1e-12
        and abs(joint_same["phi_one_random_source"] - one) < 1e-12
        and abs(joint_same["v_source"]) < 1e-12,
        joint_same,
    )
    ident = pop["s1_leniency"]["interior_shift_plus_1_identity"]
    add(
        "constant_shift_lowers_phi_and_not_erho2",
        ident["joint"]["phi_one_random_source"] < ident["joint"]["erho2_one_random_source"] - 1e-6,
        ident["joint"],
    )
    # Seeds: same seed repeats; the next replication differs.
    a = counts_from_pmfs(np.random.default_rng(np.random.SeedSequence([SEED, 0, 4, 0])), wide, 4)
    b = counts_from_pmfs(np.random.default_rng(np.random.SeedSequence([SEED, 0, 4, 0])), wide, 4)
    c = counts_from_pmfs(np.random.default_rng(np.random.SeedSequence([SEED, 0, 4, 1])), wide, 4)
    add("seed_repeat_identical", np.array_equal(a, b), True)
    add("seed_next_replication_differs", not np.array_equal(a, c), True)
    # Leakage: a fitter that sees only counts is unchanged if a test array is overwritten.
    train = a.copy()
    test = c.copy()
    fit1 = add_one(train)
    test[:] = 0
    fit2 = add_one(train)
    add("fit_ignores_test_array", np.allclose(fit1, fit2), True)
    return checks


def rng_for(scenario: int, n: int, rep: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED, scenario, n, rep, stream]))


def score_named(forecasts: dict, truth: np.ndarray) -> dict:
    return {name: mean_metrics(q, truth) for name, q in forecasts.items()}


def choose_by_validation(forecasts: dict, truth_for_construction_only, val_counts: np.ndarray, order: list) -> str:
    """Pick the forecast with the lowest mean individual-rating RPS on validation counts.

    Validation counts are the only ratings used here. `truth_for_construction_only`
    is accepted and ignored so a future edit cannot silently score the truth.
    """
    del truth_for_construction_only
    totals = {}
    n_items = val_counts.shape[0]
    for name, q in forecasts.items():
        # Per-item mean of the individual RPS, then the unweighted mean across items.
        item_means = []
        for i in range(n_items):
            acc = 0.0
            n = int(val_counts[i].sum())
            if n == 0:
                continue
            cdf = np.cumsum(q[i])[:-1]
            for y, count in enumerate(val_counts[i]):
                if count == 0:
                    continue
                ind = np.array([1.0 if y <= k else 0.0 for k in range(5)])
                acc += count * np.sum((cdf - ind) ** 2) / 5.0
            item_means.append(acc / n)
        totals[name] = float(np.mean(item_means))
    ranked = sorted(totals, key=lambda name: (totals[name], order.index(name) if name in order else 99))
    return ranked[0]


def run_monte_carlo(world) -> tuple[list, dict]:
    rows = []
    wide = world["wide"]
    shaped = world["shaped"]
    raters = world["raters"]
    marginal = raters["marginal"]
    # Pre-register streams: 1 train, 2 validation, 3 test, 4 rater-0 ... 
    for n in N_GRID:
        for rep in range(N_REP):
            # S0 and S1 share the binomial world but use separate streams so neither sees the other.
            c0 = counts_from_pmfs(rng_for(0, n, rep, 1), wide, n)
            forecasts0 = {
                "oracle": wide,
                "add_one_ce": add_one(c0),
                "binomial_projection": binomial_projection(c0 @ Y / n),
                "pooled_laplace": pooled_add_one(c0),
            }
            for method, metrics in score_named(forecasts0, wide).items():
                rows.append({"scenario": "s0_binomial", "n_train": n, "rep": rep, "method": method, **metrics})
            lenient_counts_mean = c0 @ Y / n
            forecasts1 = dict(forecasts0)
            forecasts1["lenient_point"] = point_mass_at_lenient_mean(lenient_counts_mean, 1)
            for method, metrics in score_named(forecasts1, wide).items():
                if method == "oracle":
                    continue
                rows.append({"scenario": "s1_leniency", "n_train": n, "rep": rep, "method": method, **metrics})
            # Selection at the pre-registered training size only.
            if n == 16:
                val = counts_from_pmfs(rng_for(1, n, rep, 2), wide, N_VAL)
                pool = {k: forecasts1[k] for k in PRIMARY_TIE}
                pick = choose_by_validation(pool, None, val, PRIMARY_TIE)
                pick_rev = choose_by_validation(pool, None, val, REVERSE_TIE)
                erho_pick = max(pool, key=lambda name: (-1 if pool_erho(pool[name]) is None else pool_erho(pool[name]), name))
                rows.append({"scenario": "s1_selection", "n_train": n, "rep": rep, "method": "validation_rps_primary", "selected": pick, **score_named({pick: pool[pick]}, wide)[pick]})
                rows.append({"scenario": "s1_selection", "n_train": n, "rep": rep, "method": "validation_rps_reverse_tie", "selected": pick_rev, **score_named({pick_rev: pool[pick_rev]}, wide)[pick_rev]})
                rows.append({"scenario": "s1_selection", "n_train": n, "rep": rep, "method": "max_erho2", "selected": erho_pick, **score_named({erho_pick: pool[erho_pick]}, wide)[erho_pick]})
                test = counts_from_pmfs(rng_for(0, n, rep, 3), wide, N_TEST)
                sample = sample_rps(forecasts0["add_one_ce"], test)
                exact = score_named({"add_one_ce": forecasts0["add_one_ce"]}, wide)["add_one_ce"]["rps"]
                rows.append({"scenario": "s0_evaluator_check", "n_train": n, "rep": rep, "method": "add_one_ce", "rps": sample, "exact_rps": exact, "sample_minus_exact": sample - exact})

            c2 = counts_from_pmfs(rng_for(2, n, rep, 1), shaped, n)
            forecasts2 = {
                "oracle": shaped,
                "add_one_ce": add_one(c2),
                "binomial_projection": binomial_projection(c2 @ Y / n),
                "pooled_laplace": pooled_add_one(c2),
            }
            for method, metrics in score_named(forecasts2, shaped).items():
                rows.append({"scenario": "s2_shape", "n_train": n, "rep": rep, "method": method, **metrics})

            # Six raters, n ratings each. Raters 0-3 neutral, 4 lenient, 5 harsh.
            rater_counts = []
            for r_i, p_r in enumerate([raters["neutral"]] * 4 + [raters["lenient"], raters["harsh"]]):
                rater_counts.append(counts_from_pmfs(rng_for(3, n, rep, 10 + r_i), p_r, n))
            rater_counts = np.stack(rater_counts)  # (6, items, 6)
            pooled_counts = rater_counts.sum(axis=0)
            lenient_mean = rater_counts[4] @ Y / n
            forecasts3 = {
                "oracle": marginal,
                "add_one_ce": add_one(pooled_counts),
                "binomial_projection": binomial_projection(pooled_counts @ Y / pooled_counts.sum(axis=1)),
                "pooled_laplace": pooled_add_one(pooled_counts),
                "lenient_point": point_mass_at_lenient_mean(lenient_mean, 0),
                "neutral_rater_0": add_one(rater_counts[0]),
                "lenient_rater": add_one(rater_counts[4]),
                "harsh_rater": add_one(rater_counts[5]),
            }
            for method, metrics in score_named(forecasts3, marginal).items():
                rows.append({"scenario": "s3_multifacet", "n_train": n, "rep": rep, "method": method, **metrics})
            if n in (64, 256):
                coefs = []
                for r_i in range(6):
                    # The coefficient uses the same ratings as that rater's histogram.
                    # Any order of those scores within an item gives the same mean squares.
                    coefs.append(anova_erho(matrix_from_counts(rater_counts[r_i]))["erho2_clipped"])
                stochastic_best = int(np.nanargmax([(-1 if c is None else c) for c in coefs]))
                sharp = forecasts3["lenient_point"]
                sharp_erho = panel_coeffs(sharp)["erho2"]
                rows.append({
                    "scenario": "s3_selection",
                    "n_train": n,
                    "rep": rep,
                    "method": "max_erho2_stochastic_rater",
                    "selected_rater": stochastic_best,
                    "selected_is_neutral": int(stochastic_best <= 3),
                    "rater_erho2": coefs,
                    **score_named({f"rater_{stochastic_best}": add_one(rater_counts[stochastic_best])}, marginal)[f"rater_{stochastic_best}"],
                })
                rows.append({
                    "scenario": "s3_selection",
                    "n_train": n,
                    "rep": rep,
                    "method": "max_erho2_including_sharp_lenient_point",
                    "selected": "lenient_point" if (sharp_erho is not None and sharp_erho >= max(c or -1 for c in coefs) - 1e-12) else f"rater_{stochastic_best}",
                    "sharp_erho2": sharp_erho,
                    **score_named({"lenient_point": sharp}, marginal)["lenient_point"],
                })
    return rows


def pool_erho(q: np.ndarray):
    return panel_coeffs(q)["erho2"]


def sample_rps(q: np.ndarray, counts: np.ndarray) -> float:
    item_means = []
    for i in range(q.shape[0]):
        n = int(counts[i].sum())
        acc = 0.0
        cdf = np.cumsum(q[i])[:-1]
        for y, count in enumerate(counts[i]):
            if count == 0:
                continue
            ind = np.array([1.0 if y <= k else 0.0 for k in range(5)])
            acc += count * float(np.sum((cdf - ind) ** 2) / 5.0)
        item_means.append(acc / n)
    return float(np.mean(item_means))


def collect(rows, scenario, n, method, field="rps"):
    return np.array([r[field] for r in rows if r["scenario"] == scenario and r["n_train"] == n and r["method"] == method], dtype=float)


def paired(rows, scenario, n, method_a, method_b, field="rps"):
    a = {r["rep"]: r[field] for r in rows if r["scenario"] == scenario and r["n_train"] == n and r["method"] == method_a}
    b = {r["rep"]: r[field] for r in rows if r["scenario"] == scenario and r["n_train"] == n and r["method"] == method_b}
    reps = sorted(set(a) & set(b))
    return np.array([a[i] - b[i] for i in reps], dtype=float)


def fmt_interval(d: dict) -> str:
    if d["lo"] is None:
        return f"{d['mean']:.6f}"
    return f"{d['mean']:.6f} [{d['lo']:.6f}, {d['hi']:.6f}] p-val={d['p_value']:.4g}"


def summarize(rows) -> dict:
    summary = {"contrasts": [], "levels": []}
    specs = [
        ("s0_binomial", "add_one_ce"),
        ("s0_binomial", "binomial_projection"),
        ("s0_binomial", "pooled_laplace"),
        ("s0_binomial", "oracle"),
        ("s1_leniency", "lenient_point"),
        ("s1_leniency", "add_one_ce"),
        ("s1_leniency", "binomial_projection"),
        ("s1_leniency", "pooled_laplace"),
        ("s2_shape", "add_one_ce"),
        ("s2_shape", "binomial_projection"),
        ("s2_shape", "pooled_laplace"),
        ("s2_shape", "oracle"),
        ("s3_multifacet", "add_one_ce"),
        ("s3_multifacet", "binomial_projection"),
        ("s3_multifacet", "pooled_laplace"),
        ("s3_multifacet", "lenient_point"),
        ("s3_multifacet", "neutral_rater_0"),
        ("s3_multifacet", "lenient_rater"),
        ("s3_multifacet", "harsh_rater"),
        ("s3_multifacet", "oracle"),
    ]
    for scenario, method in specs:
        for n in N_GRID:
            for field in ("rps", "ce_smooth", "tvd", "bias_forecast_minus_true", "mae_of_means", "pearson", "spearman", "kendall_tau_b", "erho2"):
                vals = collect(rows, scenario, n, method, field)
                if vals.size == 0:
                    continue
                summary["levels"].append({"scenario": scenario, "n_train": n, "method": method, "field": field, **t_interval(vals)})
    contrast_specs = [
        ("s0_binomial", "binomial_projection", "add_one_ce", "rps"),
        ("s0_binomial", "add_one_ce", "pooled_laplace", "rps"),
        ("s0_binomial", "binomial_projection", "pooled_laplace", "rps"),
        ("s0_binomial", "add_one_ce", "oracle", "rps"),
        ("s0_binomial", "binomial_projection", "oracle", "rps"),
        ("s1_leniency", "lenient_point", "pooled_laplace", "rps"),
        ("s1_leniency", "lenient_point", "add_one_ce", "rps"),
        ("s1_leniency", "add_one_ce", "pooled_laplace", "rps"),
        ("s1_leniency", "lenient_point", "pooled_laplace", "ce_smooth"),
        ("s1_leniency", "lenient_point", "pooled_laplace", "tvd"),
        ("s2_shape", "binomial_projection", "add_one_ce", "rps"),
        ("s2_shape", "binomial_projection", "add_one_ce", "ce_smooth"),
        ("s2_shape", "binomial_projection", "add_one_ce", "tvd"),
        ("s2_shape", "add_one_ce", "oracle", "rps"),
        ("s2_shape", "binomial_projection", "pooled_laplace", "rps"),
        ("s3_multifacet", "lenient_point", "pooled_laplace", "rps"),
        ("s3_multifacet", "add_one_ce", "pooled_laplace", "rps"),
        ("s3_multifacet", "lenient_point", "add_one_ce", "rps"),
        ("s3_multifacet", "neutral_rater_0", "pooled_laplace", "rps"),
        ("s3_multifacet", "lenient_rater", "pooled_laplace", "rps"),
    ]
    for scenario, a, b, field in contrast_specs:
        for n in N_GRID:
            diff = paired(rows, scenario, n, a, b, field)
            if diff.size == 0:
                continue
            summary["contrasts"].append({
                "scenario": scenario,
                "n_train": n,
                "field": field,
                "order": f"{a} minus {b}",
                "positive_means": f"{a} is worse when the field is RPS, smoothed CE, or TVD",
                **t_interval(diff),
                "fraction_positive": wilson(int(np.sum(diff > 0)), int(diff.size)),
            })
    # Selection summaries.
    summary["selection"] = []
    for method in ("validation_rps_primary", "validation_rps_reverse_tie", "max_erho2"):
        subset = [r for r in rows if r["scenario"] == "s1_selection" and r["method"] == method]
        if not subset:
            continue
        counts = {}
        for r in subset:
            counts[r["selected"]] = counts.get(r["selected"], 0) + 1
        rps = np.array([r["rps"] for r in subset])
        const = collect(rows, "s1_leniency", 16, "pooled_laplace", "rps")
        # pair by rep
        const_map = {r["rep"]: r["rps"] for r in rows if r["scenario"] == "s1_leniency" and r["n_train"] == 16 and r["method"] == "pooled_laplace"}
        diff = np.array([r["rps"] - const_map[r["rep"]] for r in subset])
        summary["selection"].append({
            "method": method,
            "n_train": 16,
            "n_validation": N_VAL,
            "selected_counts": counts,
            "rps": t_interval(rps),
            "selected_minus_pooled_laplace_rps": {**t_interval(diff), "fraction_positive": wilson(int(np.sum(diff > 0)), int(diff.size))},
        })
    for n in (64, 256):
        subset = [r for r in rows if r["scenario"] == "s3_selection" and r["n_train"] == n and r["method"] == "max_erho2_stochastic_rater"]
        k = int(sum(r["selected_is_neutral"] for r in subset))
        summary["selection"].append({
            "method": "max_erho2_stochastic_rater",
            "n_train": n,
            "neutral_fraction": wilson(k, len(subset)),
            "selected_minus_pooled_laplace_rps": t_interval(paired_selection_rps(rows, subset)),
        })
        sharp = [r for r in rows if r["scenario"] == "s3_selection" and r["n_train"] == n and r["method"] == "max_erho2_including_sharp_lenient_point"]
        k_sharp = int(sum(r["selected"] == "lenient_point" for r in sharp))
        const_map = {r["rep"]: r["rps"] for r in rows if r["scenario"] == "s3_multifacet" and r["n_train"] == n and r["method"] == "pooled_laplace"}
        diff = np.array([r["rps"] - const_map[r["rep"]] for r in sharp])
        summary["selection"].append({
            "method": "max_erho2_including_sharp_lenient_point",
            "n_train": n,
            "sharp_fraction": wilson(k_sharp, len(sharp)),
            "selected_minus_pooled_laplace_rps": {**t_interval(diff), "fraction_positive": wilson(int(np.sum(diff > 0)), len(sharp))},
        })
    check_rows = [r for r in rows if r["scenario"] == "s0_evaluator_check" and r["n_train"] == 16]
    # The evaluator check was stored at n=16 because that is the only n where the test stream is drawn.
    # Also draw the pre-registered check at n=64 inside summarize? It is only stored when n==16 in the loop.
    # Fix: the loop stores the evaluator check only at n==16. The criterion named n conceptually as a check
    # that sample RPS matches exact RPS. n=16 is a valid check. Record it.
    if check_rows:
        summary["evaluator_check_n16"] = t_interval(np.array([r["sample_minus_exact"] for r in check_rows]))
    return summary


def paired_selection_rps(rows, subset):
    if not subset:
        return np.array([])
    n = subset[0]["n_train"]
    const_map = {r["rep"]: r["rps"] for r in rows if r["scenario"] == "s3_multifacet" and r["n_train"] == n and r["method"] == "pooled_laplace"}
    return np.array([r["rps"] - const_map[r["rep"]] for r in subset], dtype=float)


def find_contrast(summary, scenario, n, order, field="rps"):
    for row in summary["contrasts"]:
        if row["scenario"] == scenario and row["n_train"] == n and row["order"] == order and row["field"] == field:
            return row
    raise KeyError((scenario, n, order, field))


def find_level(summary, scenario, n, method, field="rps"):
    for row in summary["levels"]:
        if row["scenario"] == scenario and row["n_train"] == n and row["method"] == method and row["field"] == field:
            return row
    raise KeyError((scenario, n, method, field))


def evaluate_criteria(pop, summary) -> list:
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "pass": bool(ok), "detail": py(detail)})

    agree = find_contrast(summary, "s0_binomial", 256, "binomial_projection minus add_one_ce")
    add(
        "s0_agreement_interval_inside_pm_0.01_at_256",
        agree["lo"] > -0.01 and agree["hi"] < 0.01,
        agree,
    )
    for method in ("add_one_ce", "binomial_projection"):
        diff = find_contrast(summary, "s0_binomial", 256, f"{method} minus pooled_laplace")
        add(f"s0_{method}_beats_constant_by_0.03_at_256", diff["hi"] < -0.03, diff)
    small = find_contrast(summary, "s0_binomial", 4, "binomial_projection minus add_one_ce")
    add("s0_projection_better_than_add_one_at_4", small["hi"] < 0, small)
    for n in N_GRID:
        diff = find_contrast(summary, "s1_leniency", n, "lenient_point minus pooled_laplace")
        add(f"s1_lenient_point_worse_rps_than_constant_at_{n}", diff["lo"] > 0, diff)
        level = find_level(summary, "s1_leniency", n, "lenient_point", "erho2")
        add(f"s1_lenient_point_erho2_is_1_at_{n}", abs(level["mean"] - 1.0) < 1e-12 and level["lo"] > 0.999, level)
    shape = find_contrast(summary, "s2_shape", 256, "binomial_projection minus add_one_ce")
    add("s2_projection_worse_than_add_one_by_more_than_0.01_at_256", shape["lo"] > 0.01, shape)
    add("s2_population_gap", pop["s2_shape"]["mean_vertex_rps_excess"] > 0.01, pop["s2_shape"]["mean_vertex_rps_excess"])
    for n in (64, 256):
        diff = find_contrast(summary, "s3_multifacet", n, "lenient_point minus pooled_laplace")
        add(f"s3_sharp_lenient_worse_than_constant_at_{n}", diff["lo"] > 0, diff)
        ce = find_contrast(summary, "s3_multifacet", n, "add_one_ce minus pooled_laplace")
        add(f"s3_add_one_better_than_constant_at_{n}", ce["hi"] < 0, ce)
    own = pop["s3_multifacet"]["own_coefficients"]
    best_stochastic = max(("neutral", "lenient", "harsh"), key=lambda name: own[name]["erho2"])
    add("s3_population_max_erho2_stochastic_is_neutral", best_stochastic == "neutral", {k: own[k]["erho2"] for k in own})
    add(
        "s3_population_neutral_beats_constant",
        pop["s3_multifacet"]["neutral_true"]["rps"] < pop["s3_multifacet"]["pooled_frequency"]["rps"],
        {"neutral": pop["s3_multifacet"]["neutral_true"]["rps"], "constant": pop["s3_multifacet"]["pooled_frequency"]["rps"]},
    )
    for block in summary["selection"]:
        if block["method"] == "validation_rps_primary":
            k = block["selected_counts"].get("lenient_point", 0)
            add("s1_validation_rps_picks_lenient_point_under_5_percent", k / N_REP < 0.05, block["selected_counts"])
            d = block["selected_minus_pooled_laplace_rps"]
            add("s1_validation_rps_selection_beats_constant", d["hi"] < 0, d)
        if block["method"] == "max_erho2":
            k = block["selected_counts"].get("lenient_point", 0)
            add("s1_erho2_selection_picks_lenient_point", k / N_REP >= 0.95, block["selected_counts"])
        if block["method"] == "max_erho2_stochastic_rater":
            add(
                f"s3_stochastic_selection_neutral_at_{block['n_train']}",
                block["neutral_fraction"]["lo"] > 0.95 or block["neutral_fraction"]["proportion"] >= 0.95 and block["neutral_fraction"]["lo"] > 0.90,
                block["neutral_fraction"],
            )
        if block["method"] == "max_erho2_including_sharp_lenient_point":
            add(
                f"s3_sharp_selected_at_{block['n_train']}",
                block["sharp_fraction"]["proportion"] >= 0.95,
                block["sharp_fraction"],
            )
    ev = summary.get("evaluator_check_n16")
    if ev:
        add("evaluator_sample_rps_matches_exact_rps_at_n16", ev["lo"] <= 0 <= ev["hi"], ev)
    return checks


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linewidth=0.8)
    if ax.get_xscale() == "log" or ax.get_yscale() == "log":
        ax.grid(True, which="minor", linewidth=0.4, alpha=0.45)


def plot_all(summary, pop) -> None:
    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)
    # Figure 1. Agreement.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xscale("log")
    for method, color, label in (
        ("oracle", "#222222", "True binomial distribution"),
        ("binomial_projection", "#0072B2", "Binomial-projection learner"),
        ("add_one_ce", "#E69F00", "Add-one cross-entropy learner"),
        ("pooled_laplace", "#009E73", "Pooled Laplace forecast"),
    ):
        xs, ys, lo, hi = [], [], [], []
        for n in N_GRID:
            level = find_level(summary, "s0_binomial", n, method, "rps")
            xs.append(n)
            ys.append(level["mean"])
            lo.append(level["mean"] - level["lo"])
            hi.append(level["hi"] - level["mean"])
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", color=color, label=label, capsize=3)
    style(
        ax,
        "Training ratings per item on the fixed 12 binomial items (no held-out items)",
        "Expected held-out ranked probability score (RPS) under the true conditional distribution (0 to 1, lower is better)",
    )
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_s0_agreement_expected_rps.png", dpi=160, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    ax = axes[0]
    ax.set_xscale("log")
    for method, color, label in (
        ("lenient_point", "#D55E00", "Lenient point-mass judge"),
        ("pooled_laplace", "#009E73", "Pooled Laplace forecast"),
        ("add_one_ce", "#E69F00", "Add-one cross-entropy learner"),
        ("binomial_projection", "#0072B2", "Binomial-projection learner"),
    ):
        xs, ys, lo, hi = [], [], [], []
        for n in N_GRID:
            level = find_level(summary, "s1_leniency", n, method, "rps")
            xs.append(n)
            ys.append(level["mean"])
            lo.append(level["mean"] - level["lo"])
            hi.append(level["hi"] - level["mean"])
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", color=color, label=label, capsize=3)
    style(ax, "Training ratings per item (same 12 binomial items)", "Expected held-out RPS (0 to 1, lower is better)")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.set_xscale("log")
    for method, color, label in (
        ("lenient_point", "#D55E00", "Lenient point-mass judge"),
        ("add_one_ce", "#E69F00", "Add-one cross-entropy learner"),
        ("binomial_projection", "#0072B2", "Binomial-projection learner"),
    ):
        xs, ys = [], []
        for n in N_GRID:
            level = find_level(summary, "s1_leniency", n, method, "erho2")
            xs.append(n)
            ys.append(level["mean"])
        ax.plot(xs, ys, marker="o", color=color, label=label)
    style(ax, "Training ratings per item (judge-only coefficient)", "Judge-only generalizability coefficient Eρ² (higher means more repeatable, not more accurate)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_s1_leniency_rps_and_reliability.png", dpi=160, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xscale("log")
    xs, ys, lo, hi = [], [], [], []
    for n in N_GRID:
        row = find_contrast(summary, "s2_shape", n, "binomial_projection minus add_one_ce")
        xs.append(n)
        ys.append(row["mean"])
        lo.append(row["mean"] - row["lo"])
        hi.append(row["hi"] - row["mean"])
    ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", color="#0072B2", capsize=3, label="Train-fitted difference")
    gap = pop["s2_shape"]["mean_vertex_rps_excess"]
    ax.axhline(gap, color="#D55E00", linestyle="--", label=f"Population gap at the true moments ({gap:.4f})")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    style(
        ax,
        "Training ratings per item on the fixed 8 shaped items (no held-out items)",
        "Expected RPS of the binomial projection minus expected RPS of the add-one learner (positive means add-one is better)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_s2_shape_rps_difference.png", dpi=160, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    methods = [
        ("oracle", "True marginal"),
        ("add_one_ce", "Add-one, all raters"),
        ("binomial_projection", "Binomial projection"),
        ("neutral_rater_0", "Neutral rater"),
        ("pooled_laplace", "Pooled Laplace"),
        ("lenient_rater", "Lenient rater"),
        ("lenient_point", "Lenient point mass"),
    ]
    xs = np.arange(len(methods))
    means, los, his = [], [], []
    for method, _label in methods:
        level = find_level(summary, "s3_multifacet", 64, method, "rps")
        means.append(level["mean"])
        los.append(level["mean"] - level["lo"])
        his.append(level["hi"] - level["mean"])
    ax.bar(xs, means, color="#56B4E9", yerr=[los, his], capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _m, label in methods], rotation=25, ha="right")
    style(ax, "Forecast fitted on 64 training ratings per rater (6 raters, 12 items)", "Expected held-out RPS under the rater-marginal distribution (0 to 1, lower is better)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_s3_multifacet_expected_rps.png", dpi=160, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def write_csv(rows) -> None:
    path = HERE / "results" / "replication_metrics.csv"
    fields = [
        "scenario", "n_train", "rep", "method", "selected", "selected_rater", "rps", "ce_smooth", "ce_raw",
        "tvd", "bias_forecast_minus_true", "mae_of_means", "pearson", "spearman", "kendall_tau_b",
        "erho2", "phi_own", "vu", "ve", "sample_minus_exact", "exact_rps",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: py(row.get(k)) for k in fields})


def trace_one(world) -> dict:
    """One fully written replication, for inspection. Seed stream 0, rep 0, 16 ratings."""
    wide = world["wide"]
    c0 = counts_from_pmfs(rng_for(0, 16, 0, 1), wide, 16)
    q = add_one(c0)
    point = point_mass_at_lenient_mean(c0 @ Y / 16, 1)
    return {
        "item_index_6_theta": 7 / 14,
        "true_pmf_item_6": wide[6].round(6).tolist(),
        "train_counts_item_6": c0[6].tolist(),
        "add_one_item_6": q[6].round(6).tolist(),
        "lenient_point_item_6": point[6].tolist(),
        "note": "Item index 6 has theta 0.5. The add-one row uses only that item's 16 training counts. The lenient point mass is placed at round-half-up of the training mean, plus 1.",
    }


def software_block() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": matplotlib.__version__,
        "seed": SEED,
        "n_replications": N_REP,
        "generator": "numpy PCG64 via SeedSequence([seed, scenario, n_train, replication, stream])",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-only", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    world = build_worlds()
    pop = population(world)
    checks = hard_checks(world, pop)
    (HERE / "results").mkdir(exist_ok=True)
    write_json(HERE / "results" / "population.json", pop)
    write_json(HERE / "qa" / "hard_checks.json", {"checks": checks, "all_pass": all(c["pass"] for c in checks)})
    failed = [c for c in checks if not c["pass"]]
    if failed:
        print(json.dumps(failed, indent=2))
        raise SystemExit(1)
    print("population and hard checks passed")
    print("s0 oracle RPS", pop["s0_binomial_agreement"]["oracle"]["rps"])
    print("s0 constant RPS", pop["s0_binomial_agreement"]["pooled_frequency"]["rps"])
    print("s1 lenient RPS", pop["s1_leniency"]["lenient_point"]["rps"], "erho2", pop["s1_leniency"]["lenient_point"]["erho2"])
    print("s1 constant RPS", pop["s1_leniency"]["pooled_frequency"]["rps"])
    print("s1 bias", pop["s1_leniency"]["lenient_point"]["bias_forecast_minus_true"])
    print("s1 joint", json.dumps(py(pop["s1_leniency"]["joint_human_and_lenient_point"])))
    print("s2 gap", pop["s2_shape"]["mean_vertex_rps_excess"])
    print("s2 binomial RPS", pop["s2_shape"]["binomial_projection"]["rps"], "oracle", pop["s2_shape"]["oracle_shaped"]["rps"], "constant", pop["s2_shape"]["pooled_frequency"]["rps"])
    print("s3 lenient point", pop["s3_multifacet"]["lenient_point"]["rps"], "constant", pop["s3_multifacet"]["pooled_frequency"]["rps"], "oracle", pop["s3_multifacet"]["oracle_marginal"]["rps"])
    print("s3 own", json.dumps(py(pop["s3_multifacet"]["own_coefficients"])))
    if args.population_only:
        print(f"population-only seconds {time.time() - t0:.2f}")
        return
    rows = run_monte_carlo(world)
    write_csv(rows)
    summary = summarize(rows)
    criteria = evaluate_criteria(pop, summary)
    write_json(HERE / "results" / "summary.json", summary)
    write_json(HERE / "results" / "trace_rep0.json", trace_one(world))
    write_json(HERE / "qa" / "criteria_checks.json", {"checks": criteria, "all_pass": all(c["pass"] for c in criteria), "software": software_block(), "seconds": time.time() - t0})
    plot_all(summary, pop)
    failed_c = [c for c in criteria if not c["pass"]]
    print(f"monte carlo seconds {time.time() - t0:.2f}")
    print("criteria", sum(c["pass"] for c in criteria), "/", len(criteria))
    for c in failed_c:
        print("FAIL", c["name"])
    write_json(HERE / "results" / "run_meta.json", {"seconds": time.time() - t0, "software": software_block(), "n_rows": len(rows)})


if __name__ == "__main__":
    main()
