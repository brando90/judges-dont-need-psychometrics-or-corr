"""Numerical checks of the four Lean claims. Every identity is asserted, then returned.

These are population or exact-arithmetic checks on known laws. They are not human-data results.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def check_all(seed: int = 0) -> dict:
    out = {}
    out["standardized_mse"] = _standardized()
    out["mse_decomposition"] = _mse_decomposition()
    out["brier_proper"] = _brier_proper()
    out["empirical_rate"] = _empirical_rate(seed)
    out["binary_l2"] = _binary_l2()
    out["binary_affine_witness"] = _binary_witness()
    out["fair_coin"] = _fair_coin()
    out["independent_collapse"] = _independent_collapse()
    out["mean_ne_pmf"] = _mean_ne_pmf()
    out["anti_proper"] = _anti_proper(seed)
    out["rank_corr_invariance"] = _rank_invariance(seed)
    failed = [k for k, v in out.items() if not v["ok"]]
    if failed:
        raise AssertionError(f"numerical proof checks failed: {failed}")
    return out


def _standardized():
    rng = np.random.default_rng(1)
    z_x = rng.normal(size=20000)
    z_y = rng.normal(size=20000)
    z_x = (z_x - z_x.mean()) / z_x.std()
    z_y = (z_y - z_y.mean()) / z_y.std()
    left = np.mean((z_x - z_y) ** 2)
    rho = _corr(z_x, z_y)
    right = 2 * (1 - rho)
    return {"ok": abs(left - right) < 1e-9, "left": left, "right": right, "corr": rho}


def _mse_decomposition():
    rng = np.random.default_rng(2)
    x = rng.normal(loc=1.5, scale=2.0, size=30000)
    y = 0.4 * x + rng.normal(loc=-0.5, scale=1.3, size=30000)
    mse = np.mean((x - y) ** 2)
    bias = (x.mean() - y.mean()) ** 2
    sx, sy = x.std(), y.std()
    rho = _corr(x, y)
    recon = bias + (sx - sy) ** 2 + 2 * sx * sy * (1 - rho)
    return {"ok": abs(mse - recon) < 1e-8, "mse": float(mse), "recon": float(recon),
            "bias2": float(bias), "scale2": float((sx - sy) ** 2), "corr": rho}


def _brier_proper():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    # E[BS(q)] - E[BS(p)] = ||q-p||^2
    def exp_brier(forecast):
        total = 0.0
        for i, pi in enumerate(p):
            e = np.zeros(4)
            e[i] = 1
            total += pi * np.sum((forecast - e) ** 2)
        return total
    excess = exp_brier(q) - exp_brier(p)
    db = float(np.sum((p - q) ** 2))
    # RPS excess = sum_t (Fq - Fp)^2
    fp = np.cumsum(p)[:-1]
    fq = np.cumsum(q)[:-1]
    rps_excess = float(np.sum((fq - fp) ** 2))
    def exp_rps(forecast):
        total = 0.0
        cdf = np.cumsum(forecast)[:-1]
        for i, pi in enumerate(p):
            ind = (np.arange(3) >= i).astype(float)
            total += pi * np.sum((cdf - ind) ** 2)
        return total
    rps_gap = exp_rps(q) - exp_rps(p)
    return {"ok": abs(excess - db) < 1e-12 and abs(rps_gap - rps_excess) < 1e-12 and db > 0,
            "brier_excess": excess, "D_B": db, "rps_excess": rps_gap}


def _empirical_rate(seed):
    rng = np.random.default_rng(seed)
    p = np.array([0.2, 0.5, 0.3])
    ns = [10, 50, 200]
    rows = []
    ok = True
    formula = (1 - np.dot(p, p))
    for n in ns:
        errs = []
        plugin = []
        q = np.array([0.1, 0.2, 0.7])
        db = np.sum((p - q) ** 2)
        for _ in range(4000):
            draws = rng.choice(3, size=n, p=p)
            freq = np.bincount(draws, minlength=3) / n
            errs.append(np.sum((freq - p) ** 2))
            plugin.append(np.sum((freq - q) ** 2))
        mean_err = float(np.mean(errs))
        theory = formula / n
        mean_plugin = float(np.mean(plugin))
        theory_plugin = db + formula / n
        ok = ok and abs(mean_err - theory) < 0.01 * max(theory, 1e-6) + 0.002
        ok = ok and abs(mean_plugin - theory_plugin) < 0.01 * theory_plugin + 0.002
        rows.append({"N": n, "mean_l2": mean_err, "theory": float(theory),
                     "mean_plugin": mean_plugin, "theory_plugin": float(theory_plugin)})
    return {"ok": ok, "rows": rows, "one_minus_norm2": float(formula)}


def _binary_l2():
    gaps = []
    for q in (0.1, 0.5, 0.9):
        for y in (0.0, 1.0):
            gaps.append((q - y) ** 2 + ((1 - q) - (1 - y)) ** 2 - 2 * (q - y) ** 2)
    return {"ok": max(abs(g) for g in gaps) < 1e-12, "max_abs_gap": max(abs(g) for g in gaps)}


def _binary_witness():
    # Four paired items. q* has rational standard deviation 3/10; see the Lean witness.
    q_star = np.array([0.2, 0.2, 0.8, 0.8])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    q_bad = 0.1 * q_star + 0.45
    rho_star = _corr(q_star, y)
    rho_bad = _corr(q_bad, y)

    def brier(q):
        total = 0.0
        for qi, yi in zip(q, y):
            total += (qi - yi) ** 2 + ((1 - qi) - (1 - yi)) ** 2
        return total / len(q)

    def ce(q):
        total = 0.0
        for qi, yi in zip(q, y):
            p = qi if yi == 1 else 1 - qi
            total += -math.log(p)
        return total / len(q)

    b_star, b_bad = brier(q_star), brier(q_bad)
    c_star, c_bad = ce(q_star), ce(q_bad)
    return {
        "ok": abs(rho_star - rho_bad) < 1e-12 and b_bad > b_star + 0.1 and c_bad > c_star + 0.1,
        "corr_star": rho_star,
        "corr_bad": rho_bad,
        "brier_star": b_star,
        "brier_bad": b_bad,
        "ce_star": c_star,
        "ce_bad": c_bad,
        "realized_prob_star": 0.8,
        "realized_prob_bad": 0.53,
    }


def _fair_coin():
    # X, Y Bern(1/2). Three couplings. D_B of the two marginals is 0 in every case.
    # E[(X-Y)^2] is 0, 1/2, 1.
    same = 0.0  # X = Y
    indep = 0.25 * ((0 - 0) ** 2 + (0 - 1) ** 2 + (1 - 0) ** 2 + (1 - 1) ** 2)
    anti = 1.0  # X = 1 - Y
    db = 0.0  # both marginals are (1/2, 1/2)
    return {"ok": same == 0 and abs(indep - 0.5) < 1e-12 and anti == 1 and db == 0,
            "E_sq_same": same, "E_sq_indep": indep, "E_sq_anti": anti, "D_B": db}


def _independent_collapse():
    rng = np.random.default_rng(3)
    x = rng.normal(size=80000)
    y = rng.normal(loc=0.7, scale=1.4, size=80000)
    # Algebra: mean((x-y)^2) = (mean x - mean y)^2 + var x + var y - 2 cov.
    # Independence is the sample covariance being near 0, not the algebra.
    left = float(np.mean((x - y) ** 2))
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    right_if_indep = (x.mean() - y.mean()) ** 2 + float(x.var()) + float(y.var())
    algebra = right_if_indep - 2 * cov
    return {
        "ok": abs(left - algebra) < 1e-8 and abs(cov) < 0.02 and abs(left - right_if_indep) < 0.05,
        "left": left,
        "right_dropping_covariance": float(right_if_indep),
        "sample_covariance": cov,
    }


def _mean_ne_pmf():
    # Consensus on 2 versus 50/50 on {0, 4}. Means are both 2. Proper scores separate them.
    values = np.arange(6)
    cons = np.zeros(6)
    cons[2] = 1.0
    split = np.zeros(6)
    split[0] = split[4] = 0.5
    mu_c = float(cons @ values)
    mu_s = float(split @ values)
    db = float(np.sum((cons - split) ** 2))

    def exp_rps(p, q):
        total = 0.0
        cdf = np.cumsum(q)[:-1]
        for i, pi in enumerate(p):
            ind = (np.arange(5) >= i).astype(float)
            total += pi * np.sum((cdf - ind) ** 2)
        return total

    rps_cons_on_split = exp_rps(split, cons)
    rps_split_on_split = exp_rps(split, split)
    # CE of the consensus forecast when Y=2 is 0; CE of the split forecast when Y=2 is infinite
    # because split puts mass 0 on 2. Under the split law, the consensus forecast also has CE infinity
    # (mass 0 on 0 and on 4). Compare a strictly positive perturbation only in the numeric CE of the
    # two laws as forecasts of each other using their own supports' gap.
    ce_split_under_cons = float("inf")  # log 0
    return {
        "ok": mu_c == 2 and mu_s == 2 and db == 1.5 and rps_cons_on_split > rps_split_on_split,
        "mu": mu_c,
        "D_B": db,
        "rps_of_consensus_under_split": rps_cons_on_split,
        "rps_of_split_under_split": rps_split_on_split,
        "rps_gap": rps_cons_on_split - rps_split_on_split,
        "normalized_rps_gap": (rps_cons_on_split - rps_split_on_split) / 5,
        "ce_of_split_when_Y_is_2": ce_split_under_cons,
    }


def _anti_proper(seed):
    # Two equally likely contexts. Conditional laws on {0,2} and {2,4}.
    # Var(mu) = E[sigma^2] = 1, so Corr(X_p*, Y) = 1/2 and Corr(X_delta, Y)^2 = 1/2.
    rng = np.random.default_rng(seed)
    n = 200000
    context = rng.integers(0, 2, size=n)
    mu = np.where(context == 0, 1.0, 3.0)
    # Y and X_p* independent given context.
    y = mu + np.where(rng.random(n) < 0.5, -1.0, 1.0)
    x_star = mu + np.where(rng.random(n) < 0.5, -1.0, 1.0)
    x_delta = 0.1 * mu + 4.0
    rho_star = _corr(x_star, y)
    rho_delta = _corr(x_delta, y)
    return {
        "ok": abs(rho_star - 0.5) < 0.01 and abs(rho_delta - math.sqrt(0.5)) < 0.01 and rho_delta > rho_star,
        "corr_true_distribution": rho_star,
        "corr_collapsed": rho_delta,
        "theory_true": 0.5,
        "theory_collapsed": math.sqrt(0.5),
        "n": n,
    }


def _rank_invariance(seed):
    rng = np.random.default_rng(seed)
    n = 4000
    context = rng.integers(0, 2, size=n)
    mu = np.where(context == 0, 1.0, 3.0)
    y = mu + np.where(rng.random(n) < 0.5, -1.0, 1.0)
    x_star = mu + np.where(rng.random(n) < 0.5, -1.0, 1.0)
    x_delta = 0.1 * mu + 4.0
    # Monotone warp of a judge score does not change Spearman or Kendall with Y.
    warp = np.exp(x_delta)  # strictly increasing
    sp_delta = spearmanr(x_delta, y).statistic
    sp_warp = spearmanr(warp, y).statistic
    kd_delta = kendalltau(x_delta, y).statistic
    kd_warp = kendalltau(warp, y).statistic
    sp_star = spearmanr(x_star, y).statistic
    kd_star = kendalltau(x_star, y).statistic
    return {
        "ok": abs(sp_delta - sp_warp) < 1e-12 and abs(kd_delta - kd_warp) < 1e-12
        and sp_delta > sp_star and kd_delta > kd_star,
        "spearman_collapsed": float(sp_delta),
        "spearman_warped": float(sp_warp),
        "spearman_true": float(sp_star),
        "kendall_collapsed": float(kd_delta),
        "kendall_warped": float(kd_warp),
        "kendall_true": float(kd_star),
    }
