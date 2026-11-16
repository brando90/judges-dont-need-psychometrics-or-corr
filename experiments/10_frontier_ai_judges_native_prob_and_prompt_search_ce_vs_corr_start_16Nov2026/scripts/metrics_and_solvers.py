#!/usr/bin/env python3
"""Metrics, MSE decomposition, information-gap diagnostics, certified convex/cone simplex solvers,
smooth rank-correlation optimizers, 2-parameter convex calibration head, and 48-parameter feature head
with 1-start vs 5-restart verification for Experiment 10."""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import expit, logsumexp, softmax
from scipy.stats import kendalltau, rankdata, spearmanr

GRID_01 = np.arange(6, dtype=float) / 5.0
GRID_05 = np.arange(6, dtype=float)
PROB_FLOOR = 1e-6
CERT_TOL = 1e-8
TAU_RANK = 0.1  # in 0..5 rating points (matches Exp 06)


def rating_histogram(ratings: list[int]) -> np.ndarray:
    arr = np.asarray(ratings, dtype=int)
    return np.bincount(arr, minlength=6).astype(float) / len(arr)


def smooth_probs(q: np.ndarray, eps: float = PROB_FLOOR) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return (1.0 - 6.0 * eps) * q + eps


def icc_pair(x: np.ndarray, y: np.ndarray, kind: str = "A1") -> float:
    m = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    m = m[np.all(np.isfinite(m), axis=1)]
    n, k = m.shape if m.size else (0, 0)
    if n < 3 or k != 2:
        return float("nan")
    grand = float(m.mean())
    ss_rows = k * float(((m.mean(axis=1) - grand) ** 2).sum())
    ss_cols = n * float(((m.mean(axis=0) - grand) ** 2).sum())
    ss_tot = float(((m - grand) ** 2).sum())
    ss_err = ss_tot - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    if kind == "A1":
        den = msr + (k - 1) * mse + k * (msc - mse) / n
    elif kind == "C1":
        den = msr + (k - 1) * mse
    else:
        raise ValueError(kind)
    return float("nan") if abs(den) < 1e-15 else float((msr - mse) / den)


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - x.mean()
    dy = y - y.mean()
    den = float(np.sqrt((dx @ dx) * (dy @ dy)))
    return float(dx @ dy / den) if den > 1e-14 else float("nan")


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) < 1e-13 or np.ptp(y) < 1e-13:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def kendall_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) < 1e-13 or np.ptp(y) < 1e-13:
        return float("nan")
    return float(kendalltau(x, y, variant="b").statistic)


def mse_decomposition_05(judge_05: np.ndarray, human_05: np.ndarray) -> dict[str, float]:
    """Exact Appendix D identity: MSE = bias^2 + scale_mismatch^2 + 2 s_x s_y (1 - r)."""
    x = np.asarray(judge_05, dtype=float)
    y = np.asarray(human_05, dtype=float)
    mse = float(np.mean((x - y) ** 2))
    mx, my = float(x.mean()), float(y.mean())
    sx, sy = float(x.std(ddof=0)), float(y.std(ddof=0))
    bias = mx - my
    bias_sq = bias * bias
    scale_sq = (sx - sy) ** 2
    r = pearson_safe(x, y)
    unexpl = 2.0 * sx * sy * (1.0 - r) if math.isfinite(r) else float(mse - bias_sq - scale_sq)
    recon = bias_sq + scale_sq + unexpl
    return {
        "mse_0_5": mse,
        "bias_0_5": bias,
        "bias_sq_0_5": bias_sq,
        "scale_mismatch_0_5": sx - sy,
        "scale_sq_0_5": scale_sq,
        "unexplained_corr_term_0_5": unexpl,
        "identity_max_abs_err": abs(mse - recon),
        "judge_mean_0_5": mx,
        "human_mean_0_5": my,
        "judge_sd_0_5": sx,
        "human_sd_0_5": sy,
    }


def per_artifact_metrics(p_emp: np.ndarray, q_raw: np.ndarray, eps: float = PROB_FLOOR) -> dict[str, np.ndarray]:
    """Compute per-artifact metrics for empirical histograms p_emp (N,6) and judge forecasts q_raw (N,6)."""
    p = np.asarray(p_emp, dtype=float)
    q_unsmoothed = np.asarray(q_raw, dtype=float)
    q = smooth_probs(q_unsmoothed, eps) if eps > 0 else q_unsmoothed

    fp = np.cumsum(p, axis=1)[:, :5]
    fq = np.cumsum(q, axis=1)[:, :5]
    # Individual-rating RPS averaged over each artifact's observed human ratings:
    # (1/m_i) sum_{y} (1/5) sum_{k=0}^4 (Q(k) - 1{y<=k})^2 == (1/5) sum_k [Q(k)^2 - 2 Q(k) F_hat(k) + F_hat(k)]
    rps = np.mean(fq * fq - 2.0 * fq * fp + fp, axis=1)
    ce = -np.sum(p * np.log(q), axis=1)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=1)
    brier = 1.0 + np.sum(q * q, axis=1) - 2.0 * np.sum(p * q, axis=1)
    tvd = 0.5 * np.sum(np.abs(q - p), axis=1)
    zero_miss = np.any((p > 0) & (q_unsmoothed <= 0.0), axis=1).astype(float)

    mu_p_01 = np.round(p @ GRID_01, 12)
    mu_q_01 = np.round(q @ GRID_01, 12)
    mu_p_05 = np.round(p @ GRID_05, 12)
    mu_q_05 = np.round(q @ GRID_05, 12)

    return {
        "rps": rps,
        "ce": ce,
        "kl": ce - entropy,
        "entropy": entropy,
        "brier": brier,
        "tvd": tvd,
        "zero_prob_miss_any": zero_miss,
        "mae_01": np.abs(mu_q_01 - mu_p_01),
        "mse_01": (mu_q_01 - mu_p_01) ** 2,
        "bias_01": mu_q_01 - mu_p_01,
        "human_mean_01": mu_p_01,
        "judge_mean_01": mu_q_01,
        "human_mean_05": mu_p_05,
        "judge_mean_05": mu_q_05,
    }


def summarize_panel(
    p_emp: np.ndarray, q_raw: np.ndarray, idx: np.ndarray | None = None, eps: float = PROB_FLOOR
) -> dict[str, float]:
    n = len(p_emp)
    if idx is None:
        idx = np.arange(n)
    idx = np.asarray(idx, dtype=int)
    pa = per_artifact_metrics(p_emp[idx], q_raw[idx], eps=eps)
    mu_q_01, mu_p_01 = pa["judge_mean_01"], pa["human_mean_01"]
    mu_q_05, mu_p_05 = pa["judge_mean_05"], pa["human_mean_05"]

    out = {
        "n_artifacts": int(len(idx)),
        "rps": float(pa["rps"].mean()),
        "ce": float(pa["ce"].mean()),
        "kl": float(pa["kl"].mean()),
        "brier": float(pa["brier"].mean()),
        "tvd": float(pa["tvd"].mean()),
        "zero_prob_miss_rate": float(pa["zero_prob_miss_any"].mean()),
        "pearson": pearson_safe(mu_q_01, mu_p_01),
        "spearman": spearman_safe(mu_q_01, mu_p_01),
        "kendall": kendall_safe(mu_q_01, mu_p_01),
        "icc_A1": icc_pair(mu_q_01, mu_p_01, "A1"),
        "icc_C1": icc_pair(mu_q_01, mu_p_01, "C1"),
        "mae_01": float(pa["mae_01"].mean()),
        "mse_01": float(pa["mse_01"].mean()),
        "bias_01": float(pa["bias_01"].mean()),
        "judge_mean_01": float(mu_q_01.mean()),
        "human_mean_01": float(mu_p_01.mean()),
    }
    out.update(mse_decomposition_05(mu_q_05, mu_p_05))
    return out


def artifact_disagreement_diagnostics(ratings_list: list[list[int]]) -> dict[str, np.ndarray]:
    """Compute per-artifact empirical SD, RPS disagreement floor, and Jeffreys-smoothed information gap G_i."""
    n = len(ratings_list)
    emp_sd = np.zeros(n)
    emp_rps_floor = np.zeros(n)
    info_gap_g = np.zeros(n)
    for i, rts in enumerate(ratings_list):
        arr = np.asarray(rts, dtype=float)
        emp_sd[i] = float(arr.std(ddof=0))
        p = rating_histogram(rts)
        fp = np.cumsum(p)[:5]
        emp_rps_floor[i] = float(np.mean(fp * (1.0 - fp)))
        # Jeffreys Dirichlet(0.5) smoothed distribution p_tilde:
        # G_i = RPS disagreement floor of p_tilde (captures multi-modality / spread beyond a point mass)
        counts = np.bincount(np.asarray(rts, dtype=int), minlength=6).astype(float)
        p_tilde = (counts + 0.5) / (len(rts) + 3.0)
        f_tilde = np.cumsum(p_tilde)[:5]
        info_gap_g[i] = float(np.mean(f_tilde * (1.0 - f_tilde)))
    return {
        "empirical_sd_0_5": emp_sd,
        "empirical_rps_floor": emp_rps_floor,
        "smoothed_info_gap_G": info_gap_g,
    }


# --- Certified Simplex Mixture Solvers (Class M2) ---

def fit_simplex_ce(q_cand: np.ndarray, p_emp: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Convex log-loss simplex solver with Newton active-face polishing & Frank-Wolfe gap certificate."""
    n, m, _ = q_cand.shape
    w0 = np.full(m, 1.0 / m)

    def fg(w: np.ndarray) -> tuple[float, np.ndarray]:
        w_pos = np.maximum(w, 0.0)
        s = w_pos.sum()
        w_pos = w_pos / s if s > 0 else w0
        pred = np.einsum("njr,j->nr", q_cand, w_pos)
        val = -float(np.sum(p_emp * np.log(pred)) / n)
        grad = -np.einsum("nr,njr->j", p_emp / pred, q_cand) / n
        return val, grad

    opt = minimize(
        fg,
        w0,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * m,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(m)},
        options={"maxiter": 800, "ftol": 1e-15},
    )
    w = np.maximum(opt.x, 0.0)
    w /= w.sum()

    for _ in range(12):
        val, grad = fg(w)
        if float(grad @ w - grad.min()) <= CERT_TOL / 10.0:
            break
        active = np.flatnonzero(w > 1e-10)
        if len(active) < 2:
            break
        pred = np.einsum("njr,j->nr", q_cand, w)
        hess = np.einsum("nr,njr,nkr->jk", p_emp / (pred * pred), q_cand[:, active], q_cand[:, active]) / n
        basis = np.vstack([np.eye(len(active) - 1), -np.ones(len(active) - 1)])
        tg = basis.T @ grad[active]
        red = np.linalg.lstsq(basis.T @ hess @ basis, -tg, rcond=None)[0]
        d = np.zeros(m)
        d[active] = basis @ red
        slope = float(tg @ red)
        if slope >= 0 or np.max(np.abs(d)) < 1e-15:
            break
        neg = d < 0
        step = min(1.0, float(np.min(-w[neg] / d[neg]))) if np.any(neg) else 1.0
        accepted = False
        for _ in range(20):
            trial = np.maximum(w + step * d, 0.0)
            trial /= trial.sum()
            tv, _ = fg(trial)
            if tv <= val + 1e-4 * step * slope + 1e-15:
                w = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    val, grad = fg(w)
    gap = float(grad @ w - grad.min())
    return w, {"objective": "ce", "value": val, "fw_gap": gap, "certified": bool(gap <= CERT_TOL)}


def fit_simplex_rps(q_cand: np.ndarray, p_emp: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Convex quadratic RPS simplex solver with Frank-Wolfe gap certificate."""
    n, m, _ = q_cand.shape
    fq = np.cumsum(q_cand, axis=2)[:, :, :5]  # (n, m, 5)
    fp = np.cumsum(p_emp, axis=1)[:, :5]      # (n, 5)
    w0 = np.full(m, 1.0 / m)

    def fg(w: np.ndarray) -> tuple[float, np.ndarray]:
        Q = np.einsum("njk,j->nk", fq, w)
        diff = Q - fp
        val = float(np.mean(Q * Q - 2.0 * Q * fp + fp))
        grad = 2.0 * np.einsum("nk,njk->j", diff, fq) / (5.0 * n)
        return val, grad

    opt = minimize(
        fg,
        w0,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * m,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(m)},
        options={"maxiter": 800, "ftol": 1e-15},
    )
    w = np.maximum(opt.x, 0.0)
    w /= w.sum()

    hess_full = 2.0 * np.einsum("njk,nlk->jl", fq, fq) / (5.0 * n)
    for _ in range(12):
        val, grad = fg(w)
        if float(grad @ w - grad.min()) <= CERT_TOL / 10.0:
            break
        active = np.flatnonzero(w > 1e-10)
        if len(active) < 2:
            break
        basis = np.vstack([np.eye(len(active) - 1), -np.ones(len(active) - 1)])
        tg = basis.T @ grad[active]
        red = np.linalg.lstsq(basis.T @ hess_full[np.ix_(active, active)] @ basis, -tg, rcond=None)[0]
        d = np.zeros(m)
        d[active] = basis @ red
        slope = float(tg @ red)
        if slope >= 0 or np.max(np.abs(d)) < 1e-15:
            break
        neg = d < 0
        step = min(1.0, float(np.min(-w[neg] / d[neg]))) if np.any(neg) else 1.0
        w = np.maximum(w + step * d, 0.0)
        w /= w.sum()

    val, grad = fg(w)
    gap = float(grad @ w - grad.min())
    return w, {"objective": "rps", "value": val, "fw_gap": gap, "certified": bool(gap <= CERT_TOL)}


def fit_simplex_pearson(q_cand: np.ndarray, p_emp: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact Pearson mixture solver on the simplex via cone projection (face enumeration for m<=8, NNLS for m>8)."""
    _, m, _ = q_cand.shape
    means = q_cand @ GRID_01  # (n, m)
    y = p_emp @ GRID_01       # (n,)
    a = means - means.mean(axis=0, keepdims=True)
    b = y - y.mean()
    norms = np.linalg.norm(a, axis=0)
    norm_b = float(np.linalg.norm(b))
    active = np.flatnonzero(norms > 1e-12)
    if norm_b <= 1e-12 or len(active) == 0:
        w = np.full(m, 1.0 / m)
        return w, {"objective": "pearson", "value": float("nan"), "certified": True, "status": "constant"}

    unit = a[:, active] / norms[active]
    prods = unit.T @ b
    if float(np.max(prods)) <= 0.0:
        single_r = prods / norm_b
        best_idx = int(active[int(np.argmax(single_r))])
        w = np.eye(m)[best_idx]
        return w, {"objective": "pearson", "value": float(np.max(single_r)), "certified": True, "status": "vertex"}

    if len(active) <= 8:
        best_loss = float("inf")
        best_coeff = None
        for size in range(1, len(active) + 1):
            for face in itertools.combinations(range(len(active)), size):
                ix = np.array(face)
                c = np.linalg.lstsq(unit[:, ix], b, rcond=None)[0]
                if c.min() < -1e-10:
                    continue
                c = np.maximum(c, 0.0)
                if c.sum() <= 0:
                    continue
                full = np.zeros(len(active))
                full[ix] = c
                loss = float(np.sum((unit @ full - b) ** 2))
                if loss < best_loss - 1e-12:
                    best_loss = loss
                    best_coeff = full
        assert best_coeff is not None
        coeff = best_coeff
    else:
        coeff, _ = nnls(unit, b)
        if coeff.sum() <= 0:
            j = int(np.argmax(prods))
            coeff = np.zeros(len(active))
            coeff[j] = 1.0

    z = unit @ coeff
    grad = unit.T @ (z - b)
    dual_viol = float(max(0.0, -grad.min())) / (1.0 + norm_b)
    w = np.zeros(m)
    w[active] = coeff / norms[active]
    w /= w.sum()
    mu = (np.einsum("njr,j->nr", q_cand, w)) @ GRID_01
    r_val = pearson_safe(mu, y)
    return w, {"objective": "pearson", "value": r_val, "dual_violation": dual_viol, "certified": bool(dual_viol <= 1e-6)}


def _pearson_grad_vec(a_raw: np.ndarray, b_raw: np.ndarray) -> tuple[float, np.ndarray]:
    a = a_raw - a_raw.mean()
    b = b_raw - b_raw.mean()
    na = float(np.linalg.norm(a)) + 1e-12
    nb = float(np.linalg.norm(b)) + 1e-12
    r = float((a @ b) / (na * nb))
    return r, (b / nb - r * a / na) / na


def _soft_spearman_grad_vec(m_05: np.ndarray, h_rank: np.ndarray) -> tuple[float, np.ndarray]:
    d = (m_05[:, None] - m_05[None, :]) / TAU_RANK
    sig = expit(d)
    u = sig.sum(axis=1)
    r, gu = _pearson_grad_vec(u, h_rank)
    s = sig * (1.0 - sig) / TAU_RANK
    return r, gu * s.sum(axis=1) - s @ gu


def _soft_kendall_grad_vec(m_05: np.ndarray, h_05: np.ndarray) -> tuple[float, np.ndarray]:
    n = len(m_05)
    sgn = np.sign(h_05[:, None] - h_05[None, :])
    t = np.tanh((m_05[:, None] - m_05[None, :]) / TAU_RANK)
    c = 2.0 / max(1, n * (n - 1))
    val = float(c * 0.5 * np.sum(t * sgn))
    grad = c * np.sum(((1.0 - t * t) / TAU_RANK) * sgn, axis=1)
    return val, grad


def fit_simplex_rank(
    q_cand: np.ndarray, p_emp: np.ndarray, kind: str, seed: int = 20261116
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit Spearman, Kendall, or 1/2 Pearson + 1/2 Spearman on simplex via 4,000 Dirichlet draws + 5 SLSQP polish starts."""
    means_05 = q_cand @ GRID_05  # (n, m)
    h_05 = p_emp @ GRID_05
    h_rank = rankdata(h_05)
    m = q_cand.shape[1]
    rng = np.random.default_rng(seed)

    # Sample 4,000 simplex candidates (mixed sparse and dense Dirichlet draws + vertices + uniform)
    W_grid = np.vstack(
        [
            np.eye(m),
            np.full((1, m), 1.0 / m),
            rng.dirichlet(np.full(m, 0.3), size=2000),
            rng.dirichlet(np.full(m, 1.0), size=2000),
        ]
    )
    preds_05 = W_grid @ means_05.T  # (K, n)

    def exact_score(mu: np.ndarray) -> float:
        if np.ptp(mu) < 1e-12:
            return -2.0
        if kind == "spearman":
            return spearman_safe(mu, h_05)
        if kind == "kendall":
            return kendall_safe(mu, h_05)
        if kind == "pearson_spearman":
            return 0.5 * pearson_safe(mu, h_05) + 0.5 * spearman_safe(mu, h_05)
        raise ValueError(kind)

    scores = np.array([exact_score(preds_05[k]) for k in range(len(W_grid))])
    scores = np.where(np.isfinite(scores), scores, -2.0)
    top5_idx = np.argsort(-scores)[:5]

    best_w = W_grid[int(top5_idx[0])].copy()
    best_exact = float(scores[int(top5_idx[0])])

    def smooth_fg(w: np.ndarray) -> tuple[float, np.ndarray]:
        mu = means_05 @ w
        if kind == "spearman":
            v, g_mu = _soft_spearman_grad_vec(mu, h_rank)
        elif kind == "kendall":
            v, g_mu = _soft_kendall_grad_vec(mu, h_05)
        elif kind == "pearson_spearman":
            vp, gp = _pearson_grad_vec(mu, h_05)
            vs, gs = _soft_spearman_grad_vec(mu, h_rank)
            v, g_mu = 0.5 * (vp + vs), 0.5 * (gp + gs)
        else:
            raise ValueError(kind)
        return -v, -(means_05.T @ g_mu)

    for idx0 in top5_idx:
        opt = minimize(
            smooth_fg,
            W_grid[int(idx0)].copy(),
            method="SLSQP",
            jac=True,
            bounds=[(0.0, 1.0)] * m,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(m)},
            options={"maxiter": 250, "ftol": 1e-12},
        )
        w_try = np.maximum(opt.x, 0.0)
        if w_try.sum() > 0:
            w_try /= w_try.sum()
            sc = exact_score(means_05 @ w_try)
            if math.isfinite(sc) and sc > best_exact:
                best_exact = sc
                best_w = w_try

    return best_w, {"objective": kind, "value": best_exact, "certified": False}


def fit_simplex_any(
    q_cand: np.ndarray, p_emp: np.ndarray, objective: str, seed: int = 20261116
) -> tuple[np.ndarray, dict[str, Any]]:
    if objective == "ce":
        return fit_simplex_ce(q_cand, p_emp)
    if objective == "rps":
        return fit_simplex_rps(q_cand, p_emp)
    if objective == "pearson":
        return fit_simplex_pearson(q_cand, p_emp)
    if objective in ("spearman", "kendall", "pearson_spearman"):
        return fit_simplex_rank(q_cand, p_emp, objective, seed=seed)
    raise ValueError(f"Unknown simplex objective: {objective}")


def select_single_judge(q_cand: np.ndarray, p_emp: np.ndarray, objective: str) -> int:
    """Exhaustive single-judge selector on training rows (guaranteed global optimum)."""
    m = q_cand.shape[1]
    best_j, best_val = 0, float("inf")
    for j in range(m):
        pan = summarize_panel(p_emp, q_cand[:, j, :])
        if objective in ("ce", "rps", "tvd", "brier"):
            v = pan[objective]
        elif objective in ("pearson", "spearman", "kendall"):
            v = -pan[objective] if math.isfinite(pan[objective]) else float("inf")
        elif objective == "pearson_spearman":
            pr, sp = pan["pearson"], pan["spearman"]
            v = -0.5 * (pr + sp) if (math.isfinite(pr) and math.isfinite(sp)) else float("inf")
        else:
            raise ValueError(objective)
        if v < best_val - 1e-12:
            best_val, best_j = v, j
    return best_j


# --- 2-Parameter Convex Temperature + Ordinal-Tilt Calibration Head ---

def apply_temp_tilt(q_raw: np.ndarray, alpha: float, beta: float, eps: float = PROB_FLOOR) -> np.ndarray:
    q = smooth_probs(q_raw, eps)
    logits = alpha * np.log(q) + beta * GRID_01[None, :]
    return softmax(logits, axis=1)


def fit_temp_tilt_head(
    q_train: np.ndarray, p_train: np.ndarray, objective: str = "ce", ridge: float = 1e-3
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Fit 2-parameter (alpha, beta) calibration q_{alpha,beta}(r|x) = softmax(alpha log q + beta r/5).
    At (alpha=1, beta=0), recovers raw q identically. Convex in (alpha, beta) under CE!
    """
    n = len(p_train)
    log_q = np.log(smooth_probs(q_train, PROB_FLOOR))  # (n, 6)
    h_05 = p_train @ GRID_05

    if objective == "ce":
        def fg(params: np.ndarray) -> tuple[float, np.ndarray]:
            a, b = float(params[0]), float(params[1])
            logits = a * log_q + b * GRID_01[None, :]
            lse = logsumexp(logits, axis=1, keepdims=True)
            probs = np.exp(logits - lse)
            loss = -float(np.sum(p_train * (logits - lse)) / n) + (ridge / n) * ((a - 1.0) ** 2 + b * b)
            diff = probs - p_train
            ga = float(np.sum(diff * log_q) / n) + (2.0 * ridge / n) * (a - 1.0)
            gb = float(np.sum(diff * GRID_01[None, :]) / n) + (2.0 * ridge / n) * b
            return loss, np.array([ga, gb])

        res = minimize(fg, np.array([1.0, 0.0]), method="L-BFGS-B", jac=True, bounds=[(0.0, 8.0), (-8.0, 8.0)])
        a, b = float(res.x[0]), float(res.x[1])
        return (a, b), {"converged": bool(res.success), "alpha": a, "beta": b, "convex": True}

    if objective == "rps":
        fp = np.cumsum(p_train, axis=1)[:, :5]

        def f_rps(params: np.ndarray) -> float:
            a, b = float(params[0]), float(params[1])
            probs = softmax(a * log_q + b * GRID_01[None, :], axis=1)
            fq = np.cumsum(probs, axis=1)[:, :5]
            return float(np.mean((fq - fp) ** 2)) + (ridge / n) * ((a - 1.0) ** 2 + b * b)

        best_res = None
        for init in ([1.0, 0.0], [0.5, 0.0], [1.5, -0.5], [1.5, 0.5]):
            res = minimize(f_rps, np.array(init), method="L-BFGS-B", bounds=[(0.0, 8.0), (-8.0, 8.0)])
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        a, b = float(best_res.x[0]), float(best_res.x[1])
        return (a, b), {"converged": bool(best_res.success), "alpha": a, "beta": b, "convex": False}

    if objective == "pearson":
        def f_corr(params: np.ndarray) -> tuple[float, np.ndarray]:
            a, b = float(params[0]), float(params[1])
            probs = softmax(a * log_q + b * GRID_01[None, :], axis=1)
            m = probs @ GRID_05
            r, gm = _pearson_grad_vec(m, h_05)
            dz = probs * (GRID_05[None, :] - m[:, None]) * (-gm)[:, None]
            ga = float(np.sum(dz * log_q)) + (2.0 * ridge / n) * (a - 1.0)
            gb = float(np.sum(dz * GRID_01[None, :])) + (2.0 * ridge / n) * b
            return -r + (ridge / n) * ((a - 1.0) ** 2 + b * b), np.array([ga, gb])

        best_res = None
        for init in ([1.0, 0.0], [0.5, -0.5], [0.5, 0.5], [2.0, 0.0]):
            res = minimize(f_corr, np.array(init), method="L-BFGS-B", jac=True, bounds=[(0.05, 8.0), (-8.0, 8.0)])
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        a, b = float(best_res.x[0]), float(best_res.x[1])
        return (a, b), {"converged": bool(best_res.success), "alpha": a, "beta": b, "convex": False}

    raise ValueError(objective)


# --- Class M3: 48-Parameter Softmax Feature Head + 5-Restart Check (Exp 13 / 06 / 07 / 08 on Real Data) ---

def build_pool_features(q_pool: np.ndarray) -> np.ndarray:
    """Build D=8 leakage-free artifact covariates phi(x) from the base judge pool forecasts (N, M, 6)."""
    q_s = smooth_probs(q_pool, PROB_FLOOR)
    means_01 = q_s @ GRID_01  # (N, M)
    ent = -np.sum(q_s * np.log(q_s), axis=2) / math.log(6.0)  # normalized entropy in [0, 1]
    mean_bar = means_01.mean(axis=1)
    sd_across = means_01.std(axis=1)
    ent_bar = ent.mean(axis=1)
    q_bar = q_s.mean(axis=1)  # (N, 6)
    cdf_bar = np.cumsum(q_bar, axis=1)
    p_low = cdf_bar[:, 1]          # Pr[R <= 1]
    p_mid = q_bar[:, 2] + q_bar[:, 3]  # Pr[R in {2,3}]
    p_high = q_bar[:, 4] + q_bar[:, 5] # Pr[R >= 4]
    quad = (mean_bar - 0.5) ** 2
    return np.column_stack([np.ones(len(q_pool)), mean_bar, sd_across, ent_bar, p_low, p_mid, p_high, quad])


def fit_feature_head(
    Phi_tr: np.ndarray,
    p_tr: np.ndarray,
    objective: str,
    ridge_c: float = 1.0,
    n_starts: int = 1,
    maxiter: int = 400,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit W in R^{D x 6} for q_W(r|x) = softmax(Phi @ W)_r under all 7 objectives (Exp 13/06/07/08)."""
    n, d = Phi_tr.shape
    lam = ridge_c / max(1, n)
    marg = smooth_probs(p_tr.mean(axis=0), 1e-4)
    W_base = np.zeros((d, 6))
    W_base[0] = np.log(marg) - np.mean(np.log(marg))
    rng = np.random.default_rng(10007 * seed + 97)

    starts = [W_base + 0.01 * rng.standard_normal((d, 6))]
    for _ in range(max(0, n_starts - 1)):
        starts.append(W_base + 0.35 * rng.standard_normal((d, 6)))

    fp = np.cumsum(p_tr, axis=1)[:, :5]
    h_05 = p_tr @ GRID_05
    h_rank = rankdata(h_05)

    def fg(w_flat: np.ndarray) -> tuple[float, np.ndarray]:
        W = w_flat.reshape(d, 6)
        logits = Phi_tr @ W
        lse = logsumexp(logits, axis=1, keepdims=True)
        Q = np.exp(logits - lse)
        pen = lam * float(np.sum(W * W))
        g_pen = 2.0 * lam * W

        if objective == "ce":
            loss = -float(np.sum(p_tr * (logits - lse)) / n) + pen
            grad = (Phi_tr.T @ (Q - p_tr)) / n + g_pen
            return loss, grad.ravel()

        if objective == "rps":
            cdf = np.cumsum(Q, axis=1)[:, :5]
            diff = cdf - fp  # (n, 5)
            loss = float(np.mean(diff * diff)) + pen
            # d loss / d Q_{i,r} = (2/(5n)) sum_{k=r}^4 (cdf_{i,k} - fp_{i,k})
            rev_cum = np.cumsum(diff[:, ::-1], axis=1)[:, ::-1]
            dQ = np.zeros((n, 6))
            dQ[:, :5] = (2.0 / (5.0 * n)) * rev_cum
            dot = np.sum(Q * dQ, axis=1, keepdims=True)
            dZ = Q * (dQ - dot)
            return loss, (Phi_tr.T @ dZ + g_pen).ravel()

        m_05 = Q @ GRID_05
        if objective == "pearson":
            val, gm = _pearson_grad_vec(m_05, h_05)
        elif objective == "spearman":
            val, gm = _soft_spearman_grad_vec(m_05, h_rank)
        elif objective == "kendall":
            val, gm = _soft_kendall_grad_vec(m_05, h_05)
        elif objective == "pearson_spearman":
            vp, gp = _pearson_grad_vec(m_05, h_05)
            vs, gs = _soft_spearman_grad_vec(m_05, h_rank)
            val, gm = 0.5 * (vp + vs), 0.5 * (gp + gs)
        else:
            raise ValueError(objective)
        dZ = Q * (GRID_05[None, :] - m_05[:, None]) * (-gm)[:, None]
        return -val + pen, (Phi_tr.T @ dZ + g_pen).ravel()

    best_res = None
    any_converged = False
    for W0 in starts:
        res = minimize(
            fg,
            W0.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxfun": int(maxiter * 1.6)},
        )
        if res.success:
            any_converged = True
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    assert best_res is not None
    return best_res.x.reshape(d, 6), {
        "objective": objective,
        "converged": bool(best_res.success),
        "any_start_converged": any_converged,
        "nit": int(best_res.nit),
        "fun": float(best_res.fun),
        "n_starts": n_starts,
        "maxiter": maxiter,
    }


def fit_score_then_calibrate_head(
    Phi_tr: np.ndarray,
    p_tr: np.ndarray,
    Phi_te: np.ndarray,
    ridge_c: float = 1.0,
    n_starts: int = 1,
    maxiter: int = 400,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Two-stage 'Pearson + calibrated CE head' (Exp 13): Stage 1 fits W by Pearson -> scalar score s(x);
    Stage 2 standardizes s(x) on train and fits 6-way softmax(a_r + b_r z(x)) by convex CE on train ratings."""
    W_corr, rec1 = fit_feature_head(
        Phi_tr, p_tr, "pearson", ridge_c=ridge_c, n_starts=n_starts, maxiter=maxiter, seed=seed
    )
    s_tr = softmax(Phi_tr @ W_corr, axis=1) @ GRID_05
    s_te = softmax(Phi_te @ W_corr, axis=1) @ GRID_05
    s_mean, s_sd = float(s_tr.mean()), float(s_tr.std(ddof=0))
    if s_sd < 1e-10:
        z_tr = np.zeros_like(s_tr)
        z_te = np.zeros_like(s_te)
    else:
        z_tr = (s_tr - s_mean) / s_sd
        z_te = np.clip((s_te - s_mean) / s_sd, -5.0, 5.0)
    Z_tr = np.column_stack([np.ones(len(z_tr)), z_tr])
    Z_te = np.column_stack([np.ones(len(z_te)), z_te])
    W_cal, rec2 = fit_feature_head(Z_tr, p_tr, "ce", ridge_c=1.0, n_starts=1, maxiter=400, seed=seed)
    return softmax(Z_te @ W_cal, axis=1), {
        "stage1_converged": rec1["converged"],
        "stage2_converged": rec2["converged"],
    }


def cluster_bootstrap_ci(
    per_family_fn,
    families_all: list[str],
    draws: int = 10000,
    seed: int = 20261116,
) -> dict[str, float | None]:
    """10,000-draw cluster bootstrap over unique task families."""
    uniq = sorted(set(families_all))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        v = per_family_fn(sampled)
        if v is not None and math.isfinite(v):
            vals.append(float(v))
    if not vals:
        return {"lo": None, "hi": None, "p_two_sided": None, "valid_draws": 0}
    arr = np.asarray(vals, dtype=float)
    p_le = float(np.mean(arr <= 0.0))
    p_ge = float(np.mean(arr >= 0.0))
    p_two = min(1.0, 2.0 * min(p_le, p_ge))
    return {
        "lo": float(np.quantile(arr, 0.025)),
        "hi": float(np.quantile(arr, 0.975)),
        "p_two_sided": p_two,
        "valid_draws": int(len(arr)),
    }
