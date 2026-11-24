#!/usr/bin/env python3
"""Experiment 29: Matched 1D scalar_cal vs (K+2) logit_cal calibration, scoring, and cluster bootstrap."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import kendalltau, rankdata

EXP_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = Path("/tmp/exp29_public_raw/exp29_bundle.json")
EXP24_NPZ = EXP_DIR.parent / "24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026" / "results" / "predictions.npz"
RESULTS_DIR = EXP_DIR / "results"

LAMBDA_GRID = (0.05, 0.2, 1.0, 5.0)
EPS_CE_MAIN = 6e-6
EPS_CE_ALT = 1e-2
N_BOOT = 4000
TIE_TOL = 1e-12
METRIC_NAMES = ("rps", "ce", "ce_alt", "tvd", "brier")
DIAG_KEYS = (
    "pearson", "spearman", "kendall", "lin_ccc", "forecast_mean", "human_mean",
    "mean_bias", "mean_mae", "cumulative_ece", "modal_agreement", "map_accuracy",
    "rps_low_dis", "rps_high_dis",
)

STUDY_SPECS = (
    {"study": "chaos3_qwen", "suite": "chaos3", "npz": "chaos3_qwen_predictions.npz", "objs": ("ce", "rps", "mse", "pearson"), "depths": ((1, "budgets_d1"), (5, "budgets_d5"))},
    {"study": "alpha2_qwen", "suite": "alpha2", "npz": "alpha2_qwen_predictions.npz", "objs": ("ce", "rps", "mse", "pearson"), "depths": ((1, "budgets_d1"),)},
    {"study": "dices3_qwen", "suite": "dices3", "npz": "dices3_qwen_predictions.npz", "objs": ("ce", "rps", "mse", "pearson"), "depths": ((1, "budgets_d1"),)},
    {"study": "hs2_qwen", "suite": "hs2", "npz": "hs2_qwen_extra_predictions.npz", "extra_npz": EXP24_NPZ, "objs": ("ce", "rps", "mse", "pearson"), "depths": ((1, "budgets_d1"),)},
    {"study": "chaos3_llama", "suite": "chaos3", "npz": "chaos3_llama_predictions.npz", "objs": ("ce", "rps", "mse", "pearson"), "depths": ((1, "budgets_d1"),)},
    {"study": "dices3_llama", "suite": "dices3", "npz": "dices3_llama_predictions.npz", "objs": ("ce", "rps", "mse", "pearson"), "depths": ((1, "budgets_d1"),)},
)


def u_pos_for_k(n_cat: int) -> np.ndarray:
    return np.array([(2.0 * k / (n_cat - 1)) - 1.0 for k in range(n_cat)], dtype=np.float64)


def centered_log_probs(q: np.ndarray) -> np.ndarray:
    lq = np.log(np.clip(q, 1e-5, 1.0))
    return lq - lq.mean(axis=1, keepdims=True)


def rps_per_item(q: np.ndarray, h: np.ndarray) -> np.ndarray:
    n_cat = q.shape[1]
    cq = np.cumsum(q, axis=1)[:, : n_cat - 1]
    ch = np.cumsum(h, axis=1)[:, : n_cat - 1]
    return np.mean((cq - ch) ** 2 + ch * (1.0 - ch), axis=1)


def ce_per_item(q: np.ndarray, h: np.ndarray, eps: float) -> np.ndarray:
    return -np.sum(h * np.log(np.clip(q, eps, 1.0)), axis=1)


def tvd_per_item(q: np.ndarray, h: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.abs(q - h), axis=1)


def brier_per_item(q: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.mean((q - h) ** 2, axis=1)


def _cal_loss_and_grad(theta: np.ndarray, ell: np.ndarray | None, m_std: np.ndarray, p_cal: np.ndarray, a0: np.ndarray, u_pos: np.ndarray, lam: float, use_logits: bool) -> tuple[float, np.ndarray]:
    n_items, n_cat = p_cal.shape
    a_full = np.append(theta[: n_cat - 1], -np.sum(theta[: n_cat - 1]))
    b1, b2 = float(theta[-2]), float(theta[-1])
    tau = float(theta[n_cat - 1]) if use_logits else 0.0
    logits = a_full[None, :] + b1 * (m_std[:, None] * u_pos[None, :]) + b2 * (m_std[:, None] * (u_pos[None, :] ** 2))
    if use_logits and ell is not None:
        logits = logits + tau * ell
    probs = softmax(logits, axis=1)
    da = a_full - a0
    ce = float(-np.sum(p_cal * np.log(np.clip(probs, 1e-12, 1.0))) / n_items)
    reg = 0.5 * (lam / n_items) * float(np.sum(da ** 2) + tau ** 2 + b1 ** 2 + b2 ** 2)
    diff = (probs - p_cal) / n_items
    ga_full = diff.sum(axis=0) + (lam / n_items) * da
    g_a = ga_full[: n_cat - 1] - ga_full[n_cat - 1]
    g_b1 = float(np.sum(diff * (m_std[:, None] * u_pos[None, :]))) + (lam / n_items) * b1
    g_b2 = float(np.sum(diff * (m_std[:, None] * (u_pos[None, :] ** 2)))) + (lam / n_items) * b2
    if use_logits and ell is not None:
        g_tau = float(np.sum(diff * ell)) + (lam / n_items) * tau
        grad = np.concatenate([g_a, [g_tau, g_b1, g_b2]])
    else:
        grad = np.concatenate([g_a, [g_b1, g_b2]])
    return ce + reg, grad.astype(np.float64)


def fit_calibrator_single(ell_cal: np.ndarray | None, m_std_cal: np.ndarray, p_cal: np.ndarray, lam: float, use_logits: bool) -> tuple[np.ndarray, float]:
    n_cat = p_cal.shape[1]
    prior = (p_cal.sum(axis=0) + 0.5) / (len(p_cal) + 0.5 * n_cat)
    a0 = np.log(prior) - np.mean(np.log(prior))
    u_pos = u_pos_for_k(n_cat)
    n_extra = 3 if use_logits else 2
    theta0 = np.concatenate([a0[: n_cat - 1], np.zeros(n_extra, dtype=np.float64)])
    if use_logits:
        theta0[n_cat - 1] = 1.0
    res = minimize(
        lambda th: _cal_loss_and_grad(th, ell_cal, m_std_cal, p_cal, a0, u_pos, lam, use_logits),
        theta0, jac=True, method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 600},
    )
    gnorm = float(np.linalg.norm(res.jac, ord=np.inf))
    assert gnorm < 1e-6, f"Calibrator L-BFGS-B failed: gnorm={gnorm}"
    return res.x, gnorm


def apply_calibrator(theta: np.ndarray, q: np.ndarray, mu_cal: float, sd_cal: float, use_logits: bool) -> np.ndarray:
    n_cat = q.shape[1]
    grid = np.arange(n_cat, dtype=np.float64)
    u_pos = u_pos_for_k(n_cat)
    a_full = np.append(theta[: n_cat - 1], -np.sum(theta[: n_cat - 1]))
    b1, b2 = float(theta[-2]), float(theta[-1])
    m_std = ((q @ grid) - mu_cal) / (sd_cal + 1e-6)
    logits = a_full[None, :] + b1 * (m_std[:, None] * u_pos[None, :]) + b2 * (m_std[:, None] * (u_pos[None, :] ** 2))
    if use_logits:
        logits = logits + float(theta[n_cat - 1]) * centered_log_probs(q)
    return softmax(logits, axis=1)


def fit_and_apply_cv(q_cal: np.ndarray, p_cal: np.ndarray, q_eval: np.ndarray, use_logits: bool) -> tuple[np.ndarray, dict]:
    n_items, n_cat = q_cal.shape
    grid = np.arange(n_cat, dtype=np.float64)
    ell_all = centered_log_probs(q_cal) if use_logits else None
    m_all = q_cal @ grid
    fold_ids = np.arange(n_items) % 4
    cv_rps = {}
    for lam in LAMBDA_GRID:
        f_scores = []
        for f in range(4):
            tr, va = fold_ids != f, fold_ids == f
            m_tr = m_all[tr]
            mu_f, sd_f = float(m_tr.mean()), float(m_tr.std())
            ell_tr = ell_all[tr] if ell_all is not None else None
            th_f, _ = fit_calibrator_single(ell_tr, (m_tr - mu_f) / (sd_f + 1e-6), p_cal[tr], lam, use_logits)
            f_scores.append(float(rps_per_item(apply_calibrator(th_f, q_cal[va], mu_f, sd_f, use_logits), p_cal[va]).mean()))
        cv_rps[lam] = float(np.mean(f_scores))
    best_cv = min(cv_rps.values())
    best_lam = max(lam for lam, val in cv_rps.items() if val <= best_cv + 1e-12)
    mu_full, sd_full = float(m_all.mean()), float(m_all.std())
    th_full, gnorm = fit_calibrator_single(ell_all, (m_all - mu_full) / (sd_full + 1e-6), p_cal, best_lam, use_logits)
    return apply_calibrator(th_full, q_eval, mu_full, sd_full, use_logits), {"selected_lambda": best_lam, "grad_inf_norm": gnorm}


def verify_calibrator_gradients() -> dict:
    errs = {}
    rng = np.random.default_rng(20261129)
    for n_cat in (2, 3, 5):
        q = rng.dirichlet(np.ones(n_cat), size=10)
        p_cal = rng.dirichlet(np.ones(n_cat), size=10)
        a0 = np.zeros(n_cat, dtype=np.float64)
        u_pos = u_pos_for_k(n_cat)
        m_std = ((q @ np.arange(n_cat, dtype=np.float64)) - 0.5) / 0.8
        ell = centered_log_probs(q)
        for use_logits in (False, True):
            th = rng.normal(0.0, 0.3, size=(n_cat + 2 if use_logits else n_cat + 1))
            _, g_ana = _cal_loss_and_grad(th, ell if use_logits else None, m_std, p_cal, a0, u_pos, 0.2, use_logits)
            g_fd = np.zeros_like(th)
            eps = 1e-6
            for j in range(len(th)):
                tp, tm = th.copy(), th.copy()
                tp[j] += eps
                tm[j] -= eps
                lp, _ = _cal_loss_and_grad(tp, ell if use_logits else None, m_std, p_cal, a0, u_pos, 0.2, use_logits)
                lm, _ = _cal_loss_and_grad(tm, ell if use_logits else None, m_std, p_cal, a0, u_pos, 0.2, use_logits)
                g_fd[j] = (lp - lm) / (2.0 * eps)
            rel = float(np.linalg.norm(g_ana - g_fd) / (np.linalg.norm(g_ana) + 1e-12))
            assert rel < 1e-7, f"Cal FD fail K={n_cat} logits={use_logits}: {rel}"
            errs[f"K{n_cat}_{'logit' if use_logits else 'scalar'}"] = rel
    return errs


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) <= TIE_TOL or np.ptp(y) <= TIE_TOL:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    return float((xc @ yc) / (np.linalg.norm(xc) * np.linalg.norm(yc)))


def tie_levels(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    levels = np.empty_like(order)
    levels[order] = np.r_[0, np.cumsum(np.diff(values[order]) > TIE_TOL)]
    return levels


def mode_weights(rows: np.ndarray) -> np.ndarray:
    ties = rows >= rows.max(axis=1, keepdims=True) - TIE_TOL
    return ties / ties.sum(axis=1, keepdims=True)


def cumulative_ece(q: np.ndarray, h: np.ndarray) -> float:
    n_cat = q.shape[1]
    forecast = np.clip(q.cumsum(axis=1)[:, : n_cat - 1], 0.0, 1.0)
    observed = h.cumsum(axis=1)[:, : n_cat - 1]
    bins = np.minimum((10.0 * forecast).astype(int), 9)
    tot = 0.0
    for k in range(n_cat - 1):
        tot += float(np.abs(np.bincount(bins[:, k], weights=observed[:, k] - forecast[:, k], minlength=10)).sum())
    return float(tot / forecast.size)


def compute_diagnostics(q: np.ndarray, h_test: np.ndarray) -> dict[str, float]:
    grid = np.arange(q.shape[1], dtype=np.float64)
    m, y = q @ grid, h_test @ grid
    rm, ry = tie_levels(m), tie_levels(y)
    mc, yc = m - m.mean(), y - y.mean()
    denom = float(np.mean(mc ** 2) + np.mean(yc ** 2) + (m.mean() - y.mean()) ** 2)
    lin = float(2.0 * np.mean(mc * yc) / denom) if denom > TIE_TOL ** 2 else float("nan")
    pm, hm = mode_weights(q), mode_weights(h_test)
    return {
        "pearson": safe_corr(m, y), "spearman": safe_corr(rankdata(rm), rankdata(ry)),
        "kendall": float(kendalltau(rm, ry, variant="b").statistic) if rm.max() and ry.max() else float("nan"),
        "lin_ccc": lin, "forecast_mean": float(m.mean()), "human_mean": float(y.mean()),
        "mean_bias": float((m - y).mean()), "mean_mae": float(np.abs(m - y).mean()),
        "cumulative_ece": cumulative_ece(q, h_test),
        "modal_agreement": float((pm * hm).sum(axis=1).mean()), "map_accuracy": float((pm * h_test).sum(axis=1).mean()),
    }


def load_suite_gt(suite: dict) -> dict:
    test_ids = list(suite["splits"]["test_ids"])
    n_cat = int(suite["K"])
    counts_test = np.stack([np.asarray(suite["items"][i]["counts"], dtype=np.float64) for i in test_ids])
    h_test = counts_test / counts_test.sum(axis=1, keepdims=True)
    gkeys = [suite["items"][i]["group_key"] for i in test_ids]
    uniq_g, g_inv, g_sizes = np.unique(gkeys, return_inverse=True, return_counts=True)
    mask_high = np.array([bool(suite["items"][i]["is_high_dis"]) for i in test_ids], dtype=bool)
    return {
        "K": n_cat, "test_ids": test_ids, "n_test": len(test_ids), "n_votes": int(counts_test.sum()),
        "h_test": h_test, "g_inv": g_inv, "n_groups": len(uniq_g), "g_sizes": g_sizes.astype(np.float64),
        "mask_low_dis": ~mask_high, "mask_high_dis": mask_high,
    }


def _emp_targets(suite_name: str, draw_spec: dict, item_ids: list[int], depth: int, n_cat: int) -> np.ndarray:
    out = np.zeros((len(item_ids), n_cat), dtype=np.float64)
    if suite_name == "hs2":
        for k, iid in enumerate(item_ids):
            out[k, draw_spec["sampled_vote_pairs"][str(iid)][0]] = 1.0
        return out
    for k, iid in enumerate(item_ids):
        for v in draw_spec["sampled_votes_5"][str(iid)][:depth]:
            out[k, int(v)] += 1.0 / depth
    return out


def _get_split_arrays(npz, extra_npz, key_p: str, obj: str) -> tuple[np.ndarray, np.ndarray]:
    src = npz if f"{key_p}_cal" in npz.files else extra_npz
    assert src is not None and f"{key_p}_cal" in src.files, f"Missing {key_p} ({obj})"
    return src[f"{key_p}_cal"].astype(np.float64), src[f"{key_p}_test"].astype(np.float64)


def predict_draw_methods(
    spec: dict, suite: dict, draw_spec: dict, depth: int, b_total: int, draw_id: int,
    npz, extra_npz, base_all: np.ndarray, q_base_test: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    sname, n_cat = spec["suite"], int(suite["K"])
    n_tr, n_va, n_ca = (b_total // 2) // depth, (b_total // 4) // depth, (b_total // 4) // depth
    tr_ids, va_ids, ca_ids = draw_spec["train_order"][:n_tr], draw_spec["val_order"][:n_va], draw_spec["cal_order"][:n_ca]
    p_tr = _emp_targets(sname, draw_spec, tr_ids, depth, n_cat)
    p_va = _emp_targets(sname, draw_spec, va_ids, depth, n_cat)
    p_ca = _emp_targets(sname, draw_spec, ca_ids, depth, n_cat)
    tr_hist = (p_tr.sum(axis=0) + 0.5) / (p_tr.sum() + 0.5 * n_cat)
    dev_tot = p_tr.sum(axis=0) + p_va.sum(axis=0) + p_ca.sum(axis=0)
    dev_hist = (dev_tot + 0.5) / (dev_tot.sum() + 0.5 * n_cat)
    n_test = len(q_base_test)
    q_preds = {
        "uniform": np.full((n_test, n_cat), 1.0 / n_cat, dtype=np.float64),
        "training_histogram": np.broadcast_to(tr_hist, (n_test, n_cat)).copy(),
        "all_dev_histogram": np.broadcast_to(dev_hist, (n_test, n_cat)).copy(),
        "base_raw": q_base_test,
    }
    cal_logs = []
    for use_log, reg_name in ((False, "scalar_cal"), (True, "logit_cal")):
        q_preds[f"base_{reg_name}"], meta = fit_and_apply_cv(base_all[ca_ids], p_ca, q_base_test, use_log)
        cal_logs.append({"study": spec["study"], "depth": depth, "b_total": b_total, "draw_id": draw_id, "method": f"base_{reg_name}", **meta})
    for obj in spec["objs"]:
        q_cal, q_test = _get_split_arrays(npz, extra_npz, f"draw{draw_id}_d{depth}_B{b_total}_{obj}", obj)
        q_preds[f"{obj}_raw"] = q_test
        for use_log, reg_name in ((False, "scalar_cal"), (True, "logit_cal")):
            q_preds[f"{obj}_{reg_name}"], meta = fit_and_apply_cv(q_cal, p_ca, q_test, use_log)
            cal_logs.append({"study": spec["study"], "depth": depth, "b_total": b_total, "draw_id": draw_id, "method": f"{obj}_{reg_name}", **meta})
    return q_preds, cal_logs


def score_draw(q_preds: dict[str, np.ndarray], gt: dict, losses_slice: np.ndarray, study: str, depth: int, b_total: int, draw_id: int, m_idx: dict[str, int]) -> list[dict]:
    h_test = gt["h_test"]
    rows = []
    for m_name, q in q_preds.items():
        mi = m_idx[m_name]
        losses_slice[mi, :, 0] = rps_per_item(q, h_test)
        losses_slice[mi, :, 1] = ce_per_item(q, h_test, EPS_CE_MAIN)
        losses_slice[mi, :, 2] = ce_per_item(q, h_test, EPS_CE_ALT)
        losses_slice[mi, :, 3] = tvd_per_item(q, h_test)
        losses_slice[mi, :, 4] = brier_per_item(q, h_test)
        dg = compute_diagnostics(q, h_test)
        dg.update({
            "study": study, "depth": depth, "b_total": b_total, "draw_id": draw_id, "method": m_name,
            "rps_low_dis": float(losses_slice[mi, gt["mask_low_dis"], 0].mean()),
            "rps_high_dis": float(losses_slice[mi, gt["mask_high_dis"], 0].mean()),
        })
        rows.append(dg)
    return rows


def run_ratio_cluster_bootstrap(group_sums_kdg: np.ndarray, g_counts: np.ndarray, g_sizes: np.ndarray, d_weights: np.ndarray) -> dict:
    """Cluster bootstrap of item-equal mean across heterogeneous group sizes (g_counts shape [B, G])."""
    denoms = g_counts @ g_sizes
    cond_sums = group_sums_kdg.mean(axis=1)  # [K, G]
    point = cond_sums.sum(axis=1) / float(g_sizes.sum())
    cond_boots = (cond_sums @ g_counts.T).T / denoms[:, None]
    pt_lo, pt_hi = np.quantile(cond_boots, [0.025, 0.975], axis=0)
    se_c = cond_boots.std(axis=0, ddof=1)
    vc = se_c > 1e-12
    tmax_c = np.abs((cond_boots[:, vc] - point[vc]) / se_c[vc]).max(axis=1) if vc.any() else np.zeros(len(denoms))
    crit_c = float(np.quantile(tmax_c, 0.95, method="higher"))
    cross_boots = np.einsum("bd,kdg,bg->bk", d_weights, group_sums_kdg, g_counts) / denoms[:, None]
    cr_lo, cr_hi = np.quantile(cross_boots, [0.025, 0.975], axis=0)
    se_x = cross_boots.std(axis=0, ddof=1)
    vx = se_x > 1e-12
    tmax_x = np.abs((cross_boots[:, vx] - point[vx]) / se_x[vx]).max(axis=1) if vx.any() else np.zeros(len(denoms))
    crit_x = float(np.quantile(tmax_x, 0.95, method="higher"))
    return {
        "point": point, "pt_lo": pt_lo, "pt_hi": pt_hi,
        "sim_lo": point - crit_c * se_c, "sim_hi": point + crit_c * se_c,
        "cross_pt_lo": cr_lo, "cross_pt_hi": cr_hi,
        "cross_sim_lo": point - crit_x * se_x, "cross_sim_hi": point + crit_x * se_x,
    }


def _contrast_specs() -> list[tuple[str, str, int, str]]:
    specs = []
    for target in ("ce_raw", "rps_raw"):
        for comp in ("pearson_raw", "mse_raw", "base_raw"):
            specs.extend([(target, comp, 0, "rps"), (target, comp, 1, "ce"), (target, comp, 3, "tvd")])
    specs.extend([("mse_raw", "pearson_raw", 0, "rps"), ("ce_raw", "rps_raw", 0, "rps")])
    for reg in ("scalar_cal", "logit_cal"):
        for target in (f"ce_{reg}", f"rps_{reg}"):
            for comp in (f"pearson_{reg}", f"mse_{reg}", f"base_{reg}"):
                specs.extend([(target, comp, 0, "rps"), (target, comp, 1, "ce"), (target, comp, 3, "tvd")])
    for obj in ("pearson", "mse", "base"):
        specs.append((f"{obj}_logit_cal", f"{obj}_scalar_cal", 0, "rps"))
    return specs


def _make_contrast_rows(study: str, depth: int, budgets: list[int], target: str, comp: str, d_i: int, met: str, group_losses: np.ndarray, g_counts: np.ndarray, gt: dict, d_weights: np.ndarray, m_idx: dict[str, int]) -> list[dict]:
    diff_kdg = group_losses[:, :, m_idx[comp], :, d_i] - group_losses[:, :, m_idx[target], :, d_i]
    boot = run_ratio_cluster_bootstrap(diff_kdg, g_counts, gt["g_sizes"], d_weights)
    draw_means = diff_kdg.sum(axis=2) / float(gt["n_test"])
    rows = []
    for ki, b_tot in enumerate(budgets):
        dd = draw_means[ki]
        rows.append({
            "study": study, "depth": depth, "b_total": b_tot, "n_train": (b_tot // 2) // depth,
            "n_cal": (b_tot // 4) // depth, "metric": met, "target": target, "comparator": comp,
            "diff_comparator_minus_target": float(boot["point"][ki]),
            "pt_lo_95": float(boot["pt_lo"][ki]), "pt_hi_95": float(boot["pt_hi"][ki]),
            "sim_lo_95": float(boot["sim_lo"][ki]), "sim_hi_95": float(boot["sim_hi"][ki]),
            "cross_pt_lo_95": float(boot["cross_pt_lo"][ki]), "cross_pt_hi_95": float(boot["cross_pt_hi"][ki]),
            "cross_sim_lo_95": float(boot["cross_sim_lo"][ki]), "cross_sim_hi_95": float(boot["cross_sim_hi"][ki]),
            "draw_sd": float(dd.std(ddof=1)), "draws_target_better": int((dd > 1e-12).sum()),
        })
    return rows


def _aggregate_scores(study: str, depth: int, budgets: list[int], methods: list[str], losses_all: np.ndarray, diag_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for r in diag_rows:
        grouped.setdefault((r["b_total"], r["method"]), []).append(r)
    out = []
    for bi, b_tot in enumerate(budgets):
        for mi, m_name in enumerate(methods):
            row = {"study": study, "depth": depth, "b_total": b_tot, "n_train": (b_tot // 2) // depth, "n_cal": (b_tot // 4) // depth, "method": m_name}
            for di, met in enumerate(METRIC_NAMES):
                dm = losses_all[bi, :, mi, :, di].mean(axis=1)
                row[met], row[f"{met}_draw_sd"] = float(dm.mean()), float(dm.std(ddof=1))
            sub = grouped[(b_tot, m_name)]
            for dk in DIAG_KEYS:
                vals = [x[dk] for x in sub if np.isfinite(x[dk])]
                row[dk] = float(np.mean(vals)) if vals else float("nan")
            out.append(row)
    return out


def _build_method_list(objs: tuple[str, ...]) -> list[str]:
    base = ["uniform", "training_histogram", "all_dev_histogram", "base_raw", "base_scalar_cal", "base_logit_cal"]
    for obj in objs:
        base.extend([f"{obj}_raw", f"{obj}_scalar_cal", f"{obj}_logit_cal"])
    return base


def _evaluate_depth_block(spec: dict, suite: dict, gt: dict, depth: int, b_key: str, npz, extra_npz, g_counts: np.ndarray, d_weights: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    budgets = [int(b) for b in suite[b_key]]
    n_draws = len(suite["draws"])
    methods = _build_method_list(spec["objs"])
    m_idx = {m: i for i, m in enumerate(methods)}
    base_all = npz["base_probs"].astype(np.float64)
    q_base_test = base_all[gt["test_ids"]]
    losses_all = np.zeros((len(budgets), n_draws, len(methods), gt["n_test"], 5), dtype=np.float64)
    diag_rows, cal_logs = [], []
    for bi, b_tot in enumerate(budgets):
        for draw_id in range(n_draws):
            q_preds, c_logs = predict_draw_methods(spec, suite, suite["draws"][draw_id], depth, b_tot, draw_id, npz, extra_npz, base_all, q_base_test)
            cal_logs.extend(c_logs)
            diag_rows.extend(score_draw(q_preds, gt, losses_all[bi, draw_id], spec["study"], depth, b_tot, draw_id, m_idx))
    g_losses = np.zeros((len(budgets), n_draws, len(methods), gt["n_groups"], 5), dtype=np.float64)
    for gid in range(gt["n_groups"]):
        g_losses[:, :, :, gid, :] = losses_all[:, :, :, gt["g_inv"] == gid, :].sum(axis=3)
    s_rows = _aggregate_scores(spec["study"], depth, budgets, methods, losses_all, diag_rows)
    c_rows = []
    for target, comp, di, met in _contrast_specs():
        c_rows.extend(_make_contrast_rows(spec["study"], depth, budgets, target, comp, di, met, g_losses, g_counts, gt, d_weights, m_idx))
    return s_rows, c_rows, cal_logs


def _hs2_parity_check(bundle: dict) -> dict:
    npz_extra = np.load(RESULTS_DIR / "hs2_qwen_extra_predictions.npz")
    npz_exp24 = np.load(EXP24_NPZ)
    test_ids = bundle["suites"]["hs2"]["splits"]["test_ids"]
    h_test = np.stack([np.bincount(bundle["suites"]["hs2"]["items"][i]["votes"], minlength=5) for i in test_ids], dtype=np.float64)
    h_test = h_test / h_test.sum(axis=1, keepdims=True)
    q_new = npz_extra["draw0_d1_B64_ce_test"].astype(np.float64)
    q_old = npz_exp24["draw0_d1_B64_ce_test"].astype(np.float64)
    base_diff = float(np.max(np.abs(npz_extra["base_probs"].astype(np.float64) - npz_exp24["base_probs"].astype(np.float64))))
    return {
        "max_abs_base_prob_diff": base_diff,
        "draw0_B64_ce_max_abs_prob_diff": float(np.max(np.abs(q_new - q_old))),
        "draw0_B64_ce_rps_new": float(rps_per_item(q_new, h_test).mean()),
        "draw0_B64_ce_rps_exp24": float(rps_per_item(q_old, h_test).mean()),
    }


def find_sustained(budgets: list[int], depth: int, vals: list[float]) -> dict | None:
    idx = None
    for i in range(len(budgets) - 1, -1, -1):
        if vals[i] > 0.0:
            idx = i
        else:
            break
    if idx is None:
        return None
    b_tot = int(budgets[idx])
    return {"b_total": b_tot, "n_train": (b_tot // 2) // depth}


def compute_sustained_thresholds(contrast_rows: list[dict]) -> dict:
    key_pairs = (
        ("ce_raw", "pearson_raw"), ("ce_raw", "mse_raw"),
        ("rps_raw", "pearson_raw"), ("rps_raw", "mse_raw"),
        ("ce_scalar_cal", "pearson_scalar_cal"), ("ce_scalar_cal", "mse_scalar_cal"),
        ("rps_scalar_cal", "pearson_scalar_cal"), ("ce_logit_cal", "pearson_logit_cal"),
        ("pearson_logit_cal", "pearson_scalar_cal"),
    )
    grouped: dict[tuple[str, int, str, str], list[dict]] = {}
    for r in contrast_rows:
        if r["metric"] == "rps" and (r["target"], r["comparator"]) in key_pairs:
            grouped.setdefault((r["study"], r["depth"], r["target"], r["comparator"]), []).append(r)
    out = {}
    for (study, depth, target, comp), sub in sorted(grouped.items()):
        sub_s = sorted(sub, key=lambda x: x["b_total"])
        budgets = [x["b_total"] for x in sub_s]
        out[f"{study}|d{depth}|{target}_vs_{comp}"] = {
            "cond_sim_95": find_sustained(budgets, depth, [x["sim_lo_95"] for x in sub_s]),
            "cross_sim_95": find_sustained(budgets, depth, [x["cross_sim_lo_95"] for x in sub_s]),
        }
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--allow-missing", action="store_true")
    return ap.parse_args()


def _init_suite_boot(bundle: dict, sname: str, rng: np.random.Generator, gt_cache: dict, boot_cache: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    suite = bundle["suites"][sname]
    if sname not in gt_cache:
        gt = load_suite_gt(suite)
        gt_cache[sname] = gt
        n_d, n_g = len(suite["draws"]), gt["n_groups"]
        g_c = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g), size=N_BOOT).astype(np.float64)
        d_w = rng.multinomial(n_d, np.full(n_d, 1.0 / n_d), size=N_BOOT) / n_d
        boot_cache[sname] = (g_c, d_w)
    gt = gt_cache[sname]
    return gt, boot_cache[sname][0], boot_cache[sname][1]


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_fd = verify_calibrator_gradients()
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261129)
    all_scores, all_contrasts, all_cals, suite_meta, gt_cache, boot_cache = [], [], [], {}, {}, {}
    for spec in STUDY_SPECS:
        sname = spec["suite"]
        gt, g_counts, d_weights = _init_suite_boot(bundle, sname, rng, gt_cache, boot_cache)
        npz_path = RESULTS_DIR / spec["npz"]
        if not npz_path.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"Missing predictions: {npz_path}")
        suite_meta[sname] = {"K": gt["K"], "n_test": gt["n_test"], "n_groups": gt["n_groups"], "n_votes": gt["n_votes"], "n_low_dis": int(gt["mask_low_dis"].sum()), "n_high_dis": int(gt["mask_high_dis"].sum())}
        npz = np.load(npz_path)
        extra = np.load(spec["extra_npz"]) if "extra_npz" in spec else None
        for depth, b_key in spec["depths"]:
            sr, cr, cl = _evaluate_depth_block(spec, bundle["suites"][sname], gt, depth, b_key, npz, extra, g_counts, d_weights)
            all_scores.extend(sr)
            all_contrasts.extend(cr)
            all_cals.extend(cl)

    write_csv(out_dir / "scores.csv", all_scores)
    write_csv(out_dir / "contrasts.csv", all_contrasts)
    write_csv(out_dir / "calibration_logs.csv", all_cals)
    parity = _hs2_parity_check(bundle) if (RESULTS_DIR / "hs2_qwen_extra_predictions.npz").exists() else {}
    summary = {"calibrator_fd_checks": cal_fd, "hs2_exp24_parity": parity, "sustained_win_thresholds": compute_sustained_thresholds(all_contrasts), "bootstrap_draws": N_BOOT, "bootstrap_seed": 20261129, "suites": suite_meta}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Exp 29 scoring complete ->", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
