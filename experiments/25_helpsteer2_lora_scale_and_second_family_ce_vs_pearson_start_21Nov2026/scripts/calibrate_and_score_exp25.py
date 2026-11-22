#!/usr/bin/env python3
"""Matched post-hoc calibration, held-out metric panel, and prompt-clustered uncertainty for Experiment 25."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import kendalltau, rankdata

EXP_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = Path("/tmp/exp25_helpsteer2_raw/exp25_bundle.json")
RESULTS_DIR = EXP_DIR / "results"

GRID = np.arange(5, dtype=np.float64)
U_POS = (GRID - 2.0) / 2.0  # [-1.0, -0.5, 0.0, +0.5, +1.0]
EPS_CLIP = 1e-5
EPS_CE_PRIMARY = 6e-6
EPS_CE_ALT = 1e-2
LAMBDA_GRID = (0.05, 0.2, 1.0, 5.0)
N_BOOT = 4000
TIE_TOL = 1e-12

METHODS = [
    "ce_calibrated",
    "pearson_calibrated",
    "ce_raw",
    "pearson_raw",
    "base_raw",
    "base_calibrated_matched",
    "base_calibrated_all_dev",
    "uniform",
    "training_histogram",
    "all_dev_histogram",
    "ce_exp23_4bin",
    "pearson_exp23_4bin",
    "base_exp23_4bin",
]
METRIC_NAMES = ("rps", "ce_nats", "ce_eps1e2", "tvd", "brier")
COMPARATORS = (
    "pearson_calibrated",
    "uniform",
    "training_histogram",
    "all_dev_histogram",
    "base_calibrated_matched",
    "base_calibrated_all_dev",
    "ce_raw",
)
DIAG_KEYS = (
    "rps_low_dis", "rps_high_dis", "pearson", "spearman", "kendall", "lin_ccc",
    "forecast_mean", "human_mean", "mean_bias", "mean_mae",
    "cumulative_ece", "modal_agreement", "map_accuracy", "within_prompt_pair_acc",
)
CONTRAST_SPECS = [
    ("ce_calibrated", comp, d_i, met)
    for comp in COMPARATORS
    for d_i, met in enumerate(METRIC_NAMES[:4])
] + [
    ("ce_raw", "pearson_raw", 0, "rps"),
    ("ce_raw", "pearson_raw", 1, "ce_nats"),
    ("ce_raw", "pearson_raw", 3, "tvd"),
    ("ce_exp23_4bin", "pearson_exp23_4bin", 0, "rps"),
    ("ce_exp23_4bin", "pearson_exp23_4bin", 1, "ce_nats"),
    ("ce_exp23_4bin", "pearson_exp23_4bin", 3, "tvd"),
]


def centered_log_probs(q: np.ndarray) -> np.ndarray:
    q_tilde = (1.0 - EPS_CLIP) * q + EPS_CLIP / 5.0
    lp = np.log(q_tilde)
    return lp - lp.mean(axis=1, keepdims=True)


def rps_per_item(q: np.ndarray, h: np.ndarray) -> np.ndarray:
    f = np.cumsum(q, axis=1)[:, :4]
    g = np.cumsum(h, axis=1)[:, :4]
    return np.mean(f * f + g * (1.0 - 2.0 * f), axis=1)


def ce_per_item(q: np.ndarray, h: np.ndarray, eps: float) -> np.ndarray:
    q_eps = (1.0 - eps) * q + eps / 5.0
    return -np.sum(h * np.log(q_eps), axis=1)


def tvd_per_item(q: np.ndarray, h: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.abs(q - h), axis=1)


def brier_per_item(q: np.ndarray, h: np.ndarray) -> np.ndarray:
    return 1.0 + np.sum(q * q - 2.0 * h * q, axis=1)


def fit_7param_single(
    ell_cal: np.ndarray,
    m_std_cal: np.ndarray,
    p_cal: np.ndarray,
    n_cal_votes: float,
    lam: float,
) -> tuple[np.ndarray, float]:
    """Fit strictly convex 7-parameter calibrator on (ell_cal, m_std_cal) -> p_cal."""
    n_items = len(p_cal)
    vote_per_item = n_cal_votes / n_items
    counts = (p_cal * vote_per_item).sum(axis=0)
    pi_cal = (counts + 0.5) / (n_cal_votes + 2.5)
    log_pi = np.log(pi_cal)
    a0_full = log_pi - log_pi.mean()

    theta0 = np.array([a0_full[0], a0_full[1], a0_full[2], a0_full[3], 1.0, 0.0, 0.0], dtype=np.float64)

    def loss_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a_full = np.array([theta[0], theta[1], theta[2], theta[3], -np.sum(theta[:4])], dtype=np.float64)
        tau, b1, b2 = theta[4], theta[5], theta[6]
        logits = (
            a_full[None, :]
            + tau * ell_cal
            + b1 * m_std_cal[:, None] * U_POS[None, :]
            + b2 * m_std_cal[:, None] * (U_POS[None, :] ** 2)
        )
        probs = softmax(logits, axis=1)
        ce = float(-np.sum(p_cal * np.log(np.clip(probs, 1e-15, 1.0))) / n_items)
        da = a_full - a0_full
        reg = (lam / (2.0 * n_cal_votes)) * float(np.sum(da ** 2) + tau ** 2 + b1 ** 2 + b2 ** 2)

        diff = (probs - p_cal) / n_items
        ga_full = diff.sum(axis=0) + (lam / n_cal_votes) * da
        g_a = ga_full[:4] - ga_full[4]
        g_tau = float(np.sum(diff * ell_cal)) + (lam / n_cal_votes) * tau
        g_b1 = float(np.sum(diff * (m_std_cal[:, None] * U_POS[None, :]))) + (lam / n_cal_votes) * b1
        g_b2 = float(np.sum(diff * (m_std_cal[:, None] * (U_POS[None, :] ** 2)))) + (lam / n_cal_votes) * b2
        return ce + reg, np.array([g_a[0], g_a[1], g_a[2], g_a[3], g_tau, g_b1, g_b2], dtype=np.float64)

    res = minimize(
        loss_and_grad,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 500},
    )
    grad_norm = float(np.linalg.norm(res.jac, ord=np.inf))
    assert grad_norm < 1e-6, f"Calibrator L-BFGS-B failed to converge cleanly: gnorm={grad_norm}"
    return res.x, grad_norm


def apply_7param(theta: np.ndarray, q: np.ndarray, mu_cal: float, sd_cal: float) -> np.ndarray:
    a_full = np.array([theta[0], theta[1], theta[2], theta[3], -np.sum(theta[:4])], dtype=np.float64)
    tau, b1, b2 = theta[4], theta[5], theta[6]
    ell = centered_log_probs(q)
    m = q @ GRID
    m_std = (m - mu_cal) / (sd_cal + 1e-6)
    logits = (
        a_full[None, :]
        + tau * ell
        + b1 * m_std[:, None] * U_POS[None, :]
        + b2 * m_std[:, None] * (U_POS[None, :] ** 2)
    )
    return softmax(logits, axis=1)


def fit_and_apply_7param_cv(
    q_cal: np.ndarray,
    p_cal: np.ndarray,
    q_eval: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Select lambda by 4-fold CV RPS on S_cal, then refit on all S_cal and apply to q_eval."""
    n_items = len(q_cal)
    assert n_items >= 4
    ell_all = centered_log_probs(q_cal)
    m_all = q_cal @ GRID

    fold_ids = np.arange(n_items) % 4
    cv_scores = {}
    for lam in LAMBDA_GRID:
        fold_rps = []
        for f in range(4):
            tr_mask = fold_ids != f
            va_mask = fold_ids == f
            m_tr = m_all[tr_mask]
            mu_f = float(m_tr.mean())
            sd_f = float(m_tr.std())
            m_std_tr = (m_tr - mu_f) / (sd_f + 1e-6)
            theta_f, _ = fit_7param_single(ell_all[tr_mask], m_std_tr, p_cal[tr_mask], float(tr_mask.sum()), lam)
            q_pred = apply_7param(theta_f, q_cal[va_mask], mu_f, sd_f)
            fold_rps.append(float(rps_per_item(q_pred, p_cal[va_mask]).mean()))
        cv_scores[lam] = float(np.mean(fold_rps))

    best_cv = min(cv_scores.values())
    best_lam = max(lam for lam, val in cv_scores.items() if val <= best_cv + 1e-12)
    mu_full = float(m_all.mean())
    sd_full = float(m_all.std())
    m_std_full = (m_all - mu_full) / (sd_full + 1e-6)
    theta_full, gnorm = fit_7param_single(ell_all, m_std_full, p_cal, float(n_items), best_lam)
    q_out = apply_7param(theta_full, q_eval, mu_full, sd_full)
    return q_out, {"selected_lambda": best_lam, "grad_inf_norm": gnorm, "cv_rps": cv_scores}


def fit_and_apply_exp23_4bin(
    q_unlabeled_dev: np.ndarray,
    q_cal: np.ndarray,
    p_cal: np.ndarray,
    q_eval: np.ndarray,
) -> np.ndarray:
    """Experiment 23's exact 4-bin quantile histogram smoother on expected score m_i."""
    m_dev = q_unlabeled_dev @ GRID
    cuts = np.unique(np.quantile(m_dev, (0.25, 0.50, 0.75)))
    bins_cal = np.searchsorted(cuts, q_cal @ GRID, side="right")
    bins_eval = np.searchsorted(cuts, q_eval @ GRID, side="right")
    n_bins = len(cuts) + 1
    totals = np.zeros((n_bins, 5), dtype=np.float64)
    np.add.at(totals, bins_cal, p_cal)
    n_cal_votes = len(q_cal)
    prior = (totals.sum(axis=0) + 0.5) / (n_cal_votes + 2.5)
    posterior = (totals + 8.0 * prior[None, :]) / (totals.sum(axis=1, keepdims=True) + 8.0)
    return posterior[bins_eval]


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
    forecast = np.clip(q.cumsum(axis=1)[:, :4], 0.0, 1.0)
    observed = h.cumsum(axis=1)[:, :4]
    bins = np.minimum((10.0 * forecast).astype(int), 9)
    total = 0.0
    for k in range(4):
        res = observed[:, k] - forecast[:, k]
        signed_sums = np.bincount(bins[:, k], weights=res, minlength=10)
        total += float(np.abs(signed_sums).sum())
    return float(total / forecast.size)


def within_prompt_pair_acc(m: np.ndarray, y: np.ndarray, prompt_inv: np.ndarray, n_prompts: int) -> float:
    p0 = np.full(n_prompts, -1, dtype=int)
    p1 = np.full(n_prompts, -1, dtype=int)
    for idx, pid in enumerate(prompt_inv):
        if p0[pid] == -1:
            p0[pid] = idx
        else:
            p1[pid] = idx
    valid = (p0 >= 0) & (p1 >= 0) & (np.abs(y[p0] - y[p1]) > TIE_TOL)
    if not valid.any():
        return float("nan")
    dm = m[p0[valid]] - m[p1[valid]]
    dy = y[p0[valid]] - y[p1[valid]]
    s_m = np.where(np.abs(dm) <= TIE_TOL, 0.0, np.sign(dm))
    s_y = np.sign(dy)
    return float(np.where(s_m == s_y, 1.0, np.where(s_m == 0.0, 0.5, 0.0)).mean())


def compute_diagnostics(
    q: np.ndarray,
    h_test: np.ndarray,
    prompt_inv: np.ndarray,
    n_prompts: int,
) -> dict[str, float]:
    m = q @ GRID
    y = h_test @ GRID
    rm, ry = tie_levels(m), tie_levels(y)
    mc, yc = m - m.mean(), y - y.mean()
    denom = float(np.mean(mc ** 2) + np.mean(yc ** 2) + (m.mean() - y.mean()) ** 2)
    lin = float(2.0 * np.mean(mc * yc) / denom) if denom > TIE_TOL ** 2 else float("nan")
    pm, hm = mode_weights(q), mode_weights(h_test)
    return {
        "pearson": safe_corr(m, y),
        "spearman": safe_corr(rankdata(rm), rankdata(ry)),
        "kendall": float(kendalltau(rm, ry, variant="b").statistic) if rm.max() and ry.max() else float("nan"),
        "lin_ccc": lin,
        "forecast_mean": float(m.mean()),
        "human_mean": float(y.mean()),
        "mean_bias": float((m - y).mean()),
        "mean_mae": float(np.abs(m - y).mean()),
        "cumulative_ece": cumulative_ece(q, h_test),
        "modal_agreement": float((pm * hm).sum(axis=1).mean()),
        "map_accuracy": float((pm * h_test).sum(axis=1).mean()),
        "within_prompt_pair_acc": within_prompt_pair_acc(m, y, prompt_inv, n_prompts),
    }


def run_paired_bootstrap(
    diff_by_dp: np.ndarray,
    prompt_weights: np.ndarray,
    draw_weights: np.ndarray,
) -> dict:
    """diff_by_dp shape: [n_budgets, n_draws, n_prompts]; returns pointwise, simultaneous max-t, crossed 95% CIs."""
    cond_by_p = diff_by_dp.mean(axis=1)  # [K, P]
    point = cond_by_p.mean(axis=1)  # [K]
    cond_boots = prompt_weights @ cond_by_p.T  # [B, K]
    pt_lo, pt_hi = np.quantile(cond_boots, [0.025, 0.975], axis=0)
    se_cond = cond_boots.std(axis=0, ddof=1)
    valid = se_cond > 1e-12
    tmax_cond = (
        np.abs((cond_boots[:, valid] - point[valid]) / se_cond[valid]).max(axis=1)
        if valid.any()
        else np.zeros(len(prompt_weights))
    )
    crit_cond = float(np.quantile(tmax_cond, 0.95, method="higher"))
    sim_lo = point - crit_cond * se_cond
    sim_hi = point + crit_cond * se_cond
    sim_pvals = np.ones_like(point)
    pt_pvals = np.ones_like(point)
    for k in np.flatnonzero(valid):
        t_obs = abs(point[k] / se_cond[k])
        sim_pvals[k] = float((1 + np.sum(tmax_cond >= t_obs)) / (len(prompt_weights) + 1))
        t_null = np.abs((cond_boots[:, k] - point[k]) / se_cond[k])
        pt_pvals[k] = float((1 + np.sum(t_null >= t_obs)) / (len(prompt_weights) + 1))

    cross_boots = np.einsum("bd,kdp,bp->bk", draw_weights, diff_by_dp, prompt_weights)
    cr_lo, cr_hi = np.quantile(cross_boots, [0.025, 0.975], axis=0)
    se_cr = cross_boots.std(axis=0, ddof=1)
    valid_cr = se_cr > 1e-12
    tmax_cr = (
        np.abs((cross_boots[:, valid_cr] - point[valid_cr]) / se_cr[valid_cr]).max(axis=1)
        if valid_cr.any()
        else np.zeros(len(prompt_weights))
    )
    crit_cr = float(np.quantile(tmax_cr, 0.95, method="higher"))
    cr_sim_lo = point - crit_cr * se_cr
    cr_sim_hi = point + crit_cr * se_cr
    cr_sim_pvals = np.ones_like(point)
    for k in np.flatnonzero(valid_cr):
        cr_sim_pvals[k] = float((1 + np.sum(tmax_cr >= abs(point[k] / se_cr[k]))) / (len(prompt_weights) + 1))

    return {
        "point": point,
        "pt_lo": pt_lo,
        "pt_hi": pt_hi,
        "sim_lo": sim_lo,
        "sim_hi": sim_hi,
        "pt_pval": pt_pvals,
        "sim_pval": sim_pvals,
        "cross_pt_lo": cr_lo,
        "cross_pt_hi": cr_hi,
        "cross_sim_lo": cr_sim_lo,
        "cross_sim_hi": cr_sim_hi,
        "cross_sim_pval": cr_sim_pvals,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def load_test_ground_truth(bundle: dict) -> dict:
    items = bundle["items"]
    ref_test_ids = list(bundle["studies"]["qwen3_4b"]["splits"]["test_ids"])
    assert ref_test_ids == list(bundle["studies"]["llama31_8b"]["splits"]["test_ids"])
    n_test = len(ref_test_ids)
    counts_test = np.zeros((n_test, 5), dtype=np.float64)
    test_prompts = []
    for k, iid in enumerate(ref_test_ids):
        it = items[iid]
        counts_test[k] = np.bincount(it["votes"], minlength=5)
        test_prompts.append(it["prompt"])
    h_test = counts_test / counts_test.sum(axis=1, keepdims=True)
    vote_totals = counts_test.sum(axis=1)
    unique_prompts, prompt_inv, prompt_counts = np.unique(test_prompts, return_inverse=True, return_counts=True)
    n_prompts = len(unique_prompts)
    assert n_prompts == 519 and np.all(prompt_counts == 2)
    return {
        "ref_test_ids": ref_test_ids,
        "n_test": n_test,
        "counts_test": counts_test,
        "h_test": h_test,
        "mask_low_dis": vote_totals <= 3,
        "mask_high_dis": vote_totals >= 4,
        "prompt_inv": prompt_inv,
        "n_prompts": n_prompts,
    }


def emp_from_draw(draw_spec: dict, item_ids: list[int]) -> np.ndarray:
    out = np.zeros((len(item_ids), 5), dtype=np.float64)
    for k, iid in enumerate(item_ids):
        v1, _v2 = draw_spec["sampled_vote_pairs"][str(iid)]
        out[k, v1] = 1.0
    return out


def predict_draw_methods(
    draw_spec: dict,
    b_total: int,
    draw_id: int,
    model_key: str,
    npz,
    base_probs_all: np.ndarray,
    q_base_test: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    n_tr, n_va, n_ca = b_total // 2, b_total // 4, b_total // 4
    assert n_tr + n_va + n_ca == b_total
    tr_ids = draw_spec["train_order"][:n_tr]
    va_ids = draw_spec["val_order"][:n_va]
    ca_ids = draw_spec["cal_order"][:n_ca]
    all_dev_ids = tr_ids + va_ids + ca_ids

    p_tr = emp_from_draw(draw_spec, tr_ids)
    p_va = emp_from_draw(draw_spec, va_ids)
    p_ca = emp_from_draw(draw_spec, ca_ids)
    p_all_dev = np.concatenate([p_tr, p_va, p_ca], axis=0)
    n_test = len(q_base_test)

    tr_counts = p_tr.sum(axis=0)
    tr_hist = (tr_counts + 0.5) / (tr_counts.sum() + 2.5)
    all_counts = p_all_dev.sum(axis=0)
    all_hist = (all_counts + 0.5) / (all_counts.sum() + 2.5)

    q_preds: dict[str, np.ndarray] = {
        "uniform": np.full((n_test, 5), 0.2, dtype=np.float64),
        "training_histogram": np.broadcast_to(tr_hist, (n_test, 5)).copy(),
        "all_dev_histogram": np.broadcast_to(all_hist, (n_test, 5)).copy(),
        "base_raw": q_base_test,
    }
    q_preds["base_calibrated_matched"], _ = fit_and_apply_7param_cv(base_probs_all[ca_ids], p_ca, q_base_test)
    q_preds["base_calibrated_all_dev"], _ = fit_and_apply_7param_cv(base_probs_all[all_dev_ids], p_all_dev, q_base_test)
    q_preds["base_exp23_4bin"] = fit_and_apply_exp23_4bin(
        base_probs_all[va_ids + ca_ids], base_probs_all[ca_ids], p_ca, q_base_test
    )

    cal_logs = []
    for obj in ("ce", "pearson"):
        key_p = f"draw{draw_id}_d1_B{b_total}_{obj}"
        q_cal_obj = npz[f"{key_p}_cal"].astype(np.float64)
        q_val_obj = npz[f"{key_p}_val"].astype(np.float64)
        q_test_obj = npz[f"{key_p}_test"].astype(np.float64)
        q_preds[f"{obj}_raw"] = q_test_obj
        q_preds[f"{obj}_calibrated"], cal_meta = fit_and_apply_7param_cv(q_cal_obj, p_ca, q_test_obj)
        q_preds[f"{obj}_exp23_4bin"] = fit_and_apply_exp23_4bin(
            np.concatenate([q_val_obj, q_cal_obj], axis=0), q_cal_obj, p_ca, q_test_obj
        )
        cal_logs.append({
            "study": model_key,
            "depth": 1,
            "b_total": b_total,
            "draw_id": draw_id,
            "method": f"{obj}_calibrated",
            "selected_lambda": cal_meta["selected_lambda"],
            "grad_inf_norm": cal_meta["grad_inf_norm"],
        })
    return q_preds, cal_logs


def score_draw_predictions(
    q_preds: dict[str, np.ndarray],
    gt: dict,
    losses_slice: np.ndarray,
    model_key: str,
    b_total: int,
    draw_id: int,
    m_idx: dict[str, int],
) -> list[dict]:
    h_test = gt["h_test"]
    diag_rows = []
    for m_name, q_test_m in q_preds.items():
        mi = m_idx[m_name]
        losses_slice[mi, :, 0] = rps_per_item(q_test_m, h_test)
        losses_slice[mi, :, 1] = ce_per_item(q_test_m, h_test, EPS_CE_PRIMARY)
        losses_slice[mi, :, 2] = ce_per_item(q_test_m, h_test, EPS_CE_ALT)
        losses_slice[mi, :, 3] = tvd_per_item(q_test_m, h_test)
        losses_slice[mi, :, 4] = brier_per_item(q_test_m, h_test)

        dg = compute_diagnostics(q_test_m, h_test, gt["prompt_inv"], gt["n_prompts"])
        dg.update({
            "study": model_key,
            "b_total": b_total,
            "draw_id": draw_id,
            "method": m_name,
            "rps_low_dis": float(losses_slice[mi, gt["mask_low_dis"], 0].mean()),
            "rps_high_dis": float(losses_slice[mi, gt["mask_high_dis"], 0].mean()),
        })
        diag_rows.append(dg)
    return diag_rows


def _aggregate_method_row(
    model_key: str,
    b_total: int,
    b_i: int,
    mi: int,
    m_name: str,
    losses_all: np.ndarray,
    sub_dg: list[dict],
) -> dict:
    row = {
        "study": model_key,
        "depth": 1,
        "b_total": b_total,
        "n_train": b_total // 2,
        "n_val": b_total // 4,
        "n_cal": b_total // 4,
        "method": m_name,
    }
    for d_i, met in enumerate(METRIC_NAMES):
        draw_means = losses_all[b_i, :, mi, :, d_i].mean(axis=1)
        row[met] = float(draw_means.mean())
        row[f"{met}_draw_sd"] = float(draw_means.std(ddof=1))
    for dk in DIAG_KEYS:
        vals = [r[dk] for r in sub_dg if np.isfinite(r[dk])]
        row[dk] = float(np.mean(vals)) if vals else float("nan")
    return row


def aggregate_study_scores(model_key: str, budgets: list[int], losses_all: np.ndarray, diag_rows: list[dict]) -> list[dict]:
    grouped_dg: dict[tuple[int, str], list[dict]] = {}
    for dg in diag_rows:
        grouped_dg.setdefault((dg["b_total"], dg["method"]), []).append(dg)

    score_rows = []
    for b_i, b_total in enumerate(budgets):
        for mi, m_name in enumerate(METHODS):
            score_rows.append(_aggregate_method_row(model_key, b_total, b_i, mi, m_name, losses_all, grouped_dg[(b_total, m_name)]))
    return score_rows


def find_sustained(budgets: list[int], lo_arr: np.ndarray) -> dict | None:
    first_idx = None
    for idx in range(len(budgets) - 1, -1, -1):
        if lo_arr[idx] > 0.0:
            first_idx = idx
        else:
            break
    if first_idx is None:
        return None
    return {"b_total": int(budgets[first_idx]), "n_train": int(budgets[first_idx] // 2)}


def _contrast_rows_for_spec(
    model_key: str,
    budgets: list[int],
    target_m: str,
    comp_m: str,
    d_i: int,
    met: str,
    prompt_losses: np.ndarray,
    m_idx: dict[str, int],
    prompt_weights: np.ndarray,
    draw_weights: np.ndarray,
) -> tuple[list[dict], dict, list[int]]:
    diff_dp = prompt_losses[:, :, m_idx[comp_m], :, d_i] - prompt_losses[:, :, m_idx[target_m], :, d_i]
    boot = run_paired_bootstrap(diff_dp, prompt_weights, draw_weights)
    draw_diffs = diff_dp.mean(axis=2)
    rows = []
    fav_pearson = []
    for k_i, b_tot in enumerate(budgets):
        dd = draw_diffs[k_i]
        rows.append({
            "study": model_key,
            "depth": 1,
            "b_total": b_tot,
            "n_train": b_tot // 2,
            "n_val": b_tot // 4,
            "n_cal": b_tot // 4,
            "metric": met,
            "target": target_m,
            "comparator": comp_m,
            "diff_comparator_minus_target": float(boot["point"][k_i]),
            "pt_lo_95": float(boot["pt_lo"][k_i]),
            "pt_hi_95": float(boot["pt_hi"][k_i]),
            "sim_lo_95": float(boot["sim_lo"][k_i]),
            "sim_hi_95": float(boot["sim_hi"][k_i]),
            "pt_pval": float(boot["pt_pval"][k_i]),
            "sim_pval": float(boot["sim_pval"][k_i]),
            "cross_pt_lo_95": float(boot["cross_pt_lo"][k_i]),
            "cross_pt_hi_95": float(boot["cross_pt_hi"][k_i]),
            "cross_sim_lo_95": float(boot["cross_sim_lo"][k_i]),
            "cross_sim_hi_95": float(boot["cross_sim_hi"][k_i]),
            "cross_sim_pval": float(boot["cross_sim_pval"][k_i]),
            "draw_sd": float(dd.std(ddof=1)),
            "draws_target_better": int((dd > 1e-12).sum()),
            "draws_comparator_better": int((dd < -1e-12).sum()),
        })
        if target_m == "ce_calibrated" and comp_m == "pearson_calibrated" and met == "rps" and boot["point"][k_i] < 0.0:
            fav_pearson.append(int(b_tot))
    return rows, boot, fav_pearson


def compute_study_contrasts(
    model_key: str,
    budgets: list[int],
    n_draws: int,
    losses_all: np.ndarray,
    gt: dict,
    m_idx: dict[str, int],
    prompt_weights: np.ndarray,
    draw_weights: np.ndarray,
) -> tuple[list[dict], list[int], dict]:
    n_prompts = gt["n_prompts"]
    prompt_inv = gt["prompt_inv"]
    prompt_losses = np.zeros((len(budgets), n_draws, len(METHODS), n_prompts, 5), dtype=np.float64)
    for pid in range(n_prompts):
        prompt_losses[:, :, :, pid, :] = losses_all[:, :, :, prompt_inv == pid, :].mean(axis=3)

    contrast_rows = []
    sustained_summary = {}
    budgets_favoring_pearson_rps: list[int] = []
    for target_m, comp_m, d_i, met in CONTRAST_SPECS:
        rows, boot, fav = _contrast_rows_for_spec(
            model_key, budgets, target_m, comp_m, d_i, met, prompt_losses, m_idx, prompt_weights, draw_weights
        )
        contrast_rows.extend(rows)
        budgets_favoring_pearson_rps.extend(fav)
        if target_m == "ce_calibrated" and met == "rps":
            sustained_summary[comp_m] = {
                "conditional_sim_95": find_sustained(budgets, boot["sim_lo"]),
                "crossed_sim_95": find_sustained(budgets, boot["cross_sim_lo"]),
                "crossed_pt_95": find_sustained(budgets, boot["cross_pt_lo"]),
            }
    return contrast_rows, budgets_favoring_pearson_rps, sustained_summary


def evaluate_study(
    model_key: str,
    bundle: dict,
    gt: dict,
    m_idx: dict[str, int],
    prompt_weights: np.ndarray,
    rng_boot: np.random.Generator,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    npz_path = RESULTS_DIR / f"{model_key}_predictions.npz"
    assert npz_path.exists(), f"Missing {npz_path}"
    npz = np.load(npz_path)
    study = bundle["studies"][model_key]
    budgets = [int(b) for b in study["budgets"]]
    n_draws = int(study["n_draws"])
    draw_weights = rng_boot.multinomial(n_draws, np.full(n_draws, 1.0 / n_draws), size=N_BOOT) / n_draws

    base_probs_all = npz["base_probs"].astype(np.float64)
    q_base_test = base_probs_all[gt["ref_test_ids"]]
    losses_all = np.zeros((len(budgets), n_draws, len(METHODS), gt["n_test"], 5), dtype=np.float64)
    diag_rows = []
    cal_logs = []

    for b_i, b_total in enumerate(budgets):
        for draw_id in range(n_draws):
            q_preds, draw_cal = predict_draw_methods(
                study["draws"][draw_id], b_total, draw_id, model_key, npz, base_probs_all, q_base_test
            )
            cal_logs.extend(draw_cal)
            diag_rows.extend(score_draw_predictions(q_preds, gt, losses_all[b_i, draw_id], model_key, b_total, draw_id, m_idx))

    score_rows = aggregate_study_scores(model_key, budgets, losses_all, diag_rows)
    contrast_rows, fav_pearson, sustained = compute_study_contrasts(
        model_key, budgets, n_draws, losses_all, gt, m_idx, prompt_weights, draw_weights
    )
    study_summary = {
        "n_draws": n_draws,
        "budgets": budgets,
        "budgets_favoring_pearson_calibrated_rps": fav_pearson,
        "sustained_win_thresholds_rps": sustained,
    }
    return score_rows, contrast_rows, cal_logs, study_summary


def main() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    gt = load_test_ground_truth(bundle)
    n_prompts = gt["n_prompts"]
    rng_boot = np.random.default_rng(20261125)
    prompt_weights = rng_boot.multinomial(n_prompts, np.full(n_prompts, 1.0 / n_prompts), size=N_BOOT) / n_prompts

    all_score_rows, all_contrast_rows, all_cal_logs = [], [], []
    study_summaries = {}
    m_idx = {m: i for i, m in enumerate(METHODS)}

    for model_key in ("qwen3_4b", "llama31_8b"):
        s_rows, c_rows, cal_rows, st_sum = evaluate_study(model_key, bundle, gt, m_idx, prompt_weights, rng_boot)
        all_score_rows.extend(s_rows)
        all_contrast_rows.extend(c_rows)
        all_cal_logs.extend(cal_rows)
        study_summaries[model_key] = st_sum

    write_csv(RESULTS_DIR / "scores.csv", all_score_rows)
    write_csv(RESULTS_DIR / "contrasts.csv", all_contrast_rows)
    write_csv(RESULTS_DIR / "calibration_logs.csv", all_cal_logs)
    summary = {
        "n_test_items": gt["n_test"],
        "n_test_prompts": n_prompts,
        "n_test_votes": int(gt["counts_test"].sum()),
        "n_low_disagreement_items": int(gt["mask_low_dis"].sum()),
        "n_high_disagreement_items": int(gt["mask_high_dis"].sum()),
        "bootstrap_draws": N_BOOT,
        "bootstrap_seed": 20261125,
        "studies": study_summaries,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Experiment 25 calibration & scoring complete ->", RESULTS_DIR)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
