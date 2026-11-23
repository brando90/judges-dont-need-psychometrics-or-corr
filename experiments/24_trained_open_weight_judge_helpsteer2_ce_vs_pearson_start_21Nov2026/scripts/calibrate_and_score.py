#!/usr/bin/env python3
"""Matched post-hoc calibration, held-out metric panel, and prompt-clustered uncertainty for Experiment 24."""
from __future__ import annotations

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
BUNDLE_PATH = Path("/tmp/exp24_helpsteer2_raw/prepared_bundle.json")
RESULTS_DIR = EXP_DIR / "results"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.npz"

GRID = np.arange(5, dtype=np.float64)
U_POS = (GRID - 2.0) / 2.0  # [-1.0, -0.5, 0.0, +0.5, +1.0]
EPS_CLIP = 1e-5
EPS_CE_PRIMARY = 6e-6
EPS_CE_ALT = 1e-2
LAMBDA_GRID = (0.05, 0.2, 1.0, 5.0)
N_BOOT = 4000
TIE_TOL = 1e-12


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
    # Count weighted empirical distribution with Jeffreys 0.5 prior
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

        diff = (probs - p_cal) / n_items  # [n_items, 5]
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


def apply_7param(
    theta: np.ndarray,
    q: np.ndarray,
    mu_cal: float,
    sd_cal: float,
) -> np.ndarray:
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
    depth: int,
    q_eval: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Select lambda by 4-fold CV RPS on S_cal, then refit on all S_cal and apply to q_eval."""
    n_items = len(q_cal)
    assert n_items >= 4
    ell_all = centered_log_probs(q_cal)
    m_all = q_cal @ GRID

    # Deterministic 4-fold partition (contiguous slices of already-permuted cal_order)
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
            theta_f, _ = fit_7param_single(
                ell_all[tr_mask], m_std_tr, p_cal[tr_mask], float(tr_mask.sum() * depth), lam
            )
            q_pred = apply_7param(theta_f, q_cal[va_mask], mu_f, sd_f)
            fold_rps.append(float(rps_per_item(q_pred, p_cal[va_mask]).mean()))
        cv_scores[lam] = float(np.mean(fold_rps))

    best_cv = min(cv_scores.values())
    # Tie within 1e-12 -> choose largest lambda for stability
    best_lam = max(lam for lam, val in cv_scores.items() if val <= best_cv + 1e-12)
    mu_full = float(m_all.mean())
    sd_full = float(m_all.std())
    m_std_full = (m_all - mu_full) / (sd_full + 1e-6)
    theta_full, gnorm = fit_7param_single(ell_all, m_std_full, p_cal, float(n_items * depth), best_lam)
    q_out = apply_7param(theta_full, q_eval, mu_full, sd_full)
    return q_out, {"selected_lambda": best_lam, "grad_inf_norm": gnorm, "cv_rps": cv_scores}


def fit_and_apply_exp23_4bin(
    q_unlabeled_dev: np.ndarray,
    q_cal: np.ndarray,
    p_cal: np.ndarray,
    depth: int,
    q_eval: np.ndarray,
) -> np.ndarray:
    """Experiment 23's exact 4-bin quantile histogram smoother on expected score m_i."""
    m_dev = q_unlabeled_dev @ GRID
    cuts = np.unique(np.quantile(m_dev, (0.25, 0.50, 0.75)))
    bins_cal = np.searchsorted(cuts, q_cal @ GRID, side="right")
    bins_eval = np.searchsorted(cuts, q_eval @ GRID, side="right")
    n_bins = len(cuts) + 1
    totals = np.zeros((n_bins, 5), dtype=np.float64)
    np.add.at(totals, bins_cal, p_cal * depth)
    n_cal_votes = len(q_cal) * depth
    prior = (totals.sum(axis=0) + 0.5) / (n_cal_votes + 2.5)
    posterior = (totals + 8.0 * prior[None, :]) / (totals.sum(axis=1, keepdims=True) + 8.0)
    return posterior[bins_eval]


def verify_calibrator_gradient() -> None:
    rng = np.random.default_rng(12345)
    q_cal = rng.dirichlet(np.ones(5), size=16)
    p_cal = np.eye(5)[rng.integers(0, 5, size=16)]
    ell = centered_log_probs(q_cal)
    m = q_cal @ GRID
    m_std = (m - m.mean()) / (m.std() + 1e-6)
    n_cal_votes = 16.0
    lam = 0.5
    counts = p_cal.sum(axis=0)
    pi_cal = (counts + 0.5) / (n_cal_votes + 2.5)
    a0_full = np.log(pi_cal) - np.log(pi_cal).mean()

    def fn_only(theta: np.ndarray) -> float:
        a_full = np.array([theta[0], theta[1], theta[2], theta[3], -np.sum(theta[:4])])
        logits = (
            a_full[None, :]
            + theta[4] * ell
            + theta[5] * m_std[:, None] * U_POS[None, :]
            + theta[6] * m_std[:, None] * (U_POS[None, :] ** 2)
        )
        probs = softmax(logits, axis=1)
        ce = float(-np.sum(p_cal * np.log(np.clip(probs, 1e-15, 1.0))) / 16.0)
        da = a_full - a0_full
        reg = (lam / (2.0 * n_cal_votes)) * float(np.sum(da ** 2) + theta[4] ** 2 + theta[5] ** 2 + theta[6] ** 2)
        return ce + reg

    theta_opt, gnorm = fit_7param_single(ell, m_std, p_cal, n_cal_votes, lam)
    assert gnorm < 1e-8
    # Also check analytical gradient at random test point against central finite difference
    theta_test = rng.normal(0.0, 0.3, size=7)
    eps = 1e-6
    fd = np.zeros(7)
    for k in range(7):
        e = np.zeros(7)
        e[k] = eps
        fd[k] = (fn_only(theta_test + e) - fn_only(theta_test - e)) / (2 * eps)
    # Recompute exact analytical grad at theta_test
    a_full = np.array([theta_test[0], theta_test[1], theta_test[2], theta_test[3], -np.sum(theta_test[:4])])
    logits = (
        a_full[None, :]
        + theta_test[4] * ell
        + theta_test[5] * m_std[:, None] * U_POS[None, :]
        + theta_test[6] * m_std[:, None] * (U_POS[None, :] ** 2)
    )
    probs = softmax(logits, axis=1)
    diff = (probs - p_cal) / 16.0
    ga_full = diff.sum(axis=0) + (lam / n_cal_votes) * (a_full - a0_full)
    grad = np.array([
        ga_full[0] - ga_full[4],
        ga_full[1] - ga_full[4],
        ga_full[2] - ga_full[4],
        ga_full[3] - ga_full[4],
        float(np.sum(diff * ell)) + (lam / n_cal_votes) * theta_test[4],
        float(np.sum(diff * (m_std[:, None] * U_POS[None, :]))) + (lam / n_cal_votes) * theta_test[5],
        float(np.sum(diff * (m_std[:, None] * (U_POS[None, :] ** 2)))) + (lam / n_cal_votes) * theta_test[6],
    ])
    assert np.max(np.abs(fd - grad)) < 1e-8, f"FD mismatch: {np.max(np.abs(fd - grad))}"
    print("Calibrator analytical gradient check passed (FD max err < 1e-8).", flush=True)


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


METHODS = (
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
)
METRIC_NAMES = ("rps", "ce_nats", "ce_eps1e2", "tvd", "brier")
DIAG_KEYS = (
    "rps_low_dis", "rps_high_dis", "pearson", "spearman", "kendall", "lin_ccc",
    "forecast_mean", "human_mean", "mean_bias", "mean_mae",
    "cumulative_ece", "modal_agreement", "map_accuracy", "within_prompt_pair_acc",
)
PRIMARY_COMPARATORS = (
    "pearson_calibrated",
    "uniform",
    "training_histogram",
    "all_dev_histogram",
    "base_calibrated_matched",
    "base_calibrated_all_dev",
    "ce_raw",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _emp_from_draw(draw_spec: dict, item_ids: list[int], depth: int) -> np.ndarray:
    out = np.zeros((len(item_ids), 5), dtype=np.float64)
    for k, iid in enumerate(item_ids):
        v1, v2 = draw_spec["sampled_vote_pairs"][str(iid)]
        if depth == 1:
            out[k, v1] = 1.0
        else:
            out[k, v1] += 0.5
            out[k, v2] += 0.5
    return out


def _load_test_ground_truth(items: list[dict], test_ids: list[int]) -> dict:
    n_test = len(test_ids)
    counts_test = np.zeros((n_test, 5), dtype=np.float64)
    test_prompts = []
    for k, iid in enumerate(test_ids):
        it = items[iid]
        counts_test[k] = np.bincount(it["votes"], minlength=5)
        test_prompts.append(it["prompt"])
    h_test = counts_test / counts_test.sum(axis=1, keepdims=True)
    vote_totals = counts_test.sum(axis=1)
    unique_prompts, prompt_inv, prompt_counts = np.unique(test_prompts, return_inverse=True, return_counts=True)
    n_prompts = len(unique_prompts)
    assert n_prompts == 519 and np.all(prompt_counts == 2)
    return {
        "counts_test": counts_test,
        "h_test": h_test,
        "mask_low_dis": vote_totals <= 3,
        "mask_high_dis": vote_totals >= 4,
        "prompt_inv": prompt_inv,
        "n_prompts": n_prompts,
    }


def _predict_draw_methods(
    draw_spec: dict,
    draw_id: int,
    depth: int,
    b_total: int,
    base_probs_all: np.ndarray,
    q_base_test: np.ndarray,
    npz,
    cal_logs: list[dict],
) -> dict[str, np.ndarray]:
    n_test = q_base_test.shape[0]
    n_tr, n_va, n_ca = (b_total // 2) // depth, (b_total // 4) // depth, (b_total // 4) // depth
    tr_ids = draw_spec["train_order"][:n_tr]
    va_ids = draw_spec["val_order"][:n_va]
    ca_ids = draw_spec["cal_order"][:n_ca]
    all_dev_ids = tr_ids + va_ids + ca_ids

    p_tr = _emp_from_draw(draw_spec, tr_ids, depth)
    p_va = _emp_from_draw(draw_spec, va_ids, depth)
    p_ca = _emp_from_draw(draw_spec, ca_ids, depth)
    p_all_dev = np.concatenate([p_tr, p_va, p_ca], axis=0)

    tr_counts = (p_tr * depth).sum(axis=0)
    all_counts = (p_all_dev * depth).sum(axis=0)
    q_preds: dict[str, np.ndarray] = {
        "uniform": np.full((n_test, 5), 0.2, dtype=np.float64),
        "training_histogram": np.broadcast_to((tr_counts + 0.5) / (tr_counts.sum() + 2.5), (n_test, 5)).copy(),
        "all_dev_histogram": np.broadcast_to((all_counts + 0.5) / (all_counts.sum() + 2.5), (n_test, 5)).copy(),
        "base_raw": q_base_test,
    }
    q_preds["base_calibrated_matched"], _ = fit_and_apply_7param_cv(base_probs_all[ca_ids], p_ca, depth, q_base_test)
    q_preds["base_calibrated_all_dev"], _ = fit_and_apply_7param_cv(base_probs_all[all_dev_ids], p_all_dev, depth, q_base_test)
    q_preds["base_exp23_4bin"] = fit_and_apply_exp23_4bin(
        base_probs_all[va_ids + ca_ids], base_probs_all[ca_ids], p_ca, depth, q_base_test
    )

    for obj in ("ce", "pearson"):
        key_p = f"draw{draw_id}_d{depth}_B{b_total}_{obj}"
        q_cal_obj = npz[f"{key_p}_cal"].astype(np.float64)
        q_val_obj = npz[f"{key_p}_val"].astype(np.float64)
        q_test_obj = npz[f"{key_p}_test"].astype(np.float64)
        q_preds[f"{obj}_raw"] = q_test_obj
        q_preds[f"{obj}_calibrated"], cal_meta = fit_and_apply_7param_cv(q_cal_obj, p_ca, depth, q_test_obj)
        q_preds[f"{obj}_exp23_4bin"] = fit_and_apply_exp23_4bin(
            np.concatenate([q_val_obj, q_cal_obj], axis=0), q_cal_obj, p_ca, depth, q_test_obj
        )
        cal_logs.append({
            "depth": depth,
            "b_total": b_total,
            "draw_id": draw_id,
            "method": f"{obj}_calibrated",
            "selected_lambda": cal_meta["selected_lambda"],
            "grad_inf_norm": cal_meta["grad_inf_norm"],
        })
    return q_preds


def _evaluate_all_configs(
    bundle: dict,
    configs: list[tuple[int, int]],
    base_probs_all: np.ndarray,
    q_base_test: np.ndarray,
    npz,
    gt: dict,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    n_draws = int(bundle["n_draws"])
    n_test = q_base_test.shape[0]
    m_idx = {m: i for i, m in enumerate(METHODS)}
    losses_all = np.zeros((len(configs), n_draws, len(METHODS), n_test, 5), dtype=np.float64)
    diag_rows, cal_logs = [], []
    h_test, prompt_inv, n_prompts = gt["h_test"], gt["prompt_inv"], gt["n_prompts"]

    for c_i, (depth, b_total) in enumerate(configs):
        n_tr, n_va, n_ca = (b_total // 2) // depth, (b_total // 4) // depth, (b_total // 4) // depth
        for draw_id in range(n_draws):
            q_preds = _predict_draw_methods(
                bundle["draws"][draw_id], draw_id, depth, b_total, base_probs_all, q_base_test, npz, cal_logs
            )
            for m_name, q_test_m in q_preds.items():
                mi = m_idx[m_name]
                losses_all[c_i, draw_id, mi, :, 0] = rps_per_item(q_test_m, h_test)
                losses_all[c_i, draw_id, mi, :, 1] = ce_per_item(q_test_m, h_test, EPS_CE_PRIMARY)
                losses_all[c_i, draw_id, mi, :, 2] = ce_per_item(q_test_m, h_test, EPS_CE_ALT)
                losses_all[c_i, draw_id, mi, :, 3] = tvd_per_item(q_test_m, h_test)
                losses_all[c_i, draw_id, mi, :, 4] = brier_per_item(q_test_m, h_test)
                dg = compute_diagnostics(q_test_m, h_test, prompt_inv, n_prompts)
                dg.update({
                    "depth": depth, "b_total": b_total, "n_train": n_tr * depth,
                    "n_val": n_va * depth, "n_cal": n_ca * depth, "n_train_items": n_tr,
                    "draw_id": draw_id, "method": m_name,
                    "rps_low_dis": float(losses_all[c_i, draw_id, mi, gt["mask_low_dis"], 0].mean()),
                    "rps_high_dis": float(losses_all[c_i, draw_id, mi, gt["mask_high_dis"], 0].mean()),
                })
                diag_rows.append(dg)
    return losses_all, diag_rows, cal_logs


def _aggregate_scores(configs: list[tuple[int, int]], losses_all: np.ndarray, diag_rows: list[dict]) -> list[dict]:
    score_rows = []
    for c_i, (depth, b_total) in enumerate(configs):
        n_tr, n_va, n_ca = (b_total // 2) // depth, (b_total // 4) // depth, (b_total // 4) // depth
        for mi, m_name in enumerate(METHODS):
            row = {
                "depth": depth, "b_total": b_total, "n_train": n_tr * depth,
                "n_val": n_va * depth, "n_cal": n_ca * depth, "n_train_items": n_tr,
                "n_val_items": n_va, "n_cal_items": n_ca, "method": m_name,
            }
            for d_i, met in enumerate(METRIC_NAMES):
                draw_means = losses_all[c_i, :, mi, :, d_i].mean(axis=1)
                row[met] = float(draw_means.mean())
                row[f"{met}_draw_sd"] = float(draw_means.std(ddof=1))
            sub_dg = [r for r in diag_rows if r["depth"] == depth and r["b_total"] == b_total and r["method"] == m_name]
            for dk in DIAG_KEYS:
                vals = [r[dk] for r in sub_dg if np.isfinite(r[dk])]
                row[dk] = float(np.mean(vals)) if vals else float("nan")
            score_rows.append(row)
    return score_rows


def _max_t_bands(boots: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pt_lo, pt_hi = np.quantile(boots, [0.025, 0.975], axis=0)
    se = boots.std(axis=0, ddof=1)
    valid = se > 1e-12
    tmax = np.abs((boots[:, valid] - point[valid]) / se[valid]).max(axis=1) if valid.any() else np.zeros(N_BOOT)
    crit = float(np.quantile(tmax, 0.95, method="higher"))
    sim_lo, sim_hi = point - crit * se, point + crit * se
    sim_pvals, pt_pvals = np.ones_like(point), np.ones_like(point)
    for k in np.flatnonzero(valid):
        t_obs = abs(point[k] / se[k])
        sim_pvals[k] = float((1 + np.sum(tmax >= t_obs)) / (N_BOOT + 1))
        pt_pvals[k] = float((1 + np.sum(np.abs((boots[:, k] - point[k]) / se[k]) >= t_obs)) / (N_BOOT + 1))
    return pt_lo, pt_hi, sim_lo, sim_hi, pt_pvals, sim_pvals


def _run_paired_bootstrap(diff_by_dp: np.ndarray, prompt_weights: np.ndarray, draw_weights: np.ndarray) -> dict:
    """diff_by_dp has shape [n_budgets, n_draws, n_prompts]; returns pointwise, simultaneous max-t, crossed 95% CIs."""
    cond_by_p = diff_by_dp.mean(axis=1)
    point = cond_by_p.mean(axis=1)
    pt_lo, pt_hi, sim_lo, sim_hi, pt_pvals, sim_pvals = _max_t_bands(prompt_weights @ cond_by_p.T, point)
    cross_boots = np.einsum("bd,kdp,bp->bk", draw_weights, diff_by_dp, prompt_weights)
    cr_lo, cr_hi, cr_sim_lo, cr_sim_hi, _, cr_sim_pvals = _max_t_bands(cross_boots, point)
    return {
        "point": point, "pt_lo": pt_lo, "pt_hi": pt_hi, "sim_lo": sim_lo, "sim_hi": sim_hi,
        "pt_pval": pt_pvals, "sim_pval": sim_pvals,
        "cross_pt_lo": cr_lo, "cross_pt_hi": cr_hi, "cross_sim_lo": cr_sim_lo, "cross_sim_hi": cr_sim_hi,
        "cross_sim_pval": cr_sim_pvals,
    }


def _find_sustained(lo_arr: np.ndarray, budgets: list[int]) -> dict | None:
    first_idx = None
    for idx in range(len(budgets) - 1, -1, -1):
        if lo_arr[idx] > 0.0:
            first_idx = idx
        else:
            break
    if first_idx is None:
        return None
    return {"b_total": int(budgets[first_idx]), "n_train": int(budgets[first_idx] // 2)}


def _build_primary_contrasts(
    configs: list[tuple[int, int]],
    prompt_losses: np.ndarray,
    prompt_weights: np.ndarray,
    draw_weights: np.ndarray,
) -> tuple[list[dict], dict]:
    m_idx = {m: i for i, m in enumerate(METHODS)}
    d1_indices = [i for i, (d, _) in enumerate(configs) if d == 1]
    d1_budgets = [configs[i][1] for i in d1_indices]
    contrast_specs = [
        ("ce_calibrated", comp, d_i, met) for comp in PRIMARY_COMPARATORS for d_i, met in enumerate(METRIC_NAMES[:4])
    ] + [
        ("ce_raw", "pearson_raw", 0, "rps"),
        ("ce_raw", "pearson_raw", 1, "ce_nats"),
        ("ce_raw", "pearson_raw", 3, "tvd"),
        ("ce_exp23_4bin", "pearson_exp23_4bin", 0, "rps"),
        ("ce_exp23_4bin", "pearson_exp23_4bin", 1, "ce_nats"),
        ("ce_exp23_4bin", "pearson_exp23_4bin", 3, "tvd"),
    ]
    contrast_rows, sustained_summary = [], {}
    for target_m, comp_m, d_i, met in contrast_specs:
        diff_dp = prompt_losses[d1_indices, :, m_idx[comp_m], :, d_i] - prompt_losses[d1_indices, :, m_idx[target_m], :, d_i]
        boot = _run_paired_bootstrap(diff_dp, prompt_weights, draw_weights)
        draw_diffs = diff_dp.mean(axis=2)
        for k_i, b_tot in enumerate(d1_budgets):
            dd = draw_diffs[k_i]
            contrast_rows.append({
                "depth": 1, "b_total": b_tot, "n_train": b_tot // 2, "n_val": b_tot // 4, "n_cal": b_tot // 4,
                "metric": met, "target": target_m, "comparator": comp_m,
                "diff_comparator_minus_target": float(boot["point"][k_i]),
                "pt_lo_95": float(boot["pt_lo"][k_i]), "pt_hi_95": float(boot["pt_hi"][k_i]),
                "sim_lo_95": float(boot["sim_lo"][k_i]), "sim_hi_95": float(boot["sim_hi"][k_i]),
                "pt_pval": float(boot["pt_pval"][k_i]), "sim_pval": float(boot["sim_pval"][k_i]),
                "cross_pt_lo_95": float(boot["cross_pt_lo"][k_i]), "cross_pt_hi_95": float(boot["cross_pt_hi"][k_i]),
                "cross_sim_lo_95": float(boot["cross_sim_lo"][k_i]), "cross_sim_hi_95": float(boot["cross_sim_hi"][k_i]),
                "cross_sim_pval": float(boot["cross_sim_pval"][k_i]),
                "draw_sd": float(dd.std(ddof=1)),
                "draws_target_better": int((dd > 1e-12).sum()),
                "draws_comparator_better": int((dd < -1e-12).sum()),
            })
        if target_m == "ce_calibrated" and met == "rps":
            sustained_summary[comp_m] = {
                "conditional_sim_95": _find_sustained(boot["sim_lo"], d1_budgets),
                "crossed_sim_95": _find_sustained(boot["cross_sim_lo"], d1_budgets),
                "crossed_pt_95": _find_sustained(boot["cross_pt_lo"], d1_budgets),
            }
    return contrast_rows, sustained_summary


def _build_depth_contrasts(
    bundle: dict,
    configs: list[tuple[int, int]],
    prompt_losses: np.ndarray,
    prompt_weights: np.ndarray,
    draw_weights: np.ndarray,
) -> list[dict]:
    m_idx = {m: i for i, m in enumerate(METHODS)}
    depth_rows = []
    within_pairs = (
        ("ce_calibrated", "pearson_calibrated"),
        ("ce_calibrated", "uniform"),
        ("ce_calibrated", "training_histogram"),
        ("ce_calibrated", "base_calibrated_matched"),
        ("ce_raw", "pearson_raw"),
    )
    for b_tot in bundle["budgets_d2"]:
        c1, c2 = configs.index((1, b_tot)), configs.index((2, b_tot))
        specs = [(f"d2_{comp}_minus_{tgt}", prompt_losses[[c2], :, m_idx[comp], :, 0] - prompt_losses[[c2], :, m_idx[tgt], :, 0]) for tgt, comp in within_pairs]
        specs += [(f"{m}_d2_minus_d1", prompt_losses[[c2], :, m_idx[m], :, 0] - prompt_losses[[c1], :, m_idx[m], :, 0]) for m in ("ce_calibrated", "pearson_calibrated")]
        for label, diff_dp in specs:
            boot = _run_paired_bootstrap(diff_dp, prompt_weights, draw_weights)
            dd = diff_dp[0].mean(axis=1)
            depth_rows.append({
                "comparison": label, "b_total": b_tot, "diff": float(boot["point"][0]),
                "pt_lo_95": float(boot["pt_lo"][0]), "pt_hi_95": float(boot["pt_hi"][0]),
                "cross_pt_lo_95": float(boot["cross_pt_lo"][0]), "cross_pt_hi_95": float(boot["cross_pt_hi"][0]),
                "pt_pval": float(boot["pt_pval"][0]), "draws_positive": int((dd > 1e-12).sum()),
            })
    return depth_rows


def main() -> None:
    verify_calibrator_gradient()
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    npz = np.load(PREDICTIONS_PATH)
    test_ids = list(bundle["splits"]["test_ids"])
    gt = _load_test_ground_truth(bundle["items"], test_ids)

    base_probs_all = npz["base_probs"].astype(np.float64)
    q_base_test = base_probs_all[test_ids]
    configs = [(1, b) for b in bundle["budgets_d1"]] + [(2, b) for b in bundle["budgets_d2"]]

    losses_all, diag_rows, cal_logs = _evaluate_all_configs(bundle, configs, base_probs_all, q_base_test, npz, gt)
    score_rows = _aggregate_scores(configs, losses_all, diag_rows)

    n_draws, n_prompts = int(bundle["n_draws"]), gt["n_prompts"]
    prompt_losses = np.zeros((len(configs), n_draws, len(METHODS), n_prompts, 5), dtype=np.float64)
    for pid in range(n_prompts):
        prompt_losses[:, :, :, pid, :] = losses_all[:, :, :, gt["prompt_inv"] == pid, :].mean(axis=3)

    rng_boot = np.random.default_rng(20261124)
    prompt_weights = rng_boot.multinomial(n_prompts, np.full(n_prompts, 1.0 / n_prompts), size=N_BOOT) / n_prompts
    draw_weights = rng_boot.multinomial(n_draws, np.full(n_draws, 1.0 / n_draws), size=N_BOOT) / n_draws

    contrast_rows, sustained_summary = _build_primary_contrasts(configs, prompt_losses, prompt_weights, draw_weights)
    depth_rows = _build_depth_contrasts(bundle, configs, prompt_losses, prompt_weights, draw_weights)

    _write_csv(RESULTS_DIR / "scores.csv", score_rows)
    _write_csv(RESULTS_DIR / "contrasts.csv", contrast_rows)
    _write_csv(RESULTS_DIR / "depth_contrasts.csv", depth_rows)
    _write_csv(RESULTS_DIR / "calibration_logs.csv", cal_logs)
    summary = {
        "n_test_items": len(test_ids),
        "n_test_prompts": n_prompts,
        "n_test_votes": int(gt["counts_test"].sum()),
        "n_low_disagreement_items": int(gt["mask_low_dis"].sum()),
        "n_high_disagreement_items": int(gt["mask_high_dis"].sum()),
        "sustained_win_thresholds_rps": sustained_summary,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Calibration and scoring complete ->", RESULTS_DIR)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
