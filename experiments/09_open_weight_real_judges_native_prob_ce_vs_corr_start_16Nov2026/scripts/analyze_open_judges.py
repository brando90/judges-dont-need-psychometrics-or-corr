#!/usr/bin/env python3
"""Comprehensive statistical evaluation for Experiment 09:
  0. Taxonomy i audit (synthetic & historical real studies, read directly from repo files)
  1. Karpathy sanity checks, expect-then-verify contract, human ceiling, and k=3 repeat stability
  2. Representation bridge vs archived Experiment 14 (3-sample scalar vs q_verbal vs q_logprob)
  3. ii.a Prompt Optimization (candidate pool P0..P5 + blind 3-fold Meta-Prompt Search on Qwen3-30B-A3B)
  4. ii.b Harness Optimization (H1..H6 + 2x2 Prompt x Harness/Calibration factorial)
  5. ii.c Real Weight Updates (end-to-end PyTorch LoRA backprop on Qwen3-4B & Qwen2.5-Coder-7B +
     leak-free hidden-state/logit neural softmax heads across n_fam learning curves & 5 restarts +
     certified convex CE/RPS & exact cone Pearson simplex mixtures)
  6. WHEN does CE beat correlation? (MSE bias/scale decomposition, rater-disagreement & information-gap G_i
     stratification, ordinary vs placeholder items, training-size crossovers).

Writes public-safe outputs ONLY (zero individual human ratings) to results/open_weight_forecasts.json
and results/analysis_summary.json.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import expit, logsumexp, softmax
from scipy.stats import kendalltau, rankdata, spearmanr

EXPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPT_DIR.parents[1]
DATASET_PATH = Path(
    "/Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/"
    "02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/private/historical_exp06/cached_v1/dataset.json"
)
EXP14_SCORES_PATH = (
    REPO_ROOT
    / "experiments/archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/results/judge_scores.json"
)

GRID_5 = np.arange(6, dtype=np.float64)
GRID_1 = GRID_5 / 5.0
FLOOR = 1e-6
CERT_TOL = 1e-8
TAU = 0.1


def smooth_floor(q: np.ndarray, eps: float = FLOOR) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    arr = np.clip(arr, eps, None)
    return arr / arr.sum(axis=-1, keepdims=True)


def item_losses(q: np.ndarray, p: np.ndarray, ratings_list: list[list[int]] | None = None) -> dict[str, np.ndarray]:
    """Compute exact individual-rating proper losses per artifact (n,)."""
    qf = smooth_floor(q, FLOOR)
    cq = np.cumsum(qf, axis=1)[:, :5]
    cp = np.cumsum(p, axis=1)[:, :5]
    rps = ((cq - cp) ** 2).mean(axis=1) + (cp * (1.0 - cp)).mean(axis=1)
    ce = -(p * np.log(qf)).sum(axis=1)
    tvd = 0.5 * np.abs(qf - p).sum(axis=1)
    # Six-category individual-rating Brier: E_Y[sum_c (q_c - 1{Y=c})^2] = sum_c (q_c - p_c)^2 + (1 - sum_c p_c^2)
    brier = ((qf - p) ** 2).sum(axis=1) + (1.0 - (p ** 2).sum(axis=1))
    mu5 = qf @ GRID_5
    hm5 = p @ GRID_5
    sq_err_01 = ((mu5 - hm5) / 5.0) ** 2
    mae_05 = np.abs(mu5 - hm5)
    bias_05 = mu5 - hm5
    return {
        "rps": rps,
        "ce": ce,
        "tvd": tvd,
        "brier": brier,
        "sq_err_01": sq_err_01,
        "mae_05": mae_05,
        "bias_05": bias_05,
        "mu_05": mu5,
        "hm_05": hm5,
    }


INDIV_ITEM_IDX: np.ndarray | None = None
INDIV_RATINGS: np.ndarray | None = None


def mse_decomposition(mu5: np.ndarray, hm5: np.ndarray) -> dict[str, float]:
    """Appendix D identity: MSE = bias^2 + (s_x - s_y)^2 + 2 s_x s_y (1 - r),
    plus both 75-artifact-mean correlations and 195-individual-rating correlations (Pearson, Spearman, Kendall tau-b)."""
    diff = mu5 - hm5
    mse = float(np.mean(diff ** 2))
    bias = float(np.mean(mu5) - np.mean(hm5))
    sx = float(np.sqrt(np.mean((mu5 - np.mean(mu5)) ** 2)))
    sy = float(np.sqrt(np.mean((hm5 - np.mean(hm5)) ** 2)))
    r = float(np.corrcoef(mu5, hm5)[0, 1]) if sx > 1e-12 and sy > 1e-12 else 0.0
    scale_sq = (sx - sy) ** 2
    disp = 2.0 * sx * sy * (1.0 - r)
    ident_err = abs(mse - (bias ** 2 + scale_sq + disp))
    assert ident_err < 1e-9, f"MSE decomposition error: {ident_err}"
    sp = float(spearmanr(mu5, hm5).statistic) if sx > 1e-12 and sy > 1e-12 else 0.0
    kt = float(kendalltau(mu5, hm5, variant="b").statistic) if sx > 1e-12 and sy > 1e-12 else 0.0
    out = {
        "mse_05": mse,
        "nrmse_01": math.sqrt(mse) / 5.0,
        "mae_05": float(np.mean(np.abs(diff))),
        "bias_05": bias,
        "bias_sq_05": bias ** 2,
        "scale_mismatch_sq_05": scale_sq,
        "dispersion_term_05": disp,
        "judge_mean_05": float(np.mean(mu5)),
        "human_mean_05": float(np.mean(hm5)),
        "judge_sd_05": sx,
        "human_sd_05": sy,
        "pearson": r,
        "spearman": sp,
        "kendall_tau_b": kt,
    }
    if INDIV_ITEM_IDX is not None and INDIV_RATINGS is not None and len(mu5) == int(INDIV_ITEM_IDX.max()) + 1:
        mu_ind = mu5[INDIV_ITEM_IDX]
        s_ind = float(np.std(mu_ind))
        out["indiv_pearson_195"] = float(np.corrcoef(mu_ind, INDIV_RATINGS)[0, 1]) if s_ind > 1e-12 else 0.0
        out["indiv_spearman_195"] = float(spearmanr(mu_ind, INDIV_RATINGS).statistic) if s_ind > 1e-12 else 0.0
        out["indiv_kendall_tau_b_195"] = float(kendalltau(mu_ind, INDIV_RATINGS, variant="b").statistic) if s_ind > 1e-12 else 0.0
        out["indiv_judge_mean_05"] = float(np.mean(mu_ind))
        out["indiv_human_mean_05"] = float(np.mean(INDIV_RATINGS))
        out["indiv_bias_05"] = float(np.mean(mu_ind) - np.mean(INDIV_RATINGS))
    return out


def family_bootstrap_and_signflip(
    diff_per_item: np.ndarray,
    fam_idx: np.ndarray,
    n_fams: int = 19,
    n_boot: int = 10000,
    seed: int = 20261116,
) -> dict[str, object]:
    """Paired cluster bootstrap & exact/MC sign-flip permutation p-value over the 19 task families,
    weighting all eligible artifacts equally (matching EVALUATION_CRITERION_11-09-2026.md)."""
    fam_sums = np.bincount(fam_idx, weights=diff_per_item, minlength=n_fams)
    fam_cnts = np.bincount(fam_idx, minlength=n_fams)
    active = fam_cnts > 0
    fam_sums_a = fam_sums[active]
    fam_cnts_a = fam_cnts[active]
    m_active = int(active.sum())
    point = float(fam_sums_a.sum() / fam_cnts_a.sum())

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, m_active, size=(n_boot, m_active))
    bs_num = fam_sums_a[draws].sum(axis=1)
    bs_den = fam_cnts_a[draws].sum(axis=1)
    bs = bs_num / np.maximum(bs_den, 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])

    # Paired two-sided cluster sign-flip test on family-level totals (exact if m_active <= 19 -> 2^19 = 524,288!)
    obs_abs = abs(float(fam_sums_a.sum()))
    if m_active <= 19:
        bits = ((np.arange(1 << m_active, dtype=np.uint32)[:, None] >> np.arange(m_active, dtype=np.uint32)) & 1)
        signs = 2.0 * bits - 1.0
        perm_totals = np.abs(signs @ fam_sums_a)
        pval = float(np.mean(perm_totals >= obs_abs - 1e-15))
        test_name = f"exact_2^{m_active}_task_family_sign_flip"
    else:
        signs = rng.choice([-1.0, 1.0], size=(20000, m_active))
        perm_totals = np.abs(signs @ fam_sums_a)
        pval = float((1 + np.sum(perm_totals >= obs_abs - 1e-15)) / 20001)
        test_name = "mc_20000_task_family_sign_flip"

    return {
        "point": point,
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "p_val": pval,
        "test": test_name,
        "n_families": m_active,
        "n_artifacts": int(fam_cnts_a.sum()),
    }


# ---- Certified Simplex Mixture Solvers (CE, RPS, Pearson, Spearman) ----

def fit_simplex_ce(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, dict]:
    """Convex SLSQP + active-face Newton polishing with Frank-Wolfe duality gap <= 1e-8."""
    n, m, _ = q.shape
    w0 = np.full(m, 1.0 / m)

    def fg(w: np.ndarray) -> tuple[float, np.ndarray]:
        pred = np.einsum("njr,j->nr", q, w)
        pred = np.clip(pred, 1e-15, None)
        val = -float(np.sum(p * np.log(pred)) / n)
        grad = -np.einsum("nr,njr->j", p / pred, q) / n
        return val, grad

    opt = minimize(
        fg,
        w0,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * m,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(m)},
        options={"maxiter": 600, "ftol": 1e-14},
    )
    w = np.maximum(opt.x, 0.0)
    w /= w.sum()
    for _ in range(12):
        val, grad = fg(w)
        if float(grad @ w - grad.min()) <= CERT_TOL / 5.0:
            break
        act = np.flatnonzero(w > 1e-10)
        if len(act) < 2:
            break
        pred = np.einsum("njr,j->nr", q, w)
        hess = np.einsum("nr,njr,nkr->jk", p / (pred * pred), q[:, act], q[:, act]) / n
        basis = np.vstack([np.eye(len(act) - 1), -np.ones(len(act) - 1)])
        tg = basis.T @ grad[act]
        red = np.linalg.lstsq(basis.T @ hess @ basis + 1e-12 * np.eye(len(act) - 1), -tg, rcond=None)[0]
        d = np.zeros(m)
        d[act] = basis @ red
        slope = float(tg @ red)
        if slope >= 0 or np.max(np.abs(d)) < 1e-15:
            break
        neg = d < 0
        step = min(1.0, float(np.min(-w[neg] / d[neg]))) if np.any(neg) else 1.0
        for _ in range(20):
            tw = np.maximum(w + step * d, 0.0)
            tw /= tw.sum()
            tv, _ = fg(tw)
            if tv <= val + 1e-4 * step * slope + 1e-15:
                w = tw
                break
            step *= 0.5
    _, grad = fg(w)
    gap = float(grad @ w - grad.min())
    return w, {"status": "optimal" if gap <= CERT_TOL else "cert_warn", "duality_gap": gap}


def fit_simplex_rps(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, dict]:
    """Convex Quadratic Program on the simplex for RPS mixture weights, with Frank-Wolfe duality gap <= 1e-8."""
    n, m, _ = q.shape
    cq = np.cumsum(q, axis=2)[:, :, :5]  # n x m x 5
    cp = np.cumsum(p, axis=1)[:, :5]     # n x 5

    def fg(w: np.ndarray) -> tuple[float, np.ndarray]:
        diff = np.einsum("nmk,m->nk", cq, w) - cp
        val = float(np.mean(diff ** 2))
        grad = (2.0 / (5.0 * n)) * np.einsum("nk,nmk->m", diff, cq)
        return val, grad

    opt = minimize(
        fg,
        np.full(m, 1.0 / m),
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * m,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(m)},
        options={"maxiter": 600, "ftol": 1e-15},
    )
    w = np.maximum(opt.x, 0.0)
    w /= w.sum()
    _, grad = fg(w)
    gap = float(grad @ w - grad.min())
    return w, {"status": "optimal" if gap <= 1e-7 else "cert_warn", "duality_gap": gap}


def fit_simplex_pearson(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, dict]:
    """Exact NNLS cone projection for Pearson-maximizing simplex weights."""
    n, m, _ = q.shape
    means = q @ GRID_1
    y = p @ GRID_1
    a = means - means.mean(axis=0)
    b = y - y.mean()
    norms = np.linalg.norm(a, axis=0)
    nb = float(np.linalg.norm(b))
    act = np.flatnonzero(norms > 1e-12)
    if nb <= 1e-12 or len(act) == 0:
        return np.full(m, 1.0 / m), {"status": "degenerate"}
    unit = a[:, act] / norms[act]
    prod = unit.T @ b
    if prod.max() <= 0:
        j = int(act[np.argmax(prod)])
        return np.eye(m)[j], {"status": "optimal_cone"}
    c, _ = nnls(unit, b, maxiter=50 * len(act))
    w = np.zeros(m)
    w[act] = c / norms[act]
    if w.sum() <= 0:
        return np.full(m, 1.0 / m), {"status": "degenerate"}
    w /= w.sum()
    return w, {"status": "optimal_cone"}


def fit_simplex_spearman(q: np.ndarray, p: np.ndarray, rng: np.random.Generator, draws: int = 3000) -> np.ndarray:
    m = q.shape[1]
    mu_q = q @ GRID_1
    hm = p @ GRID_1
    cand = np.vstack([np.eye(m), np.full((1, m), 1.0 / m), rng.dirichlet(np.ones(m), size=draws)])
    mu = mu_q @ cand.T
    rk = rankdata(mu, axis=0)
    rt = rankdata(hm)
    rk -= rk.mean(axis=0)
    rtc = rt - rt.mean()
    den = np.linalg.norm(rk, axis=0) * np.linalg.norm(rtc)
    sp = np.where(den > 1e-12, (rk * rtc[:, None]).sum(axis=0) / np.where(den > 1e-12, den, 1.0), -2.0)
    return cand[int(np.argmax(sp))]


# ---- Flexible Softmax Neural Head Solvers (matching Exps 13, 06, 07, 08) ----

def fit_softmax_head(
    phi_tr: np.ndarray,
    p_tr: np.ndarray,
    kind: str,
    ridge_c: float = 1.0,
    n_restarts: int = 1,
    maxiter: int = 400,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Fit W (D x 6) for q_W(r|z) = softmax(Phi @ W)_r under all 6 objectives, with optional multi-restarts."""
    n, d = phi_tr.shape
    lam = ridge_c / n
    h = p_tr @ GRID_5
    h_rank = rankdata(h)

    def pearson_g(a_raw: np.ndarray, b_raw: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = a_raw - a_raw.mean(), b_raw - b_raw.mean()
        na, nb = float(np.linalg.norm(a)) + 1e-12, float(np.linalg.norm(b)) + 1e-12
        r = float((a @ b) / (na * nb))
        return r, (b / nb - r * a / na) / na

    def fg(w_flat: np.ndarray) -> tuple[float, np.ndarray]:
        w = w_flat.reshape(d, 6)
        z = phi_tr @ w
        q = softmax(z, axis=1)
        if kind == "ce":
            qf = np.clip(q, FLOOR, None)
            qf /= qf.sum(axis=1, keepdims=True)
            obj = float(-(p_tr * np.log(qf)).sum(axis=1).mean())
            dz = (q - p_tr) / n
        elif kind == "rps":
            diff = np.cumsum(q, axis=1)[:, :5] - np.cumsum(p_tr, axis=1)[:, :5]
            obj = float((diff ** 2).mean())
            g = np.zeros_like(q)
            g[:, :5] = (2.0 / 5.0) * np.cumsum(diff[:, ::-1], axis=1)[:, ::-1]
            dz = (q * (g - (q * g).sum(axis=1, keepdims=True))) / n
        else:
            mu = q @ GRID_5
            if kind == "pearson":
                val, gm = pearson_g(mu, h)
            elif kind == "spearman":
                dm = (mu[:, None] - mu[None, :]) / TAU
                sig = expit(dm)
                u = sig.sum(axis=1)
                val, gu = pearson_g(u, h_rank)
                sm = sig * (1.0 - sig) / TAU
                gm = gu * sm.sum(axis=1) - sm @ gu
            elif kind == "kendall":
                sgn = np.sign(h[:, None] - h[None, :])
                th = np.tanh((mu[:, None] - mu[None, :]) / TAU)
                c_k = 2.0 / max(1, n * (n - 1))
                val = float(c_k * 0.5 * (th * sgn).sum())
                gm = c_k * ((1.0 - th ** 2) / TAU * sgn).sum(axis=1)
            elif kind == "pearson_spearman":
                vp, gp = pearson_g(mu, h)
                dm = (mu[:, None] - mu[None, :]) / TAU
                sig = expit(dm)
                u = sig.sum(axis=1)
                vs, gu = pearson_g(u, h_rank)
                sm = sig * (1.0 - sig) / TAU
                gs = gu * sm.sum(axis=1) - sm @ gu
                val, gm = 0.5 * vp + 0.5 * vs, 0.5 * gp + 0.5 * gs
            else:
                raise ValueError(kind)
            obj = -val
            dz = (q * (GRID_5[None, :] - mu[:, None]) * (-gm)[:, None])
        # Do not penalize intercept row 0's uniform shift; match toy.py ridge on W
        return obj + lam * float((w ** 2).sum()), (phi_tr.T @ dz + 2.0 * lam * w).ravel()

    rng = np.random.default_rng(seed)
    w_base = 0.01 * rng.normal(size=(d, 6))
    w_base[0] += np.log(np.clip(p_tr.mean(axis=0), 1e-3, None))
    starts = [w_base]
    for _ in range(n_restarts - 1):
        starts.append(w_base + 0.5 * rng.normal(size=(d, 6)))

    best_res = None
    for w0 in starts:
        res = minimize(fg, w0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxfun": int(maxiter * 2)})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    assert best_res is not None
    return best_res.x.reshape(d, 6), {"converged": bool(best_res.success), "nit": int(best_res.nit), "fun": float(best_res.fun)}


def leak_free_gram_features(gram_all: np.ndarray, logits_base_all: np.ndarray, tr_idx: np.ndarray, te_idx: np.ndarray, k_pcs: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Compute leak-free kernel PCA coordinates strictly from training Gram submatrix G[tr, tr],
    plus standardized base expected score and entropy features."""
    n_tr = len(tr_idx)
    g_tt = gram_all[np.ix_(tr_idx, tr_idx)]
    row_m = g_tt.mean(axis=1, keepdims=True)
    col_m = g_tt.mean(axis=0, keepdims=True)
    tot_m = g_tt.mean()
    g_tt_c = g_tt - row_m - col_m + tot_m

    eigvals, eigvecs = np.linalg.eigh(g_tt_c)
    order = np.argsort(-eigvals)[:k_pcs]
    pos_vals = np.maximum(eigvals[order], 1e-8)
    alpha = eigvecs[:, order] / np.sqrt(pos_vals)[None, :]
    pcs_tr = g_tt_c @ alpha

    g_et = gram_all[np.ix_(te_idx, tr_idx)]
    g_et_c = g_et - g_et.mean(axis=1, keepdims=True) - col_m + tot_m
    pcs_te = g_et_c @ alpha

    # Standardize PCs on train only
    sd = pcs_tr.std(axis=0, keepdims=True) + 1e-8
    pcs_tr /= sd
    pcs_te /= sd

    q_b_tr = softmax(logits_base_all[tr_idx], axis=1)
    q_b_te = softmax(logits_base_all[te_idx], axis=1)
    s_tr = q_b_tr @ GRID_5
    s_te = q_b_te @ GRID_5
    ent_tr = -(q_b_tr * np.log(np.clip(q_b_tr, 1e-12, None))).sum(axis=1)
    ent_te = -(q_b_te * np.log(np.clip(q_b_te, 1e-12, None))).sum(axis=1)

    ms, sds = s_tr.mean(), s_tr.std() + 1e-8
    me, sde = ent_tr.mean(), ent_tr.std() + 1e-8

    phi_tr = np.column_stack([np.ones(n_tr), (s_tr - ms) / sds, (ent_tr - me) / sde, pcs_tr])
    phi_te = np.column_stack([np.ones(len(te_idx)), (s_te - ms) / sds, (ent_te - me) / sde, pcs_te])
    return phi_tr, phi_te


def audit_part_i_synthetic_and_historical() -> dict:
    """Programmatically verify numbers from synthetic (i) and historical real experiments."""
    exp01 = json.loads(
        (REPO_ROOT / "experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/results/summary.json").read_text()
    )
    exp02 = json.loads(
        (REPO_ROOT / "experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/results/results.json").read_text()
    )
    exp12 = json.loads(
        (REPO_ROOT / "experiments/archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026/results/results.json").read_text()
    )
    exp14_open = json.loads(
        (REPO_ROOT / "experiments/archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/results/open/results.json").read_text()
    )
    return {
        "synthetic_studies": {
            "prompt_changes_tested": False,
            "harness_changes_tested": False,
            "what_changed": "Fixed 2- or 9-candidate selection (exps 00, 01; 0 weights) and 78-weight linear-softmax basis fitting on scalar x in [-1,1] (exps 13, 06, 07, 08).",
            "exp01_1rating_ce_ident_rate": exp01["1"]["ce"]["oracle_in_tie_count"],
            "exp01_replications": exp01["1"]["ce"]["replications"],
        },
        "historical_real_studies": {
            "prompt_changes_tested": False,
            "harness_changes_tested": False,
            "neural_weight_updates_tested": False,
            "what_changed": "Simplex mixture weights (5 or 44 scalars summing to 1) or single-judge pick over fixed 3-sample scalar outputs.",
            "exp02_ce_minus_pearson_ce": exp02["ce_minus_pearson_refit"]["ce"],
            "exp12_corr_minus_ce_rps": exp12["paired"]["rps: corr_mixture_pearson minus ce_mixture"],
            "exp14_open39_corr_minus_ce_rps": exp14_open["paired"]["rps: corr_mixture_pearson minus ce_mixture"],
            "exp14_open39_ce_mixture_test_ce": exp14_open["metrics"]["ce"]["ce_mixture"],
            "exp14_open39_constant_test_ce": exp14_open["metrics"]["ce"]["constant"],
        },
    }


def compute_human_ceiling_and_info_gap(rows: list[dict]) -> dict:
    """Compute leave-one-rater-out human RPS/correlation ceiling and per-artifact Dirichlet-smoothed info gap G_i."""
    n = len(rows)
    p_emp = np.zeros((n, 6), dtype=np.float64)
    sd_emp = np.zeros(n, dtype=np.float64)
    info_gap_g = np.zeros(n, dtype=np.float64)
    loro_rps = []
    loro_mu = []
    loro_target = []

    for i, r in enumerate(rows):
        rats = [int(x) for x in r["ratings"]]
        m = len(rats)
        for v in rats:
            p_emp[i, v] += 1.0 / m
        sd_emp[i] = float(np.std(rats, ddof=0))
        # Dirichlet(alpha=0.25) smoothed distribution P_tilde vs maximum-entropy distribution matching E[R]
        p_sm = (np.array([rats.count(c) for c in range(6)], dtype=np.float64) + 0.25) / (m + 1.5)
        mu_sm = float(p_sm @ GRID_5)
        # Max-entropy distribution on {0..5} with mean mu_sm: q_me(r) propto exp(lambda * r)
        def me_obj(lmb: np.ndarray) -> float:
            z = float(lmb[0]) * GRID_5
            return float(logsumexp(z) - float(lmb[0]) * mu_sm)

        def me_grad(lmb: np.ndarray) -> np.ndarray:
            return np.array([softmax(float(lmb[0]) * GRID_5) @ GRID_5 - mu_sm])

        opt_me = minimize(me_obj, [0.0], method="BFGS", jac=me_grad, options={"gtol": 1e-10})
        z_me = float(opt_me.x[0]) * GRID_5
        q_me = softmax(z_me)
        assert abs(float(q_me @ GRID_5) - mu_sm) < 1e-7, "Maximum-entropy forecast must match the smoothed rating mean"
        # Information gap G_i = KL(p_sm || q_maxent_with_same_mean): excess non-mean distributional structure!
        info_gap_g[i] = float(np.sum(p_sm * (np.log(p_sm) - np.log(np.clip(q_me, 1e-12, None)))))

        if m >= 2:
            item_loro = []
            for k in range(m):
                others = rats[:k] + rats[k + 1 :]
                q_others = np.array([(others.count(c) + 0.1) / (len(others) + 0.6) for c in range(6)])
                cq = np.cumsum(q_others)[:5]
                ind = np.array([1.0 if rats[k] <= c else 0.0 for c in range(5)])
                item_loro.append(float(np.mean((cq - ind) ** 2)))
                loro_mu.append(float(np.mean(others)))
                loro_target.append(float(rats[k]))
            loro_rps.append(float(np.mean(item_loro)))

    return {
        "p_emp": p_emp,
        "sd_emp": sd_emp,
        "info_gap_g": info_gap_g,
        "n_disagree_items": int(np.sum(sd_emp > 0)),
        "n_unanimous_items": int(np.sum(sd_emp == 0)),
        "human_loro_mean_rps": float(np.mean(loro_rps)),
        "human_loro_pair_pearson": float(np.corrcoef(loro_mu, loro_target)[0, 1]),
        "human_loro_pair_spearman": float(spearmanr(loro_mu, loro_target).statistic),
    }


def generate_grouped_splits(fam_idx: np.ndarray, n_fams: int = 19, n_train_fams: int = 15, n_splits: int = 200, seed: int = 20261116) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate 200 stratified/grouped task-family splits where every split holds out (19 - n_train_fams) test families."""
    rng = np.random.default_rng(seed + 1000 * n_train_fams)
    n_test_fams = min(4, n_fams - n_train_fams)
    splits = []
    for _ in range(n_splits):
        perm = rng.permutation(n_fams)
        te_f = set(perm[:n_test_fams].tolist())
        tr_f = set(perm[n_test_fams : n_test_fams + n_train_fams].tolist())
        tr = np.array([i for i, f in enumerate(fam_idx) if f in tr_f], dtype=int)
        te = np.array([i for i, f in enumerate(fam_idx) if f in te_f], dtype=int)
        splits.append((tr, te))
    return splits


def evaluate_pool_across_splits(
    q_pool: np.ndarray,
    p_emp: np.ndarray,
    fam_idx: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    include_val_tuned: bool = True,
) -> dict[str, np.ndarray]:
    """Evaluate single-judge selection & simplex mixtures across grouped task-family splits.
    Returns per-artifact average test loss/prediction matrices across splits where each artifact was held out."""
    n, j_cnt, _ = q_pool.shape
    qf_pool = smooth_floor(q_pool, FLOOR)

    methods = [
        "constant",
        "equal_mixture",
        "corr_single_train",      # ② single
        "ce_single_train",        # ④ single (CE)
        "rps_single_train",       # ④ single (RPS)
        "corr_mix_pearson",       # ② Pearson mixture
        "corr_mix_spearman",      # ② Spearman mixture
        "ce_mix_train",           # ④ CE mixture
        "rps_mix_train",          # ④ RPS mixture
    ]
    if include_val_tuned:
        methods.extend(["corr_val_tuned_1", "loss_val_tuned_3"])

    sum_q = {m: np.zeros((n, 6), dtype=np.float64) for m in methods}
    sum_rps = {m: np.zeros(n, dtype=np.float64) for m in methods}
    sum_ce = {m: np.zeros(n, dtype=np.float64) for m in methods}
    sum_tvd = {m: np.zeros(n, dtype=np.float64) for m in methods}
    cnt = np.zeros(n, dtype=np.float64)
    cert_fails = {"ce": 0, "rps": 0}

    for s_idx, (tr, te) in enumerate(splits):
        rng = np.random.default_rng(70000 + s_idx)
        pt = p_emp[tr]
        qt = qf_pool[tr]
        hm = pt @ GRID_5
        mu_qt = qt @ GRID_5

        marg = smooth_floor(pt.mean(axis=0), FLOOR)
        preds: dict[str, np.ndarray] = {}
        preds["constant"] = np.tile(marg, (len(te), 1))
        preds["equal_mixture"] = qf_pool[te].mean(axis=1)

        pe_scores = np.array([np.corrcoef(mu_qt[:, j], hm)[0, 1] if mu_qt[:, j].std() > 1e-12 else -2.0 for j in range(j_cnt)])
        preds["corr_single_train"] = qf_pool[te, int(np.argmax(pe_scores))]

        ce_scores = np.array([-(pt * np.log(qt[:, j])).sum(axis=1).mean() for j in range(j_cnt)])
        preds["ce_single_train"] = qf_pool[te, int(np.argmin(ce_scores))]

        rps_scores = np.array([item_losses(qt[:, j], pt)["rps"].mean() for j in range(j_cnt)])
        preds["rps_single_train"] = qf_pool[te, int(np.argmin(rps_scores))]

        w_pe, _ = fit_simplex_pearson(qt, pt)
        preds["corr_mix_pearson"] = np.einsum("njr,j->nr", qf_pool[te], w_pe)

        w_sp = fit_simplex_spearman(qt, pt, rng, draws=1500)
        preds["corr_mix_spearman"] = np.einsum("njr,j->nr", qf_pool[te], w_sp)

        w_ce, rec_ce = fit_simplex_ce(qt, pt)
        if rec_ce["status"] != "optimal":
            cert_fails["ce"] += 1
        preds["ce_mix_train"] = np.einsum("njr,j->nr", qf_pool[te], w_ce)

        w_rps, rec_rps = fit_simplex_rps(qt, pt)
        if rec_rps["status"] != "optimal":
            cert_fails["rps"] += 1
        preds["rps_mix_train"] = np.einsum("njr,j->nr", qf_pool[te], w_rps)

        if include_val_tuned:
            # Inner 3-fold family split of tr for Formulations ① and ③ (tuning shrinkage lambda toward train marginal!)
            tr_fams = sorted(set(fam_idx[tr].tolist()))
            perm_in = [tr_fams[i] for i in rng.permutation(len(tr_fams))]
            val_f = set(perm_in[: max(2, len(tr_fams) // 4)])
            in_tr = np.array([i for i in tr if fam_idx[i] not in val_f], dtype=int)
            in_va = np.array([i for i in tr if fam_idx[i] in val_f], dtype=int)

            marg_in = smooth_floor(p_emp[in_tr].mean(axis=0), FLOOR)
            w_pe_in, _ = fit_simplex_pearson(qf_pool[in_tr], p_emp[in_tr])
            q_pe_va = np.einsum("njr,j->nr", qf_pool[in_va], w_pe_in)
            j_pe_in = int(np.argmax([np.corrcoef(qf_pool[in_tr, j] @ GRID_5, p_emp[in_tr] @ GRID_5)[0, 1] if (qf_pool[in_tr, j] @ GRID_5).std() > 1e-12 else -2.0 for j in range(j_cnt)]))
            # ① chooses between single best Pearson judge and Pearson mixture by validation Pearson
            r_mix_va = np.corrcoef(q_pe_va @ GRID_5, p_emp[in_va] @ GRID_5)[0, 1] if (q_pe_va @ GRID_5).std() > 1e-12 else -2.0
            r_sng_va = np.corrcoef(qf_pool[in_va, j_pe_in] @ GRID_5, p_emp[in_va] @ GRID_5)[0, 1] if (qf_pool[in_va, j_pe_in] @ GRID_5).std() > 1e-12 else -2.0
            preds["corr_val_tuned_1"] = preds["corr_mix_pearson"] if r_mix_va >= r_sng_va else preds["corr_single_train"]

            # ③ tunes convex shrinkage (1 - alpha) * Mix + alpha * Marginal over alpha in {0.0, 0.05, 0.15, 0.30, 0.50} & CE vs RPS mix on inner val RPS
            w_ce_in, _ = fit_simplex_ce(qf_pool[in_tr], p_emp[in_tr])
            w_rps_in, _ = fit_simplex_rps(qf_pool[in_tr], p_emp[in_tr])
            best_val_rps = float("inf")
            best_cfg = ("rps", 0.05)
            for base_kind, w_in in (("ce", w_ce_in), ("rps", w_rps_in)):
                q_va_base = np.einsum("njr,j->nr", qf_pool[in_va], w_in)
                for alpha in (0.0, 0.05, 0.15, 0.30, 0.50):
                    q_cand = (1.0 - alpha) * q_va_base + alpha * marg_in[None, :]
                    vr = float(item_losses(q_cand, p_emp[in_va])["rps"].mean())
                    if vr < best_val_rps:
                        best_val_rps = vr
                        best_cfg = (base_kind, alpha)
            base_te = preds["ce_mix_train"] if best_cfg[0] == "ce" else preds["rps_mix_train"]
            preds["loss_val_tuned_3"] = (1.0 - best_cfg[1]) * base_te + best_cfg[1] * marg[None, :]

        cnt[te] += 1.0
        for m in methods:
            ls = item_losses(preds[m], p_emp[te])
            sum_q[m][te] += smooth_floor(preds[m], FLOOR)
            sum_rps[m][te] += ls["rps"]
            sum_ce[m][te] += ls["ce"]
            sum_tvd[m][te] += ls["tvd"]

    assert np.all(cnt > 0), "Every artifact must be held out at least once!"
    out = {"cert_fails": cert_fails, "per_item_rps": {}, "per_item_ce": {}, "per_item_tvd": {}, "avg_q": {}}
    for m in methods:
        out["per_item_rps"][m] = sum_rps[m] / cnt
        out["per_item_ce"][m] = sum_ce[m] / cnt
        out["per_item_tvd"][m] = sum_tvd[m] / cnt
        out["avg_q"][m] = sum_q[m] / cnt[:, None]
    return out


def summarize_split_eval(eval_out: dict, p_emp: np.ndarray, fam_idx: np.ndarray) -> dict:
    hm5 = p_emp @ GRID_5
    methods_summary = {}
    for m, rps_vec in eval_out["per_item_rps"].items():
        ce_vec = eval_out["per_item_ce"][m]
        tvd_vec = eval_out["per_item_tvd"][m]
        q_avg = eval_out["avg_q"][m]
        ls_avg = item_losses(q_avg, p_emp)
        decomp = mse_decomposition(ls_avg["mu_05"], hm5)
        methods_summary[m] = {
            "rps": float(rps_vec.mean()),
            "ce": float(ce_vec.mean()),
            "tvd": float(tvd_vec.mean()),
            "brier": float(ls_avg["brier"].mean()),
            **decomp,
        }

    contrasts = {}
    pairs = [
        ("corr_mix_pearson", "ce_mix_train"),
        ("corr_mix_pearson", "rps_mix_train"),
        ("corr_mix_spearman", "ce_mix_train"),
        ("corr_single_train", "ce_single_train"),
        ("corr_single_train", "rps_single_train"),
        ("ce_mix_train", "constant"),
        ("rps_mix_train", "constant"),
    ]
    if "loss_val_tuned_3" in eval_out["per_item_rps"]:
        pairs.extend(
            [
                ("corr_val_tuned_1", "loss_val_tuned_3"),
                ("ce_mix_train", "loss_val_tuned_3"),
                ("rps_mix_train", "loss_val_tuned_3"),
                ("loss_val_tuned_3", "constant"),
            ]
        )
    for m_a, m_b in pairs:
        contrasts[f"rps: {m_a} minus {m_b}"] = family_bootstrap_and_signflip(
            eval_out["per_item_rps"][m_a] - eval_out["per_item_rps"][m_b], fam_idx
        )
        contrasts[f"ce: {m_a} minus {m_b}"] = family_bootstrap_and_signflip(
            eval_out["per_item_ce"][m_a] - eval_out["per_item_ce"][m_b], fam_idx
        )
        contrasts[f"tvd: {m_a} minus {m_b}"] = family_bootstrap_and_signflip(
            eval_out["per_item_tvd"][m_a] - eval_out["per_item_tvd"][m_b], fam_idx
        )
    return {"methods": methods_summary, "contrasts": contrasts, "cert_fails": eval_out["cert_fails"]}


def main() -> None:
    global INDIV_ITEM_IDX, INDIV_RATINGS
    raw_dir = EXPT_DIR / "results/raw_mercury2"
    manifest = json.loads((EXPT_DIR / "results/input_manifest.json").read_text(encoding="utf-8"))
    ds_bytes = DATASET_PATH.read_bytes()
    ds_sha256 = hashlib.sha256(ds_bytes).hexdigest()
    ds = json.loads(ds_bytes.decode("utf-8"))
    rows = ds["rows"]
    n_items = len(rows)
    INDIV_ITEM_IDX = np.array([i for i, r in enumerate(rows) for _ in r["ratings"]], dtype=int)
    INDIV_RATINGS = np.array([float(x) for r in rows for x in r["ratings"]], dtype=np.float64)
    item_to_idx = {r["item"]: i for i, r in enumerate(rows)}
    unique_fams = sorted({r["family"] for r in rows})
    fam_idx = np.array([unique_fams.index(r["family"]) for r in rows], dtype=int)
    ordinary_mask = np.array([bool(r["ordinary"]) for r in rows], dtype=bool)

    ceil_info = compute_human_ceiling_and_info_gap(rows)
    p_emp = ceil_info["p_emp"]
    hm5 = p_emp @ GRID_5

    # Load all mercury2 model JSONL files
    model_keys = [
        "q3_30b_a3b",
        "q3_14b_think",
        "q3_14b_nothink",
        "q25c_14b",
        "q25_14b_awq",
        "ds_r1_qwen_7b",
        "q3_4b_2507",
        "llama31_8b",
        "hermes3_8b",
        "olmo3_7b",
    ]
    records_by_model: dict[str, list[dict]] = {}
    for mk in model_keys:
        p_f = raw_dir / f"{mk}.jsonl"
        if p_f.exists():
            records_by_model[mk] = [json.loads(line) for line in p_f.read_text(encoding="utf-8").splitlines() if line.strip()]

    print(f"Loaded mercury2 JSONLs for {len(records_by_model)} models: {list(records_by_model.keys())}")

    # ---- 1. Assemble Forecasts & Sanity Controls ----
    forecasts_logprob: dict[str, np.ndarray] = {}
    forecasts_verbal: dict[str, np.ndarray] = {}
    parse_stats: dict[str, dict] = {}
    sanity_controls: dict[str, dict] = {}
    repeat_stability: dict[str, dict] = {}

    cached_fc_path = EXPT_DIR / "results/open_weight_forecasts.json"
    if not records_by_model and cached_fc_path.exists():
        cached_fc = json.loads(cached_fc_path.read_text(encoding="utf-8"))
        forecasts_logprob = {k: np.array(v, dtype=np.float64) for k, v in cached_fc["logprob_forecasts"].items()}
        forecasts_verbal = {k: np.array(v, dtype=np.float64) for k, v in cached_fc["verbal_forecasts"].items()}
        parse_stats = cached_fc.get("parse_stats", {})
        sanity_controls = cached_fc.get("sanity_controls", {})
        repeat_stability = cached_fc.get("repeat_stability", {})
        print("Loaded forecasts from committed results/open_weight_forecasts.json fallback.")

    for mk, recs in records_by_model.items():
        # Group main + harness cells
        by_tag: dict[str, list[dict]] = {}
        ctl_recs = []
        rep_recs: dict[int, dict[str, list[float]]] = {1: {}, 2: {}, 3: {}}
        for r in recs:
            knd = r["kind"]
            if knd.startswith("control_"):
                ctl_recs.append(r)
            elif knd.startswith("repeat_"):
                r_id = int(knd.split("_")[1])
                q_use = r["q_verbal"] if r["q_verbal"] is not None else r["q_logprob"]
                if q_use is not None:
                    rep_recs[r_id][r["item"]] = q_use
            elif knd == "main_prompt":
                by_tag.setdefault(f"{mk}__{r['prompt_id']}", []).append(r)
            elif knd == "harness_H3_ast":
                by_tag.setdefault(f"{mk}__H3_ast", []).append(r)
            elif knd == "harness_H4_two_call":
                by_tag.setdefault(f"{mk}__H4_two_call", []).append(r)

        for tag, rlist in sorted(by_tag.items()):
            q_lp = np.full((n_items, 6), 1.0 / 6.0, dtype=np.float64)
            q_vb = np.full((n_items, 6), 1.0 / 6.0, dtype=np.float64)
            ok_lp, ok_vb = 0, 0
            for r in rlist:
                idx = item_to_idx[r["item"]]
                if r["q_logprob"] is not None:
                    q_lp[idx] = r["q_logprob"]
                    ok_lp += 1
                if r["q_verbal"] is not None:
                    q_vb[idx] = r["q_verbal"]
                    ok_vb += 1
            forecasts_logprob[tag] = q_lp
            forecasts_verbal[tag] = q_vb
            parse_stats[tag] = {"n_scheduled": n_items, "logprob_ok": ok_lp, "verbal_ok": ok_vb}

        if ctl_recs:
            by_ck: dict[str, list[dict]] = {}
            for r in ctl_recs:
                by_ck.setdefault(r["kind"], []).append(r)
            c_sum = {}
            for ck, clist in by_ck.items():
                mus_lp = [float(np.dot(r["q_logprob"], GRID_5)) for r in clist if r["q_logprob"] is not None]
                mus_vb = [float(np.dot(r["q_verbal"], GRID_5)) for r in clist if r["q_verbal"] is not None]
                entry = {
                    "n": len(clist),
                    "mean_score_logprob_05": float(np.mean(mus_lp)) if mus_lp else None,
                    "mean_score_verbal_05": float(np.mean(mus_vb)) if mus_vb else None,
                }
                if ck == "control_label_in_prompt":
                    targets = [int(r["expected_synthetic_label"]) for r in clist]
                    preds_vb = [float(np.dot(r["q_verbal"], GRID_5)) if r["q_verbal"] is not None else float(r["scalar_score"] or 2.5) for r in clist]
                    preds_lp = [float(np.dot(r["q_logprob"], GRID_5)) if r["q_logprob"] is not None else 2.5 for r in clist]
                    modal_matches = sum(
                        1 for r, t in zip(clist, targets)
                        if (r["q_verbal"] is not None and int(np.argmax(r["q_verbal"])) == t) or r["scalar_score"] == t
                    )
                    entry["targets"] = targets
                    entry["modal_exact_matches"] = f"{modal_matches}/{len(targets)}"
                    entry["pearson_verbal"] = float(np.corrcoef(preds_vb, targets)[0, 1]) if np.std(preds_vb) > 1e-9 else 0.0
                    entry["pearson_logprob"] = float(np.corrcoef(preds_lp, targets)[0, 1]) if np.std(preds_lp) > 1e-9 else 0.0
                c_sum[ck] = entry
            sanity_controls[mk] = c_sum

        if rep_recs[1] and rep_recs[2] and rep_recs[3]:
            common_items = sorted(set(rep_recs[1]) & set(rep_recs[2]) & set(rep_recs[3]))
            if common_items:
                q3_arr = np.array([[rep_recs[r_i][it_k] for r_i in (1, 2, 3)] for it_k in common_items])  # n x 3 x 6
                mu3 = q3_arr @ GRID_5  # n x 3
                rms_sd_05 = float(np.sqrt(np.mean(np.var(mu3, axis=1, ddof=1))))
                rms_prob_sd = float(np.sqrt(np.mean(np.var(q3_arr, axis=1, ddof=1))))
                repeat_stability[mk] = {
                    "n_items": len(common_items),
                    "repeats_per_item": 3,
                    "rms_expected_score_sd_05": rms_sd_05,
                    "rms_expected_score_sd_01": rms_sd_05 / 5.0,
                    "rms_bin_prob_sd": rms_prob_sd,
                }
                # Also build H6 multi-sample ensemble forecast for this model!
                if len(common_items) == n_items:
                    q_ens = np.zeros((n_items, 6), dtype=np.float64)
                    for it_k in common_items:
                        idx = item_to_idx[it_k]
                        q_ens[idx] = (
                            0.5 * forecasts_logprob[f"{mk}__P2_disagreement_calibrated"][idx]
                            + 0.5 * np.mean([rep_recs[1][it_k], rep_recs[2][it_k], rep_recs[3][it_k]], axis=0)
                        )
                    forecasts_logprob[f"{mk}__H6_multi_sample_ensemble"] = q_ens

    # Theorem-count syntactic baseline (linearly interpolating adjacent integer categories per TTJ exp 02 spec!)
    q_thm_baseline = np.zeros((n_items, 6), dtype=np.float64)
    for a in manifest["artifacts"]:
        s5 = 5.0 * float(a["theorem_count_ratio_0_1"])
        lo_i = min(4, int(math.floor(s5)))
        frac = s5 - lo_i
        q_thm_baseline[a["index"], lo_i] = 1.0 - frac
        q_thm_baseline[a["index"], lo_i + 1] = frac
    thm_losses = item_losses(q_thm_baseline, p_emp)
    thm_summary = {
        "rps": float(thm_losses["rps"].mean()),
        "ce": float(thm_losses["ce"].mean()),
        "tvd": float(thm_losses["tvd"].mean()),
        **mse_decomposition(thm_losses["mu_05"], hm5),
    }

    # In-sample single-judge summary table (explicitly labeled in-sample descriptive!)
    per_judge_insample = {}
    for tag in sorted(forecasts_logprob):
        for rep_name, q_m in (("logprob", forecasts_logprob[tag]), ("verbal", forecasts_verbal.get(tag))):
            if q_m is None:
                continue
            ls = item_losses(q_m, p_emp)
            dec = mse_decomposition(ls["mu_05"], hm5)
            per_judge_insample[f"{tag}__{rep_name}"] = {
                "rps": float(ls["rps"].mean()),
                "ce": float(ls["ce"].mean()),
                "tvd": float(ls["tvd"].mean()),
                "pearson": dec["pearson"],
                "spearman": dec["spearman"],
                "judge_mean_05": dec["judge_mean_05"],
                "bias_05": dec["bias_05"],
            }

    splits_200 = generate_grouped_splits(fam_idx, n_fams=19, n_train_fams=15, n_splits=200, seed=20261116)

    # ---- 2. Representation Bridge: Archived Exp 14 (3-sample scalar) vs Native Verbal vs Native Logprob ----
    exp14_raw = json.loads(EXP14_SCORES_PATH.read_text(encoding="utf-8"))
    # Reconstruct 3-sample histograms for the 39 open-weight judges in archived Exp 14
    open39_names = [k for k in exp14_raw["candidate_order"] if k not in ("H0_sol", "H0_opus", "H1_sol", "H2_sol", "H2_opus")]
    item_order_14 = [x["item"] if isinstance(x, dict) else x for x in exp14_raw["items"]]
    q_exp14_39 = np.zeros((n_items, len(open39_names), 6), dtype=np.float64)
    for j_i, jn in enumerate(open39_names):
        sc_map = dict(zip(item_order_14, exp14_raw["outputs"][jn]))
        for it_str, idx in item_to_idx.items():
            s3 = sc_map[it_str]
            if s3:
                for sv in s3:
                    q_exp14_39[idx, j_i, int(sv)] += 1.0 / len(s3)
            else:
                q_exp14_39[idx, j_i, :] = 1.0 / 6.0

    core_p012_tags = [
        f"{mk}__{pid}"
        for mk in model_keys
        for pid in ("P0_human_mirror", "P1_conventions", "P2_disagreement_calibrated")
        if f"{mk}__{pid}" in forecasts_logprob
    ]
    q_pool_verbal = np.stack([forecasts_verbal[t] for t in core_p012_tags], axis=1)
    q_pool_logprob = np.stack([forecasts_logprob[t] for t in core_p012_tags], axis=1)

    eval_exp14_39 = evaluate_pool_across_splits(q_exp14_39, p_emp, fam_idx, splits_200, include_val_tuned=True)
    eval_verbal_pool = evaluate_pool_across_splits(q_pool_verbal, p_emp, fam_idx, splits_200, include_val_tuned=True)
    eval_logprob_pool = evaluate_pool_across_splits(q_pool_logprob, p_emp, fam_idx, splits_200, include_val_tuned=True)

    rep_bridge = {
        "exp14_3sample_scalar_39judges": summarize_split_eval(eval_exp14_39, p_emp, fam_idx),
        "exp09_verbal_6prob_pool": summarize_split_eval(eval_verbal_pool, p_emp, fam_idx),
        "exp09_logprob_6prob_pool": summarize_split_eval(eval_logprob_pool, p_emp, fam_idx),
        "direct_representation_contrasts": {
            "ce_mix_test_ce: exp14_3sample minus exp09_logprob": family_bootstrap_and_signflip(
                eval_exp14_39["per_item_ce"]["ce_mix_train"] - eval_logprob_pool["per_item_ce"]["ce_mix_train"], fam_idx
            ),
            "ce_mix_test_rps: exp14_3sample minus exp09_logprob": family_bootstrap_and_signflip(
                eval_exp14_39["per_item_rps"]["ce_mix_train"] - eval_logprob_pool["per_item_rps"]["ce_mix_train"], fam_idx
            ),
        },
    }

    # ---- 3. ii.a Prompt Optimization (Candidate Pool P0..P5 + Blind Meta-Prompt Search on Qwen3-30B-A3B) ----
    p0_to_p5_models = [mk for mk in ("q3_30b_a3b", "q25c_14b", "q3_4b_2507") if f"{mk}__P5_cdf_threshold" in forecasts_logprob]
    prompt_opt_by_model = {}
    for mk in p0_to_p5_models:
        p_tags = [f"{mk}__{pid}" for pid in manifest["prompt_ids"]]
        # Evaluate prompt selection on verbalized probabilities (where prompt instructions directly govern calibration!) and logprobs
        for mode_name, store in (("verbal", forecasts_verbal), ("logprob", forecasts_logprob)):
            qp = np.stack([store[t] for t in p_tags], axis=1)
            ev = evaluate_pool_across_splits(qp, p_emp, fam_idx, splits_200, include_val_tuned=True)
            prompt_opt_by_model[f"{mk}__{mode_name}"] = summarize_split_eval(ev, p_emp, fam_idx)

    # Multi-model x 6-prompt pool
    all_prompt_tags = [f"{mk}__{pid}" for mk in p0_to_p5_models for pid in manifest["prompt_ids"]]
    ev_prompts_verbal = evaluate_pool_across_splits(
        np.stack([forecasts_verbal[t] for t in all_prompt_tags], axis=1), p_emp, fam_idx, splits_200, include_val_tuned=True
    )
    ev_prompts_logprob = evaluate_pool_across_splits(
        np.stack([forecasts_logprob[t] for t in all_prompt_tags], axis=1), p_emp, fam_idx, splits_200, include_val_tuned=True
    )

    # Meta-prompt search evaluation (if meta_prompt_opt_q3_30b.json exists)
    meta_prompt_summary = None
    meta_file = raw_dir / "meta_prompt_opt_q3_30b.json"
    if meta_file.exists():
        meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
        q_seed = np.zeros((n_items, 6), dtype=np.float64)
        q_corr_tr = np.zeros((n_items, 6), dtype=np.float64)
        q_corr_va = np.zeros((n_items, 6), dtype=np.float64)
        q_ce_tr = np.zeros((n_items, 6), dtype=np.float64)
        q_ce_va = np.zeros((n_items, 6), dtype=np.float64)
        for fold in meta_data["folds"]:
            te_i = np.array(fold["test_indices"], dtype=int)
            tc = fold["tracks"]["corr"]
            tl = fold["tracks"]["ce"]
            q_seed[te_i] = np.array(tc["test_forecasts_by_id"]["cand_0_seed_P0"])
            q_corr_tr[te_i] = np.array(tc["test_forecasts_by_id"][tc["selected_train_id"]])
            q_corr_va[te_i] = np.array(tc["test_forecasts_by_id"][tc["selected_val_id"]])
            q_ce_tr[te_i] = np.array(tl["test_forecasts_by_id"][tl["selected_train_id"]])
            q_ce_va[te_i] = np.array(tl["test_forecasts_by_id"][tl["selected_val_id"]])

        ls_seed = item_losses(q_seed, p_emp)
        ls_c2 = item_losses(q_corr_tr, p_emp)
        ls_c1 = item_losses(q_corr_va, p_emp)
        ls_l4 = item_losses(q_ce_tr, p_emp)
        ls_l3 = item_losses(q_ce_va, p_emp)
        meta_prompt_summary = {
            "model": meta_data["model"],
            "split_design": "3-fold grouped task-family out-of-fold evaluation covering all 19 families / 75 artifacts",
            "arms": {
                "seed_P0": {"rps": float(ls_seed["rps"].mean()), "ce": float(ls_seed["ce"].mean()), "tvd": float(ls_seed["tvd"].mean()), **mse_decomposition(ls_seed["mu_05"], hm5)},
                "corr_prompt_train_2": {"rps": float(ls_c2["rps"].mean()), "ce": float(ls_c2["ce"].mean()), "tvd": float(ls_c2["tvd"].mean()), **mse_decomposition(ls_c2["mu_05"], hm5)},
                "corr_prompt_val_1": {"rps": float(ls_c1["rps"].mean()), "ce": float(ls_c1["ce"].mean()), "tvd": float(ls_c1["tvd"].mean()), **mse_decomposition(ls_c1["mu_05"], hm5)},
                "ce_prompt_train_4": {"rps": float(ls_l4["rps"].mean()), "ce": float(ls_l4["ce"].mean()), "tvd": float(ls_l4["tvd"].mean()), **mse_decomposition(ls_l4["mu_05"], hm5)},
                "ce_prompt_val_3": {"rps": float(ls_l3["rps"].mean()), "ce": float(ls_l3["ce"].mean()), "tvd": float(ls_l3["tvd"].mean()), **mse_decomposition(ls_l3["mu_05"], hm5)},
            },
            "contrasts": {
                "rps: corr_prompt_train_2 minus ce_prompt_train_4": family_bootstrap_and_signflip(ls_c2["rps"] - ls_l4["rps"], fam_idx),
                "ce: corr_prompt_train_2 minus ce_prompt_train_4": family_bootstrap_and_signflip(ls_c2["ce"] - ls_l4["ce"], fam_idx),
                "rps: corr_prompt_val_1 minus ce_prompt_val_3": family_bootstrap_and_signflip(ls_c1["rps"] - ls_l3["rps"], fam_idx),
                "ce: corr_prompt_val_1 minus ce_prompt_val_3": family_bootstrap_and_signflip(ls_c1["ce"] - ls_l3["ce"], fam_idx),
                "rps: ce_prompt_train_4 minus ce_prompt_val_3": family_bootstrap_and_signflip(ls_l4["rps"] - ls_l3["rps"], fam_idx),
            },
            "fold_selected_prompts": [
                {
                    "fold": f["fold"],
                    "corr_train_id": f["tracks"]["corr"]["selected_train_id"],
                    "corr_val_id": f["tracks"]["corr"]["selected_val_id"],
                    "ce_train_id": f["tracks"]["ce"]["selected_train_id"],
                    "ce_val_id": f["tracks"]["ce"]["selected_val_id"],
                }
                for f in meta_data["folds"]
            ],
        }

    # ---- 4. ii.b Harness Optimization (H1..H6 + Temperature/Prior Harness Grid + 2x2 Prompt x Harness Factorial) ----
    harness_candidates: list[tuple[str, np.ndarray]] = []
    for mk in ("q3_30b_a3b", "q25c_14b", "q3_4b_2507"):
        if f"{mk}__P1_conventions" in forecasts_verbal:
            harness_candidates.append((f"{mk}__H1_single_verbal", forecasts_verbal[f"{mk}__P1_conventions"]))
            harness_candidates.append((f"{mk}__H2_single_logprob", forecasts_logprob[f"{mk}__P1_conventions"]))
        if f"{mk}__H3_ast" in forecasts_logprob:
            harness_candidates.append((f"{mk}__H3_ast_logprob", forecasts_logprob[f"{mk}__H3_ast"]))
            harness_candidates.append((f"{mk}__H3_ast_verbal", forecasts_verbal[f"{mk}__H3_ast"]))
        if f"{mk}__H4_two_call" in forecasts_logprob:
            harness_candidates.append((f"{mk}__H4_two_call_logprob", forecasts_logprob[f"{mk}__H4_two_call"]))
            harness_candidates.append((f"{mk}__H4_two_call_verbal", forecasts_verbal[f"{mk}__H4_two_call"]))
        if f"{mk}__H6_multi_sample_ensemble" in forecasts_logprob:
            harness_candidates.append((f"{mk}__H6_multi_sample_ensemble", forecasts_logprob[f"{mk}__H6_multi_sample_ensemble"]))
    for mk_th in ("q3_14b_think", "q3_14b_nothink", "ds_r1_qwen_7b"):
        if f"{mk_th}__P1_conventions" in forecasts_logprob:
            harness_candidates.append((f"{mk_th}__H5_logprob", forecasts_logprob[f"{mk_th}__P1_conventions"]))
            harness_candidates.append((f"{mk_th}__H5_verbal", forecasts_verbal[f"{mk_th}__P1_conventions"]))
        if f"{mk_th}__H3_ast" in forecasts_logprob:
            harness_candidates.append((f"{mk_th}__H3_ast_logprob", forecasts_logprob[f"{mk_th}__H3_ast"]))

    # Add temperature-scaled + prior-smoothed harness configurations (T in {0.7, 1.0, 1.8, 3.0}, alpha in {0.0, 0.08})
    expanded_harnesses: list[tuple[str, np.ndarray]] = list(harness_candidates)
    for h_name, q_h in harness_candidates[:10]:
        logits_h = np.log(smooth_floor(q_h, 1e-6))
        for temp in (0.7, 1.8, 3.0):
            q_t = softmax(logits_h / temp, axis=1)
            expanded_harnesses.append((f"{h_name}__T{temp}", q_t))

    q_harness_pool = np.stack([q_h for _, q_h in expanded_harnesses], axis=1)
    ev_harness = evaluate_pool_across_splits(q_harness_pool, p_emp, fam_idx, splits_200, include_val_tuned=True)
    harness_opt_summary = summarize_split_eval(ev_harness, p_emp, fam_idx)
    harness_opt_summary["n_harness_configurations"] = len(expanded_harnesses)

    # Thinking vs Non-Thinking isolated ablation on Qwen3-14B
    thinking_ablation = None
    if "q3_14b_think__P1_conventions" in forecasts_logprob and "q3_14b_nothink__P1_conventions" in forecasts_logprob:
        ls_th = item_losses(forecasts_logprob["q3_14b_think__P1_conventions"], p_emp)
        ls_nt = item_losses(forecasts_logprob["q3_14b_nothink__P1_conventions"], p_emp)
        thinking_ablation = {
            "q3_14b_think_logprob": {"rps": float(ls_th["rps"].mean()), "ce": float(ls_th["ce"].mean()), **mse_decomposition(ls_th["mu_05"], hm5)},
            "q3_14b_nothink_logprob": {"rps": float(ls_nt["rps"].mean()), "ce": float(ls_nt["ce"].mean()), **mse_decomposition(ls_nt["mu_05"], hm5)},
            "rps_nothink_minus_think": family_bootstrap_and_signflip(ls_nt["rps"] - ls_th["rps"], fam_idx),
        }

    # 2x2 Factorial: (Prompt Chosen by Corr vs CE) x (Harness/Mixture Calibration Chosen by Corr vs CE)
    p_corr_4 = {"corr_prompt__corr_harness": np.zeros(n_items), "corr_prompt__ce_harness": np.zeros(n_items),
                "ce_prompt__corr_harness": np.zeros(n_items), "ce_prompt__ce_harness": np.zeros(n_items)}
    c_corr_4 = {k: np.zeros(n_items) for k in p_corr_4}
    cnt_2x2 = np.zeros(n_items)
    if p0_to_p5_models:
        mk0 = p0_to_p5_models[0]
        p_ids = manifest["prompt_ids"]
        q_vb_m0 = np.stack([smooth_floor(forecasts_verbal[f"{mk0}__{pid}"], FLOOR) for pid in p_ids], axis=1)
        q_lp_m0 = np.stack([smooth_floor(forecasts_logprob[f"{mk0}__{pid}"], FLOOR) for pid in p_ids], axis=1)
        for tr, te in splits_200:
            pt = p_emp[tr]
            hm_tr = pt @ GRID_5
            # Step 1: Choose prompt pid on verbal outputs by Train Pearson vs Train CE
            pe_p = [np.corrcoef(q_vb_m0[tr, j] @ GRID_5, hm_tr)[0, 1] if (q_vb_m0[tr, j] @ GRID_5).std() > 1e-12 else -2.0 for j in range(len(p_ids))]
            ce_p = [-(pt * np.log(q_vb_m0[tr, j])).sum(axis=1).mean() for j in range(len(p_ids))]
            j_p_corr = int(np.argmax(pe_p))
            j_p_ce = int(np.argmin(ce_p))

            # Step 2: Given chosen prompt j, build harness family = {raw verbal, raw logprob, T=1.8 logprob, T=3.0 logprob, train_marginal}
            marg_tr = smooth_floor(pt.mean(axis=0), FLOOR)
            for p_label, j_sel in (("corr_prompt", j_p_corr), ("ce_prompt", j_p_ce)):
                h_tr_cands = []
                h_te_cands = []
                for q_src in (q_vb_m0[:, j_sel], q_lp_m0[:, j_sel]):
                    h_tr_cands.append(q_src[tr])
                    h_te_cands.append(q_src[te])
                    lg = np.log(q_src)
                    for temp in (1.8, 3.0):
                        h_tr_cands.append(softmax(lg[tr] / temp, axis=1))
                        h_te_cands.append(softmax(lg[te] / temp, axis=1))
                # Corr harness choice: Pearson simplex mixture over {verbal, logprob, T=1.8, T=3.0} (marginal has 0 variance so ignored by Pearson!)
                qh_tr = np.stack(h_tr_cands, axis=1)
                qh_te = np.stack(h_te_cands, axis=1)
                w_hc, _ = fit_simplex_pearson(qh_tr, pt)
                q_te_hc = np.einsum("njr,j->nr", qh_te, w_hc)
                # CE harness choice: CE simplex mixture over {verbal, logprob, T=1.8, T=3.0, train_marginal}
                qh_tr_ce = np.concatenate([qh_tr, np.tile(marg_tr, (len(tr), 1))[:, None, :]], axis=1)
                qh_te_ce = np.concatenate([qh_te, np.tile(marg_tr, (len(te), 1))[:, None, :]], axis=1)
                w_hl, _ = fit_simplex_ce(qh_tr_ce, pt)
                q_te_hl = np.einsum("njr,j->nr", qh_te_ce, w_hl)

                ls_hc = item_losses(q_te_hc, p_emp[te])
                ls_hl = item_losses(q_te_hl, p_emp[te])
                p_corr_4[f"{p_label}__corr_harness"][te] += ls_hc["rps"]
                c_corr_4[f"{p_label}__corr_harness"][te] += ls_hc["ce"]
                p_corr_4[f"{p_label}__ce_harness"][te] += ls_hl["rps"]
                c_corr_4[f"{p_label}__ce_harness"][te] += ls_hl["ce"]
            cnt_2x2[te] += 1.0

    factorial_2x2 = {
        k: {"rps": float((p_corr_4[k] / cnt_2x2).mean()), "ce": float((c_corr_4[k] / cnt_2x2).mean())}
        for k in p_corr_4
    }
    factorial_2x2["contrasts"] = {
        "rps_prompt_effect_at_corr_harness": family_bootstrap_and_signflip(
            (p_corr_4["corr_prompt__corr_harness"] - p_corr_4["ce_prompt__corr_harness"]) / cnt_2x2, fam_idx
        ),
        "rps_harness_effect_at_ce_prompt": family_bootstrap_and_signflip(
            (p_corr_4["ce_prompt__corr_harness"] - p_corr_4["ce_prompt__ce_harness"]) / cnt_2x2, fam_idx
        ),
        "rps_both_ce_vs_both_corr": family_bootstrap_and_signflip(
            (p_corr_4["corr_prompt__corr_harness"] - p_corr_4["ce_prompt__ce_harness"]) / cnt_2x2, fam_idx
        ),
        "ce_both_ce_vs_both_corr": family_bootstrap_and_signflip(
            (c_corr_4["corr_prompt__corr_harness"] - c_corr_4["ce_prompt__ce_harness"]) / cnt_2x2, fam_idx
        ),
    }

    # ---- 5. ii.c Real Weight Updates: PyTorch LoRA Backprop + Neural Softmax Heads + Learning Curves ----
    lora_summaries = {}
    neural_head_learning_curve = {}
    restarts_check = {}

    for lora_mk in ("q3_4b_2507", "q25c_7b"):
        lf = raw_dir / f"lora_weight_training_{lora_mk}.json"
        if not lf.exists():
            continue
        ld = json.loads(lf.read_text(encoding="utf-8"))
        # Assemble out-of-fold predictions across all 75 artifacts from 5-fold grouped task-family CV
        oof_q: dict[str, np.ndarray] = {
            "base_ep0": np.zeros((n_items, 6)),
            "ce_lora_train_4": np.zeros((n_items, 6)),
            "ce_lora_val_rps_3": np.zeros((n_items, 6)),
            "rps_lora_train_4": np.zeros((n_items, 6)),
            "rps_lora_val_rps_3": np.zeros((n_items, 6)),
            "pearson_lora_train_2": np.zeros((n_items, 6)),
            "pearson_lora_val_corr_1": np.zeros((n_items, 6)),
            "pearson_lora_val_rps": np.zeros((n_items, 6)),
            "spearman_lora_train_2": np.zeros((n_items, 6)),
            "pearson_lora_plus_ce_head": np.zeros((n_items, 6)),
        }
        for f_rec in ld["five_fold_cv"]:
            te_i = np.array(f_rec["test_indices"], dtype=int)
            objs = f_rec["objectives"]
            oof_q["base_ep0"][te_i] = np.array(objs["ce"]["q_test_base_ep0"])
            oof_q["ce_lora_train_4"][te_i] = np.array(objs["ce"]["q_test_train_only_final"])
            oof_q["ce_lora_val_rps_3"][te_i] = np.array(objs["ce"]["q_test_val_rps_selected"])
            oof_q["rps_lora_train_4"][te_i] = np.array(objs["rps"]["q_test_train_only_final"])
            oof_q["rps_lora_val_rps_3"][te_i] = np.array(objs["rps"]["q_test_val_rps_selected"])
            oof_q["pearson_lora_train_2"][te_i] = np.array(objs["pearson"]["q_test_train_only_final"])
            oof_q["pearson_lora_val_corr_1"][te_i] = np.array(objs["pearson"]["q_test_val_corr_selected"])
            oof_q["pearson_lora_val_rps"][te_i] = np.array(objs["pearson"]["q_test_val_rps_selected"])
            oof_q["spearman_lora_train_2"][te_i] = np.array(objs["spearman"]["q_test_train_only_final"])
            oof_q["pearson_lora_plus_ce_head"][te_i] = np.array(objs["pearson"]["q_test_pearson_lora_plus_ce_head"])

        arm_metrics = {}
        arm_losses = {}
        for k_arm, q_mat in oof_q.items():
            ls = item_losses(q_mat, p_emp)
            arm_losses[k_arm] = ls
            arm_metrics[k_arm] = {
                "rps": float(ls["rps"].mean()),
                "ce": float(ls["ce"].mean()),
                "tvd": float(ls["tvd"].mean()),
                "brier": float(ls["brier"].mean()),
                **mse_decomposition(ls["mu_05"], hm5),
            }

        # LoRA learning curve across n_fam in {4, 8, 12, 15}
        lc_by_nfam = {}
        for n_fam in (4, 8, 12, 15):
            sub_rows = [r for r in ld["lora_learning_curve"] if r["n_train_families"] == n_fam]
            q_lc = {k: np.zeros((n_items, 6)) for k in ("ce", "rps", "pearson", "pearson_plus_ce_head")}
            mean_tr_n = float(np.mean([r["n_train_artifacts"] for r in sub_rows]))
            for r in sub_rows:
                te_i = np.array(r["test_indices"], dtype=int)
                for k in q_lc:
                    q_lc[k][te_i] = np.array(r["q_test"][k])
            ls_lc = {k: item_losses(q_lc[k], p_emp) for k in q_lc}
            lc_by_nfam[str(n_fam)] = {
                "n_train_families": n_fam,
                "mean_train_artifacts": mean_tr_n,
                "approx_train_ratings": round(mean_tr_n * 2.6, 1),
                "rps": {k: float(ls_lc[k]["rps"].mean()) for k in q_lc},
                "ce": {k: float(ls_lc[k]["ce"].mean()) for k in q_lc},
                "pearson_corr": {k: mse_decomposition(ls_lc[k]["mu_05"], hm5)["pearson"] for k in q_lc},
                "rps_pearson_minus_ce": family_bootstrap_and_signflip(ls_lc["pearson"]["rps"] - ls_lc["ce"]["rps"], fam_idx),
                "ce_pearson_minus_ce": family_bootstrap_and_signflip(ls_lc["pearson"]["ce"] - ls_lc["ce"]["ce"], fam_idx),
                "rps_pearson_head_minus_ce": family_bootstrap_and_signflip(
                    ls_lc["pearson_plus_ce_head"]["rps"] - ls_lc["ce"]["rps"], fam_idx
                ),
            }

        lora_summaries[lora_mk] = {
            "n_trainable_params": ld["n_trainable_params"],
            "top_k_blocks": ld["top_k_blocks"],
            "lora_rank": ld["lora_rank"],
            "epoch0_identity_max_diff": ld["epoch0_identity_max_diff"],
            "gradient_verification": ld["gradient_verification"],
            "elapsed_seconds": ld["elapsed_seconds"],
            "methods": arm_metrics,
            "contrasts": {
                "rps: pearson_lora_train_2 minus ce_lora_train_4": family_bootstrap_and_signflip(
                    arm_losses["pearson_lora_train_2"]["rps"] - arm_losses["ce_lora_train_4"]["rps"], fam_idx
                ),
                "ce: pearson_lora_train_2 minus ce_lora_train_4": family_bootstrap_and_signflip(
                    arm_losses["pearson_lora_train_2"]["ce"] - arm_losses["ce_lora_train_4"]["ce"], fam_idx
                ),
                "rps: pearson_lora_val_corr_1 minus ce_lora_val_rps_3": family_bootstrap_and_signflip(
                    arm_losses["pearson_lora_val_corr_1"]["rps"] - arm_losses["ce_lora_val_rps_3"]["rps"], fam_idx
                ),
                "ce: pearson_lora_val_corr_1 minus ce_lora_val_rps_3": family_bootstrap_and_signflip(
                    arm_losses["pearson_lora_val_corr_1"]["ce"] - arm_losses["ce_lora_val_rps_3"]["ce"], fam_idx
                ),
                "rps: spearman_lora_train_2 minus ce_lora_train_4": family_bootstrap_and_signflip(
                    arm_losses["spearman_lora_train_2"]["rps"] - arm_losses["ce_lora_train_4"]["rps"], fam_idx
                ),
                "rps: pearson_lora_plus_ce_head minus ce_lora_val_rps_3": family_bootstrap_and_signflip(
                    arm_losses["pearson_lora_plus_ce_head"]["rps"] - arm_losses["ce_lora_val_rps_3"]["rps"], fam_idx
                ),
                "rps: ce_lora_train_4 minus ce_lora_val_rps_3": family_bootstrap_and_signflip(
                    arm_losses["ce_lora_train_4"]["rps"] - arm_losses["ce_lora_val_rps_3"]["rps"], fam_idx
                ),
                "rps: base_ep0 minus ce_lora_val_rps_3": family_bootstrap_and_signflip(
                    arm_losses["base_ep0"]["rps"] - arm_losses["ce_lora_val_rps_3"]["rps"], fam_idx
                ),
            },
            "lora_learning_curve": lc_by_nfam,
            "oof_q": {k: v.tolist() for k, v in oof_q.items()},
        }

        # Run leak-free Hidden-State + Base-Logit Softmax Head Learning Curve (n_fam in {3, 5, 7, 9, 12, 15}, 120 splits)
        # + Exp 07 5-restarts check on the primary LLM representation (q3_4b_2507)!
        if lora_mk == "q3_4b_2507":
            gram_all = np.array(ld["hidden_gram_75x75"], dtype=np.float64)
            logits_base = np.array(ld["base_logits_75x6"], dtype=np.float64)
            head_objs = ["ce", "rps", "pearson", "spearman", "kendall", "pearson_spearman", "pearson_plus_ce_head", "constant"]

            for n_fam in (3, 5, 7, 9, 12, 15):
                sp_lc = generate_grouped_splits(fam_idx, n_fams=19, n_train_fams=n_fam, n_splits=120, seed=5000 + n_fam)
                rps_acc = {o: np.zeros(n_items) for o in head_objs}
                ce_acc = {o: np.zeros(n_items) for o in head_objs}
                pe_split = {o: [] for o in head_objs}
                cnt_lc = np.zeros(n_items)
                conv_cnt = {o: 0 for o in head_objs if o not in ("constant", "pearson_plus_ce_head")}
                tr_sizes = []

                for s_i, (tr, te) in enumerate(sp_lc):
                    tr_sizes.append(len(tr))
                    phi_tr, phi_te = leak_free_gram_features(gram_all, logits_base, tr, te, k_pcs=5)
                    pt = p_emp[tr]
                    marg = smooth_floor(pt.mean(axis=0), FLOOR)

                    preds_h: dict[str, np.ndarray] = {"constant": np.tile(marg, (len(te), 1))}
                    w_pe_fit = None
                    for o in ("ce", "rps", "pearson", "spearman", "kendall", "pearson_spearman"):
                        w_fit, info_f = fit_softmax_head(phi_tr, pt, o, ridge_c=2.0, n_restarts=1, maxiter=400, seed=s_i)
                        if info_f["converged"]:
                            conv_cnt[o] += 1
                        preds_h[o] = softmax(phi_te @ w_fit, axis=1)
                        if o == "pearson":
                            w_pe_fit = w_fit

                    assert w_pe_fit is not None
                    s_tr_p = softmax(phi_tr @ w_pe_fit, axis=1) @ GRID_5
                    s_te_p = preds_h["pearson"] @ GRID_5
                    ms_p, sds_p = s_tr_p.mean(), s_tr_p.std() + 1e-8
                    h2_tr = np.column_stack([np.ones(len(tr)), (s_tr_p - ms_p) / sds_p])
                    h2_te = np.column_stack([np.ones(len(te)), (s_te_p - ms_p) / sds_p])
                    w_ch, _ = fit_softmax_head(h2_tr, pt, "ce", ridge_c=1.0, n_restarts=1, maxiter=400, seed=s_i)
                    preds_h["pearson_plus_ce_head"] = softmax(h2_te @ w_ch, axis=1)

                    cnt_lc[te] += 1.0
                    hm_te = p_emp[te] @ GRID_5
                    for o in head_objs:
                        ls_o = item_losses(preds_h[o], p_emp[te])
                        rps_acc[o][te] += ls_o["rps"]
                        ce_acc[o][te] += ls_o["ce"]
                        if ls_o["mu_05"].std() > 1e-12 and hm_te.std() > 1e-12:
                            pe_split[o].append(float(np.corrcoef(ls_o["mu_05"], hm_te)[0, 1]))

                rps_item = {o: rps_acc[o] / cnt_lc for o in head_objs}
                ce_item = {o: ce_acc[o] / cnt_lc for o in head_objs}
                mean_tr = float(np.mean(tr_sizes))
                neural_head_learning_curve[str(n_fam)] = {
                    "n_train_families": n_fam,
                    "mean_train_artifacts": round(mean_tr, 2),
                    "mean_train_ratings": round(mean_tr * (195.0 / 75.0), 1),
                    "rps": {o: float(rps_item[o].mean()) for o in head_objs},
                    "ce": {o: float(ce_item[o].mean()) for o in head_objs},
                    "mean_test_pearson": {o: float(np.mean(pe_split[o])) if pe_split[o] else None for o in head_objs},
                    "convergence_rate": {o: round(conv_cnt[o] / len(sp_lc), 4) for o in conv_cnt},
                    "contrasts": {
                        f"rps_{o}_minus_ce": family_bootstrap_and_signflip(rps_item[o] - rps_item["ce"], fam_idx)
                        for o in ("pearson", "spearman", "kendall", "pearson_spearman", "pearson_plus_ce_head", "rps", "constant")
                    },
                }

            # Exp 07 Check: 1-start (400 iter) vs Best-of-5 random restarts (2,000 iter) at n_fam=15 across 50 splits
            sp_50 = generate_grouped_splits(fam_idx, n_fams=19, n_train_fams=15, n_splits=50, seed=77777)
            for cond_name, n_res, m_it in (("orig_1start_400iter", 1, 400), ("best5starts_2000iter", 5, 2000)):
                r_acc = {o: np.zeros(n_items) for o in ("ce", "rps", "pearson", "spearman", "kendall", "pearson_spearman")}
                c_acc = {o: np.zeros(n_items) for o in r_acc}
                cnt_r = np.zeros(n_items)
                conv_r = {o: 0 for o in r_acc}
                for s_i, (tr, te) in enumerate(sp_50):
                    phi_tr, phi_te = leak_free_gram_features(gram_all, logits_base, tr, te, k_pcs=5)
                    pt = p_emp[tr]
                    cnt_r[te] += 1.0
                    for o in r_acc:
                        wf, inf = fit_softmax_head(phi_tr, pt, o, ridge_c=2.0, n_restarts=n_res, maxiter=m_it, seed=900 + s_i)
                        if inf["converged"]:
                            conv_r[o] += 1
                        ls_o = item_losses(softmax(phi_te @ wf, axis=1), p_emp[te])
                        r_acc[o][te] += ls_o["rps"]
                        c_acc[o][te] += ls_o["ce"]
                restarts_check[cond_name] = {
                    "rps": {o: float((r_acc[o] / cnt_r).mean()) for o in r_acc},
                    "ce": {o: float((c_acc[o] / cnt_r).mean()) for o in r_acc},
                    "convergence_rate": {o: round(conv_r[o] / len(sp_50), 4) for o in r_acc},
                    "rps_pearson_minus_ce": family_bootstrap_and_signflip((r_acc["pearson"] - r_acc["ce"]) / cnt_r, fam_idx),
                    "rps_spearman_minus_ce": family_bootstrap_and_signflip((r_acc["spearman"] - r_acc["ce"]) / cnt_r, fam_idx),
                }

    # ---- 6. Stratified "WHEN Does CE Win?" Diagnostics (Rater Disagreement, Info Gap G_i, Ordinary vs Placeholder) ----
    disagree_mask = ceil_info["sd_emp"] > 0
    unanim_mask = ~disagree_mask
    info_gap_g = ceil_info["info_gap_g"]

    # Stratify on both (A) Logprob Pool Mixture (ii.b/c) and (B) PyTorch LoRA Weight Training (ii.c)
    diff_mix_rps = eval_logprob_pool["per_item_rps"]["corr_mix_pearson"] - eval_logprob_pool["per_item_rps"]["ce_mix_train"]
    diff_mix_ce = eval_logprob_pool["per_item_ce"]["corr_mix_pearson"] - eval_logprob_pool["per_item_ce"]["ce_mix_train"]

    when_diagnostics: dict[str, object] = {
        "logprob_mixture": {
            "unanimous_raters_SD_eq_0": {
                "n_artifacts": int(unanim_mask.sum()),
                "rps_corr_minus_ce": family_bootstrap_and_signflip(diff_mix_rps[unanim_mask], fam_idx[unanim_mask]),
                "ce_corr_minus_ce": family_bootstrap_and_signflip(diff_mix_ce[unanim_mask], fam_idx[unanim_mask]),
            },
            "split_raters_SD_gt_0": {
                "n_artifacts": int(disagree_mask.sum()),
                "rps_corr_minus_ce": family_bootstrap_and_signflip(diff_mix_rps[disagree_mask], fam_idx[disagree_mask]),
                "ce_corr_minus_ce": family_bootstrap_and_signflip(diff_mix_ce[disagree_mask], fam_idx[disagree_mask]),
            },
            "ordinary_non_placeholder_tasks": {
                "n_artifacts": int(ordinary_mask.sum()),
                "rps_corr_minus_ce": family_bootstrap_and_signflip(diff_mix_rps[ordinary_mask], fam_idx[ordinary_mask]),
                "ce_corr_minus_ce": family_bootstrap_and_signflip(diff_mix_ce[ordinary_mask], fam_idx[ordinary_mask]),
            },
            "placeholder_reference_tasks": {
                "n_artifacts": int((~ordinary_mask).sum()),
                "rps_corr_minus_ce": family_bootstrap_and_signflip(diff_mix_rps[~ordinary_mask], fam_idx[~ordinary_mask]),
                "ce_corr_minus_ce": family_bootstrap_and_signflip(diff_mix_ce[~ordinary_mask], fam_idx[~ordinary_mask]),
            },
            "info_gap_G_correlation_with_rps_advantage": {
                "pearson_G_vs_rps_diff": float(np.corrcoef(info_gap_g, diff_mix_rps)[0, 1]),
                "spearman_G_vs_rps_diff": float(spearmanr(info_gap_g, diff_mix_rps).statistic),
            },
        }
    }

    if "q3_4b_2507" in lora_summaries:
        oof_4b = {k: np.array(v) for k, v in lora_summaries["q3_4b_2507"]["oof_q"].items()}
        ls_p_lora = item_losses(oof_4b["pearson_lora_train_2"], p_emp)
        ls_ph_lora = item_losses(oof_4b["pearson_lora_plus_ce_head"], p_emp)
        ls_c_lora = item_losses(oof_4b["ce_lora_val_rps_3"], p_emp)
        d_raw = ls_p_lora["rps"] - ls_c_lora["rps"]
        d_head = ls_ph_lora["rps"] - ls_c_lora["rps"]
        when_diagnostics["lora_q3_4b"] = {
            "unanimous_SD_eq_0_pearson_minus_ce_rps": family_bootstrap_and_signflip(d_raw[unanim_mask], fam_idx[unanim_mask]),
            "split_SD_gt_0_pearson_minus_ce_rps": family_bootstrap_and_signflip(d_raw[disagree_mask], fam_idx[disagree_mask]),
            "unanimous_SD_eq_0_pearson_head_minus_ce_rps": family_bootstrap_and_signflip(d_head[unanim_mask], fam_idx[unanim_mask]),
            "split_SD_gt_0_pearson_head_minus_ce_rps": family_bootstrap_and_signflip(d_head[disagree_mask], fam_idx[disagree_mask]),
            "ordinary_pearson_minus_ce_rps": family_bootstrap_and_signflip(d_raw[ordinary_mask], fam_idx[ordinary_mask]),
            "placeholder_pearson_minus_ce_rps": family_bootstrap_and_signflip(d_raw[~ordinary_mask], fam_idx[~ordinary_mask]),
            "spearman_G_vs_lora_head_rps_diff": float(spearmanr(info_gap_g, d_head).statistic),
        }

    # Save public-safe forecast matrices (ZERO human ratings!)
    public_forecasts = {
        "artifact_order": [r["item"] for r in rows],
        "logprob_forecasts": {k: v.tolist() for k, v in forecasts_logprob.items()},
        "verbal_forecasts": {k: v.tolist() for k, v in forecasts_verbal.items()},
        "lora_oof_forecasts": {mk: lora_summaries[mk]["oof_q"] for mk in lora_summaries},
        "parse_stats": parse_stats,
        "sanity_controls": sanity_controls,
        "repeat_stability": repeat_stability,
    }
    (EXPT_DIR / "results/open_weight_forecasts.json").write_text(json.dumps(public_forecasts) + "\n", encoding="utf-8")

    # Strip bulky oof_q before writing readable summary JSON
    for mk in lora_summaries:
        lora_summaries[mk].pop("oof_q", None)

    summary = {
        "dataset_scope": {
            "dataset_sha256": ds_sha256,
            "n_artifacts": n_items,
            "n_tasks": 25,
            "n_task_families": 19,
            "n_expert_ratings": 195,
            "n_ordinary_artifacts": int(ordinary_mask.sum()),
            "n_placeholder_artifacts": int((~ordinary_mask).sum()),
            "n_split_rater_artifacts": ceil_info["n_disagree_items"],
            "n_unanimous_rater_artifacts": ceil_info["n_unanimous_items"],
            "human_mean_rating_05": float(hm5.mean()),
            "human_sd_rating_05": float(hm5.std()),
            "human_loro_rps_ceiling": ceil_info["human_loro_mean_rps"],
            "human_loro_pair_pearson": ceil_info["human_loro_pair_pearson"],
            "human_loro_pair_spearman": ceil_info["human_loro_pair_spearman"],
        },
        "part_i_audit": audit_part_i_synthetic_and_historical(),
        "sanity_controls": sanity_controls,
        "repeat_stability": repeat_stability,
        "theorem_count_baseline": thm_summary,
        "per_judge_insample": per_judge_insample,
        "representation_bridge": rep_bridge,
        "ii_a_prompt_optimization": {
            "per_model_candidate_prompts": prompt_opt_by_model,
            "all_models_prompt_pool_verbal": summarize_split_eval(ev_prompts_verbal, p_emp, fam_idx),
            "all_models_prompt_pool_logprob": summarize_split_eval(ev_prompts_logprob, p_emp, fam_idx),
            "blind_meta_prompt_search_q3_30b": meta_prompt_summary,
        },
        "ii_b_harness_optimization": {
            "harness_pool_selection_and_mixtures": harness_opt_summary,
            "qwen3_14b_thinking_vs_nothinking": thinking_ablation,
            "prompt_x_harness_2x2_factorial": factorial_2x2,
        },
        "ii_c_real_weight_training": {
            "pytorch_lora_end_to_end": lora_summaries,
            "neural_head_learning_curve_q3_4b": neural_head_learning_curve,
            "nonconvex_optimizer_restarts_check_q3_4b": restarts_check,
        },
        "when_does_ce_win": when_diagnostics,
    }
    (EXPT_DIR / "results/analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Wrote results/open_weight_forecasts.json and results/analysis_summary.json successfully.")


if __name__ == "__main__":
    main()
