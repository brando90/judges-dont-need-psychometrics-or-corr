#!/usr/bin/env python3
"""Phase 5 Comprehensive Statistical Analysis:
1. Exact Exp-02 5x5 grouped task-family CV (seeds 11062026..11062030) comparing:
   - Historical 5 cached 3-sample scalar frontier judges (verifying <1e-5 parity with Exp 02),
   - Zero-exposure 15 native-probability & logprob frontier judges (7 OpenAI/Mistral models),
   - Combined 20 frontier judges.
2. 250 seeded stratified grouped task-family splits (15 train families / 4 held-out test families) across:
   - Formulations ① (Val-Pearson), ② (Train-Pearson/Spearman/Kendall/Mix), ③ (Val-RPS), ④ (Train-CE / Train-RPS),
   - Classes M1 (single judge), M2 (simplex mixture with convex/cone certificates), M3 (48-param softmax feature head + Pearson+CE head),
   - Optimizer sensitivity check (Exp 07 on real data): 1 start x 400 iters vs best-of-5 starts x 2,000 iters across 50 splits.
3. Training-size learning curve (n_fam in {3, 5, 7, 9, 12, 15} training task families) on matched held-out test splits.
4. Axis iii.b Frontier Judge Harness Optimization (H1..H4) + 2x2 Interaction (Prompt Search Objective x Harness Objective)
   both on the locked 9/5/5 blind split AND across the 250 grouped splits for zero-exposure P0 harnesses.
5. Exact MSE Bias + Scale + Correlation decomposition & per-artifact Information-Gap G_i / SD>0 / Ordinary-vs-Placeholder stratification.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from scipy.special import softmax

from common import (
    PROBABILITY_FLOOR,
    RESULTS_DIR,
    load_private_dataset_and_inputs,
    utc_now_iso,
    write_public_json,
)
from metrics_and_solvers import (
    GRID_01,
    GRID_05,
    apply_temp_tilt,
    artifact_disagreement_diagnostics,
    build_pool_features,
    cluster_bootstrap_ci,
    fit_feature_head,
    fit_score_then_calibrate_head,
    fit_simplex_any,
    fit_temp_tilt_head,
    pearson_safe,
    per_artifact_metrics,
    rating_histogram,
    select_single_judge,
    smooth_probs,
    spearman_safe,
    summarize_panel,
)
from run_arm_b_prompt_search import LOCKED_TEST_FAMILIES, LOCKED_TRAIN_FAMILIES, LOCKED_VAL_FAMILIES

EXP02_SEEDS = (11062026, 11062027, 11062028, 11062029, 11062030)
EXP02_HIST_ORDER = ("H0_sol", "H0_opus", "H1_sol", "H2_sol", "H2_opus")
SHRINKAGE_GRID = (0.0, 0.05, 0.15, 0.30, 0.50, 0.75, 1.0)
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
LEARNING_CURVE_NFAM = (3, 5, 7, 9, 12, 15)


def exp02_partitions(rows: list[dict[str, Any]], seeds: tuple[int, ...] = EXP02_SEEDS) -> list[np.ndarray]:
    families = sorted({r["family"] for r in rows})
    out = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(families)
        byfam = {g: i % 5 for i, g in enumerate(order)}
        out.append(np.array([byfam[r["family"]] for r in rows], dtype=int))
    return out


def build_250_grouped_splits(rows: list[dict[str, Any]], n_splits: int = 250, seed: int = 20261116) -> list[dict[str, Any]]:
    """Generate 250 stratified grouped task-family splits (15 train families / 4 held-out test families)."""
    fams = sorted({r["family"] for r in rows})
    ord_fams = sorted({r["family"] for r in rows if r["ordinary"]})
    ph_fams = sorted({r["family"] for r in rows if not r["ordinary"]})
    rng = np.random.default_rng(seed)
    splits = []
    for s_idx in range(n_splits):
        # Hold out 3 ordinary families + 1 placeholder family = 4 held-out families (balanced!)
        te_ord = list(rng.choice(ord_fams, size=3, replace=False))
        te_ph = list(rng.choice(ph_fams, size=1, replace=False))
        te_fams = set(te_ord + te_ph)
        tr_fams_ordered = list(rng.permutation([f for f in fams if f not in te_fams]))
        splits.append({"split_id": s_idx, "train_families_ordered": tr_fams_ordered, "test_families": sorted(te_fams)})
    return splits


def inner_cv_select_shrinkage(
    q_tr: np.ndarray, p_tr: np.ndarray, fams_tr: np.ndarray, fit_obj: str, val_criterion: str
) -> tuple[float, np.ndarray]:
    """5-fold grouped task-family inner CV to select simplex shrinkage lambda in SHRINKAGE_GRID (Formulations ① and ③)."""
    uniq_fams = sorted(set(fams_tr.tolist()))
    fold_of = {g: i % min(5, len(uniq_fams)) for i, g in enumerate(uniq_fams)}
    f_arr = np.array([fold_of[g] for g in fams_tr], dtype=int)
    n_folds = len(set(f_arr.tolist()))
    m = q_tr.shape[1]
    u_w = np.full(m, 1.0 / m)

    oof_base = np.zeros((len(p_tr), 6), dtype=float)
    oof_unif = np.einsum("njr,j->nr", q_tr, u_w)
    for k in range(n_folds):
        itr = np.flatnonzero(f_arr != k)
        iva = np.flatnonzero(f_arr == k)
        if len(itr) < 2:
            oof_base[iva] = oof_unif[iva]
            continue
        w_k, _ = fit_simplex_any(q_tr[itr], p_tr[itr], fit_obj, seed=2026 + k)
        oof_base[iva] = np.einsum("njr,j->nr", q_tr[iva], w_k)

    best_lam = 0.0
    best_val = float("inf")
    for lam in SHRINKAGE_GRID:
        q_cand_oof = (1.0 - lam) * oof_base + lam * oof_unif
        pan = summarize_panel(p_tr, q_cand_oof)
        if val_criterion == "rps":
            score = pan["rps"]
        elif val_criterion == "ce":
            score = pan["ce"]
        elif val_criterion == "pearson":
            score = -pan["pearson"] if math.isfinite(pan["pearson"]) else 1e6
        else:
            raise ValueError(val_criterion)
        if score < best_val - 1e-12:
            best_val, best_lam = score, lam

    w_full, _ = fit_simplex_any(q_tr, p_tr, fit_obj, seed=20261116)
    return best_lam, (1.0 - best_lam) * w_full + best_lam * u_w


def evaluate_exp02_5x5_cv(
    rows: list[dict[str, Any]],
    p_emp: np.ndarray,
    q_hist: np.ndarray,
    q_native: np.ndarray,
    native_names: list[str],
) -> dict[str, Any]:
    """Run exact Exp-02 5x5 grouped task-family CV across Historical (5), Native-Prob (15), and Combined (20) pools."""
    parts = exp02_partitions(rows, EXP02_SEEDS)
    families = np.array([r["family"] for r in rows])
    uniq_fams = sorted(set(families.tolist()))
    fam_idx_map = {g: np.flatnonzero(families == g) for g in uniq_fams}

    pools = {
        "historical_5_scalar": q_hist,
        "native_15_frontier": q_native,
        "combined_20_frontier": np.concatenate([q_native, q_hist], axis=1),
    }
    methods = (
        "mix_ce_f4",
        "mix_rps_f4",
        "mix_ce_valrps_f3",
        "mix_pearson_f2",
        "mix_pearson_valcorr_f1",
        "mix_spearman_f2",
        "mix_kendall_f2",
        "mix_half_p_s_f2",
        "single_ce_f4",
        "single_rps_f4",
        "single_pearson_f2",
        "single_spearman_f2",
        "equal_mixture",
        "constant_marginal",
    )

    out_pools: dict[str, Any] = {}
    cert_counts = {"ce_total": 0, "ce_certified": 0, "rps_total": 0, "rps_certified": 0, "pearson_total": 0, "pearson_certified": 0}

    for pname, q_pool in pools.items():
        m = q_pool.shape[1]
        preds_by_method: dict[str, list[np.ndarray]] = {k: [] for k in methods}

        for si, outer in enumerate(parts):
            fold_preds = {k: np.zeros_like(p_emp) for k in methods}
            for fold in range(5):
                tr = np.flatnonzero(outer != fold)
                te = np.flatnonzero(outer == fold)
                fams_tr = families[tr]

                w_ce, r_ce = fit_simplex_any(q_pool[tr], p_emp[tr], "ce")
                cert_counts["ce_total"] += 1
                cert_counts["ce_certified"] += int(r_ce["certified"])
                fold_preds["mix_ce_f4"][te] = np.einsum("njr,j->nr", q_pool[te], w_ce)

                w_rps, r_rps = fit_simplex_any(q_pool[tr], p_emp[tr], "rps")
                cert_counts["rps_total"] += 1
                cert_counts["rps_certified"] += int(r_rps["certified"])
                fold_preds["mix_rps_f4"][te] = np.einsum("njr,j->nr", q_pool[te], w_rps)

                _, w_ce_f3 = inner_cv_select_shrinkage(q_pool[tr], p_emp[tr], fams_tr, "ce", "rps")
                fold_preds["mix_ce_valrps_f3"][te] = np.einsum("njr,j->nr", q_pool[te], w_ce_f3)

                w_pr, r_pr = fit_simplex_any(q_pool[tr], p_emp[tr], "pearson")
                cert_counts["pearson_total"] += 1
                cert_counts["pearson_certified"] += int(r_pr["certified"])
                fold_preds["mix_pearson_f2"][te] = np.einsum("njr,j->nr", q_pool[te], w_pr)

                _, w_pr_f1 = inner_cv_select_shrinkage(q_pool[tr], p_emp[tr], fams_tr, "pearson", "pearson")
                fold_preds["mix_pearson_valcorr_f1"][te] = np.einsum("njr,j->nr", q_pool[te], w_pr_f1)

                w_sp, _ = fit_simplex_any(q_pool[tr], p_emp[tr], "spearman", seed=100 * si + fold)
                fold_preds["mix_spearman_f2"][te] = np.einsum("njr,j->nr", q_pool[te], w_sp)

                w_kd, _ = fit_simplex_any(q_pool[tr], p_emp[tr], "kendall", seed=200 * si + fold)
                fold_preds["mix_kendall_f2"][te] = np.einsum("njr,j->nr", q_pool[te], w_kd)

                w_ps, _ = fit_simplex_any(q_pool[tr], p_emp[tr], "pearson_spearman", seed=300 * si + fold)
                fold_preds["mix_half_p_s_f2"][te] = np.einsum("njr,j->nr", q_pool[te], w_ps)

                for obj_s, key_s in (
                    ("ce", "single_ce_f4"),
                    ("rps", "single_rps_f4"),
                    ("pearson", "single_pearson_f2"),
                    ("spearman", "single_spearman_f2"),
                ):
                    j_sel = select_single_judge(q_pool[tr], p_emp[tr], obj_s)
                    fold_preds[key_s][te] = q_pool[te, j_sel, :]

                fold_preds["equal_mixture"][te] = q_pool[te].mean(axis=1)
                const_pmf = smooth_probs(p_emp[tr].mean(axis=0), PROBABILITY_FLOOR)
                fold_preds["constant_marginal"][te] = const_pmf[None, :]

            for k in methods:
                #Note: forecast matrices in q_pool are already smoothed once at PROBABILITY_FLOOR, so evaluate with eps=0 to match Exp 02 exact log-loss!
                preds_by_method[k].append(fold_preds[k])

        # Point estimates + 10,000-draw task-family bootstrap intervals
        method_summaries: dict[str, Any] = {}
        per_item_mean_across_seeds: dict[str, dict[str, np.ndarray]] = {}
        for k in methods:
            pans = [summarize_panel(p_emp, qq, eps=0.0) for qq in preds_by_method[k]]
            avg_pan = {mk: float(np.mean([p[mk] for p in pans])) for mk in pans[0]}
            # Per-artifact average across the 5 partitions for fast exact cluster bootstrap on RPS/CE/TVD
            pas = [per_artifact_metrics(p_emp, qq, eps=0.0) for qq in preds_by_method[k]]
            per_item_mean_across_seeds[k] = {
                mk: np.mean([pa[mk] for pa in pas], axis=0) for mk in ("rps", "ce", "tvd", "brier")
            }
            ci_rps = cluster_bootstrap_ci(
                lambda sampled, kk=k: float(
                    np.mean(np.concatenate([per_item_mean_across_seeds[kk]["rps"][fam_idx_map[g]] for g in sampled]))
                ),
                uniq_fams,
                draws=10000,
                seed=20261116,
            )
            ci_ce = cluster_bootstrap_ci(
                lambda sampled, kk=k: float(
                    np.mean(np.concatenate([per_item_mean_across_seeds[kk]["ce"][fam_idx_map[g]] for g in sampled]))
                ),
                uniq_fams,
                draws=10000,
                seed=20261116,
            )
            method_summaries[k] = {
                **{mk: round(v, 6) if isinstance(v, float) and math.isfinite(v) else v for mk, v in avg_pan.items()},
                "rps_ci95": ci_rps,
                "ce_ci95": ci_ce,
            }

        def _pair_diff(m_corr: str, m_loss: str) -> dict[str, Any]:
            res_d = {}
            for met in ("rps", "ce", "tvd"):
                diff_vec = per_item_mean_across_seeds[m_corr][met] - per_item_mean_across_seeds[m_loss][met]
                est = float(diff_vec.mean())
                ci = cluster_bootstrap_ci(
                    lambda sampled, dv=diff_vec: float(np.mean(np.concatenate([dv[fam_idx_map[g]] for g in sampled]))),
                    uniq_fams,
                    draws=10000,
                    seed=20261116,
                )
                res_d[f"corr_minus_loss_{met}"] = {"estimate": round(est, 6), **ci}
            return res_d

        contrasts = {
            "mix_pearson_f2_minus_mix_ce_f4": _pair_diff("mix_pearson_f2", "mix_ce_f4"),
            "mix_pearson_f2_minus_mix_rps_f4": _pair_diff("mix_pearson_f2", "mix_rps_f4"),
            "mix_pearson_valcorr_f1_minus_mix_ce_valrps_f3": _pair_diff("mix_pearson_valcorr_f1", "mix_ce_valrps_f3"),
            "mix_spearman_f2_minus_mix_ce_f4": _pair_diff("mix_spearman_f2", "mix_ce_f4"),
            "mix_ce_f4_minus_mix_ce_valrps_f3_form4_vs_form3": _pair_diff("mix_ce_f4", "mix_ce_valrps_f3"),
            "single_pearson_f2_minus_single_ce_f4": _pair_diff("single_pearson_f2", "single_ce_f4"),
            "single_pearson_f2_minus_single_rps_f4": _pair_diff("single_pearson_f2", "single_rps_f4"),
        }
        out_pools[pname] = {
            "methods": method_summaries,
            "contrasts": contrasts,
            "_per_item_rps_pearson_f2": per_item_mean_across_seeds["mix_pearson_f2"]["rps"].tolist(),
            "_per_item_rps_ce_f4": per_item_mean_across_seeds["mix_ce_f4"]["rps"].tolist(),
            "_per_item_rps_rps_f4": per_item_mean_across_seeds["mix_rps_f4"]["rps"].tolist(),
        }

    return {"pools": out_pools, "certificates": cert_counts}


def evaluate_250_splits_and_learning_curve(
    rows: list[dict[str, Any]], p_emp: np.ndarray, q_native: np.ndarray
) -> dict[str, Any]:
    """Evaluate 250 grouped task-family splits:
    - Full n_fam=15 comparison across M1 (single), M2 (simplex), and M3 (48-param feature head, all 7 objectives!),
    - Training-size learning curve across n_fam in {3, 5, 7, 9, 12, 15},
    - 50-split optimizer restart check (1 start x 400 iters vs best-of-5 starts x 2,000 iters) on M3!
    """
    splits = build_250_grouped_splits(rows, n_splits=250, seed=20261116)
    families = np.array([r["family"] for r in rows])
    n_ratings_per_row = np.array([len(r["_private_ratings"]) for r in rows], dtype=int)
    Phi_all = build_pool_features(q_native)  # (75, 8) — strictly zero human-label leakage!

    # 1. Learning curve across n_fam in {3, 5, 7, 9, 12, 15} for both M2 (simplex) and M3 (feature head)
    lc_methods = (
        "m3_ce_f4",
        "m3_ce_valrps_f3",
        "m3_rps_f4",
        "m3_pearson_f2",
        "m3_pearson_valcorr_f1",
        "m3_spearman_f2",
        "m3_kendall_f2",
        "m3_half_p_s_f2",
        "m3_pearson_plus_ce_head",
        "m2_mix_ce_f4",
        "m2_mix_rps_f4",
        "m2_mix_pearson_f2",
        "constant_marginal",
    )
    lc_records: dict[int, dict[str, Any]] = {}

    for nf in LEARNING_CURVE_NFAM:
        # Store per-split test metrics so we can pair by split AND compute family-bootstrap / split-bootstrap intervals
        split_metrics: dict[str, list[dict[str, float]]] = {m: [] for m in lc_methods}
        train_rating_counts = []

        for sp in splits:
            sub_tr_fams = set(sp["train_families_ordered"][:nf])
            te_fams = set(sp["test_families"])
            tr = np.array([i for i in range(len(rows)) if families[i] in sub_tr_fams], dtype=int)
            te = np.array([i for i in range(len(rows)) if families[i] in te_fams], dtype=int)
            train_rating_counts.append(int(n_ratings_per_row[tr].sum()))

            # M3 48-param feature head fits (fixed ridge c=1 for ②/④)
            for obj, mname in (
                ("ce", "m3_ce_f4"),
                ("rps", "m3_rps_f4"),
                ("pearson", "m3_pearson_f2"),
                ("spearman", "m3_spearman_f2"),
                ("kendall", "m3_kendall_f2"),
                ("pearson_spearman", "m3_half_p_s_f2"),
            ):
                W, _ = fit_feature_head(Phi_all[tr], p_emp[tr], obj, ridge_c=1.0, n_starts=1, maxiter=400, seed=sp["split_id"])
                q_te = softmax(Phi_all[te] @ W, axis=1)
                split_metrics[mname].append(summarize_panel(p_emp[te], q_te))

            # Pearson + calibrated CE head
            q_te_phead, _ = fit_score_then_calibrate_head(
                Phi_all[tr], p_emp[tr], Phi_all[te], ridge_c=1.0, n_starts=1, maxiter=400, seed=sp["split_id"]
            )
            split_metrics["m3_pearson_plus_ce_head"].append(summarize_panel(p_emp[te], q_te_phead))

            # Validation-tuned ridge c in {0.1, 1.0, 10.0} via 3-fold family CV inside tr for ③ and ①
            tr_fam_list = sorted(sub_tr_fams)
            fold_tr_map = {g: idx_g % min(3, len(tr_fam_list)) for idx_g, g in enumerate(tr_fam_list)}
            f_tr_arr = np.array([fold_tr_map[families[i]] for i in tr], dtype=int)

            for fit_o, sel_crit, mname in (
                ("ce", "rps", "m3_ce_valrps_f3"),
                ("pearson", "pearson", "m3_pearson_valcorr_f1"),
            ):
                best_c, best_sc = 1.0, float("inf")
                for c_cand in (0.1, 1.0, 10.0):
                    oof_q = np.zeros((len(tr), 6), dtype=float)
                    for fk in range(len(set(f_tr_arr.tolist()))):
                        itr = np.flatnonzero(f_tr_arr != fk)
                        iva = np.flatnonzero(f_tr_arr == fk)
                        W_k, _ = fit_feature_head(Phi_all[tr[itr]], p_emp[tr[itr]], fit_o, ridge_c=c_cand, n_starts=1, maxiter=250, seed=fk)
                        oof_q[iva] = softmax(Phi_all[tr[iva]] @ W_k, axis=1)
                    pan_oof = summarize_panel(p_emp[tr], oof_q)
                    sc = pan_oof["rps"] if sel_crit == "rps" else (-pan_oof["pearson"] if math.isfinite(pan_oof["pearson"]) else 1e6)
                    if sc < best_sc - 1e-12:
                        best_sc, best_c = sc, c_cand
                W_best, _ = fit_feature_head(Phi_all[tr], p_emp[tr], fit_o, ridge_c=best_c, n_starts=1, maxiter=400, seed=sp["split_id"])
                split_metrics[mname].append(summarize_panel(p_emp[te], softmax(Phi_all[te] @ W_best, axis=1)))

            # M2 simplex mixtures
            w_ce, _ = fit_simplex_any(q_native[tr], p_emp[tr], "ce")
            split_metrics["m2_mix_ce_f4"].append(summarize_panel(p_emp[te], np.einsum("njr,j->nr", q_native[te], w_ce), eps=0.0))
            w_rps, _ = fit_simplex_any(q_native[tr], p_emp[tr], "rps")
            split_metrics["m2_mix_rps_f4"].append(summarize_panel(p_emp[te], np.einsum("njr,j->nr", q_native[te], w_rps), eps=0.0))
            w_pr, _ = fit_simplex_any(q_native[tr], p_emp[tr], "pearson")
            split_metrics["m2_mix_pearson_f2"].append(summarize_panel(p_emp[te], np.einsum("njr,j->nr", q_native[te], w_pr), eps=0.0))

            const_pmf = smooth_probs(p_emp[tr].mean(axis=0), PROBABILITY_FLOOR)
            split_metrics["constant_marginal"].append(summarize_panel(p_emp[te], np.tile(const_pmf, (len(te), 1)), eps=0.0))

        # Aggregate across the 250 splits + paired bootstrap over splits
        rng_b = np.random.default_rng(20261116 + nf)
        boot_idx = rng_b.integers(0, len(splits), size=(10000, len(splits)))

        agg_nf: dict[str, Any] = {}
        for mname in lc_methods:
            rps_arr = np.array([x["rps"] for x in split_metrics[mname]])
            ce_arr = np.array([x["ce"] for x in split_metrics[mname]])
            tvd_arr = np.array([x["tvd"] for x in split_metrics[mname]])
            pr_arr = np.array([x["pearson"] for x in split_metrics[mname]])
            sp_arr = np.array([x["spearman"] for x in split_metrics[mname]])
            rps_b = rps_arr[boot_idx].mean(axis=1)
            ce_b = ce_arr[boot_idx].mean(axis=1)
            agg_nf[mname] = {
                "rps": round(float(rps_arr.mean()), 6),
                "rps_lo": round(float(np.quantile(rps_b, 0.025)), 6),
                "rps_hi": round(float(np.quantile(rps_b, 0.975)), 6),
                "ce": round(float(ce_arr.mean()), 6),
                "ce_lo": round(float(np.quantile(ce_b, 0.025)), 6),
                "ce_hi": round(float(np.quantile(ce_b, 0.975)), 6),
                "tvd": round(float(tvd_arr.mean()), 6),
                "pearson": round(float(np.nanmean(pr_arr)), 6),
                "spearman": round(float(np.nanmean(sp_arr)), 6),
                "mse_0_5": round(float(np.mean([x["mse_0_5"] for x in split_metrics[mname]])), 6),
                "bias_0_5": round(float(np.mean([x["bias_0_5"] for x in split_metrics[mname]])), 6),
                "bias_sq_0_5": round(float(np.mean([x["bias_sq_0_5"] for x in split_metrics[mname]])), 6),
                "scale_sq_0_5": round(float(np.mean([x["scale_sq_0_5"] for x in split_metrics[mname]])), 6),
                "unexplained_corr_term_0_5": round(float(np.mean([x["unexplained_corr_term_0_5"] for x in split_metrics[mname]])), 6),
                "judge_mean_0_5": round(float(np.mean([x["judge_mean_0_5"] for x in split_metrics[mname]])), 6),
                "human_mean_0_5": round(float(np.mean([x["human_mean_0_5"] for x in split_metrics[mname]])), 6),
            }

        def _paired_lc_diff(mc: str, ml: str, met: str = "rps") -> dict[str, float]:
            d = np.array([x[met] for x in split_metrics[mc]]) - np.array([x[met] for x in split_metrics[ml]])
            db = d[boot_idx].mean(axis=1)
            ple = float(np.mean(db <= 0.0))
            pge = float(np.mean(db >= 0.0))
            return {
                "estimate": round(float(d.mean()), 6),
                "lo": round(float(np.quantile(db, 0.025)), 6),
                "hi": round(float(np.quantile(db, 0.975)), 6),
                "p_two_sided": round(min(1.0, 2.0 * min(ple, pge)), 6),
            }

        lc_records[nf] = {
            "n_train_families": nf,
            "mean_train_ratings": round(float(np.mean(train_rating_counts)), 2),
            "methods": agg_nf,
            "contrasts": {
                "m3_pearson_f2_minus_m3_ce_f4_rps": _paired_lc_diff("m3_pearson_f2", "m3_ce_f4", "rps"),
                "m3_pearson_f2_minus_m3_ce_f4_ce": _paired_lc_diff("m3_pearson_f2", "m3_ce_f4", "ce"),
                "m3_pearson_valcorr_f1_minus_m3_ce_valrps_f3_rps": _paired_lc_diff("m3_pearson_valcorr_f1", "m3_ce_valrps_f3", "rps"),
                "m3_pearson_head_minus_m3_ce_f4_rps": _paired_lc_diff("m3_pearson_plus_ce_head", "m3_ce_f4", "rps"),
                "m2_mix_pearson_f2_minus_m2_mix_ce_f4_rps": _paired_lc_diff("m2_mix_pearson_f2", "m2_mix_ce_f4", "rps"),
                "m2_mix_pearson_f2_minus_m2_mix_rps_f4_rps": _paired_lc_diff("m2_mix_pearson_f2", "m2_mix_rps_f4", "rps"),
                "m3_ce_f4_minus_m3_ce_valrps_f3_rps_4vs3": _paired_lc_diff("m3_ce_f4", "m3_ce_valrps_f3", "rps"),
            },
        }
        print(
            f"[Learning Curve n_fam={nf:2d} (~{np.mean(train_rating_counts):5.1f} ratings)] "
            f"M3: Pearson RPS={agg_nf['m3_pearson_f2']['rps']:.4f} vs CE RPS={agg_nf['m3_ce_f4']['rps']:.4f} "
            f"(diff={lc_records[nf]['contrasts']['m3_pearson_f2_minus_m3_ce_f4_rps']['estimate']:+.4f})"
        )

    # 2. Optimizer restart check (Exp 07 on real frontier covariates: 50 splits, 1x400 vs 5x2000)
    restart_check: dict[str, Any] = {}
    for obj in ("ce", "rps", "pearson", "spearman", "kendall", "pearson_spearman"):
        conv_1, conv_5 = [], []
        rps_1, rps_5 = [], []
        ce_1, ce_5 = [], []
        for sp in splits[:50]:
            tr_f = set(sp["train_families_ordered"])
            te_f = set(sp["test_families"])
            tr = np.array([i for i in range(len(rows)) if families[i] in tr_f], dtype=int)
            te = np.array([i for i in range(len(rows)) if families[i] in te_f], dtype=int)

            W1, rec1 = fit_feature_head(Phi_all[tr], p_emp[tr], obj, ridge_c=1.0, n_starts=1, maxiter=400, seed=sp["split_id"])
            W5, rec5 = fit_feature_head(Phi_all[tr], p_emp[tr], obj, ridge_c=1.0, n_starts=5, maxiter=2000, seed=sp["split_id"])
            p1 = summarize_panel(p_emp[te], softmax(Phi_all[te] @ W1, axis=1))
            p5 = summarize_panel(p_emp[te], softmax(Phi_all[te] @ W5, axis=1))
            conv_1.append(float(rec1["converged"]))
            conv_5.append(float(rec5["any_start_converged"]))
            rps_1.append(p1["rps"])
            rps_5.append(p5["rps"])
            ce_1.append(p1["ce"])
            ce_5.append(p5["ce"])
        restart_check[obj] = {
            "splits_evaluated": 50,
            "converged_rate_1start_400iter": round(float(np.mean(conv_1)), 4),
            "converged_rate_5starts_2000iter": round(float(np.mean(conv_5)), 4),
            "test_rps_1start_400iter": round(float(np.mean(rps_1)), 6),
            "test_rps_5starts_2000iter": round(float(np.mean(rps_5)), 6),
            "test_ce_1start_400iter": round(float(np.mean(ce_1)), 6),
            "test_ce_5starts_2000iter": round(float(np.mean(ce_5)), 6),
        }

    return {"learning_curve": lc_records, "optimizer_restart_check_50splits": restart_check}


def evaluate_harnesses_and_2x2_interaction(
    rows: list[dict[str, Any]], p_emp: np.ndarray, phase4_data: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate Axis iii.b Frontier Harness Optimization (H1..H4) and the 2x2 Prompt x Harness Interaction
    on the locked 9/5/5 split (30 held-out test artifacts, 5 families, 81 ratings).
    """
    hf = {k: np.array(v, dtype=float) for k, v in phase4_data["harness_forecasts"].items()}
    tr_idx = np.array([i for i, r in enumerate(rows) if r["family"] in LOCKED_TRAIN_FAMILIES], dtype=int)
    va_idx = np.array([i for i, r in enumerate(rows) if r["family"] in LOCKED_VAL_FAMILIES], dtype=int)
    te_idx = np.array([i for i, r in enumerate(rows) if r["family"] in LOCKED_TEST_FAMILIES], dtype=int)

    test_fams = [rows[int(i)]["family"] for i in te_idx]
    fam_to_local_te = {
        g: np.array([k for k, gi in enumerate(test_fams) if gi == g], dtype=int) for g in sorted(set(test_fams))
    }

    # For each prompt track pkey in ("P0_untuned", "P_corr_val", "P_ce_val"), build the 7 candidate harness architectures:
    # 1. H1_single_gpt61
    # 2. H1_single_mistral35
    # 3. H2_ast_gpt61
    # 4. H3_twostep_gpt61
    # 5. H4a_crossfamily_equal
    # 6. H4b_simplex_plus_cal (simplex mixture over {H1_oai, H1_mis, H2_oai, H3_oai} + 2-param calibration, fitted by Corr vs CE!)
    # 7. H4e_discrete_3vote_scalar
    per_prompt_harnesses: dict[str, Any] = {}
    interaction_test_forecasts: dict[str, np.ndarray] = {}

    for pkey in ("P0_untuned", "P_corr_val", "P_ce_val"):
        base_4 = np.stack(
            [
                smooth_probs(hf[f"H1_single_gpt61__{pkey}"], PROBABILITY_FLOOR),
                smooth_probs(hf[f"H1_single_gpt54mini__{pkey}"], PROBABILITY_FLOOR),
                smooth_probs(hf[f"H1_single_mistral35__{pkey}"], PROBABILITY_FLOOR),
                smooth_probs(hf[f"H2_ast_gpt61__{pkey}"], PROBABILITY_FLOOR),
                smooth_probs(hf[f"H3_twostep_gpt61__{pkey}"], PROBABILITY_FLOOR),
            ],
            axis=1,
        )  # (75, 5, 6)

        # Fit H4 learned harness under Correlation (②/①) vs Proper Loss (④/③) on Train (+ Val selection)
        w_h4_corr, _ = fit_simplex_any(base_4[tr_idx], p_emp[tr_idx], "pearson")
        q_h4_corr_raw = np.einsum("njr,j->nr", base_4, w_h4_corr)
        (a_hc, b_hc), _ = fit_temp_tilt_head(q_h4_corr_raw[tr_idx], p_emp[tr_idx], objective="pearson")
        q_h4_corr_cal = apply_temp_tilt(q_h4_corr_raw, a_hc, b_hc)

        w_h4_ce, _ = fit_simplex_any(base_4[tr_idx], p_emp[tr_idx], "ce")
        q_h4_ce_raw = np.einsum("njr,j->nr", base_4, w_h4_ce)
        (a_hl, b_hl), _ = fit_temp_tilt_head(q_h4_ce_raw[tr_idx], p_emp[tr_idx], objective="ce")
        q_h4_ce_cal = apply_temp_tilt(q_h4_ce_raw, a_hl, b_hl)

        w_h4_rps, _ = fit_simplex_any(base_4[tr_idx], p_emp[tr_idx], "rps")
        q_h4_rps_raw = np.einsum("njr,j->nr", base_4, w_h4_rps)

        # Also Pearson + calibrated CE head on the 4-harness pool
        Phi_h = build_pool_features(base_4)
        q_te_phead, _ = fit_score_then_calibrate_head(Phi_h[tr_idx], p_emp[tr_idx], Phi_h[te_idx], ridge_c=1.0)
        q_va_phead, _ = fit_score_then_calibrate_head(Phi_h[tr_idx], p_emp[tr_idx], Phi_h[va_idx], ridge_c=1.0)
        q_tr_phead, _ = fit_score_then_calibrate_head(Phi_h[tr_idx], p_emp[tr_idx], Phi_h[tr_idx], ridge_c=1.0)
        q_full_phead = np.zeros((len(rows), 6))
        q_full_phead[tr_idx], q_full_phead[va_idx], q_full_phead[te_idx] = q_tr_phead, q_va_phead, q_te_phead

        # Candidate harness search space for discrete+continuous harness optimization:
        cand_space = {
            "H1_single_gpt61": hf[f"H1_single_gpt61__{pkey}"],
            "H1_single_gpt54mini": hf[f"H1_single_gpt54mini__{pkey}"],
            "H1_single_mistral35": hf[f"H1_single_mistral35__{pkey}"],
            "H2_ast_gpt61": hf[f"H2_ast_gpt61__{pkey}"],
            "H3_twostep_gpt61": hf[f"H3_twostep_gpt61__{pkey}"],
            "H4_crossfamily_equal": hf[f"H4_crossfamily_equal__{pkey}"],
            "H4_simplex_corr": q_h4_corr_raw,
            "H4_simplex_plus_cal_corr": q_h4_corr_cal,
            "H4_simplex_ce": q_h4_ce_raw,
            "H4_simplex_rps": q_h4_rps_raw,
            "H4_simplex_plus_cal_ce": q_h4_ce_cal,
            "H4_pearson_plus_ce_head": q_full_phead,
            "H4_discrete_3vote_scalar": hf[f"H4_discrete_3vote_scalar__{pkey}"],
        }

        # Correlation harness optimizer chooses among correlation-eligible harnesses (raw + corr-fitted, never peeking at CE-fitted!)
        corr_eligible = (
            "H1_single_gpt61",
            "H1_single_gpt54mini",
            "H1_single_mistral35",
            "H2_ast_gpt61",
            "H3_twostep_gpt61",
            "H4_crossfamily_equal",
            "H4_simplex_corr",
            "H4_simplex_plus_cal_corr",
            "H4_discrete_3vote_scalar",
        )
        # Proper-loss harness optimizer chooses among proper-loss-eligible harnesses (raw + CE/RPS-fitted)
        loss_eligible = (
            "H1_single_gpt61",
            "H1_single_gpt54mini",
            "H1_single_mistral35",
            "H2_ast_gpt61",
            "H3_twostep_gpt61",
            "H4_crossfamily_equal",
            "H4_simplex_ce",
            "H4_simplex_rps",
            "H4_simplex_plus_cal_ce",
        )

        val_scores = {hk: summarize_panel(p_emp[va_idx], qm[va_idx]) for hk, qm in cand_space.items()}
        tr_scores = {hk: summarize_panel(p_emp[tr_idx], qm[tr_idx]) for hk, qm in cand_space.items()}
        te_scores = {hk: summarize_panel(p_emp[te_idx], qm[te_idx]) for hk, qm in cand_space.items()}

        best_h_f1_corr_val = max(corr_eligible, key=lambda hk: (val_scores[hk]["pearson"], val_scores[hk]["spearman"]))
        best_h_f2_corr_tr = max(corr_eligible, key=lambda hk: (tr_scores[hk]["pearson"], tr_scores[hk]["spearman"]))
        best_h_f3_loss_val = min(loss_eligible, key=lambda hk: (val_scores[hk]["rps"], val_scores[hk]["ce"]))
        best_h_f4_loss_tr = min(loss_eligible, key=lambda hk: (tr_scores[hk]["ce"], tr_scores[hk]["rps"]))

        interaction_test_forecasts[f"{pkey}__H_untuned_H1"] = cand_space["H1_single_gpt61"][te_idx]
        interaction_test_forecasts[f"{pkey}__H_corr_val_f1"] = cand_space[best_h_f1_corr_val][te_idx]
        interaction_test_forecasts[f"{pkey}__H_corr_tr_f2"] = cand_space[best_h_f2_corr_tr][te_idx]
        interaction_test_forecasts[f"{pkey}__H_loss_val_f3"] = cand_space[best_h_f3_loss_val][te_idx]
        interaction_test_forecasts[f"{pkey}__H_loss_tr_f4"] = cand_space[best_h_f4_loss_tr][te_idx]

        per_prompt_harnesses[pkey] = {
            "heldout_test_by_harness": te_scores,
            "val_by_harness": val_scores,
            "selected_harnesses": {
                "F1_val_pearson_selected": {"chosen": best_h_f1_corr_val, "heldout_test": te_scores[best_h_f1_corr_val]},
                "F2_train_pearson_selected": {"chosen": best_h_f2_corr_tr, "heldout_test": te_scores[best_h_f2_corr_tr]},
                "F3_val_rps_selected": {"chosen": best_h_f3_loss_val, "heldout_test": te_scores[best_h_f3_loss_val]},
                "F4_train_ce_selected": {"chosen": best_h_f4_loss_tr, "heldout_test": te_scores[best_h_f4_loss_tr]},
            },
        }

    # Build explicit 3x3 / 2x2 Interaction Contrasts with 10,000-draw task-family bootstrap CIs on the 30 held-out test items
    p_te = p_emp[te_idx]

    def _ic_diff(key_a: str, key_b: str) -> dict[str, Any]:
        pa_a = per_artifact_metrics(p_te, interaction_test_forecasts[key_a])
        pa_b = per_artifact_metrics(p_te, interaction_test_forecasts[key_b])
        res = {}
        for met in ("rps", "ce", "tvd"):
            dv = pa_a[met] - pa_b[met]
            ci = cluster_bootstrap_ci(
                lambda sampled, v=dv: float(np.mean(np.concatenate([v[fam_to_local_te[g]] for g in sampled]))),
                test_fams,
                draws=10000,
                seed=20261116,
            )
            res[f"a_minus_b_{met}"] = {"estimate": round(float(dv.mean()), 6), **ci}
        return res

    interaction_contrasts = {
        "harness_only_at_P0__F1_corr_minus_F3_loss": _ic_diff("P0_untuned__H_corr_val_f1", "P0_untuned__H_loss_val_f3"),
        "harness_only_at_P0__F2_corr_minus_F4_loss": _ic_diff("P0_untuned__H_corr_tr_f2", "P0_untuned__H_loss_tr_f4"),
        "prompt_only_at_H1__P_corr_minus_P_ce": _ic_diff("P_corr_val__H_untuned_H1", "P_ce_val__H_untuned_H1"),
        "joint_2x2_val__PromptCorr_HarnessCorr_minus_PromptCE_HarnessLoss": _ic_diff(
            "P_corr_val__H_corr_val_f1", "P_ce_val__H_loss_val_f3"
        ),
        "joint_2x2_train__PromptCorr_HarnessCorr_minus_PromptCE_HarnessLoss": _ic_diff(
            "P_corr_val__H_corr_tr_f2", "P_ce_val__H_loss_tr_f4"
        ),
        "untuned_P0_H1_minus_joint_PromptCE_HarnessLoss": _ic_diff(
            "P0_untuned__H_untuned_H1", "P_ce_val__H_loss_val_f3"
        ),
    }

    return {
        "per_prompt_harnesses": per_prompt_harnesses,
        "interaction_contrasts": interaction_contrasts,
    }


def evaluate_disagreement_and_strata(
    rows: list[dict[str, Any]],
    p_emp: np.ndarray,
    rps_pearson_per_item: np.ndarray,
    rps_ce_per_item: np.ndarray,
) -> dict[str, Any]:
    """Stratified held-out RPS analysis (SD > 0 vs SD == 0; Ordinary vs Placeholder) and Info-Gap G_i correlation (Exp 05)."""
    disag = artifact_disagreement_diagnostics([r["_private_ratings"] for r in rows])
    sd_pos = disag["empirical_sd_0_5"] > 0
    ord_mask = np.array([r["ordinary"] for r in rows], dtype=bool)
    families = np.array([r["family"] for r in rows])
    uniq_fams = sorted(set(families.tolist()))

    diff_rps = rps_pearson_per_item - rps_ce_per_item  # positive => CE-trained better!
    g_vec = disag["smoothed_info_gap_G"]

    def _stratum_summary(mask: np.ndarray) -> dict[str, Any]:
        sub_fams = sorted(set(families[mask].tolist()))
        local_map = {g: np.flatnonzero((families == g) & mask) for g in sub_fams}
        ci = cluster_bootstrap_ci(
            lambda sampled: float(np.mean(np.concatenate([diff_rps[local_map[g]] for g in sampled]))),
            sub_fams,
            draws=10000,
            seed=20261116,
        )
        return {
            "n_artifacts": int(mask.sum()),
            "n_families": len(sub_fams),
            "test_rps_pearson_trained": round(float(rps_pearson_per_item[mask].mean()), 6),
            "test_rps_ce_trained": round(float(rps_ce_per_item[mask].mean()), 6),
            "pearson_minus_ce_rps": {"estimate": round(float(diff_rps[mask].mean()), 6), **ci},
        }

    fam_all_map = {g: np.flatnonzero(families == g) for g in uniq_fams}
    corr_g_r = pearson_safe(g_vec, diff_rps)
    corr_g_sp = spearman_safe(g_vec, diff_rps)
    ci_g_r = cluster_bootstrap_ci(
        lambda sampled: pearson_safe(
            np.concatenate([g_vec[fam_all_map[g]] for g in sampled]),
            np.concatenate([diff_rps[fam_all_map[g]] for g in sampled]),
        ),
        uniq_fams,
        draws=10000,
        seed=20261116,
    )
    ci_g_sp = cluster_bootstrap_ci(
        lambda sampled: spearman_safe(
            np.concatenate([g_vec[fam_all_map[g]] for g in sampled]),
            np.concatenate([diff_rps[fam_all_map[g]] for g in sampled]),
        ),
        uniq_fams,
        draws=10000,
        seed=20261116,
    )

    return {
        "strata": {
            "all_75_artifacts": _stratum_summary(np.ones(len(rows), dtype=bool)),
            "rater_disagreement_sd_gt_0": _stratum_summary(sd_pos),
            "unanimous_raters_sd_eq_0": _stratum_summary(~sd_pos),
            "ordinary_non_placeholder_60": _stratum_summary(ord_mask),
            "placeholder_security_15": _stratum_summary(~ord_mask),
        },
        "info_gap_G_correlation_with_rps_gain": {
            "pearson_r": {"estimate": round(corr_g_r, 6), **ci_g_r},
            "spearman_rho": {"estimate": round(corr_g_sp, 6), **ci_g_sp},
            # Safe public aggregate bins (3 terciles of G_i — never individual private ratings!)
            "tercile_means": [
                {
                    "tercile": t_idx + 1,
                    "n_artifacts": int(len(t_ix)),
                    "mean_G": round(float(g_vec[t_ix].mean()), 6),
                    "mean_pearson_minus_ce_rps": round(float(diff_rps[t_ix].mean()), 6),
                }
                for t_idx, t_ix in enumerate(np.array_split(np.argsort(g_vec), 3))
            ],
        },
    }


def main() -> None:
    data = load_private_dataset_and_inputs()
    rows = data["rows"]
    p_emp = np.array([rating_histogram(r["_private_ratings"]) for r in rows])

    fc = json.loads((RESULTS_DIR / "judge_forecasts.json").read_text(encoding="utf-8"))
    native_names = [s["judge_id"] for s in fc["arm_a_specs"]]
    q_native_raw = np.stack(
        [np.array([x["probabilities"] for x in fc["native_prob_forecasts"][jid]]) for jid in native_names],
        axis=1,
    )
    q_native = smooth_probs(q_native_raw, PROBABILITY_FLOOR)

    q_hist_raw = np.stack(
        [np.array(fc["historical_3sample_forecasts"][hname]) for hname in EXP02_HIST_ORDER],
        axis=1,
    )
    q_hist = smooth_probs(q_hist_raw, PROBABILITY_FLOOR)

    print("Running Exp-02 5x5 grouped task-family CV across 3 pools...")
    exp02_cv = evaluate_exp02_5x5_cv(rows, p_emp, q_hist, q_native, native_names)
    hist_ce = exp02_cv["pools"]["historical_5_scalar"]["methods"]["mix_ce_f4"]["ce"]
    hist_pr = exp02_cv["pools"]["historical_5_scalar"]["methods"]["mix_pearson_f2"]["ce"]
    nat_ce = exp02_cv["pools"]["native_15_frontier"]["methods"]["mix_ce_f4"]["ce"]
    nat_pr = exp02_cv["pools"]["native_15_frontier"]["methods"]["mix_pearson_f2"]["ce"]
    print(
        f"  Historical 5 scalar 5x5 CV CE: CE-mix={hist_ce:.6f} (Exp02=5.491441), Pearson-mix={hist_pr:.6f} (Exp02=5.751842)\n"
        f"  Native-15 frontier 5x5 CV CE:  CE-mix={nat_ce:.6f}, Pearson-mix={nat_pr:.6f}"
    )

    print("Running 250 grouped task-family splits, learning curve, and 50-split restart check...")
    splits250 = evaluate_250_splits_and_learning_curve(rows, p_emp, q_native)

    print("Running Axis iii.b Frontier Harness Optimization & 2x2 Interaction...")
    phase4_data = json.loads((RESULTS_DIR / "phase4_arm_c_harnesses.json").read_text(encoding="utf-8"))
    harness_2x2 = evaluate_harnesses_and_2x2_interaction(rows, p_emp, phase4_data)

    print("Running Rater-Disagreement / Info-Gap G_i & Strata Diagnostics...")
    # Evaluate Disagreement strata on the 15-judge native frontier pool (M3 feature head across Exp-02 5x5 OOF predictions)
    parts = exp02_partitions(rows, EXP02_SEEDS)
    Phi_nat = build_pool_features(q_native)
    oof_m3_pr = np.zeros(len(rows))
    oof_m3_ce = np.zeros(len(rows))
    for si, outer in enumerate(parts):
        q_pr_s = np.zeros_like(p_emp)
        q_ce_s = np.zeros_like(p_emp)
        for fold in range(5):
            tr = np.flatnonzero(outer != fold)
            te = np.flatnonzero(outer == fold)
            W_pr, _ = fit_feature_head(Phi_nat[tr], p_emp[tr], "pearson", ridge_c=1.0, seed=si)
            W_ce, _ = fit_feature_head(Phi_nat[tr], p_emp[tr], "ce", ridge_c=1.0, seed=si)
            q_pr_s[te] = softmax(Phi_nat[te] @ W_pr, axis=1)
            q_ce_s[te] = softmax(Phi_nat[te] @ W_ce, axis=1)
        oof_m3_pr += per_artifact_metrics(p_emp, q_pr_s)["rps"] / len(parts)
        oof_m3_ce += per_artifact_metrics(p_emp, q_ce_s)["rps"] / len(parts)

    strata_m3 = evaluate_disagreement_and_strata(rows, p_emp, oof_m3_pr, oof_m3_ce)
    strata_m2 = evaluate_disagreement_and_strata(
        rows,
        p_emp,
        np.array(exp02_cv["pools"]["native_15_frontier"]["_per_item_rps_pearson_f2"]),
        np.array(exp02_cv["pools"]["native_15_frontier"]["_per_item_rps_ce_f4"]),
    )
    # Strip internal per-item arrays before public export so zero item-level human residuals are exposed
    for pdict in exp02_cv["pools"].values():
        for k in list(pdict.keys()):
            if k.startswith("_"):
                del pdict[k]

    write_public_json(
        RESULTS_DIR / "analysis_heldout_tables.json",
        {
            "created_utc": utc_now_iso(),
            "exp02_5x5_grouped_cv": exp02_cv,
            "splits250_and_learning_curve": splits250,
            "harnesses_and_2x2_interaction": harness_2x2,
            "disagreement_and_strata_m3_feature_head": strata_m3,
            "disagreement_and_strata_m2_simplex_mixture": strata_m2,
        },
    )
    print("Saved results/analysis_heldout_tables.json successfully!")


if __name__ == "__main__":
    main()
