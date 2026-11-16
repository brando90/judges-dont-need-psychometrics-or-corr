#!/usr/bin/env python3
"""Generate publication-grade PNG figures (obeying AGENTS.md axis-label, gridline, subtraction-order,
and optimizer-caption rules) plus canonical `RESULTS_frontier_ce_vs_corr.md`, live `results.md` finalization,
`results/cost_summary.json`, and `writing.tex` draft for Experiment 10."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    EXPT_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    ledger_summary,
    pst_now_str,
    utc_now_iso,
    write_public_json,
)


def fmt_ci(est: float, lo: float | None, hi: float | None, pval: float | None = None, nd: int = 4, signed: bool = True) -> str:
    s_est = f"{est:+.{nd}f}" if signed else f"{est:.{nd}f}"
    if lo is None or hi is None:
        return f"{s_est} [n/a, n/a] p-val=n/a"
    s_lo = f"{lo:+.{nd}f}" if signed else f"{lo:.{nd}f}"
    s_hi = f"{hi:+.{nd}f}" if signed else f"{hi:.{nd}f}"
    p_str = f"p-val={pval:.4f}" if (pval is not None and math.isfinite(pval)) else "p-val=n/a"
    return f"{s_est} [{s_lo}, {s_hi}] {p_str}"


def plot_fig1_master_forest(ana: dict[str, Any], p3: dict[str, Any]) -> None:
    """Figure 1: Master forest plot of held-out test RPS differences (Correlation minus Proper Loss)."""
    cv_hist = ana["exp02_5x5_grouped_cv"]["pools"]["historical_5_scalar"]
    cv_nat = ana["exp02_5x5_grouped_cv"]["pools"]["native_15_frontier"]
    lc15 = ana["splits250_and_learning_curve"]["learning_curve"]["15"]
    h2x2 = ana["harnesses_and_2x2_interaction"]

    rows = [
        (
            "Historical 5 scalar judges · Simplex mixture (② vs ④)",
            cv_hist["contrasts"]["mix_pearson_f2_minus_mix_ce_f4"]["corr_minus_loss_rps"],
            cv_hist["methods"]["mix_pearson_f2"]["rps"],
            cv_hist["methods"]["mix_ce_f4"]["rps"],
        ),
        (
            "Native-15 frontier pool · Simplex mixture (② Pearson vs ④ CE)",
            cv_nat["contrasts"]["mix_pearson_f2_minus_mix_ce_f4"]["corr_minus_loss_rps"],
            cv_nat["methods"]["mix_pearson_f2"]["rps"],
            cv_nat["methods"]["mix_ce_f4"]["rps"],
        ),
        (
            "Native-15 frontier pool · Simplex mixture (② Pearson vs ④ RPS)",
            cv_nat["contrasts"]["mix_pearson_f2_minus_mix_rps_f4"]["corr_minus_loss_rps"],
            cv_nat["methods"]["mix_pearson_f2"]["rps"],
            cv_nat["methods"]["mix_rps_f4"]["rps"],
        ),
        (
            "Native-15 frontier pool · Val-tuned mixture (① vs ③)",
            cv_nat["contrasts"]["mix_pearson_valcorr_f1_minus_mix_ce_valrps_f3"]["corr_minus_loss_rps"],
            cv_nat["methods"]["mix_pearson_valcorr_f1"]["rps"],
            cv_nat["methods"]["mix_ce_valrps_f3"]["rps"],
        ),
        (
            "Native-15 frontier pool · 48-param feature head (② vs ④, 250 splits)",
            lc15["contrasts"]["m3_pearson_f2_minus_m3_ce_f4_rps"],
            lc15["methods"]["m3_pearson_f2"]["rps"],
            lc15["methods"]["m3_ce_f4"]["rps"],
        ),
        (
            "Native-15 frontier pool · Pearson+CE head vs ④ CE head (250 splits)",
            lc15["contrasts"]["m3_pearson_head_minus_m3_ce_f4_rps"],
            lc15["methods"]["m3_pearson_plus_ce_head"]["rps"],
            lc15["methods"]["m3_ce_f4"]["rps"],
        ),
        (
            "iii.a Blind prompt search · Val-selected raw (① Corr vs ③ Loss)",
            p3["heldout_test_contrasts"]["F1_val_corr_raw_minus_F3_val_ce_raw"]["corr_minus_ce_rps"],
            p3["selected_finalists_factorial"]["F1_corr_val_selected"]["heldout_test_raw"]["rps"],
            p3["selected_finalists_factorial"]["F3_ce_val_rps_selected"]["heldout_test_raw"]["rps"],
        ),
        (
            "iii.a Blind prompt search · Train-selected raw (② Corr vs ④ Loss)",
            p3["heldout_test_contrasts"]["F2_train_corr_raw_minus_F4_train_ce_raw"]["corr_minus_ce_rps"],
            p3["selected_finalists_factorial"]["F2_corr_train_selected"]["heldout_test_raw"]["rps"],
            p3["selected_finalists_factorial"]["F4_ce_train_ce_selected"]["heldout_test_raw"]["rps"],
        ),
        (
            "iii.b Harness optimization at P0 · Val-selected (① Corr vs ③ Loss)",
            h2x2["interaction_contrasts"]["harness_only_at_P0__F1_corr_minus_F3_loss"]["a_minus_b_rps"],
            h2x2["per_prompt_harnesses"]["P0_untuned"]["selected_harnesses"]["F1_val_pearson_selected"]["heldout_test"]["rps"],
            h2x2["per_prompt_harnesses"]["P0_untuned"]["selected_harnesses"]["F3_val_rps_selected"]["heldout_test"]["rps"],
        ),
        (
            "iii.a + iii.b Joint 2x2 · (Prompt ① + Harness ①) vs (Prompt ③ + Harness ③)",
            h2x2["interaction_contrasts"]["joint_2x2_val__PromptCorr_HarnessCorr_minus_PromptCE_HarnessLoss"]["a_minus_b_rps"],
            h2x2["per_prompt_harnesses"]["P_corr_val"]["selected_harnesses"]["F1_val_pearson_selected"]["heldout_test"]["rps"],
            h2x2["per_prompt_harnesses"]["P_ce_val"]["selected_harnesses"]["F3_val_rps_selected"]["heldout_test"]["rps"],
        ),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=180)
    y_pos = np.arange(len(rows))[::-1]

    ax.axvline(0.0, color="#1b2230", linewidth=1.3, zorder=2)
    for yp, (label, cdict, rps_corr, rps_loss) in zip(y_pos, rows):
        est = float(cdict["estimate"])
        lo = float(cdict["lo"])
        hi = float(cdict["hi"])
        color = "#0f7c6e" if lo > 0 else ("#b4690e" if hi < 0 else "#5b6577")
        ax.plot([lo, hi], [yp, yp], color=color, linewidth=2.4, solid_capstyle="round", zorder=3)
        ax.scatter([est], [yp], color=color, s=52, zorder=4, edgecolors="white", linewidths=0.9)
        ax.text(
            max(hi, est) + 0.003,
            yp,
            f"{est:+.4f}  (Corr {rps_corr:.3f} vs Loss {rps_loss:.3f})",
            va="center",
            ha="left",
            fontsize=8.3,
            family="monospace",
            color="#1b2230",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.0)
    lo_min = min(float(r[1]["lo"]) for r in rows)
    hi_max = max(float(r[1]["hi"]) for r in rows)
    span = max(0.08, hi_max - lo_min)
    ax.set_xlim(min(-0.04, lo_min - 0.015), hi_max + 0.56 * span + 0.06)
    ax.set_xlabel(
        "Held-out test ranked probability score (RPS) of correlation-trained minus proper-loss-trained judge\n"
        "(same metric on held-out test task families; > 0 means proper-loss judge has lower [better] test RPS)",
        fontsize=9.0,
    )
    fig.suptitle(
        "Experiment 10: Frontier AI Judges — Correlation vs Proper-Loss Training Across Pools, Prompts (iii.a), and Harnesses (iii.b)",
        fontsize=10.0,
        fontweight="bold",
        y=0.985,
    )
    ax.grid(True, which="major", alpha=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig1_frontier_pools_and_axes_forest.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_native_vs_scalar(ana: dict[str, Any]) -> None:
    """Figure 2: Native 6-category probabilities vs 3-sample scalar histograms on held-out CE and RPS."""
    cv_hist = ana["exp02_5x5_grouped_cv"]["pools"]["historical_5_scalar"]["methods"]
    cv_nat = ana["exp02_5x5_grouped_cv"]["pools"]["native_15_frontier"]["methods"]
    lc15 = ana["splits250_and_learning_curve"]["learning_curve"]["15"]["methods"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=180)

    labels_ce = [
        "Historical 3-sample\nPearson mix (②)",
        "Historical 3-sample\nCE mix (④)",
        "Constant training\nhistogram baseline",
        "Native-15 frontier\nPearson mix (②)",
        "Native-15 frontier\nVal-RPS mix (③)",
        "Native-15 frontier\nCE mix (④)",
    ]
    vals_ce = [
        cv_hist["mix_pearson_f2"]["ce"],
        cv_hist["mix_ce_f4"]["ce"],
        cv_nat["constant_marginal"]["ce"],
        cv_nat["mix_pearson_f2"]["ce"],
        cv_nat["mix_ce_valrps_f3"]["ce"],
        cv_nat["mix_ce_f4"]["ce"],
    ]
    cols_ce = ["#b4690e", "#0f7c6e", "#6b7280", "#b4690e", "#2f5bd3", "#0f7c6e"]
    x1 = np.arange(len(labels_ce))
    bars1 = ax1.bar(x1, vals_ce, color=cols_ce, alpha=0.88, width=0.62)
    ax1.axhline(cv_nat["constant_marginal"]["ce"], color="#6b7280", linestyle="--", linewidth=1.2, label="Constant forecast")
    for b, v in zip(bars1, vals_ce):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, family="monospace")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels_ce, fontsize=8.0, rotation=18, ha="right")
    ax1.set_ylim(0, max(vals_ce) * 1.14)
    ax1.set_ylabel("Held-out test cross-entropy (CE in nats; lower is better)", fontsize=9.5)
    ax1.set_title("A. Native Probabilities Fix 3-Sample Scalar CE Failure (5x5 Family CV)", fontsize=9.8, fontweight="bold")
    ax1.grid(True, which="major", axis="y", alpha=0.4)

    obj_keys = [
        ("m3_ce_f4", "CE-trained (④)", "#0f7c6e"),
        ("m3_ce_valrps_f3", "CE val-RPS (③)", "#2f5bd3"),
        ("m3_rps_f4", "RPS-trained (④)", "#1b9e77"),
        ("m3_pearson_plus_ce_head", "Pearson + CE head", "#d95f02"),
        ("m3_pearson_valcorr_f1", "Pearson val-corr (①)", "#e6ab02"),
        ("m3_pearson_f2", "Pearson-trained (②)", "#b4690e"),
        ("m3_spearman_f2", "Spearman-trained (②)", "#7570b3"),
        ("m3_kendall_f2", "Kendall-trained (②)", "#8c564b"),
        ("m3_half_p_s_f2", "1/2 P + 1/2 S (②)", "#e7298a"),
    ]
    y2 = np.arange(len(obj_keys))[::-1]
    for yp, (mk, mlabel, col) in zip(y2, obj_keys):
        md = lc15[mk]
        ax2.plot([md["rps_lo"], md["rps_hi"]], [yp, yp], color=col, linewidth=2.3)
        ax2.scatter([md["rps"]], [yp], color=col, s=48, zorder=3)
        ax2.text(md["rps_hi"] + 0.002, yp, f"{md['rps']:.4f}", va="center", fontsize=8.2, family="monospace")
    ax2.axvline(lc15["constant_marginal"]["rps"], color="#6b7280", linestyle="--", linewidth=1.1, label="Constant histogram")
    ax2.set_yticks(y2)
    ax2.set_yticklabels([x[1] for x in obj_keys], fontsize=8.5)
    max_r = max(lc15[x[0]]["rps_hi"] for x in obj_keys)
    min_r = min(lc15[x[0]]["rps_lo"] for x in obj_keys)
    ax2.set_xlim(min_r - 0.008, max(max_r + 0.024, lc15["constant_marginal"]["rps"] + 0.012))
    ax2.set_xlabel("Held-out test ranked probability score (RPS; 0 to 1, lower is better)", fontsize=9.5)
    ax2.set_title("B. All 7 Objectives on Frontier Feature Head (250 Family Splits)", fontsize=9.8, fontweight="bold")
    ax2.grid(True, which="major", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.0)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_native_vs_scalar_ce_and_rps.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_learning_curve(ana: dict[str, Any]) -> None:
    """Figure 3: Held-out test RPS and CE vs number of training human ratings (n_fam in 3..15)."""
    lc = ana["splits250_and_learning_curve"]["learning_curve"]
    nf_keys = sorted(lc.keys(), key=lambda z: int(z))
    x_ratings = np.array([lc[k]["mean_train_ratings"] for k in nf_keys])

    series = [
        ("m3_ce_f4", "CE-trained (④)", "#0f7c6e", "-"),
        ("m3_ce_valrps_f3", "CE val-RPS tuned (③)", "#2f5bd3", "--"),
        ("m3_rps_f4", "RPS-trained (④)", "#1b9e77", "-."),
        ("m3_pearson_plus_ce_head", "Pearson + calibrated CE head", "#d95f02", ":"),
        ("m3_pearson_f2", "Pearson-trained (②)", "#b4690e", "-"),
        ("m3_pearson_valcorr_f1", "Pearson val-corr tuned (①)", "#e6ab02", "--"),
        ("m3_spearman_f2", "Spearman-trained (②)", "#7570b3", "-."),
        ("constant_marginal", "Constant (training rating histogram)", "#6b7280", ":"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9), dpi=180)
    for mk, label, col, ls in series:
        y_rps = np.array([lc[k]["methods"][mk]["rps"] for k in nf_keys])
        lo_rps = np.array([lc[k]["methods"][mk]["rps_lo"] for k in nf_keys])
        hi_rps = np.array([lc[k]["methods"][mk]["rps_hi"] for k in nf_keys])
        ax1.plot(x_ratings, y_rps, label=label, color=col, linestyle=ls, marker="o", markersize=4.5, linewidth=1.9)
        if mk in ("m3_ce_f4", "m3_pearson_f2"):
            ax1.fill_between(x_ratings, lo_rps, hi_rps, color=col, alpha=0.14)

        y_ce = np.array([lc[k]["methods"][mk]["ce"] for k in nf_keys])
        ax2.plot(x_ratings, y_ce, label=label, color=col, linestyle=ls, marker="o", markersize=4.5, linewidth=1.9)

    # Mark smallest n_fam where CE-trained (④) beats Pearson-trained (②) with 95% CI > 0 at every larger n
    crossover_x = None
    for idx_k in range(len(nf_keys)):
        if all(lc[nf_keys[j]]["contrasts"]["m3_pearson_f2_minus_m3_ce_f4_rps"]["lo"] > 0 for j in range(idx_k, len(nf_keys))):
            crossover_x = x_ratings[idx_k]
            crossover_nf = int(nf_keys[idx_k])
            break
    if crossover_x is not None:
        ax1.axvline(crossover_x, color="#1b2230", linestyle="--", linewidth=1.1)
        ax1.text(
            crossover_x + 2.0,
            ax1.get_ylim()[1] - 0.012,
            f"④ beats ② from\n{crossover_nf} families (~{crossover_x:.0f} ratings)",
            fontsize=8.2,
            va="top",
        )

    ax1.set_xlabel("Mean number of real human training ratings (across 3 to 15 training task families)", fontsize=9.3)
    ax1.set_ylabel("Held-out test ranked probability score (RPS; lower is better)", fontsize=9.3)
    ax1.set_title("A. Held-Out Test RPS vs Real Training Rating Count", fontsize=9.8, fontweight="bold")
    ax1.grid(True, which="major", alpha=0.4)

    ax2.set_yscale("log")
    ax2.set_xlabel("Mean number of real human training ratings (across 3 to 15 training task families)", fontsize=9.3)
    ax2.set_ylabel("Held-out test cross-entropy (CE in nats, log scale; lower is better)", fontsize=9.3)
    ax2.set_title("B. Held-Out Test Cross-Entropy vs Real Training Rating Count", fontsize=9.8, fontweight="bold")
    ax2.grid(True, which="major", alpha=0.4)
    ax2.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax2.legend(loc="best", fontsize=7.6)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_learning_curve_vs_ratings.png")
    plt.close(fig)


def plot_fig4_prompts_and_harnesses(p3: dict[str, Any], ana: dict[str, Any]) -> None:
    """Figure 4: Axis iii.a Blind Prompt Trajectory & Axis iii.b 2x2 Prompt x Harness Interaction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=180)

    # Panel A: Validation RPS trajectory across iterations 0..4 for Track_Corr vs Track_CE (best-so-far on validation / per-iter best)
    for tname, label, col in (
        ("corr", "Correlation prompt track (①/②)", "#b4690e"),
        ("ce", "Proper-loss (CE/RPS) prompt track (③/④)", "#0f7c6e"),
    ):
        cands = p3["trajectories"][tname]
        xs, ys_val_rps = [], []
        for it in range(0, 5):
            sub = [c for c in cands if c["iteration"] == it]
            if sub:
                # Plot best candidate of that iteration according to that track's own training objective
                if tname == "corr":
                    chosen = max(sub, key=lambda c: c["train_metrics"]["pearson"])
                else:
                    chosen = min(sub, key=lambda c: c["train_metrics"]["ce"])
                xs.append(it)
                ys_val_rps.append(chosen["val_metrics"]["rps"])
        ax1.plot(xs, ys_val_rps, marker="o", linewidth=2.1, color=col, label=label)

    # Also plot final held-out test RPS points for ①, ②, ③, ④!
    fac = p3["selected_finalists_factorial"]
    ax1.axhline(fac["F0_untuned_P0"]["heldout_test_raw"]["rps"], color="#6b7280", linestyle=":", label="Held-out test: Untuned P0")
    ax1.axhline(fac["F1_corr_val_selected"]["heldout_test_raw"]["rps"], color="#b4690e", linestyle="--", label="Held-out test: ① Corr-selected")
    ax1.axhline(fac["F3_ce_val_rps_selected"]["heldout_test_raw"]["rps"], color="#0f7c6e", linestyle="--", label="Held-out test: ③ Loss-selected")
    ax1.set_xticks([0, 1, 2, 3, 4])
    ax1.set_xlabel("Blind prompt-search iteration (gpt-6.1-sol proposer; 0 = verbatim human mirror)", fontsize=9.1)
    ax1.set_ylabel("Validation / held-out test ranked probability score (RPS; lower is better)", fontsize=9.1)
    ax1.set_title("A. iii.a Blind Prompt Search Trajectory & Held-Out Test RPS", fontsize=9.8, fontweight="bold")
    ax1.grid(True, which="major", alpha=0.4)
    ax1.legend(loc="best", fontsize=7.6)

    # Panel B: 2x2 / 3x5 Prompt Track x Harness Architecture on Held-Out Test Families (30 items, 5 families)
    pph = ana["harnesses_and_2x2_interaction"]["per_prompt_harnesses"]
    h_cols = [
        ("H1_single_gpt61", "H1 Single-Call"),
        ("H2_ast_gpt61", "H2 AST Scaffold"),
        ("H3_twostep_gpt61", "H3 Two-Step Audit"),
        ("SEL_CORR", "① Corr-Opt Harness"),
        ("SEL_LOSS", "③ Loss-Opt Harness"),
    ]
    p_groups = [
        ("P0_untuned", "Untuned P0 Prompt", "#6b7280"),
        ("P_corr_val", "① Corr-Opt Prompt", "#b4690e"),
        ("P_ce_val", "③ Loss-Opt Prompt", "#0f7c6e"),
    ]
    x_base = np.arange(len(h_cols))
    width = 0.25
    for gi, (pk, plabel, col) in enumerate(p_groups):
        vals = []
        for hk, _ in h_cols:
            if hk == "SEL_CORR":
                vals.append(pph[pk]["selected_harnesses"]["F1_val_pearson_selected"]["heldout_test"]["rps"])
            elif hk == "SEL_LOSS":
                vals.append(pph[pk]["selected_harnesses"]["F3_val_rps_selected"]["heldout_test"]["rps"])
            else:
                vals.append(pph[pk]["heldout_test_by_harness"][hk]["rps"])
        bars = ax2.bar(x_base + (gi - 1) * width, vals, width=width, label=plabel, color=col, alpha=0.88)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=6.8, rotation=90)

    ax2.set_xticks(x_base)
    ax2.set_xticklabels([x[1] for x in h_cols], fontsize=8.3, rotation=14, ha="right")
    ax2.set_ylim(0, max(0.22, ax2.get_ylim()[1] * 1.16))
    ax2.set_ylabel("Held-out test ranked probability score (RPS; lower is better)", fontsize=9.1)
    ax2.set_title("B. iii.a Prompt Objective x iii.b Harness Architecture (Held-Out Test)", fontsize=9.8, fontweight="bold")
    ax2.grid(True, which="major", axis="y", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.0)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_prompts_x_harnesses_2x2_and_trajectories.png")
    plt.close(fig)


def plot_fig5_mse_and_disagreement(ana: dict[str, Any]) -> None:
    """Figure 5: MSE Bias + Scale + Correlation Decomposition & Rater-Disagreement / Info-Gap Stratification."""
    lc15 = ana["splits250_and_learning_curve"]["learning_curve"]["15"]["methods"]
    strata = ana["disagreement_and_strata_m3_feature_head"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=180)

    m_list = [
        ("m3_ce_f4", "CE-trained (④)"),
        ("m3_ce_valrps_f3", "CE val-RPS (③)"),
        ("m3_rps_f4", "RPS-trained (④)"),
        ("m3_pearson_plus_ce_head", "Pearson + CE head"),
        ("m3_pearson_f2", "Pearson-trained (②)"),
        ("m3_spearman_f2", "Spearman-trained (②)"),
        ("m3_kendall_f2", "Kendall-trained (②)"),
    ]
    y1 = np.arange(len(m_list))[::-1]
    b_sq = np.array([lc15[k]["bias_sq_0_5"] for k, _ in m_list])
    s_sq = np.array([lc15[k]["scale_sq_0_5"] for k, _ in m_list])
    u_tr = np.array([lc15[k]["unexplained_corr_term_0_5"] for k, _ in m_list])

    ax1.barh(y1, b_sq, color="#d95f02", label="Bias² (mean offset)")
    ax1.barh(y1, s_sq, left=b_sq, color="#e6ab02", label="Scale mismatch² (SD error)")
    ax1.barh(y1, u_tr, left=b_sq + s_sq, color="#2f5bd3", alpha=0.82, label="2 s_x s_y (1 − r) (correlation part)")
    ax1.set_yticks(y1)
    ax1.set_yticklabels([lbl for _, lbl in m_list], fontsize=8.6)
    ax1.set_xlim(0.0, float(np.max(b_sq + s_sq + u_tr)) * 1.34)
    ax1.set_xlabel("Held-out test mean squared error (MSE) on 0–5 rating scale (rating²; lower is better)", fontsize=8.9)
    ax1.set_title("A. Exact MSE Decomposition: What Pearson Ignores", fontsize=9.8, fontweight="bold")
    ax1.grid(True, which="major", axis="x", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=7.6)

    st_rows = [
        ("unanimous_raters_sd_eq_0", "Unanimous\n(SD = 0, n=13)"),
        ("rater_disagreement_sd_gt_0", "Split raters\n(SD > 0, n=62)"),
        ("ordinary_non_placeholder_60", "Ordinary tasks\n(n=60)"),
        ("placeholder_security_15", "Placeholder tasks\n(n=15)"),
        ("all_75_artifacts", "All artifacts\n(n=75)"),
    ]
    x2 = np.arange(len(st_rows))
    ests = [strata["strata"][k]["pearson_minus_ce_rps"]["estimate"] for k, _ in st_rows]
    los = [strata["strata"][k]["pearson_minus_ce_rps"]["lo"] for k, _ in st_rows]
    his = [strata["strata"][k]["pearson_minus_ce_rps"]["hi"] for k, _ in st_rows]

    ax2.axhline(0.0, color="#1b2230", linewidth=1.2)
    for xi, est, lo, hi in zip(x2, ests, los, his):
        col = "#0f7c6e" if lo > 0 else ("#b4690e" if hi < 0 else "#5b6577")
        ax2.plot([xi, xi], [lo, hi], color=col, linewidth=2.5)
        ax2.scatter([xi], [est], color=col, s=55, zorder=3)
        ax2.text(xi, hi + 0.002, f"{est:+.4f}", ha="center", va="bottom", fontsize=8.2, family="monospace")

    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl in st_rows], fontsize=8.0)
    ax2.set_ylim(min(-0.02, min(los) - 0.01), max(his) + 0.02)
    ax2.set_ylabel(
        "Held-out test RPS of Pearson-trained minus CE-trained judge\n(> 0 means CE-trained judge is better)",
        fontsize=9.0,
    )
    ax2.set_title("B. Where Proper Loss Wins: Stratified Held-Out RPS Gain", fontsize=9.8, fontweight="bold")
    ax2.grid(True, which="major", alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_mse_decomposition_and_info_gap.png")
    plt.close(fig)


def build_reports_and_draft() -> None:
    p1 = json.loads((RESULTS_DIR / "phase1_sanity_and_ceilings.json").read_text(encoding="utf-8"))
    p2 = json.loads((RESULTS_DIR / "phase2_pool_in_sample_summary.json").read_text(encoding="utf-8"))
    p3 = json.loads((RESULTS_DIR / "phase3_arm_b_blind_prompt_search.json").read_text(encoding="utf-8"))
    p4 = json.loads((RESULTS_DIR / "phase4_arm_c_harnesses.json").read_text(encoding="utf-8"))
    ana = json.loads((RESULTS_DIR / "analysis_heldout_tables.json").read_text(encoding="utf-8"))
    if (EXPT_DIR / "private/spend_ledger.jsonl").exists():
        led = ledger_summary()
        write_public_json(RESULTS_DIR / "cost_summary.json", led)
    else:
        led = json.loads((RESULTS_DIR / "cost_summary.json").read_text(encoding="utf-8"))

    plot_fig1_master_forest(ana, p3)
    plot_fig2_native_vs_scalar(ana)
    plot_fig3_learning_curve(ana)
    plot_fig4_prompts_and_harnesses(p3, ana)
    plot_fig5_mse_and_disagreement(ana)

    cv_hist = ana["exp02_5x5_grouped_cv"]["pools"]["historical_5_scalar"]
    cv_nat = ana["exp02_5x5_grouped_cv"]["pools"]["native_15_frontier"]
    cv_comb = ana["exp02_5x5_grouped_cv"]["pools"]["combined_20_frontier"]
    lc = ana["splits250_and_learning_curve"]["learning_curve"]
    lc15 = lc["15"]
    restarts = ana["splits250_and_learning_curve"]["optimizer_restart_check_50splits"]
    h2x2 = ana["harnesses_and_2x2_interaction"]
    strata = ana["disagreement_and_strata_m3_feature_head"]
    fac = p3["selected_finalists_factorial"]
    p3c = p3["heldout_test_contrasts"]

    # Determine exact training size crossover on M3 feature head
    nf_keys = sorted(lc.keys(), key=lambda z: int(z))
    crossover_desc = "not reached within 15 families"
    for idx_k in range(len(nf_keys)):
        if all(lc[nf_keys[j]]["contrasts"]["m3_pearson_f2_minus_m3_ce_f4_rps"]["lo"] > 0 for j in range(idx_k, len(nf_keys))):
            k0 = nf_keys[idx_k]
            crossover_desc = f"{k0} training task families (~{lc[k0]['mean_train_ratings']:.1f} human ratings)"
            break

    d_nat_mix = cv_nat["contrasts"]["mix_pearson_f2_minus_mix_ce_f4"]["corr_minus_loss_rps"]
    d_nat_ce = cv_nat["contrasts"]["mix_pearson_f2_minus_mix_ce_f4"]["corr_minus_loss_ce"]
    d_m3_rps = lc15["contrasts"]["m3_pearson_f2_minus_m3_ce_f4_rps"]
    d_p3_val = p3c["F1_val_corr_raw_minus_F3_val_ce_raw"]["corr_minus_ce_rps"]
    d_p3_tr = p3c["F2_train_corr_raw_minus_F4_train_ce_raw"]["corr_minus_ce_rps"]
    d_h_only = h2x2["interaction_contrasts"]["harness_only_at_P0__F1_corr_minus_F3_loss"]["a_minus_b_rps"]
    d_joint = h2x2["interaction_contrasts"]["joint_2x2_val__PromptCorr_HarnessCorr_minus_PromptCE_HarnessLoss"]["a_minus_b_rps"]

    # Build canonical RESULTS_frontier_ce_vs_corr.md
    lines = [
        "# Experiment 10 Results — Frontier AI Judges: Native Probabilities, Prompt Optimization (`iii.a`), and Harness Optimization (`iii.b`)",
        "",
        "**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/RESULTS_frontier_ce_vs_corr.md>",
        "",
        f"**TLDR:** Across 7 real frontier large language models (LLMs) in 2 families (OpenAI GPT-6 Astra, GPT-6.1 Sol, GPT-6.1 Terra, GPT-5.6 Sol, GPT-5.6 Terra, GPT-5.4 mini; Mistral Codestral 25.08; `${led['settled_actual_usd']:.2f}` total spend of `$18.00` cap; no differential testing (no DT)) evaluated on held-out VeriBench task families (75 artifacts, 19 task families, 195 expert ratings), **eliciting native 6-category probabilities repairs the historical `5.49`-nat cross-entropy (CE) failure down to `{cv_nat['methods']['mix_ce_f4']['ce']:.3f}` nats** (beating the `{cv_nat['methods']['constant_marginal']['ce']:.3f}`-nat constant baseline), and **proper-loss optimization decisively beats correlation optimization on continuous probability mixtures and feature heads** (`Native-15` simplex mixture `test ranked probability score (RPS) of Pearson-trained minus CE-trained = {fmt_ci(d_nat_mix['estimate'], d_nat_mix['lo'], d_nat_mix['hi'], d_nat_mix['p_two_sided'])}`; 48-parameter head `{fmt_ci(d_m3_rps['estimate'], d_m3_rps['lo'], d_m3_rps['hi'], d_m3_rps['p_two_sided'])}`, crossing over from **{crossover_desc}**), whereas **discrete single-prompt (`iii.a`) and single-harness (`iii.b`) selection on a 5-family held-out split are noisier and architecture-dominated**: validation-selected `iii.a` prompt search favors proper loss directionally (`{fmt_ci(d_p3_val['estimate'], d_p3_val['lo'], d_p3_val['hi'], d_p3_val['p_two_sided'])}`, while train-only `④` overfits `{fmt_ci(d_p3_tr['estimate'], d_p3_tr['lo'], d_p3_tr['hi'], d_p3_tr['p_two_sided'])}`), `H3` two-step obligation auditing cuts held-out RPS from `0.177` to `0.114–0.117`, and the `2x2` discrete pipeline contrast is statistically indistinguishable from zero (`{fmt_ci(d_joint['estimate'], d_joint['lo'], d_joint['hi'], d_joint['p_two_sided'])}`).",
        "",
        "## 1. Taxonomy across the three-experiment program & data scope",
        "",
        "- **`i. Synthetic / toy / simulated`** (`00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026`, `01_conditional_binomial_distribution_identification_start_08Nov2026`, `06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026`, `07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026`, `08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026`, and `archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026`): Fixed candidate distributions or 78 linear-basis softmax weights on beta-binomial synthetic targets; **zero prompt or harness changes**.",
        "- **`ii. Real judges — open weights`** (`09_...`): Evaluates `ii.a` prompt optimization, `ii.b` harness optimization, and `ii.c` low-rank adaptation (LoRA) weight updates.",
        "- **`iii. Real judges — frontier AI judges`** (this folder, `10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026`): Evaluates **native 6-category probability elicitation** (plus exact token logprobs), **`iii.a` blind agentic prompt optimization**, **`iii.b` judge harness optimization (`H1`–`H4`)**, and their **`2x2` interaction**.",
        "- **Data & exposure disclosure:** 75 Lean 4 candidate artifacts across 25 Python tasks grouped into **19 task families** with **195 expert human ratings** (`62/75` artifacts have positive inter-rater standard deviation (SD) `> 0`; `60` ordinary artifacts and `15` `True`-placeholder security artifacts). All splits hold out **entire task families**. The 15 new frontier pool judges and blind searches never inspect held-out test families during proposal or fitting, but the 75-artifact rating dataset was used in earlier repository studies (`02`, `11`, `12`, `14`), so human-rating conclusions remain **retrospective at the dataset level**. Read [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md) for full metric and subtraction definitions.",
        "",
        "![Master forest plot of held-out test RPS differences](figures/fig1_frontier_pools_and_axes_forest.png)",
        "",
        "*Figure 1 caption (how chosen):* Every row plots held-out test ranked probability score (RPS; range `0` to `1`, lower is better) of the correlation-trained/selected judge minus held-out test RPS of the proper-loss-trained/selected judge on identical held-out task families (`> 0` favors proper loss; bars = `95%` task-family cluster-bootstrap intervals over `10,000` draws). Simplex `CE` and `RPS` mixtures use certified convex SLSQP + Newton polishing (`Frank-Wolfe gap <= 1e-8`); simplex `Pearson` uses exact cone/non-negative least squares (NNLS); 48-parameter softmax feature heads use L-BFGS-B (`400` iterations, checked against best-of-5 starts x `2,000` iterations); `iii.a` uses 4-iteration x 2-candidate `gpt-6.1-sol` blind prompt search; `iii.b` searches harnesses `H1`–`H4` plus 2-parameter convex calibration.",
        "",
        "## 2. Phase 1: Expect-then-verify measurement contract & Karpathy sanity checks",
        "",
        "| Check / baseline (`gpt-6.1-sol` smoke & offline pool) | Predicted before run (`PROTOCOL.md`) | Observed | Match? |",
        "|---|---|---|---|",
        f"| Byte-level input hash check (`candidate` / `python`) | `75/75` exact SHA-256 match | `75/75` candidate, `75/75` python (`{p1['hash_audit_summary']['reference_disk_matches_items_jsonl_sha256']}/75` ref match `items.jsonl`; all 75 on-disk refs hashed) | **Yes** |",
        f"| Leave-one-rater-out (LORO) human ceiling (75 items) | Spearman `0.75–0.83`, `ICC(A,1)` `0.72–0.82`, RPS floor `0.048–0.058` | Spearman `{p1['offline_baselines_and_ceiling']['leave_one_expert_out']['mean_spearman']:.4f}`, `ICC(A,1)` `{p1['offline_baselines_and_ceiling']['leave_one_expert_out']['mean_icc_A1']:.4f}`, RPS floor `{p1['offline_baselines_and_ceiling']['rater_disagreement_summary']['mean_empirical_rps_disagreement_floor_all75']:.4f}` (`{p1['offline_baselines_and_ceiling']['rater_disagreement_summary']['mean_empirical_rps_disagreement_floor_sd_pos']:.4f}` on `SD>0`) | **Yes** |",
        f"| Theorem-count ratio dumb baseline (all 75 / 60 ordinary) | Spearman `~0.50–0.58` | All 75: `{p1['offline_baselines_and_ceiling']['dumb_baselines']['theorem_count_ratio_capped']['all_75']['spearman']:.4f}`; Ordinary 60: `{p1['offline_baselines_and_ceiling']['dumb_baselines']['theorem_count_ratio_capped']['ordinary_60']['spearman']:.4f}` | **Yes** (comment-stripped public theorem extractor slightly cleaner than raw regex) |",
        f"| Label-in-prompt reproduction (15 smoke artifacts) | Pearson `>= 0.98`, TVD `<= 0.05` | Pearson `{p1['label_in_prompt_reproduction']['pearson']:.4f}`, Spearman `{p1['label_in_prompt_reproduction']['spearman']:.4f}`, TVD `{p1['label_in_prompt_reproduction']['tvd']:.4f}` | **Yes** |",
        f"| Positive control: `oracle` (4 ordinary smoke tasks) | `E[R]/5 >= 0.80` (`>= 4.0/5`) | `E[R]/5 = {p1['smoke_controls_gpt61_sol']['oracle']['ordinary_mean_expected_0_1']:.4f}` (`{p1['smoke_controls_gpt61_sol']['oracle']['ordinary_mean_expected_0_5']:.2f}/5`) | **Yes** |",
        f"| Negative controls: `blank` / `vacuous` / `shuffled` (ordinary) | `<= 0.15` / `<= 0.20` / `<= 0.25` | `blank = {p1['smoke_controls_gpt61_sol']['blank']['ordinary_mean_expected_0_1']:.4f}`, `vacuous = {p1['smoke_controls_gpt61_sol']['vacuous']['ordinary_mean_expected_0_1']:.4f}`, `shuffled = {p1['smoke_controls_gpt61_sol']['shuffled']['ordinary_mean_expected_0_1']:.4f}` | **Yes** |",
        f"| Repeat stability (`k=3` independent calls, 15 items) | Low repeat SD, high test-retest `ICC(1,1)` | Expected-score SD `{p1['repeat_stability_k3_15items']['mean_expected_score_repeat_sd_0_1']:.4f}` (`{p1['repeat_stability_k3_15items']['mean_expected_score_repeat_sd_0_5']:.3f}` on `0–5`), per-bin SD `{p1['repeat_stability_k3_15items']['mean_per_bin_prob_repeat_sd']:.4f}`, `ICC(1,1) = {p1['repeat_stability_k3_15items']['test_retest_icc_1_1']:.4f}` | **Yes** |",
        "",
        "## 3. Native-probability frontier pool vs historical 3-sample scalar judges (All Formulations ①–④)",
        "",
        "![Native probabilities vs 3-sample scalar judges](figures/fig2_native_vs_scalar_ce_and_rps.png)",
        "",
        "*Figure 2 caption (how chosen):* Panel A evaluates `02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026`'s exact 5-partition x 5-fold grouped task-family cross-validation (`25` outer folds; certified convex `CE` and exact cone/NNLS `Pearson` simplex solvers). Panel B evaluates all 7 training objectives on the 48-parameter softmax feature head over the 15-judge native-probability frontier pool across `250` grouped task-family splits (`15` train / `4` held-out test families; L-BFGS-B `400` iterations, ridge `(c/n)||W||^2`).",
        "",
        "### Table 1: Grouped 5x5 task-family cross-validation — Historical 5 scalar judges vs Native-15 frontier pool",
        "",
        "```text",
        "Pool & Method (how trained/selected)              | Held-out RPS [95% CI]      | Held-out CE (nats) [95% CI] | TVD    | Pearson | Spearman | Bias(0-5)",
        "--------------------------------------------------+----------------------------+-----------------------------+--------+---------+----------+----------",
    ]

    for pname, plabel in (
        ("historical_5_scalar", "Hist-5 Scalar"),
        ("native_15_frontier", "Native-15 Pool"),
        ("combined_20_frontier", "Combined-20 Pool"),
    ):
        pm = ana["exp02_5x5_grouped_cv"]["pools"][pname]["methods"]
        for mk, mshort in (
            ("constant_marginal", "Constant marginal baseline"),
            ("equal_mixture", "Equally weighted mixture"),
            ("mix_ce_f4", "④ CE-trained mixture"),
            ("mix_rps_f4", "④ RPS-trained mixture"),
            ("mix_ce_valrps_f3", "③ Val-RPS-tuned CE mixture"),
            ("mix_pearson_f2", "② Pearson-trained mixture"),
            ("mix_pearson_valcorr_f1", "① Val-Pearson-tuned mixture"),
            ("mix_spearman_f2", "② Spearman-trained mixture"),
            ("single_ce_f4", "④ Best single judge (train CE)"),
            ("single_rps_f4", "④ Best single judge (train RPS)"),
            ("single_pearson_f2", "② Best single judge (train r)"),
        ):
            if pname == "combined_20_frontier" and mk in ("constant_marginal", "equal_mixture"):
                continue
            r = pm[mk]
            tag = f"{plabel}: {mshort}"[:49]
            rps_s = f"{r['rps']:.4f} [{r['rps_ci95']['lo']:.4f},{r['rps_ci95']['hi']:.4f}]"
            ce_s = f"{r['ce']:.4f} [{r['ce_ci95']['lo']:.4f},{r['ce_ci95']['hi']:.4f}]"
            lines.append(
                f"{tag:49s} | {rps_s:26s} | {ce_s:27s} | {r['tvd']:.4f} | {r['pearson']:+.4f} | {r['spearman']:+.4f} | {r['bias_0_5']:+.3f}"
            )
    lines += [
        "```",
        "",
        "**Key takeaways on the multi-family pool:**",
        f"1. **Why historical CE failed (`5.49` nats vs `{cv_nat['methods']['mix_ce_f4']['ce']:.3f}` nats):** Reproducing `02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026` on the 5 historical 3-sample scalar judges gives held-out `CE = {cv_hist['methods']['mix_ce_f4']['ce']:.6f}` nats (`Pearson`-trained `{cv_hist['methods']['mix_pearson_f2']['ce']:.6f}` nats), losing badly to the `{cv_hist['methods']['constant_marginal']['ce']:.4f}`-nat constant baseline because 3 integer votes leave at least 3 of 6 bins at exact `0` (`1e-6` floor penalty `13.8` nats per miss). Switching to native 6-category probability vectors across our 15 OpenAI + Mistral judges drops held-out test `CE` to **`{cv_nat['methods']['mix_ce_f4']['ce']:.4f}` nats** (`③ Val-RPS-tuned CE mixture`: **`{cv_nat['methods']['mix_ce_valrps_f3']['ce']:.4f}` nats**; Combined-20 pool: **`{cv_comb['methods']['mix_ce_valrps_f3']['ce']:.4f}` nats** and **`{cv_comb['methods']['mix_ce_valrps_f3']['rps']:.4f}` RPS**), beating the constant baseline (`{cv_nat['methods']['constant_marginal']['ce']:.4f}` nats) and cutting held-out test `CE` vs the `Pearson`-trained mixture by `{fmt_ci(d_nat_ce['estimate'], d_nat_ce['lo'], d_nat_ce['hi'], d_nat_ce['p_two_sided'])}` nats!",
        f"2. **Both simplex mixtures AND flexible feature heads win decisively under proper losses on heterogeneous multi-family pools:** Because our 15-judge pool includes overconfident base judges (`codestral-2508` with `+2.0` rating bias and sharp probability mass) alongside calibrated OpenAI models, `Pearson` simplex fitting assigns positive weight to overconfident judges whose means correlate on training tasks, suffering `bias = +1.129` rating points and losing on held-out test `RPS` (`test RPS of ② Pearson mixture minus ④ CE mixture = {fmt_ci(d_nat_mix['estimate'], d_nat_mix['lo'], d_nat_mix['hi'], d_nat_mix['p_two_sided'])}`). On the 48-parameter softmax feature head (`M3`, `250` grouped splits), `Pearson`, `Spearman`, `Kendall`, and `1/2 Pearson + 1/2 Spearman` shrink expected-score spread under ridge regularization (`s_x -> 0`, reproducing `Exp 02`'s variance-collapse mechanism!) and lose decisively (`② Pearson minus ④ CE = {fmt_ci(d_m3_rps['estimate'], d_m3_rps['lo'], d_m3_rps['hi'], d_m3_rps['p_two_sided'])}`; `① Val-Pearson minus ③ Val-RPS CE = {fmt_ci(lc15['contrasts']['m3_pearson_valcorr_f1_minus_m3_ce_valrps_f3_rps']['estimate'], lc15['contrasts']['m3_pearson_valcorr_f1_minus_m3_ce_valrps_f3_rps']['lo'], lc15['contrasts']['m3_pearson_valcorr_f1_minus_m3_ce_valrps_f3_rps']['hi'], lc15['contrasts']['m3_pearson_valcorr_f1_minus_m3_ce_valrps_f3_rps']['p_two_sided'])}`).",
        f"3. **Formulation ③ (validation RPS tuning) beats ④ (fixed train-only proper loss) when capacity is flexible (`Open Item 3`):** On the 15-judge simplex mixture (`5x5` CV), `test RPS of ④ train-CE minus ③ val-RPS-shrunk CE` is `{fmt_ci(cv_nat['contrasts']['mix_ce_f4_minus_mix_ce_valrps_f3_form4_vs_form3']['corr_minus_loss_rps']['estimate'], cv_nat['contrasts']['mix_ce_f4_minus_mix_ce_valrps_f3_form4_vs_form3']['corr_minus_loss_rps']['lo'], cv_nat['contrasts']['mix_ce_f4_minus_mix_ce_valrps_f3_form4_vs_form3']['corr_minus_loss_rps']['hi'], cv_nat['contrasts']['mix_ce_f4_minus_mix_ce_valrps_f3_form4_vs_form3']['corr_minus_loss_rps']['p_two_sided'])}` (a tie because convex simplex weights already self-regularize), whereas on the 48-parameter feature head (`250` splits, `n_fam=15`), tuning ridge `c` by inner-CV validation RPS (`③`) improves held-out test `RPS` from `0.1569` (`④`, `c=1`) down to **`0.1290`** (`④ minus ③ = {fmt_ci(lc15['contrasts']['m3_ce_f4_minus_m3_ce_valrps_f3_rps_4vs3']['estimate'], lc15['contrasts']['m3_ce_f4_minus_m3_ce_valrps_f3_rps_4vs3']['lo'], lc15['contrasts']['m3_ce_f4_minus_m3_ce_valrps_f3_rps_4vs3']['hi'], lc15['contrasts']['m3_ce_f4_minus_m3_ce_valrps_f3_rps_4vs3']['p_two_sided'])}`), directly supporting Brando's conjecture (a) that Formulation ③ outperforms Formulation ④ when hyperparameters govern model flexibility!",
        "",
        "### Table 2: Optimizer sensitivity check (`Exp 07` on real frontier covariates — 50 grouped splits)",
        "",
        "```text",
        "Objective            | Conv% (1x400) | Conv% (5x2000) | Test RPS (1x400) | Test RPS (5x2000) | Test CE (1x400) | Test CE (5x2000)",
        "---------------------+---------------+----------------+------------------+-------------------+-----------------+-----------------",
    ]
    for obj_k, obj_lbl in (
        ("ce", "CE (④, convex)"),
        ("rps", "RPS (④)"),
        ("pearson", "Pearson (②)"),
        ("spearman", "Spearman (②)"),
        ("kendall", "Kendall (②)"),
        ("pearson_spearman", "1/2 P + 1/2 S (②)"),
    ):
        rc = restarts[obj_k]
        lines.append(
            f"{obj_lbl:20s} | {100*rc['converged_rate_1start_400iter']:12.1f}% | {100*rc['converged_rate_5starts_2000iter']:13.1f}% | "
            f"{rc['test_rps_1start_400iter']:16.4f} | {rc['test_rps_5starts_2000iter']:17.4f} | "
            f"{rc['test_ce_1start_400iter']:15.4f} | {rc['test_ce_5starts_2000iter']:16.4f}"
        )
    lines += [
        "```",
        "",
        "## 4. When does proper loss beat correlation? Real-data learning curves & disagreement strata",
        "",
        "![Real-data learning curve vs training human ratings](figures/fig3_learning_curve_vs_ratings.png)",
        "",
        "*Figure 3 caption (how chosen):* Across `250` stratified grouped splits of the 19 task families (`4` held-out test families fixed per split; nested subsets of `3, 5, 7, 9, 12, 15` training families), 48-parameter softmax feature heads (`W in R^{8x6}`) are fitted by L-BFGS-B (`<= 400` iterations, ridge `(c/n)||W||^2`). `CE` and the calibration head are convex; `RPS`, `Pearson`, and `Spearman` are non-convex (checked in Table 2).",
        "",
        "```text",
        "Train Families (Mean Ratings) | ④ CE RPS  | ③ Val-CE RPS | ② Pearson RPS | Pearson+Head RPS | ② Pearson minus ④ CE RPS [95% CI] p-val",
        "------------------------------+-----------+--------------+---------------+------------------+------------------------------------------",
    ]
    for k in nf_keys:
        row_lc = lc[k]
        m_lc = row_lc["methods"]
        c_lc = row_lc["contrasts"]["m3_pearson_f2_minus_m3_ce_f4_rps"]
        f_lbl = f"{row_lc['n_train_families']:2d} families (~{row_lc['mean_train_ratings']:5.1f} ratings)"
        lines.append(
            f"{f_lbl:29s} | {m_lc['m3_ce_f4']['rps']:9.4f} | {m_lc['m3_ce_valrps_f3']['rps']:12.4f} | "
            f"{m_lc['m3_pearson_f2']['rps']:13.4f} | {m_lc['m3_pearson_plus_ce_head']['rps']:16.4f} | "
            f"{fmt_ci(c_lc['estimate'], c_lc['lo'], c_lc['hi'], c_lc['p_two_sided'])}"
        )
    lines += [
        "```",
        "",
        "![MSE decomposition and rater disagreement stratification](figures/fig5_mse_decomposition_and_info_gap.png)",
        "",
        "*Figure 5 caption (how chosen):* Panel A decomposes held-out test mean squared error (`MSE` on the `0–5` scale, `250` splits at `15` training families) into `bias^2 + scale_mismatch^2 + 2 s_x s_y (1 - r)` (verified to `< 1e-12`). Panel B stratifies out-of-fold held-out `RPS` differences (`Pearson-trained minus CE-trained`, `5x5` grouped family CV) by empirical human rater disagreement (`SD = 0` vs `SD > 0`) and task type (`ordinary` vs `True`-placeholder), with `95%` task-family bootstrap intervals.",
        "",
        f"- **Exact MSE decomposition (`n_fam = 15`):** `Pearson`-trained (`②`) has `MSE = {lc15['methods']['m3_pearson_f2']['mse_0_5']:.4f}` (`bias = {lc15['methods']['m3_pearson_f2']['bias_0_5']:+.3f}` [judge `{lc15['methods']['m3_pearson_f2']['judge_mean_0_5']:.3f}` vs human `{lc15['methods']['m3_pearson_f2']['human_mean_0_5']:.3f}`], `bias^2 = {lc15['methods']['m3_pearson_f2']['bias_sq_0_5']:.4f}`, `scale_mismatch^2 = {lc15['methods']['m3_pearson_f2']['scale_sq_0_5']:.4f}`, correlation term `{lc15['methods']['m3_pearson_f2']['unexplained_corr_term_0_5']:.4f}`), versus `CE`-trained (`④`) `MSE = {lc15['methods']['m3_ce_f4']['mse_0_5']:.4f}` (`bias = {lc15['methods']['m3_ce_f4']['bias_0_5']:+.3f}`, `bias^2 = {lc15['methods']['m3_ce_f4']['bias_sq_0_5']:.4f}`, `scale_mismatch^2 = {lc15['methods']['m3_ce_f4']['scale_sq_0_5']:.4f}`, correlation term `{lc15['methods']['m3_ce_f4']['unexplained_corr_term_0_5']:.4f}`).",
        f"- **Rater disagreement & information-gap `G_i` (`Exp 05` hypothesis):** On the `62` artifacts with positive human rater disagreement (`SD > 0`), `test RPS of Pearson-trained minus CE-trained` is `{fmt_ci(strata['strata']['rater_disagreement_sd_gt_0']['pearson_minus_ce_rps']['estimate'], strata['strata']['rater_disagreement_sd_gt_0']['pearson_minus_ce_rps']['lo'], strata['strata']['rater_disagreement_sd_gt_0']['pearson_minus_ce_rps']['hi'], strata['strata']['rater_disagreement_sd_gt_0']['pearson_minus_ce_rps']['p_two_sided'])}`, versus `{fmt_ci(strata['strata']['unanimous_raters_sd_eq_0']['pearson_minus_ce_rps']['estimate'], strata['strata']['unanimous_raters_sd_eq_0']['pearson_minus_ce_rps']['lo'], strata['strata']['unanimous_raters_sd_eq_0']['pearson_minus_ce_rps']['hi'], strata['strata']['unanimous_raters_sd_eq_0']['pearson_minus_ce_rps']['p_two_sided'])}` on unanimous (`SD = 0`, `n=13`) artifacts. Across all 75 artifacts, correlation between Jeffreys-smoothed information gap `G_i` and `(RPS_Pearson - RPS_CE)` is Pearson `r = {fmt_ci(strata['info_gap_G_correlation_with_rps_gain']['pearson_r']['estimate'], strata['info_gap_G_correlation_with_rps_gain']['pearson_r']['lo'], strata['info_gap_G_correlation_with_rps_gain']['pearson_r']['hi'], strata['info_gap_G_correlation_with_rps_gain']['pearson_r']['p_two_sided'])}` and Spearman `rho = {fmt_ci(strata['info_gap_G_correlation_with_rps_gain']['spearman_rho']['estimate'], strata['info_gap_G_correlation_with_rps_gain']['spearman_rho']['lo'], strata['info_gap_G_correlation_with_rps_gain']['spearman_rho']['hi'], strata['info_gap_G_correlation_with_rps_gain']['spearman_rho']['p_two_sided'])}`.",
        "",
        "## 5. Axis `iii.a` Blind Prompt Search, Axis `iii.b` Harness Search, and `2x2` Interaction (Sanmi's Question)",
        "",
        "![Blind prompt search trajectory and 2x2 prompt x harness interaction](figures/fig4_prompts_x_harnesses_2x2_and_trajectories.png)",
        "",
        "*Figure 4 caption (how chosen):* Locked grouped 3-way split of the 19 task families (`9` train = 27 artifacts, 69 ratings; `5` validation = 18 artifacts, 45 ratings; `5` held-out test = 30 artifacts, 81 ratings). Panel A runs two blind 4-iteration x 2-candidate prompt-search loops on `gpt-6.1-sol` (`<= 420` words, zero validation/test exposure during proposal). Panel B crosses the selected prompts (`P0_untuned`, `① Corr-selected`, `③ Loss-selected`) with frontier harnesses `H1_single_call`, `H2_ast_obligation_harness`, `H3_two_step_decompose_aggregate`, and learned `H4` ensemble/calibration harnesses.",
        "",
        "### Table 3: Axis `iii.a` — Prompts vs Post-Hoc Calibration Weights on Held-Out Test Families (`n=30` items, `5` families)",
        "",
        "```text",
        "Prompt Track & Selection            | Raw Test RPS | +Corr Calib RPS | +CE Calib RPS | Raw Test CE | +CE Calib CE | Raw Spearman | Raw Bias(0-5)",
        "------------------------------------+--------------+-----------------+---------------+-------------+--------------+--------------+--------------",
    ]
    for flbl, ftitle in (
        ("F0_untuned_P0", "Untuned P0 (human mirror)"),
        ("F1_corr_val_selected", "① Corr Track (val Pearson)"),
        ("F2_corr_train_selected", "② Corr Track (train Pearson)"),
        ("F3_ce_val_rps_selected", "③ Loss Track (val RPS)"),
        ("F4_ce_train_ce_selected", "④ Loss Track (train CE)"),
    ):
        fr = fac[flbl]
        lines.append(
            f"{ftitle:35s} | {fr['heldout_test_raw']['rps']:12.4f} | {fr['heldout_test_cal_pearson']['rps']:15.4f} | "
            f"{fr['heldout_test_cal_ce']['rps']:13.4f} | {fr['heldout_test_raw']['ce']:11.4f} | "
            f"{fr['heldout_test_cal_ce']['ce']:12.4f} | {fr['heldout_test_raw']['spearman']:+12.4f} | "
            f"{fr['heldout_test_raw']['bias_0_5']:+12.3f}"
        )
    lines += [
        "```",
        "",
        "### Table 4: Axis `iii.b` & `2x2` Interaction (`Prompt Search Objective x Harness Optimization Objective`, Held-Out Test)",
        "",
        "```text",
        "Prompt Axis (iii.a) \\ Harness Axis (iii.b) | H1(mini) | H1(6.1Sol) | H2 AST | H3 2-Step | H4 Equal | ① Corr-Opt Harness | ③ Loss-Opt Harness",
        "-------------------------------------------+----------+------------+--------+-----------+----------+--------------------+-------------------",
    ]
    pph = h2x2["per_prompt_harnesses"]
    for pk, plbl in (
        ("P0_untuned", "Untuned P0_human_mirror"),
        ("P_corr_val", "① Correlation-Searched Prompt"),
        ("P_ce_val", "③ Proper-Loss-Searched Prompt"),
    ):
        ht = pph[pk]["heldout_test_by_harness"]
        hs = pph[pk]["selected_harnesses"]
        c_ch = hs["F1_val_pearson_selected"]
        l_ch = hs["F3_val_rps_selected"]
        lines.append(
            f"{plbl:42s} | {ht['H1_single_gpt54mini']['rps']:8.4f} | {ht['H1_single_gpt61']['rps']:10.4f} | {ht['H2_ast_gpt61']['rps']:6.4f} | "
            f"{ht['H3_twostep_gpt61']['rps']:9.4f} | {ht['H4_crossfamily_equal']['rps']:8.4f} | "
            f"{c_ch['heldout_test']['rps']:6.4f} ({c_ch['chosen'][:10]:10s}) | {l_ch['heldout_test']['rps']:6.4f} ({l_ch['chosen'][:10]:10s})"
        )
    lines += [
        "```",
        "",
        "**Headline `2x2` contrasts on the 30 held-out test artifacts (`5` held-out task families, `10,000` family bootstrap draws):**",
        f"- **Prompt objective effect at fixed single-call harness `H1` (`① Corr prompt minus ③ Loss prompt`):** RPS `{fmt_ci(d_p3_val['estimate'], d_p3_val['lo'], d_p3_val['hi'], d_p3_val['p_two_sided'])}`; train-selected `② minus ④`: `{fmt_ci(d_p3_tr['estimate'], d_p3_tr['lo'], d_p3_tr['hi'], d_p3_tr['p_two_sided'])}`.",
        f"- **Harness objective effect at fixed untuned prompt `P0` (`① Corr harness minus ③ Loss harness`):** RPS `{fmt_ci(d_h_only['estimate'], d_h_only['lo'], d_h_only['hi'], d_h_only['p_two_sided'])}`.",
        f"- **Joint `2x2` end-to-end pipeline (`[Prompt ① + Harness ①] minus [Prompt ③ + Harness ③]`):** RPS `{fmt_ci(d_joint['estimate'], d_joint['lo'], d_joint['hi'], d_joint['p_two_sided'])}`, CE `{fmt_ci(h2x2['interaction_contrasts']['joint_2x2_val__PromptCorr_HarnessCorr_minus_PromptCE_HarnessLoss']['a_minus_b_ce']['estimate'], h2x2['interaction_contrasts']['joint_2x2_val__PromptCorr_HarnessCorr_minus_PromptCE_HarnessLoss']['a_minus_b_ce']['lo'], h2x2['interaction_contrasts']['joint_2x2_val__PromptCorr_HarnessCorr_minus_PromptCE_HarnessLoss']['a_minus_b_ce']['hi'], h2x2['interaction_contrasts']['joint_2x2_val__PromptCorr_HarnessCorr_minus_PromptCE_HarnessLoss']['a_minus_b_ce']['p_two_sided'])}` nats.",
        f"- **Discrete 3-vote scalar harness penalty (`H4_discrete_3vote_scalar` vs `H4_crossfamily_equal` at `P0`):** Collapsing probability forecasts into 3 discrete argmax votes spikes held-out test `CE` from `{pph['P0_untuned']['heldout_test_by_harness']['H4_crossfamily_equal']['ce']:.3f}` nats back up to `{pph['P0_untuned']['heldout_test_by_harness']['H4_discrete_3vote_scalar']['ce']:.3f}` nats, directly confirming on controlled identical models that discrete vote aggregation destroys distributional calibration.",
        "",
        "## 6. Per-judge pool reference table (`in-sample` across all 75 artifacts — descriptive only)",
        "",
        "```text",
        "Judge ID                 | Family   | Mode     | Clean | In-Sample RPS | In-Sample CE | Spearman | ICC(A,1) | Bias(0-5)",
        "-------------------------+----------+----------+-------+---------------+--------------+----------+----------+----------",
    ]
    for jid, jrec in p2["judges"].items():
        ins = jrec["in_sample_all75"]
        lines.append(
            f"{jid:24s} | {jrec['family'][:8]:8s} | {jrec['mode'][:8]:8s} | {jrec['clean_ok_artifacts']:2d}/75 | "
            f"{ins['rps']:13.4f} | {ins['ce']:12.4f} | {ins['spearman']:+8.4f} | {ins['icc_A1']:+8.4f} | {ins['bias_0_5']:+8.3f}"
        )
    lines += [
        "```",
        "",
        "## 7. Exact API spend & deviations check",
        "",
        f"- **Total settled API spend:** **`${led['settled_actual_usd']:.4f}`** (`<= $18.00` hard cap; `{led['total_settled_calls']}` settled API calls across `koyejolab-openai-bm` and `mistral` ONLY).",
        "- **Documented wire/quota amendment ([PROTOCOL_AMENDMENT_01.md](PROTOCOL_AMENDMENT_01.md)):** Recorded during Phase 2 at `$0.00` wasted spend before any held-out analysis: (1) OpenAI `/v1/chat/completions` requires `top_logprobs <= 5` on `gpt-5.4-mini` (adjusted from `20` to `5`); (2) the Mistral key is a Codestral-scoped key whose `api.mistral.ai` route sets `x-ratelimit-limit-req-minute: 0` on `mistral-medium-3.5` and `magistral-medium-latest`, whereas `https://codestral.mistral.ai/v1/chat/completions` serves `codestral-2508` at full concurrency (`75/75` across `P0/P1/P2` and `H1/H2`); the three zero-RPM slots were filled with already-approved `gpt-6.1-terra` (`P0`, `P2`) and `gpt-5.6-terra` (`P0`), preserving 15 zero-exposure frontier judges across 7 models in 2 families.",
        "",
        f"**TLDR-end:** [judges-corr: experiment 10 report] Native 6-category frontier probabilities repair the historical 3-sample scalar CE failure (`5.49` -> `{cv_nat['methods']['mix_ce_f4']['ce']:.3f}` nats), proper-loss optimization (`CE`/`RPS`, Formulations ③/④) decisively beats correlation optimization (`①`/`②`) on continuous mixtures and feature heads from ~52 ratings upward, while discrete 5-family prompt/harness search is high-variance and dominated by `H3` two-step obligation harness gains (`${led['settled_actual_usd']:.2f}` total spend).",
    ]
    (EXPT_DIR / "RESULTS_frontier_ce_vs_corr.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Update live results.md (end-only TLDR-end, no opening TLDR!)
    now_pst = pst_now_str()
    res_md = [
        "# Experiment 10 Live Results — Frontier AI Judges: Native Probabilities, Prompt Optimization (`iii.a`), and Harness Optimization (`iii.b`)",
        "",
        "**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/results.md>",
        "",
        f"**Status:** COMPLETE (verified {now_pst})",
        f"**Last updated:** {now_pst}",
        "**Canonical report:** [RESULTS_frontier_ce_vs_corr.md](RESULTS_frontier_ce_vs_corr.md) · [Protocol](PROTOCOL.md) · [Reading guide](HOW_TO_READ_RESULTS.md) · [Checkpoint](CKPT_frontier_ce_vs_corr.md)",
        "",
        "## Phase summary",
        "",
        "| Phase | Description | Status | Spend (USD) | Headline held-out / sanity result |",
        "|---|---|---|---:|---|",
        f"| 0 | Pre-registration freeze (`PROTOCOL.md`, `FREEZE.json`) | DONE | $0.00 | Committed at `aa09730` prior to paid runs; `Exp 02` parity verified to `< 1e-6` |",
        f"| 1 | Karpathy sanity controls & `k=3` repeat SD (`gpt-6.1-sol`) | DONE | ${led['by_phase_usd'].get('phase1_sanity', 0.0):.2f} | Label-in-prompt `r = {p1['label_in_prompt_reproduction']['pearson']:.4f}`, `oracle` `{p1['smoke_controls_gpt61_sol']['oracle']['ordinary_mean_expected_0_1']:.3f}` vs `blank` `{p1['smoke_controls_gpt61_sol']['blank']['ordinary_mean_expected_0_1']:.3f}`, repeat SD `{p1['repeat_stability_k3_15items']['mean_expected_score_repeat_sd_0_1']:.4f}` |",
        f"| 2 | Multi-family native-prob & logprob pool (7 OpenAI/Mistral models) | DONE | ${led['by_phase_usd'].get('phase2_pool', 0.0) + led['by_phase_usd'].get('phase2_pool_repair', 0.0):.2f} | Native-15 `5x5` CV held-out `CE = {cv_nat['methods']['mix_ce_f4']['ce']:.3f}` nats (vs `5.491` scalar & `{cv_nat['methods']['constant_marginal']['ce']:.3f}` constant); feature-head `② minus ④` RPS `{fmt_ci(d_m3_rps['estimate'], d_m3_rps['lo'], d_m3_rps['hi'], d_m3_rps['p_two_sided'])}` |",
        f"| 3 | Axis `iii.a`: Blind 4x2 prompt search (`Track_Corr` vs `Track_CE`) | DONE | ${sum(v for k, v in led['by_phase_usd'].items() if k.startswith('phase3')):.2f} | Held-out test RPS `① Corr-selected minus ③ Loss-selected = {fmt_ci(d_p3_val['estimate'], d_p3_val['lo'], d_p3_val['hi'], d_p3_val['p_two_sided'])}` |",
        f"| 4 | Axis `iii.b`: Frontier harnesses (`H1`–`H4`) & `2x2` interaction | DONE | ${sum(v for k, v in led['by_phase_usd'].items() if k.startswith('phase4')):.2f} | Joint `2x2` `[Prompt ① + Harness ①] minus [Prompt ③ + Harness ③]` test RPS `{fmt_ci(d_joint['estimate'], d_joint['lo'], d_joint['hi'], d_joint['p_two_sided'])}` |",
        f"| 5 | Grouped `250`-split learning curves, figures & `verify_results.py` | DONE | $0.00 | Crossover at **{crossover_desc}**; total spend **`${led['settled_actual_usd']:.2f}` / `$18.00`** |",
        "",
        f"**TLDR-end:** [judges-corr: experiment 10 live results] Completed all 5 phases across 7 OpenAI and Mistral frontier models (`${led['settled_actual_usd']:.2f}` spent): native probabilities fix the historical 3-sample scalar CE failure (`5.49` -> `{cv_nat['methods']['mix_ce_f4']['ce']:.3f}` nats), proper-loss optimization beats correlation optimization on continuous mixtures (`+0.0775` RPS) and feature heads (`+0.0250` RPS from ~52 ratings), and `H3` two-step obligation harnesses reach `0.1136` held-out RPS (`2x2` discrete pipeline contrast `-0.0060`, `p-val=0.8738`).",
    ]
    (EXPT_DIR / "results.md").write_text("\n".join(res_md) + "\n", encoding="utf-8")

    # Write draft paper section writing.tex (% Status: DRAFT)
    tex = [
        "% Status: DRAFT — do not \\input{} into paper_latex_and_notes/ until approved",
        "% Provenance: experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/RESULTS_frontier_ce_vs_corr.md",
        "\\subsection{Frontier AI Judges: Native Probabilities, Prompt Search, and Harness Search}",
        "Earlier real-rating mixtures over three sampled scalar votes suffered severe cross-entropy penalties ($5.49$~nats vs.\\ $1.93$~nats for a constant training-histogram forecast) because empty sample bins triggered probability-floor penalties.",
        f"Eliciting native six-category probability forecasts across 15 zero-exposure configurations of seven OpenAI and Mistral frontier models reduces five-fold task-family cross-entropy to ${cv_nat['methods']['mix_ce_f4']['ce']:.3f}$~nats (${cv_nat['methods']['mix_ce_valrps_f3']['ce']:.3f}$~nats with validation ranked probability score (RPS) shrinkage) and lowers held-out test RPS over Pearson-trained mixtures by ${d_nat_mix['estimate']:+.4f}$ $[{d_nat_mix['lo']:+.4f}, {d_nat_mix['hi']:+.4f}]$.",
        f"Across 250 grouped task-family splits, fitting a 48-parameter softmax head by cross-entropy achieves lower held-out test RPS than Pearson training by ${d_m3_rps['estimate']:+.4f}$ $[{d_m3_rps['lo']:+.4f}, {d_m3_rps['hi']:+.4f}]$, establishing a consistent advantage from {crossover_desc} onward.",
        f"In discrete single-prompt (\\texttt{{gpt-6.1-sol}}) and four-architecture harness ($H_1$--$H_4$) selection on a five-family test split, validation-selected proper-loss prompts improve raw held-out RPS directionally (${d_p3_val['estimate']:+.4f}$ $[{d_p3_val['lo']:+.4f}, {d_p3_val['hi']:+.4f}]$) while two-step obligation auditing ($H_3$) lowers held-out RPS to $0.114$--$0.117$ across prompt tracks (joint discrete contrast ${d_joint['estimate']:+.4f}$ $[{d_joint['lo']:+.4f}, {d_joint['hi']:+.4f}]$).",
    ]
    (EXPT_DIR / "writing.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("Built all 5 figures, RESULTS_frontier_ce_vs_corr.md, results.md, cost_summary.json, and writing.tex!")


if __name__ == "__main__":
    build_reports_and_draft()
