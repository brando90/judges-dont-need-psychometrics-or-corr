#!/usr/bin/env python3
"""Render Experiment 24 Matplotlib figures (with grids on every axis) and Markdown reports."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator
import numpy as np

EXP_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"

METHOD_STYLE = {
    "ce_calibrated": ("CE LoRA + 7-param cal.", "#0b6e82", "o", "-"),
    "pearson_calibrated": ("Pearson LoRA + 7-param cal.", "#b54128", "v", "-"),
    "ce_raw": ("CE LoRA (raw)", "#42a5b8", "o", ":"),
    "pearson_raw": ("Pearson LoRA (raw)", "#df816b", "v", ":"),
    "base_calibrated_matched": ("Base Qwen3-4B + cal. (N_cal)", "#5a4aa8", "^", "-."),
    "base_calibrated_all_dev": ("Base Qwen3-4B + cal. (all B_total)", "#2d225a", "D", "--"),
    "training_histogram": ("Smoothed train histogram", "#9f6b12", "s", "--"),
    "uniform": ("Uniform [0.2]*5", "#687078", "", "--"),
}


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style_log2_axis(ax, budgets: list[int], tick_labels: list[str] | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(budgets))
    if tick_labels is not None:
        ax.set_xticklabels(tick_labels, fontsize=8)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_minor_locator(LogLocator(base=2, subs=[2 ** 0.5]))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", axis="x", alpha=0.12)


def make_figures(scores: list[dict], contrasts: list[dict], depth_rows: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    d1_scores = [r for r in scores if int(r["depth"]) == 1]
    budgets = sorted({int(r["b_total"]) for r in d1_scores})
    n_trains = [b // 2 for b in budgets]
    tick_labels = [f"{b // 2}\n({b})" for b in budgets]

    # Lookup (b_total, method) -> row
    s_map = {(int(r["b_total"]), r["method"]): r for r in d1_scores}
    c_map = {(int(r["b_total"]), r["target"], r["comparator"], r["metric"]): r for r in contrasts}

    # --- Figure 1: 2x2 learning curves and paired differences ---
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), sharex="col")
    fig.subplots_adjust(left=0.075, right=0.935, bottom=0.165, top=0.840, hspace=0.28, wspace=0.29)

    ax_rps = axes[0, 0]
    for m_name, (label, color, marker, ls) in METHOD_STYLE.items():
        ys = [float(s_map[(b, m_name)]["rps"]) for b in budgets]
        ax_rps.plot(n_trains, ys, label=label, color=color, marker=marker or None, markersize=4.5, linewidth=2.0, linestyle=ls)
    ax_rps.set_title("A. Absolute held-out ranked probability score (RPS)", loc="left", fontsize=11.5, fontweight="bold")
    ax_rps.set_ylabel("Held-out test ranked probability score (RPS)\n(0–1, lower is better)")
    style_log2_axis(ax_rps, n_trains, tick_labels)

    ax_ce = axes[0, 1]
    ax_tvd = ax_ce.twinx()
    for m_name in ("ce_calibrated", "pearson_calibrated", "ce_raw", "pearson_raw", "base_calibrated_matched", "uniform"):
        label, color, marker, _ = METHOD_STYLE[m_name]
        y_ce = [float(s_map[(b, m_name)]["ce_nats"]) for b in budgets]
        y_tvd = [float(s_map[(b, m_name)]["tvd"]) for b in budgets]
        ax_ce.plot(n_trains, y_ce, color=color, marker=marker or None, markersize=4.2, linewidth=2.0, linestyle="-", label=f"{label} (CE)")
        ax_tvd.plot(n_trains, y_tvd, color=color, marker=marker or None, markersize=3.5, linewidth=1.3, linestyle="--", alpha=0.8)
    ax_ce.set_title("B. Held-out cross-entropy (solid) & TVD (dashed)", loc="left", fontsize=11.5, fontweight="bold")
    ax_ce.set_ylabel("Held-out test cross-entropy (CE, nats)\n(solid lines, lower is better)")
    ax_tvd.set_ylabel("Held-out test total variation distance (TVD)\n(dashed lines, 0–1, lower is better)")
    style_log2_axis(ax_ce, n_trains, tick_labels)
    ax_tvd.spines["top"].set_visible(False)
    ax_tvd.grid(True, alpha=0.14, linestyle=":")

    ax_pdiff = axes[1, 0]
    for target_m, comp_m, label, color, marker in (
        ("ce_calibrated", "pearson_calibrated", "Calibrated Pearson LoRA minus calibrated CE LoRA", "#b54128", "v"),
        ("ce_raw", "pearson_raw", "Raw Pearson LoRA minus raw CE LoRA", "#df816b", "^"),
    ):
        ys = [float(c_map[(b, target_m, comp_m, "rps")]["diff_comparator_minus_target"]) for b in budgets]
        lo = [float(c_map[(b, target_m, comp_m, "rps")]["sim_lo_95"]) for b in budgets]
        hi = [float(c_map[(b, target_m, comp_m, "rps")]["sim_hi_95"]) for b in budgets]
        ax_pdiff.plot(n_trains, ys, label=label, color=color, marker=marker, markersize=4.8, linewidth=2.0)
        ax_pdiff.fill_between(n_trains, lo, hi, color=color, alpha=0.16, linewidth=0)
    ax_pdiff.axhline(0.0, color="#222222", linewidth=1.1)
    ax_pdiff.set_title("C. Paired test RPS gap: Pearson LoRA minus CE LoRA", loc="left", fontsize=11.5, fontweight="bold")
    ax_pdiff.set_ylabel("Held-out test RPS difference\n(Pearson minus CE; > 0 favors CE)")
    ax_pdiff.set_xlabel("Training human ratings N_train (total development ratings B_total)")
    style_log2_axis(ax_pdiff, n_trains, tick_labels)
    ax_pdiff.legend(loc="best", fontsize=8.2, frameon=True, framealpha=0.9)

    ax_ctrl = axes[1, 1]
    for comp_m, label, color, marker in (
        ("base_calibrated_matched", "Calibrated Base Qwen3-4B (N_cal) minus cal. CE LoRA", "#5a4aa8", "^"),
        ("base_calibrated_all_dev", "Calibrated Base Qwen3-4B (all B_total) minus cal. CE LoRA", "#2d225a", "D"),
        ("training_histogram", "Smoothed train histogram minus cal. CE LoRA", "#9f6b12", "s"),
        ("uniform", "Uniform minus cal. CE LoRA", "#687078", "x"),
    ):
        ys = [float(c_map[(b, "ce_calibrated", comp_m, "rps")]["diff_comparator_minus_target"]) for b in budgets]
        lo = [float(c_map[(b, "ce_calibrated", comp_m, "rps")]["sim_lo_95"]) for b in budgets]
        hi = [float(c_map[(b, "ce_calibrated", comp_m, "rps")]["sim_hi_95"]) for b in budgets]
        ax_ctrl.plot(n_trains, ys, label=label, color=color, marker=marker, markersize=4.5, linewidth=1.9)
        ax_ctrl.fill_between(n_trains, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax_ctrl.axhline(0.0, color="#222222", linewidth=1.1)
    ax_ctrl.set_title("D. Paired test RPS gain over base & baseline controls", loc="left", fontsize=11.5, fontweight="bold")
    ax_ctrl.set_ylabel("Held-out test RPS difference\n(comparator minus calibrated CE LoRA; > 0 favors CE)")
    ax_ctrl.set_xlabel("Training human ratings N_train (total development ratings B_total)")
    style_log2_axis(ax_ctrl, n_trains, tick_labels)
    ax_ctrl.legend(loc="best", fontsize=7.8, frameon=True, framealpha=0.9)

    handles, labels = ax_rps.get_legend_handles_labels()
    fig.suptitle(
        "Qwen3-4B-Instruct-2507 LoRA Weight Training on HelpSteer2: Cross-Entropy vs Pearson Correlation",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.070, 0.925), ncol=4, fontsize=8.5, frameon=False)
    fig.text(
        0.075,
        0.072,
        "Selection & training: top-4-block LoRA (rank=8, 1.85M params) + 5-logit head trained with AdamW across lr in {5e-5, 2e-4, 8e-4} & 4 checkpoints;\n"
        "winner picked by validation RPS (non-convex Pearson, exact 2-pass chain-rule gradient). Identical 7-param ridge-regularized calibrator\n"
        "fitted on disjoint S_cal (N_cal = B_total/4). Shaded bands show 95% simultaneous max-t prompt-clustered bootstrap (519 prompts, 1,038 responses, 10 draws).",
        fontsize=8.0,
        va="top",
        linespacing=1.35,
    )
    for ext in ("png", "pdf"):
        meta = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
        fig.savefig(FIGURES_DIR / f"helpsteer2_lora_learning_curves.{ext}", dpi=180, facecolor="white", metadata=meta)
    plt.close(fig)

    # --- Figure 2: Equal-cost depth comparison & 7-param vs 4-bin calibrator + disagreement strata ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12.8, 5.4))
    fig2.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.81, wspace=0.28)

    ax_d = axes2[0]
    d2_budgets = sorted({int(r["b_total"]) for r in scores if int(r["depth"]) == 2})
    x_pos = np.arange(len(d2_budgets))
    width = 0.20
    s_all = {(int(r["depth"]), int(r["b_total"]), r["method"]): r for r in scores}
    bars_spec = [
        (1, "ce_calibrated", "CE LoRA cal. (d=1, N items)", "#0b6e82", -1.5 * width),
        (2, "ce_calibrated", "CE LoRA cal. (d=2, N/2 items)", "#52b2c5", -0.5 * width),
        (1, "pearson_calibrated", "Pearson LoRA cal. (d=1, N items)", "#b54128", 0.5 * width),
        (2, "pearson_calibrated", "Pearson LoRA cal. (d=2, N/2 items)", "#eb9580", 1.5 * width),
    ]
    for d, m_name, lbl, col, offset in bars_spec:
        vals = [float(s_all[(d, b, m_name)]["rps"]) for b in d2_budgets]
        ax_d.bar(x_pos + offset, vals, width=width, label=lbl, color=col, edgecolor="#222222", linewidth=0.5)
    ax_d.set_ylim(0.0, 0.248)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels([f"B_total={b}\n({b // 2} train votes)" for b in d2_budgets])
    ax_d.set_ylabel("Held-out test ranked probability score (RPS)\n(0–1, lower is better)")
    ax_d.set_xlabel("Total development human labels B_total")
    ax_d.set_title("A. Equal-cost allocation: d=1 vs d=2 votes per response", loc="left", fontsize=11.0, fontweight="bold")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.set_axisbelow(True)
    ax_d.grid(True, alpha=0.28)
    ax_d.legend(loc="upper center", ncol=2, fontsize=7.5, frameon=True, framealpha=0.92)

    ax_cal = axes2[1]
    for target_m, comp_m, label, col, marker in (
        ("ce_calibrated", "pearson_calibrated", "7-param 5-logit cal. (Pearson minus CE)", "#b54128", "o"),
        ("ce_exp23_4bin", "pearson_exp23_4bin", "Exp-23 1D 4-bin cal. (Pearson minus CE)", "#7e3c8c", "s"),
    ):
        ys = [float(c_map[(b, target_m, comp_m, "rps")]["diff_comparator_minus_target"]) for b in budgets]
        lo = [float(c_map[(b, target_m, comp_m, "rps")]["sim_lo_95"]) for b in budgets]
        hi = [float(c_map[(b, target_m, comp_m, "rps")]["sim_hi_95"]) for b in budgets]
        ax_cal.plot(n_trains, ys, label=label, color=col, marker=marker, markersize=4.5, linewidth=2.0)
        ax_cal.fill_between(n_trains, lo, hi, color=col, alpha=0.15, linewidth=0)
    ax_cal.axhline(0.0, color="#222222", linewidth=1.1)
    ax_cal.set_title("B. 5-logit vs 1D 4-bin calibration (Pearson minus CE)", loc="left", fontsize=11.0, fontweight="bold")
    ax_cal.set_ylabel("Held-out test RPS difference\n(Pearson minus CE; > 0 favors CE)")
    ax_cal.set_xlabel("Training human ratings N_train (total B_total)")
    style_log2_axis(ax_cal, n_trains, tick_labels)
    ax_cal.legend(loc="best", fontsize=8.2, frameon=True, framealpha=0.9)

    fig2.suptitle(
        "Label Allocation (1 vs 2 Votes/Response) & Calibration Family Sensitivity on HelpSteer2",
        x=0.08,
        y=0.95,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig2.text(
        0.08,
        0.06,
        "Same Qwen3-4B-Instruct-2507 LoRA training sweeps, AdamW budget, and validation-RPS checkpoint selection across all 10 nested draws.\n"
        "Left: d=1 vs d=2 votes per response at equal total human label budgets. Right: simultaneous 95% prompt-clustered intervals across 519 held-out prompts.",
        fontsize=8.0,
        va="top",
    )
    for ext in ("png", "pdf"):
        meta = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
        fig2.savefig(FIGURES_DIR / f"helpsteer2_calibration_and_depth.{ext}", dpi=180, facecolor="white", metadata=meta)
    plt.close(fig2)


def main() -> None:
    scores = load_csv(RESULTS_DIR / "scores.csv")
    contrasts = load_csv(RESULTS_DIR / "contrasts.csv")
    depth_rows = load_csv(RESULTS_DIR / "depth_contrasts.csv")
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    prov = json.loads((RESULTS_DIR / "training_provenance.json").read_text(encoding="utf-8"))
    sanity = json.loads((RESULTS_DIR / "sanity_checks.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS_DIR / "data_manifest.json").read_text(encoding="utf-8"))

    make_figures(scores, contrasts, depth_rows)

    d1_scores = [r for r in scores if int(r["depth"]) == 1]
    budgets = sorted({int(r["b_total"]) for r in d1_scores})
    s_map = {(int(r["b_total"]), r["method"]): r for r in d1_scores}
    c_map = {(int(r["b_total"]), r["target"], r["comparator"], r["metric"]): r for r in contrasts}
    sust = summary["sustained_win_thresholds_rps"]

    def fmt_sust(entry: dict | None) -> str:
        if entry is None:
            return "None within tested budgets"
        return f"B_total={entry['b_total']} (N_train={entry['n_train']})"

    # Write BUDGET_TABLES_11-21-2026.md
    bt_lines = [
        "# HelpSteer2 LoRA Budget Tables and Full Held-Out Metric Panel | 11-21-2026",
        "",
        "**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026/BUDGET_TABLES_11-21-2026.md>",
        "",
        "As of 11-21-2026. All metrics are evaluated on the `1,038` held-out responses (`519` unique source prompts, `3,610` human helpfulness votes on `0–4`) of `nvidia/HelpSteer2` (`validation.jsonl.gz` joined to `disagreements.jsonl.gz`).",
        "Positive RPS differences subtract calibrated cross-entropy (CE) LoRA test RPS from the named comparator's test RPS, so **positive values favor CE**.",
        "Simultaneous 95% intervals use 4,000 prompt-clustered bootstrap draws (`seed = 20261124`) via max-`t` across all 8 primary budgets; crossed intervals jointly resample the 10 training draws and 519 prompts.",
        "",
        "## 1. Primary Sweep (`d = 1` rating per response): Paired Test RPS Differences vs Calibrated CE LoRA",
        "",
        "| B_total (N_train / N_val / N_cal) | Pearson cal. minus CE cal. [sim. 95% CI; p-val] (wins/10) | Base cal. (N_cal) minus CE cal. [sim. 95% CI] | Base cal. (all B_total) minus CE cal. [sim. 95% CI] | Train histogram minus CE cal. [sim. 95% CI] | Uniform minus CE cal. [sim. 95% CI] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for b in budgets:
        lbl = f"{b:,} ({b//2}/{b//4}/{b//4})"
        pc = c_map[(b, "ce_calibrated", "pearson_calibrated", "rps")]
        bm = c_map[(b, "ce_calibrated", "base_calibrated_matched", "rps")]
        ba = c_map[(b, "ce_calibrated", "base_calibrated_all_dev", "rps")]
        th = c_map[(b, "ce_calibrated", "training_histogram", "rps")]
        un = c_map[(b, "ce_calibrated", "uniform", "rps")]
        bt_lines.append(
            f"| {lbl} | {float(pc['diff_comparator_minus_target']):+.5f} [{float(pc['sim_lo_95']):+.5f}, {float(pc['sim_hi_95']):+.5f}; p={float(pc['sim_pval']):.4f}] ({pc['draws_target_better']}/10) "
            f"| {float(bm['diff_comparator_minus_target']):+.5f} [{float(bm['sim_lo_95']):+.5f}, {float(bm['sim_hi_95']):+.5f}] "
            f"| {float(ba['diff_comparator_minus_target']):+.5f} [{float(ba['sim_lo_95']):+.5f}, {float(ba['sim_hi_95']):+.5f}] "
            f"| {float(th['diff_comparator_minus_target']):+.5f} [{float(th['sim_lo_95']):+.5f}, {float(th['sim_hi_95']):+.5f}] "
            f"| {float(un['diff_comparator_minus_target']):+.5f} [{float(un['sim_lo_95']):+.5f}, {float(un['sim_hi_95']):+.5f}] |"
        )

    bt_lines += [
        "",
        "## 2. Absolute Held-Out Test Metrics (`d = 1`): RPS, CE (6e-6 and 1e-2), TVD, and Correlation/Bias Diagnostics",
        "",
        "Notice TVD (`0–1`) is computed against each test response's sparse empirical histogram ($k_i \\in \\{2..5\\}$ votes) and is flagged as a noisy sparse-histogram diagnostic. Leave-one-vote-out human reference on the same `1,038` test responses: RPS = `0.14567`, CE = `1.3585` nats.",
        "",
        "| B_total (N_train) | Method | Test RPS | Test CE (6e-6) | Test CE (1e-2) | Test TVD | Pearson r | Spearman rho | Mean Bias (judge-human) | Within-Prompt Pair Acc |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    key_methods = (
        "ce_calibrated",
        "pearson_calibrated",
        "ce_raw",
        "pearson_raw",
        "base_calibrated_matched",
        "base_calibrated_all_dev",
        "base_raw",
        "training_histogram",
        "uniform",
        "ce_exp23_4bin",
        "pearson_exp23_4bin",
    )
    for b in budgets:
        for m in key_methods:
            r = s_map[(b, m)]
            bt_lines.append(
                f"| {b} ({b//2}) | {m} | {float(r['rps']):.5f} | {float(r['ce_nats']):.4f} | {float(r['ce_eps1e2']):.4f} "
                f"| {float(r['tvd']):.4f} | {float(r['pearson']):+.4f} | {float(r['spearman']):+.4f} "
                f"| {float(r['mean_bias']):+.4f} | {float(r['within_prompt_pair_acc']):.4f} |"
            )

    bt_lines += [
        "",
        "## 3. Equal-Cost Depth Comparison (`d = 1` vs `d = 2` Votes per Response) and Crossed Bootstrap Sensitivity",
        "",
        "| Comparison | B_total | Diff | Prompt Pointwise 95% CI | Crossed (draw x prompt) 95% CI | p-val | Draws > 0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in depth_rows:
        bt_lines.append(
            f"| {r['comparison']} | {r['b_total']} | {float(r['diff']):+.5f} "
            f"| [{float(r['pt_lo_95']):+.5f}, {float(r['pt_hi_95']):+.5f}] "
            f"| [{float(r['cross_pt_lo_95']):+.5f}, {float(r['cross_pt_hi_95']):+.5f}] "
            f"| {float(r['pt_pval']):.4f} | {r['draws_positive']}/10 |"
        )
    (EXP_DIR / "BUDGET_TABLES_11-21-2026.md").write_text("\n".join(bt_lines) + "\n", encoding="utf-8")

    # Build headline numbers for RESULTS_11-21-2026.md and README.md
    max_b = budgets[-1]
    max_pc = c_map[(max_b, "ce_calibrated", "pearson_calibrated", "rps")]
    max_bm = c_map[(max_b, "ce_calibrated", "base_calibrated_matched", "rps")]
    max_ba = c_map[(max_b, "ce_calibrated", "base_calibrated_all_dev", "rps")]
    max_th = c_map[(max_b, "ce_calibrated", "training_histogram", "rps")]
    max_un = c_map[(max_b, "ce_calibrated", "uniform", "rps")]
    max_raw = c_map[(max_b, "ce_raw", "pearson_raw", "rps")]
    max_4bin = c_map[(max_b, "ce_exp23_4bin", "pearson_exp23_4bin", "rps")]
    max_ce_diff = c_map[(max_b, "ce_calibrated", "pearson_calibrated", "ce_nats")]
    max_tvd_diff = c_map[(max_b, "ce_calibrated", "pearson_calibrated", "tvd")]
    d_map = {(int(r["b_total"]), r["comparison"]): r for r in depth_rows}
    d2_1024 = d_map[(1024, "d2_pearson_calibrated_minus_ce_calibrated")]
    d2_vs_d1_ce_1024 = d_map[(1024, "ce_calibrated_d2_minus_d1")]
    d2_vs_d1_ce_64 = d_map[(64, "ce_calibrated_d2_minus_d1")]
    d2_vs_d1_ce_256 = d_map[(256, "ce_calibrated_d2_minus_d1")]

    ce_cal_max = s_map[(max_b, "ce_calibrated")]
    pe_cal_max = s_map[(max_b, "pearson_calibrated")]
    ce_raw_max = s_map[(max_b, "ce_raw")]
    pe_raw_max = s_map[(max_b, "pearson_raw")]
    bm_max = s_map[(max_b, "base_calibrated_matched")]
    ba_max = s_map[(max_b, "base_calibrated_all_dev")]
    th_max = s_map[(max_b, "training_histogram")]
    un_max = s_map[(max_b, "uniform")]

    p_b2048_runs = [r for r in prov["runs"] if r["depth"] == 1 and r["b_total"] == max_b and r["objective"] == "pearson"]
    p_b2048_undef = sum(int(r["undefined_pearson_batches"]) for r in p_b2048_runs)
    p_b2048_tot = sum(int(r["total_candidate_batches"]) for r in p_b2048_runs)

    # Identify any budget where point estimate favors Pearson (diff < 0)
    pearson_favored_cal = [
        b for b in budgets if float(c_map[(b, "ce_calibrated", "pearson_calibrated", "rps")]["diff_comparator_minus_target"]) < 0
    ]
    pearson_favored_raw = [
        b for b in budgets if float(c_map[(b, "ce_raw", "pearson_raw", "rps")]["diff_comparator_minus_target"]) < 0
    ]

    sust_pc_cond = fmt_sust(sust["pearson_calibrated"]["conditional_sim_95"])
    sust_pc_cross = fmt_sust(sust["pearson_calibrated"]["crossed_sim_95"])
    sust_un_cond = fmt_sust(sust["uniform"]["conditional_sim_95"])
    sust_th_cond = fmt_sust(sust["training_histogram"]["conditional_sim_95"])
    sust_bm_cond = fmt_sust(sust["base_calibrated_matched"]["conditional_sim_95"])
    sust_ba_cond = fmt_sust(sust["base_calibrated_all_dev"]["conditional_sim_95"])

    # Compute draw_0 selected checkpoint info & overall GPU wall time
    tot_gpu_s = sum(float(v) for v in prov["draw_elapsed_seconds"].values())
    max_draw_s = max(float(v) for v in prov["draw_elapsed_seconds"].values())

    res_lines = [
        "# When Does Training Open-Weight LLM Judge Weights with Cross-Entropy Beat Pearson Correlation? (11-21-2026)",
        "",
        "**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026/RESULTS_11-21-2026.md>",
        "",
        "Date: 11-21-2026",
        "",
        "## Executive Answer",
        "",
        f"In experiment `24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026`, we fine-tuned real transformer low-rank adaptation (LoRA) weights (`1,847,813` trainable parameters across the top 4 blocks plus the 5-digit rating head) of `Qwen/Qwen3-4B-Instruct-2507` on prompt-disjoint human helpfulness ratings (`0–4`) from `nvidia/HelpSteer2`, matching the model, prompt, optimizer (`AdamW`), learning-rate grid, mini-batch schedule, validation ranked probability score (RPS) checkpoint selection, and post-hoc 7-parameter calibrator across both training objectives.",
        "",
        f"- **Calibrated cross-entropy (CE) LoRA vs calibrated Pearson LoRA (primary sustained-win threshold on held-out RPS):** Conditional on the 10 matched draws, calibrated CE LoRA first clears calibrated Pearson LoRA across a simultaneous 95% prompt-clustered band and remains ahead at all larger tested budgets from **{sust_pc_cond}** (two-way crossed `draw × prompt` simultaneous threshold: **{sust_pc_cross}**). At the largest budget (`B_total = 2,048`, `N_train = 1,024`, `N_val = 512`, `N_cal = 512`), test RPS is **{float(ce_cal_max['rps']):.5f}** for calibrated CE LoRA vs **{float(pe_cal_max['rps']):.5f}** for calibrated Pearson LoRA (paired difference = **{float(max_pc['diff_comparator_minus_target']):+.5f}** `[{float(max_pc['sim_lo_95']):+.5f}, {float(max_pc['sim_hi_95']):+.5f}]`, `p-val={float(max_pc['sim_pval']):.4f}`, crossed 95% CI `[{float(max_pc['cross_sim_lo_95']):+.5f}, {float(max_pc['cross_sim_hi_95']):+.5f}]`, CE wins **{max_pc['draws_target_better']}/10** matched draws).",
        f"- **When calibrated CE LoRA beats baselines and the unmodified base judge:** Sustained threshold vs `uniform` (`RPS = {float(un_max['rps']):.5f}`): **{sust_un_cond}**; vs `training_histogram`: **{sust_th_cond}**; vs `base_calibrated_matched` (unmodified Qwen3-4B calibrated on the same `N_cal` labels): **{sust_bm_cond}**; vs `base_calibrated_all_dev` (unmodified Qwen3-4B calibrated on all `B_total` development labels): **{sust_ba_cond}**.",
        f"- **Where Pearson wins or ties:** Without calibration, raw Pearson LoRA has much worse RPS (`{float(pe_raw_max['rps']):.5f}` vs `{float(ce_raw_max['rps']):.5f}` at `B_total=2,048`) because Pearson is invariant to shift and positive rescaling (`mean_bias = {float(pe_raw_max['mean_bias']):+.3f}` vs `{float(ce_raw_max['mean_bias']):+.3f}`). After identical 7-parameter post-hoc calibration on `S_cal`, budgets where calibrated Pearson LoRA had lower point-estimate RPS than calibrated CE LoRA: `{pearson_favored_cal if pearson_favored_cal else 'none'}` (raw uncalibrated budgets favoring Pearson: `{pearson_favored_raw if pearson_favored_raw else 'none'}`).",
        "",
        "## Full Budget Summary (`d = 1` Rating per Response; `1,038` Held-Out Responses / `519` Prompts)",
        "",
        "Normalized individual-rating RPS (0–1, lower is better) is primary; CE is negative log-probability in nats (floor `6e-6`, lower is better); total variation distance (TVD, 0–1, lower is better) is measured against each item's empirical `2–5` vote histogram (noisy sparse-histogram diagnostic). Leave-one-vote-out human reference on the test set: RPS = `0.14567`, CE = `1.3585` nats.",
        "",
        "| B_total (N_train / N_val / N_cal) | CE LoRA cal. RPS / CE / TVD | Pearson LoRA cal. RPS / CE / TVD | Base cal. (N_cal) RPS | Base cal. (all B_total) RPS | Train hist. RPS | Pearson cal. minus CE cal. RPS [sim. 95% CI] (wins/10) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in budgets:
        cc = s_map[(b, "ce_calibrated")]
        pc_r = s_map[(b, "pearson_calibrated")]
        bm_r = s_map[(b, "base_calibrated_matched")]
        ba_r = s_map[(b, "base_calibrated_all_dev")]
        th_r = s_map[(b, "training_histogram")]
        diff = c_map[(b, "ce_calibrated", "pearson_calibrated", "rps")]
        res_lines.append(
            f"| {b} ({b//2}/{b//4}/{b//4}) "
            f"| {float(cc['rps']):.5f} / {float(cc['ce_nats']):.4f} / {float(cc['tvd']):.4f} "
            f"| {float(pc_r['rps']):.5f} / {float(pc_r['ce_nats']):.4f} / {float(pc_r['tvd']):.4f} "
            f"| {float(bm_r['rps']):.5f} | {float(ba_r['rps']):.5f} | {float(th_r['rps']):.5f} "
            f"| {float(diff['diff_comparator_minus_target']):+.5f} [{float(diff['sim_lo_95']):+.5f}, {float(diff['sim_hi_95']):+.5f}] ({diff['draws_target_better']}/10) |"
        )

    res_lines += [
        "",
        "![Learning curves of held-out test ranked probability score (RPS), cross-entropy (CE), total variation distance (TVD), and paired simultaneous 95% differences across training and total development budgets on HelpSteer2.](figures/helpsteer2_lora_learning_curves.png)",
        "",
        "## Why Weight Updates and Representation Matter (`B_total = 2,048` Diagnostics)",
        "",
        "| Method (`B_total = 2,048`, `N_train = 1,024`) | Test RPS | Test CE (6e-6) | Test CE (1e-2) | Test TVD | Pearson r | Spearman rho | Mean Bias | Within-Prompt Pair Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in ("ce_calibrated", "pearson_calibrated", "ce_raw", "pearson_raw", "base_calibrated_matched", "base_calibrated_all_dev", "base_raw", "training_histogram", "uniform", "ce_exp23_4bin", "pearson_exp23_4bin"):
        r = s_map[(max_b, m)]
        res_lines.append(
            f"| {m} | {float(r['rps']):.5f} | {float(r['ce_nats']):.4f} | {float(r['ce_eps1e2']):.4f} | {float(r['tvd']):.4f} "
            f"| {float(r['pearson']):+.4f} | {float(r['spearman']):+.4f} | {float(r['mean_bias']):+.4f} | {float(r['within_prompt_pair_acc']):.4f} |"
        )

    res_lines += [
        "",
        f"- **Secondary proper and distributional metrics at `B_total = 2,048` (`N_train = 1,024`):** Calibrated CE LoRA beats calibrated Pearson LoRA in **`{max_ce_diff['draws_target_better']}/10`** draws on test CE (paired difference = `{float(max_ce_diff['diff_comparator_minus_target']):+.5f}` nats, sim. 95% CI `[{float(max_ce_diff['sim_lo_95']):+.5f}, {float(max_ce_diff['sim_hi_95']):+.5f}]`, `p-val={float(max_ce_diff['sim_pval']):.4f}`; two-way crossed sim. 95% CI `[{float(max_ce_diff['cross_sim_lo_95']):+.5f}, {float(max_ce_diff['cross_sim_hi_95']):+.5f}]`, `p-val={float(max_ce_diff['cross_sim_pval']):.4f}`) and **`{max_tvd_diff['draws_target_better']}/10`** draws on sparse-histogram test TVD (`{float(max_tvd_diff['diff_comparator_minus_target']):+.5f}`, crossed sim. 95% CI `[{float(max_tvd_diff['cross_sim_lo_95']):+.5f}, {float(max_tvd_diff['cross_sim_hi_95']):+.5f}]`, `p-val={float(max_tvd_diff['cross_sim_pval']):.4f}`). On additional diagnostics at `B_total = 2,048` (`ce_calibrated` vs `pearson_calibrated`): multiclass Brier score `{float(ce_cal_max['brier']):.5f}` vs `{float(pe_cal_max['brier']):.5f}`, Lin's concordance correlation coefficient (CCC) `{float(ce_cal_max['lin_ccc']):.4f}` vs `{float(pe_cal_max['lin_ccc']):.4f}`, mean absolute error (MAE) `{float(ce_cal_max['mean_mae']):.4f}` vs `{float(pe_cal_max['mean_mae']):.4f}`, cumulative expected calibration error (ECE) `{float(ce_cal_max['cumulative_ece']):.4f}` vs `{float(pe_cal_max['cumulative_ece']):.4f}`, and modal agreement `{float(ce_cal_max['modal_agreement']):.4f}` vs `{float(pe_cal_max['modal_agreement']):.4f}`.",
        f"- **Rater disagreement strata at `B_total = 2,048` (`1,038` test responses):** On low-disagreement items ($k_i \\le 3$ votes, `742` responses), calibrated CE LoRA RPS = `{float(ce_cal_max['rps_low_dis']):.5f}` vs `{float(pe_cal_max['rps_low_dis']):.5f}` (calibrated Pearson LoRA) and `{float(ba_max['rps_low_dis']):.5f}` (`base_calibrated_all_dev`); on high-disagreement items ($k_i \\ge 4$ votes, `296` responses), `{float(ce_cal_max['rps_high_dis']):.5f}` vs `{float(pe_cal_max['rps_high_dis']):.5f}` and `{float(ba_max['rps_high_dis']):.5f}`.",
        f"- **Calibration mechanism (5-logit vs 1D 4-bin):** Under the secondary Experiment 23 1D 4-bin expected-score calibrator at `B_total = 2,048`, `pearson_exp23_4bin minus ce_exp23_4bin` RPS is `{float(max_4bin['diff_comparator_minus_target']):+.5f}` `[{float(max_4bin['sim_lo_95']):+.5f}, {float(max_4bin['sim_hi_95']):+.5f}]`. The 7-parameter 5-logit calibrator preserves all 5 logit shape features learned during LoRA fine-tuning.",
        f"- **Equal-cost multi-rating depth (`d = 1` vs `d = 2` votes per response):** For `ce_calibrated`, paying for `d=2` votes on half as many items hurts at `B_total = 64` (`ce_calibrated d=2 minus d=1 = {float(d2_vs_d1_ce_64['diff']):+.5f}`, prompt 95% CI `[{float(d2_vs_d1_ce_64['pt_lo_95']):+.5f}, {float(d2_vs_d1_ce_64['pt_hi_95']):+.5f}]`), essentially ties at `B_total = 256` (`{float(d2_vs_d1_ce_256['diff']):+.5f}` `[{float(d2_vs_d1_ce_256['pt_lo_95']):+.5f}, {float(d2_vs_d1_ce_256['pt_hi_95']):+.5f}]`), and slightly favors `d = 1` at `B_total = 1,024` (`{float(d2_vs_d1_ce_1024['diff']):+.5f}`, where calibrated CE still beats calibrated Pearson under `d=2` by `{float(d2_1024['diff']):+.5f}` `[{float(d2_1024['pt_lo_95']):+.5f}, {float(d2_1024['pt_hi_95']):+.5f}]`, `p-val={float(d2_1024['pt_pval']):.4f}`, `{d2_1024['draws_positive']}/10` draws). See [BUDGET_TABLES_11-21-2026.md](BUDGET_TABLES_11-21-2026.md) §3 and `figures/helpsteer2_calibration_and_depth.png`.",
        "",
        "![Equal-cost label depth (d=1 vs d=2 votes per response) and 5-logit vs Experiment 23 1D 4-bin calibration comparison on HelpSteer2.](figures/helpsteer2_calibration_and_depth.png)",
        "",
        "## Provenance, Sanity Verification, and Compute Cost",
        "",
        f"- **Model & trainable modules:** `{prov['model_snapshot']}`, top `4` transformer blocks (`layers 32..35`) with LoRA (`rank = 8`, `alpha = 16.0`) on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` plus `5`-logit rating head initialized from `lm_head` digit tokens (`{prov['n_trainable_params']:,}` trainable parameters).",
        f"- **Equivalence & sanity checks (`results/sanity_checks.json`):** Full-36-layer vs cached-`H^(32)` Epoch-0 max probability diff = `{sanity['epoch0_full_vs_h32_max_prob_diff']:.2e}`; 2-step `AdamW` equivalence max prob diff = `{sanity['two_step_full_vs_h32_equivalence']['ce']['max_prob_diff_after_2_steps']:.2e}` (CE) and `{sanity['two_step_full_vs_h32_equivalence']['pearson']['max_prob_diff_after_2_steps']:.2e}` (Pearson); two-pass Pearson chain-rule relative gradient error = `{sanity['gradient_verification']['pearson_two_pass_rel_error']:.2e}`; `label_in_prompt` hits = `{sanity['karpathy_controls']['label_in_prompt_hits']}/5`; blank-response mean = `{sanity['karpathy_controls']['blank_response_mean_grade']:.3f}`.",
        f"- **Weight-update & reload verification (`results/training_provenance.json`):** `{prov['nonzero_delta_lora_count']}/{prov['n_selected_runs']}` selected runs changed LoRA weights (`||Delta_LoRA||_F > 0`) and `{prov['nonzero_delta_head_count']}/{prov['n_selected_runs']}` changed the rating head; `22/22` disk-saved `draw_id = 0` checkpoints reloaded into fresh instances within `< 1e-7` validation RPS difference; undefined Pearson mini-batches = `{p_b2048_undef}/{p_b2048_tot}` at `B_total = 2,048` (`{prov['total_undefined_pearson_batches']}` across all candidate runs).",
        f"- **Compute cost:** `$0.00` on group-owned `mercury2` NVIDIA RTX A4000 GPUs (10 GPUs in parallel; slowest draw wall-clock `{max_draw_s:.1f}s`, aggregate GPU time `{tot_gpu_s:.1f}s`); `$0.00` in new human annotation fees.",
        "",
        "See [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md) · [BUDGET_TABLES_11-21-2026.md](BUDGET_TABLES_11-21-2026.md) · [PROTOCOL_11-21-2026.md](PROTOCOL_11-21-2026.md)",
    ]
    (EXP_DIR / "RESULTS_11-21-2026.md").write_text("\n".join(res_lines) + "\n", encoding="utf-8")

    # Write HOW_TO_READ_RESULTS.md
    htr_lines = [
        "# How to Read Experiment 24 Results (11-21-2026)",
        "",
        "**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026/HOW_TO_READ_RESULTS.md>",
        "",
        "Date: 11-21-2026",
        "",
        "1. **Which data:** `nvidia/HelpSteer2` (`disagreements/disagreements.jsonl.gz` joined to `train.jsonl.gz` and `validation.jsonl.gz`, commit `990b2711a36180dd19d9c94b8627844866f8982a`), `helpfulness` votes on integer scale `0–4` (`K = 5`). The held-out test set consists of all `1,038` responses across the `519` unique prompts of `validation.jsonl.gz` (`3,610` votes), strictly prompt-disjoint from `train_pool` (`1,600` prompts), `val_pool` (`800` prompts), and `cal_pool` (`800` prompts). Prior exposure: HelpSteer2 is a public 2024 benchmark and `Qwen/Qwen3-4B-Instruct-2507` pretraining exposure to public HelpSteer2 text is unknown; all supervised splits here are strictly prompt-disjoint.",
        "2. **Which metrics:** Primary metric is normalized individual-rating **ranked probability score (RPS)**: average squared error between cumulative forecast and observed rating indicators across the four ordered thresholds `0..3` (`(0..1)`, lower is better; uniform baseline `~0.224`, leave-one-vote-out human reference `0.14567`). Secondary metrics: **cross-entropy (CE)** in natural log units (nats, `-log q_eps(y)`, floor `6e-6` and sensitivity `1e-2`, uniform = `1.6094` nats), **total variation distance (TVD)** (`0.5 * sum |q - h|`, range `0–1`, noisy sparse-histogram diagnostic since each test item has `2–5` votes), and multiclass Brier score (`0–2`), plus Pearson `r`, Spearman `rho`, Kendall `tau-b`, Lin's concordance correlation coefficient (CCC), signed `mean_bias = forecast_mean - human_mean`, mean absolute error (MAE), cumulative expected calibration error (ECE), modal agreement, and within-prompt pairwise preference accuracy.",
        "3. **What is subtracted:** Every difference subtracts `test_loss(CE-trained judge)` from `test_loss(named comparator)`, written explicitly as `comparator minus CE-trained` — positive values mean the CE-trained judge has lower held-out loss (is better). Methods are always named by their training objective and calibration (`calibrated CE-trained LoRA`, `calibrated Pearson-trained LoRA`, etc.).",
        "4. **Uncertainty:** 4,000 paired bootstrap resamples (`seed = 20261124`) clustered by the `519` held-out source prompts. Simultaneous 95% bands use the max-`t` critical value across all 8 primary budgets (`B_total` 16–2,048). Two-way crossed intervals jointly resample the `10` nested training draws and `519` held-out prompts.",
        "5. **How the judge was chosen:** For every draw and budget, both CE and Pearson objectives initialize identically (`Qwen/Qwen3-4B-Instruct-2507` top-4-block LoRA `rank=8` + 5-logit head, `1,847,813` trainable parameters), run identical `AdamW` mini-batches (`lr in {5e-5, 2e-4, 8e-4}`, `4` checkpoints), and select the checkpoint minimizing validation RPS on `S_val` (`N_val = B_total/4`). Pearson optimization is non-convex, so Pearson results reflect this exact optimizer, start, and budget. Both trained judges and the unmodified base judge then fit an identical strictly convex 7-parameter L-BFGS-B post-hoc calibrator (`4`-fold CV on `S_cal`, `N_cal = B_total/4`).",
    ]
    (EXP_DIR / "HOW_TO_READ_RESULTS.md").write_text("\n".join(htr_lines) + "\n", encoding="utf-8")

    # Write README.md
    readme_lines = [
        "# Experiment 24 (`24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026`): Trained Open-Weight LoRA Judge (CE vs Pearson) on HelpSteer2 (11-21-2026)",
        "",
        "**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026/README.md>",
        "",
        f"**Bottom line (11-21-2026):** Updating real `Qwen/Qwen3-4B-Instruct-2507` LoRA weights (`1,847,813` trainable parameters) with cross-entropy (CE) beats matched Pearson correlation LoRA weight training on held-out ranked probability score (RPS) under identical post-hoc calibration on `nvidia/HelpSteer2` (`1,038` responses / `519` prompts) at sustained threshold **{sust_pc_cond}** (two-way crossed threshold: **{sust_pc_cross}**). At `B_total = 2,048` (`N_train = 1,024`, `N_val = 512`, `N_cal = 512`), calibrated CE LoRA reaches test RPS **{float(ce_cal_max['rps']):.5f}** vs **{float(pe_cal_max['rps']):.5f}** for calibrated Pearson LoRA (paired difference = **{float(max_pc['diff_comparator_minus_target']):+.5f}** `[{float(max_pc['sim_lo_95']):+.5f}, {float(max_pc['sim_hi_95']):+.5f}]`, `p-val={float(max_pc['sim_pval']):.4f}`, **{max_pc['draws_target_better']}/10** draws won) and **{float(max_bm['diff_comparator_minus_target']):+.5f}** better RPS than the calibrated unmodified base judge on `N_cal` labels (`{float(bm_max['rps']):.5f}` -> `{float(ce_cal_max['rps']):.5f}`).",
        "",
        "## Files",
        "- [RESULTS_11-21-2026.md](RESULTS_11-21-2026.md): Full verdict, budget learning curve, calibration comparison, and provenance.",
        "- [BUDGET_TABLES_11-21-2026.md](BUDGET_TABLES_11-21-2026.md): Every budget table (`d=1` and `d=2`), CE/TVD/bias/ranking diagnostics, and crossed uncertainty.",
        "- [PROTOCOL_11-21-2026.md](PROTOCOL_11-21-2026.md): Frozen pre-run protocol (SHA-256 in `results/protocol.sha256`).",
        "- [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md): Metric definitions, splits, calibration, and optimizer disclosures.",
        "- [CKPT_helpsteer2_lora.md](CKPT_helpsteer2_lora.md): Run checkpoint and verification log.",
        "- `scripts/`: `prepare_helpsteer2.py`, `train_lora_helpsteer2.py`, `calibrate_and_score.py`, `plot_and_report.py`, `verify.py`, `send_run_email.py`.",
        "- `figures/`: `helpsteer2_lora_learning_curves.{png,pdf}`, `helpsteer2_calibration_and_depth.{png,pdf}`.",
    ]
    (EXP_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print("Generated figures and all Markdown reports cleanly.")


if __name__ == "__main__":
    main()
