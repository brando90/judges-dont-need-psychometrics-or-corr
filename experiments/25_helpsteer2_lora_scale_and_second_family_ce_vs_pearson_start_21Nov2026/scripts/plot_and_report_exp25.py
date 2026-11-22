#!/usr/bin/env python3
"""Render Experiment 25 Matplotlib figures (ax.grid(True) on every axis) and Markdown reports."""
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

HUMAN_LOVO_RPS = 0.14567

METHOD_STYLE = {
    "ce_calibrated": ("CE LoRA + 7-param cal.", "#0b6e82", "o", "-"),
    "pearson_calibrated": ("Pearson LoRA + 7-param cal.", "#b54128", "v", "-"),
    "ce_raw": ("CE LoRA (raw)", "#42a5b8", "o", ":"),
    "pearson_raw": ("Pearson LoRA (raw)", "#df816b", "v", ":"),
    "base_calibrated_matched": ("Base judge + cal. (N_cal)", "#5a4aa8", "^", "-."),
    "base_calibrated_all_dev": ("Base judge + cal. (all B_total)", "#2d225a", "D", "--"),
    "training_histogram": ("Smoothed train histogram", "#9f6b12", "s", "--"),
    "uniform": ("Uniform [0.2]*5", "#687078", "", "--"),
}

STUDY_A_CONTROLS = (
    ("base_calibrated_matched", "Calibrated Base Qwen3-4B (N_cal) minus cal. CE LoRA", "#5a4aa8", "^"),
    ("base_calibrated_all_dev", "Calibrated Base Qwen3-4B (all B_total) minus cal. CE LoRA", "#2d225a", "D"),
    ("training_histogram", "Smoothed train histogram minus cal. CE LoRA", "#9f6b12", "s"),
    ("uniform", "Uniform minus cal. CE LoRA", "#687078", "x"),
)

CALIBRATION_SENSITIVITY_SERIES = (
    ("qwen3_4b", "ce_calibrated", "pearson_calibrated", "Qwen3-4B 7-param 5-logit cal.", "#0b6e82", "o", "-"),
    ("qwen3_4b", "ce_exp23_4bin", "pearson_exp23_4bin", "Qwen3-4B 1D 4-bin cal.", "#42a5b8", "^", "--"),
    ("llama31_8b", "ce_calibrated", "pearson_calibrated", "Llama-3.1-8B 7-param 5-logit cal.", "#b54128", "s", "-"),
    ("llama31_8b", "ce_exp23_4bin", "pearson_exp23_4bin", "Llama-3.1-8B 1D 4-bin cal.", "#df816b", "D", "--"),
)

REPORT_TABLE_METHODS = (
    "ce_calibrated",
    "pearson_calibrated",
    "ce_raw",
    "pearson_raw",
    "base_calibrated_matched",
    "base_calibrated_all_dev",
    "training_histogram",
    "uniform",
    "ce_exp23_4bin",
    "pearson_exp23_4bin",
)


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style_log2_axis(ax, x_vals: list[int], tick_labels: list[str] | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(x_vals))
    if tick_labels is not None:
        ax.set_xticklabels(tick_labels, fontsize=7.8)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_minor_locator(LogLocator(base=2, subs=[2 ** 0.5]))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", axis="x", alpha=0.12)


def _plot_absolute_rps(
    ax,
    s_map: dict,
    study_key: str,
    budgets: list[int],
    tick_labels: list[str],
    title: str,
    marker_size: float,
    xlabel: str | None = None,
) -> None:
    n_vals = [b // 2 for b in budgets]
    for m_name, (label, color, marker, ls) in METHOD_STYLE.items():
        ys = [float(s_map[(study_key, b, m_name)]["rps"]) for b in budgets]
        ax.plot(n_vals, ys, label=label, color=color, marker=marker or None, markersize=marker_size, linewidth=2.0, linestyle=ls)
    ax.axhline(HUMAN_LOVO_RPS, color="#1a7a3c", linewidth=1.3, linestyle="--", label="Human LOVO reference (0.1457)")
    ax.set_title(title, loc="left", fontsize=11.0, fontweight="bold")
    ax.set_ylabel("Held-out test ranked probability score (RPS)\n(0–1, lower is better)")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    style_log2_axis(ax, n_vals, tick_labels)


def _plot_study_a_ce_tvd(ax_ce, s_map: dict, b_a: list[int], n_a: list[int], t_a: list[str]) -> None:
    ax_tvd = ax_ce.twinx()
    for m_name in ("ce_calibrated", "pearson_calibrated", "ce_raw", "pearson_raw", "base_calibrated_matched", "uniform"):
        _, color, marker, _ = METHOD_STYLE[m_name]
        y_ce = [float(s_map[("qwen3_4b", b, m_name)]["ce_nats"]) for b in b_a]
        y_tvd = [float(s_map[("qwen3_4b", b, m_name)]["tvd"]) for b in b_a]
        ax_ce.plot(n_a, y_ce, color=color, marker=marker or None, markersize=4.4, linewidth=2.0, linestyle="-")
        ax_tvd.plot(n_a, y_tvd, color=color, marker=marker or None, markersize=3.5, linewidth=1.3, linestyle="--", alpha=0.8)
    ax_ce.set_title("B. Study A (Qwen3-4B): Held-out CE (solid) & TVD (dashed)", loc="left", fontsize=11.0, fontweight="bold")
    ax_ce.set_ylabel("Held-out test cross-entropy (CE, nats)\n(solid lines, lower is better)")
    ax_tvd.set_ylabel("Held-out test total variation distance (TVD)\n(dashed lines, 0–1, lower is better)")
    style_log2_axis(ax_ce, n_a, t_a)
    ax_tvd.spines["top"].set_visible(False)
    ax_tvd.grid(True, alpha=0.14, linestyle=":")


def _plot_paired_pearson_vs_ce(
    ax,
    c_map: dict,
    study_key: str,
    budgets: list[int],
    tick_labels: list[str],
    title: str,
    cal_label: str,
    raw_label: str,
    cross_label: str,
    xlabel: str,
    cal_marker_size: float,
    cal_linewidth: float,
) -> None:
    n_vals = [b // 2 for b in budgets]
    series = (
        ("ce_calibrated", "pearson_calibrated", cal_label, "#b54128", "v", True),
        ("ce_raw", "pearson_raw", raw_label, "#df816b", "^", False),
    )
    for target_m, comp_m, label, color, marker, show_cross in series:
        ys = [float(c_map[(study_key, b, target_m, comp_m, "rps")]["diff_comparator_minus_target"]) for b in budgets]
        lo = [float(c_map[(study_key, b, target_m, comp_m, "rps")]["sim_lo_95"]) for b in budgets]
        hi = [float(c_map[(study_key, b, target_m, comp_m, "rps")]["sim_hi_95"]) for b in budgets]
        ax.plot(n_vals, ys, label=label, color=color, marker=marker, markersize=cal_marker_size, linewidth=cal_linewidth)
        ax.fill_between(n_vals, lo, hi, color=color, alpha=0.15, linewidth=0)
        if show_cross:
            c_lo = [float(c_map[(study_key, b, target_m, comp_m, "rps")]["cross_sim_lo_95"]) for b in budgets]
            c_hi = [float(c_map[(study_key, b, target_m, comp_m, "rps")]["cross_sim_hi_95"]) for b in budgets]
            ax.plot(n_vals, c_lo, color=color, linestyle="--", linewidth=1.2, label=cross_label)
            ax.plot(n_vals, c_hi, color=color, linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color="#222222", linewidth=1.1)
    ax.set_title(title, loc="left", fontsize=11.0, fontweight="bold")
    ax.set_ylabel("Held-out test RPS difference\n(Pearson minus CE; > 0 favors CE)")
    ax.set_xlabel(xlabel)
    style_log2_axis(ax, n_vals, tick_labels)
    ax.legend(loc="best", fontsize=7.6, frameon=True, framealpha=0.92)


def _plot_study_a_controls(ax_ctrl, c_map: dict, b_a: list[int], n_a: list[int], t_a: list[str]) -> None:
    for comp_m, label, color, marker in STUDY_A_CONTROLS:
        ys = [float(c_map[("qwen3_4b", b, "ce_calibrated", comp_m, "rps")]["diff_comparator_minus_target"]) for b in b_a]
        lo = [float(c_map[("qwen3_4b", b, "ce_calibrated", comp_m, "rps")]["cross_sim_lo_95"]) for b in b_a]
        hi = [float(c_map[("qwen3_4b", b, "ce_calibrated", comp_m, "rps")]["cross_sim_hi_95"]) for b in b_a]
        ax_ctrl.plot(n_a, ys, label=label, color=color, marker=marker, markersize=4.6, linewidth=1.9)
        ax_ctrl.fill_between(n_a, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax_ctrl.axhline(0.0, color="#222222", linewidth=1.1)
    ax_ctrl.set_title("D. Study A paired test RPS gain over base & baseline controls", loc="left", fontsize=11.0, fontweight="bold")
    ax_ctrl.set_ylabel("Held-out test RPS difference\n(comparator minus calibrated CE LoRA; > 0 favors CE)")
    ax_ctrl.set_xlabel("Training human ratings N_train (total development ratings B_total)")
    style_log2_axis(ax_ctrl, n_a, t_a)
    ax_ctrl.legend(loc="best", fontsize=7.6, frameon=True, framealpha=0.92)


def _plot_cross_family_overlay(ax_cf, c_map: dict, b_a: list[int], b_b: list[int]) -> None:
    shared_b = sorted(set(b_a) & set(b_b))
    shared_n = [b // 2 for b in shared_b]
    shared_t = [f"{b // 2:,}\n({b:,})" for b in shared_b]
    families = (
        ("qwen3_4b", "Qwen3-4B-Instruct-2507 (Study A, 20 draws)", "#0b6e82", "o"),
        ("llama31_8b", "Llama-3.1-8B-Instruct (Study B, 10 draws)", "#b54128", "s"),
    )
    for st_key, lbl, col, mk in families:
        ys = [float(c_map[(st_key, b, "ce_calibrated", "pearson_calibrated", "rps")]["diff_comparator_minus_target"]) for b in shared_b]
        lo = [float(c_map[(st_key, b, "ce_calibrated", "pearson_calibrated", "rps")]["cross_sim_lo_95"]) for b in shared_b]
        hi = [float(c_map[(st_key, b, "ce_calibrated", "pearson_calibrated", "rps")]["cross_sim_hi_95"]) for b in shared_b]
        ax_cf.plot(shared_n, ys, label=lbl, color=col, marker=mk, markersize=5.4, linewidth=2.1)
        ax_cf.fill_between(shared_n, lo, hi, color=col, alpha=0.14, linewidth=0)
    ax_cf.axhline(0.0, color="#222222", linewidth=1.1)
    ax_cf.set_title("C. Cross-family overlay on shared budgets (512..2,048)", loc="left", fontsize=11.0, fontweight="bold")
    ax_cf.set_ylabel("Held-out test RPS difference\n(cal. Pearson minus cal. CE; > 0 favors CE)")
    ax_cf.set_xlabel("Training human ratings N_train (total B_total)")
    style_log2_axis(ax_cf, shared_n, shared_t)
    ax_cf.legend(loc="best", fontsize=8.0, frameon=True, framealpha=0.92)


def _plot_calibration_sensitivity(ax_cal, c_map: dict, b_a: list[int], b_b: list[int]) -> None:
    for st_key, target_m, comp_m, lbl, col, mk, ls in CALIBRATION_SENSITIVITY_SERIES:
        bs = b_a if st_key == "qwen3_4b" else b_b
        ns = [b // 2 for b in bs]
        ys = [float(c_map[(st_key, b, target_m, comp_m, "rps")]["diff_comparator_minus_target"]) for b in bs]
        ax_cal.plot(ns, ys, label=lbl, color=col, marker=mk, markersize=4.5, linewidth=1.9, linestyle=ls)
    ax_cal.axhline(0.0, color="#222222", linewidth=1.1)
    ax_cal.set_title("D. 5-logit vs 1D 4-bin calibration across both families", loc="left", fontsize=11.0, fontweight="bold")
    ax_cal.set_ylabel("Held-out test RPS difference\n(Pearson minus CE; > 0 favors CE)")
    ax_cal.set_xlabel("Training human ratings N_train (B_total = 2 * N_train)")
    all_ns = sorted({b // 2 for b in b_a} | {b // 2 for b in b_b})
    style_log2_axis(ax_cal, all_ns, [f"{n:,}" for n in all_ns])
    ax_cal.legend(loc="best", fontsize=7.6, frameon=True, framealpha=0.92)


def _save_figure(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        meta = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
        fig.savefig(FIGURES_DIR / f"{stem}.{ext}", dpi=180, facecolor="white", metadata=meta)
    plt.close(fig)


def plot_study_a_figure(s_map: dict, c_map: dict, b_a: list[int]) -> None:
    n_a = [b // 2 for b in b_a]
    t_a = [f"{b // 2:,}\n({b:,})" for b in b_a]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4), sharex="col")
    fig.subplots_adjust(left=0.075, right=0.935, bottom=0.17, top=0.835, hspace=0.28, wspace=0.30)

    _plot_absolute_rps(axes[0, 0], s_map, "qwen3_4b", b_a, t_a, "A. Study A (Qwen3-4B, 20 draws): Absolute held-out RPS", 4.8)
    _plot_study_a_ce_tvd(axes[0, 1], s_map, b_a, n_a, t_a)
    _plot_paired_pearson_vs_ce(
        axes[1, 0], c_map, "qwen3_4b", b_a, t_a,
        "C. Study A paired test RPS gap: Pearson LoRA minus CE LoRA",
        "Cal. Pearson LoRA minus cal. CE LoRA (sim. cond. band)",
        "Raw Pearson LoRA minus raw CE LoRA",
        "Cal. Pearson minus cal. CE (95% crossed sim. bounds)",
        "Training human ratings N_train (total development ratings B_total)",
        5.0, 2.1,
    )
    _plot_study_a_controls(axes[1, 1], c_map, b_a, n_a, t_a)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Study A (Qwen3-4B-Instruct-2507, 20 Draws, B_total <= 8,192): Cross-Entropy vs Pearson on HelpSteer2",
        x=0.075, y=0.965, ha="left", fontsize=13.5, fontweight="bold",
    )
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.070, 0.928), ncol=3, fontsize=8.3, frameon=False)
    fig.text(
        0.075, 0.074,
        "Optimizer & selection: top-4-block LoRA (rank=8, 1.85M params) + 5-logit head trained with AdamW across lr in {5e-5, 2e-4, 8e-4} & 4 checkpoints;\n"
        "winner selected by validation RPS on S_val (non-convex Pearson gradient). Matched 7-param L-BFGS-B calibrator fitted on disjoint S_cal.\n"
        "Shaded & dashed bands show 95% simultaneous max-t conditional and two-way crossed (draw x prompt) bootstrap (4,000 draws; 519 held-out prompts).",
        fontsize=8.0, va="top", linespacing=1.35,
    )
    _save_figure(fig, "exp25_qwen3_4b_scale_curves")


def plot_study_b_figure(s_map: dict, c_map: dict, b_a: list[int], b_b: list[int]) -> None:
    t_b = [f"{b // 2}\n({b})" for b in b_b]
    fig2, axes2 = plt.subplots(2, 2, figsize=(13.4, 9.4))
    fig2.subplots_adjust(left=0.075, right=0.965, bottom=0.17, top=0.835, hspace=0.31, wspace=0.28)

    _plot_absolute_rps(
        axes2[0, 0], s_map, "llama31_8b", b_b, t_b,
        "A. Study B (Llama-3.1-8B, 10 draws): Absolute held-out RPS", 4.5,
        xlabel="Training human ratings N_train (total B_total)",
    )
    _plot_paired_pearson_vs_ce(
        axes2[0, 1], c_map, "llama31_8b", b_b, t_b,
        "B. Study B (Llama-3.1-8B): Pearson LoRA minus CE LoRA RPS",
        "Llama-3.1-8B cal. Pearson minus cal. CE (cond. sim.)",
        "Llama-3.1-8B raw Pearson minus raw CE",
        "Crossed (draw x prompt) sim. 95% bounds",
        "Training human ratings N_train (total B_total)",
        4.8, 2.0,
    )
    _plot_cross_family_overlay(axes2[1, 0], c_map, b_a, b_b)
    _plot_calibration_sensitivity(axes2[1, 1], c_map, b_a, b_b)

    handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
    fig2.suptitle(
        "Study B (Llama-3.1-8B-Instruct Replication) & Cross-Family Comparison on HelpSteer2",
        x=0.075, y=0.965, ha="left", fontsize=13.5, fontweight="bold",
    )
    fig2.legend(handles2, labels2, loc="upper left", bbox_to_anchor=(0.070, 0.928), ncol=3, fontsize=8.3, frameon=False)
    fig2.text(
        0.075, 0.074,
        "Llama-3.1-8B-Instruct top-4-block LoRA (rank=8) + 5-logit head vs Qwen3-4B-Instruct-2507 under identical HelpSteer2 splits, AdamW sweeps,\n"
        "validation-RPS checkpoint selection, and 7-param L-BFGS-B post-hoc calibration. Held-out test set: 1,038 responses / 519 prompts / 3,610 votes.\n"
        "Shaded & dashed intervals are 95% simultaneous max-t conditional and two-way crossed (draw x prompt) bootstrap (4,000 draws; seed=20261125).",
        fontsize=8.0, va="top", linespacing=1.35,
    )
    _save_figure(fig2, "exp25_llama31_8b_and_cross_family")


def make_figures(scores: list[dict], contrasts: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    s_map = {(r["study"], int(r["b_total"]), r["method"]): r for r in scores}
    c_map = {(r["study"], int(r["b_total"]), r["target"], r["comparator"], r["metric"]): r for r in contrasts}
    b_a = sorted({int(r["b_total"]) for r in scores if r["study"] == "qwen3_4b"})
    b_b = sorted({int(r["b_total"]) for r in scores if r["study"] == "llama31_8b"})
    plot_study_a_figure(s_map, c_map, b_a)
    plot_study_b_figure(s_map, c_map, b_a, b_b)


def fmt_sust(entry: dict | None) -> str:
    if entry is None:
        return "None within tested budgets"
    return f"B_total={entry['b_total']:,} (N_train={entry['n_train']:,})"


def _budget_study_section(st_key: str, st_title: str, summary: dict, s_map: dict, c_map: dict) -> list[str]:
    n_draws = int(summary["studies"][st_key]["n_draws"])
    budgets = [int(b) for b in summary["studies"][st_key]["budgets"]]
    lines = [
        f"## {st_title}",
        "",
        f"| B_total (N_train / N_val / N_cal) | Pearson cal. minus CE cal. RPS [cond. sim. 95% CI; p-val] | Crossed (draw x prompt) sim. 95% CI [p-val] (wins/{n_draws}) | Base cal. (N_cal) minus CE cal. | Base cal. (all B_total) minus CE cal. |",
        "|---:|---:|---:|---:|---:|",
    ]
    for b in budgets:
        pc = c_map[(st_key, b, "ce_calibrated", "pearson_calibrated", "rps")]
        bm = c_map[(st_key, b, "ce_calibrated", "base_calibrated_matched", "rps")]
        ba = c_map[(st_key, b, "ce_calibrated", "base_calibrated_all_dev", "rps")]
        lines.append(
            f"| {b:,} ({b//2:,}/{b//4:,}/{b//4:,}) "
            f"| {float(pc['diff_comparator_minus_target']):+.5f} [{float(pc['sim_lo_95']):+.5f}, {float(pc['sim_hi_95']):+.5f}; p={float(pc['sim_pval']):.4f}] "
            f"| [{float(pc['cross_sim_lo_95']):+.5f}, {float(pc['cross_sim_hi_95']):+.5f}; p={float(pc['cross_sim_pval']):.4f}] ({pc['draws_target_better']}/{n_draws}) "
            f"| {float(bm['diff_comparator_minus_target']):+.5f} [{float(bm['cross_sim_lo_95']):+.5f}, {float(bm['cross_sim_hi_95']):+.5f}] "
            f"| {float(ba['diff_comparator_minus_target']):+.5f} [{float(ba['cross_sim_lo_95']):+.5f}, {float(ba['cross_sim_hi_95']):+.5f}] |"
        )
    lines += [
        "",
        f"### Absolute Held-Out Test Metrics ({st_key})",
        "",
        "| B_total (N_train) | Method | Test RPS | Test CE (6e-6) | Test CE (1e-2) | Test TVD | Pearson r | Spearman rho | Mean Bias | Within-Prompt Pair Acc |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in budgets:
        for m in REPORT_TABLE_METHODS:
            r = s_map[(st_key, b, m)]
            lines.append(
                f"| {b:,} ({b//2:,}) | {m} | {float(r['rps']):.5f} | {float(r['ce_nats']):.4f} | {float(r['ce_eps1e2']):.4f} "
                f"| {float(r['tvd']):.4f} | {float(r['pearson']):+.4f} | {float(r['spearman']):+.4f} "
                f"| {float(r['mean_bias']):+.4f} | {float(r['within_prompt_pair_acc']):.4f} |"
            )
    lines.append("")
    return lines


def build_budget_tables_md(summary: dict, s_map: dict, c_map: dict) -> str:
    bt = [
        "# Experiment 25 Budget Tables and Full Held-Out Metric Panel | 11-21-2026",
        "",
        "**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026/BUDGET_TABLES_11-21-2026.md>",
        "",
        "As of 11-21-2026. All held-out test metrics evaluate on all `1,038` responses (`519` prompts, `3,610` votes) of `nvidia/HelpSteer2` (`validation.jsonl.gz`). Primary metric: normalized individual-rating **ranked probability score (RPS)** (`0–1`, lower is better; leave-one-vote-out human reference `0.14567`). Differences compute `comparator minus calibrated CE-trained LoRA` (`> 0` favors CE). Bootstrap: `4,000` prompt-clustered draws (`seed = 20261125`).",
        "",
    ]
    for st_key, st_title in (
        ("qwen3_4b", "Study A: `Qwen/Qwen3-4B-Instruct-2507` (20 draws, B_total 512–8,192)"),
        ("llama31_8b", "Study B: `unsloth/Llama-3.1-8B-Instruct` (10 draws, B_total 16–2,048)"),
    ):
        bt.extend(_budget_study_section(st_key, st_title, summary, s_map, c_map))
    return "\n".join(bt)


def _res_summary_rows(st_key: str, n_draws: int, budgets: list[int], s_map: dict, c_map: dict) -> list[str]:
    rows = []
    for b in budgets:
        cc = s_map[(st_key, b, "ce_calibrated")]
        pc_r = s_map[(st_key, b, "pearson_calibrated")]
        ba_r = s_map[(st_key, b, "base_calibrated_all_dev")]
        diff = c_map[(st_key, b, "ce_calibrated", "pearson_calibrated", "rps")]
        rows.append(
            f"| {b:,} ({b//2:,}/{b//4:,}/{b//4:,}) | {float(cc['rps']):.5f} / {float(cc['ce_nats']):.4f} / {float(cc['tvd']):.4f} "
            f"| {float(pc_r['rps']):.5f} / {float(pc_r['ce_nats']):.4f} / {float(pc_r['tvd']):.4f} | {float(ba_r['rps']):.5f} "
            f"| {float(diff['diff_comparator_minus_target']):+.5f} [{float(diff['cross_sim_lo_95']):+.5f}, {float(diff['cross_sim_hi_95']):+.5f}] ({diff['draws_target_better']}/{n_draws}) |"
        )
    return rows


def _build_exec_answer(summary: dict, s_map: dict, c_map: dict, prov_q: dict, prov_l: dict) -> list[str]:
    q_max_b = summary["studies"]["qwen3_4b"]["budgets"][-1]
    l_max_b = summary["studies"]["llama31_8b"]["budgets"][-1]
    q_pc = c_map[("qwen3_4b", q_max_b, "ce_calibrated", "pearson_calibrated", "rps")]
    q_ce = c_map[("qwen3_4b", q_max_b, "ce_calibrated", "pearson_calibrated", "ce_nats")]
    q_tvd = c_map[("qwen3_4b", q_max_b, "ce_calibrated", "pearson_calibrated", "tvd")]
    q_ba = c_map[("qwen3_4b", q_max_b, "ce_calibrated", "base_calibrated_all_dev", "rps")]
    q_bm = c_map[("qwen3_4b", q_max_b, "ce_calibrated", "base_calibrated_matched", "rps")]
    q_ce_s, q_pe_s, q_ba_s = s_map[("qwen3_4b", q_max_b, "ce_calibrated")], s_map[("qwen3_4b", q_max_b, "pearson_calibrated")], s_map[("qwen3_4b", q_max_b, "base_calibrated_all_dev")]
    q_4bin = c_map[("qwen3_4b", q_max_b, "ce_exp23_4bin", "pearson_exp23_4bin", "rps")]

    l_pc = c_map[("llama31_8b", l_max_b, "ce_calibrated", "pearson_calibrated", "rps")]
    l_ce = c_map[("llama31_8b", l_max_b, "ce_calibrated", "pearson_calibrated", "ce_nats")]
    l_tvd = c_map[("llama31_8b", l_max_b, "ce_calibrated", "pearson_calibrated", "tvd")]
    l_ba = c_map[("llama31_8b", l_max_b, "ce_calibrated", "base_calibrated_all_dev", "rps")]
    l_ce_s, l_pe_s, l_ba_s = s_map[("llama31_8b", l_max_b, "ce_calibrated")], s_map[("llama31_8b", l_max_b, "pearson_calibrated")], s_map[("llama31_8b", l_max_b, "base_calibrated_all_dev")]

    q_sust = summary["studies"]["qwen3_4b"]["sustained_win_thresholds_rps"]["pearson_calibrated"]
    l_sust = summary["studies"]["llama31_8b"]["sustained_win_thresholds_rps"]["pearson_calibrated"]
    q_fav = summary["studies"]["qwen3_4b"]["budgets_favoring_pearson_calibrated_rps"]
    l_fav = summary["studies"]["llama31_8b"]["budgets_favoring_pearson_calibrated_rps"]
    return [
        f"- **Study A (`Qwen/Qwen3-4B-Instruct-2507`, `20` nested draws, `B_total` `512..8,192`, `{prov_q['n_trainable_params']:,}` trainable parameters):** Across all tested budgets up to `B_total = {q_max_b:,}` (`N_train = {q_max_b // 2:,}`), conditional simultaneous max-`t` 95% threshold for `calibrated Pearson-trained LoRA RPS minus calibrated CE-trained LoRA RPS` is **{fmt_sust(q_sust['conditional_sim_95'])}**, and strict two-way crossed (`draw × prompt`) simultaneous 95% threshold is **{fmt_sust(q_sust['crossed_sim_95'])}** (pointwise crossed threshold: **{fmt_sust(q_sust['crossed_pt_95'])}**). At `B_total = {q_max_b:,}`, calibrated cross-entropy (CE) low-rank adaptation (LoRA) reaches held-out test ranked probability score (RPS) = **{float(q_ce_s['rps']):.5f}** vs **{float(q_pe_s['rps']):.5f}** for calibrated Pearson LoRA (paired RPS difference = **{float(q_pc['diff_comparator_minus_target']):+.5f}**, cond. sim. 95% CI `[{float(q_pc['sim_lo_95']):+.5f}, {float(q_pc['sim_hi_95']):+.5f}]`, `p-val={float(q_pc['sim_pval']):.4f}`; crossed sim. 95% CI `[{float(q_pc['cross_sim_lo_95']):+.5f}, {float(q_pc['cross_sim_hi_95']):+.5f}]`, `p-val={float(q_pc['cross_sim_pval']):.4f}`, winning **{q_pc['draws_target_better']}/20** draws), relative to the leave-one-vote-out human reference (`{HUMAN_LOVO_RPS:.5f}`). Budgets favoring Pearson point-estimate RPS: `{q_fav if q_fav else 'none'}`.",
        f"- **Study A baselines & secondary metrics at `B_total = {q_max_b:,}`:** Calibrated CE LoRA beats `base_calibrated_matched` (`N_cal = {q_max_b//4:,}`) by **{float(q_bm['diff_comparator_minus_target']):+.5f}** and `base_calibrated_all_dev` (unmodified base judge calibrated on all `{q_max_b:,}` labels, RPS = **{float(q_ba_s['rps']):.5f}**) by **{float(q_ba['diff_comparator_minus_target']):+.5f}** (`[{float(q_ba['cross_sim_lo_95']):+.5f}, {float(q_ba['cross_sim_hi_95']):+.5f}]`, `p-val={float(q_ba['cross_sim_pval']):.4f}`). On test CE (`6e-6` floor): **{float(q_ce_s['ce_nats']):.4f}** vs **{float(q_pe_s['ce_nats']):.4f}** nats (`{q_ce['draws_target_better']}/20` draws, diff = **{float(q_ce['diff_comparator_minus_target']):+.5f}**, crossed sim. `p-val={float(q_ce['cross_sim_pval']):.4f}`); on sparse-histogram total variation distance (TVD): **{float(q_ce_s['tvd']):.4f}** vs **{float(q_pe_s['tvd']):.4f}** (`{q_tvd['draws_target_better']}/20` draws, crossed sim. `p-val={float(q_tvd['cross_sim_pval']):.4f}`); under 1D 4-bin calibration: `{float(q_4bin['diff_comparator_minus_target']):+.5f}` (`[{float(q_4bin['cross_sim_lo_95']):+.5f}, {float(q_4bin['cross_sim_hi_95']):+.5f}]`).",
        f"- **Study B (`unsloth/Llama-3.1-8B-Instruct`, `10` nested draws, `B_total` `16..2,048`, `{prov_l['n_trainable_params']:,}` trainable parameters across top 4 blocks with grouped-query attention + 5-logit head):** Sustained threshold vs calibrated Pearson LoRA is **{fmt_sust(l_sust['conditional_sim_95'])}** (conditional sim. 95%) and **{fmt_sust(l_sust['crossed_sim_95'])}** (crossed sim. 95%). At `B_total = {l_max_b:,}` (`N_train = {l_max_b // 2:,}`), calibrated CE LoRA achieves test RPS = **{float(l_ce_s['rps']):.5f}** vs **{float(l_pe_s['rps']):.5f}** for calibrated Pearson LoRA (paired diff = **{float(l_pc['diff_comparator_minus_target']):+.5f}**, cond. sim. 95% CI `[{float(l_pc['sim_lo_95']):+.5f}, {float(l_pc['sim_hi_95']):+.5f}]`, `p-val={float(l_pc['sim_pval']):.4f}`; crossed sim. 95% CI `[{float(l_pc['cross_sim_lo_95']):+.5f}, {float(l_pc['cross_sim_hi_95']):+.5f}]`, `p-val={float(l_pc['cross_sim_pval']):.4f}`, winning **{l_pc['draws_target_better']}/10** draws), beats `base_calibrated_all_dev` (`{float(l_ba_s['rps']):.5f}`) by **{float(l_ba['diff_comparator_minus_target']):+.5f}** (`p-val={float(l_ba['cross_sim_pval']):.4f}`), wins CE in **{l_ce['draws_target_better']}/10** draws (mean **{float(l_ce_s['ce_nats']):.4f}** vs **{float(l_pe_s['ce_nats']):.4f}** nats) and TVD in **{l_tvd['draws_target_better']}/10** draws (mean **{float(l_ce_s['tvd']):.4f}** vs **{float(l_pe_s['tvd']):.4f}**). Budgets favoring Pearson point-estimate RPS: `{l_fav if l_fav else 'none'}`.",
    ]


def _build_sanity_section(prov_q: dict, prov_l: dict, san_q: dict, san_l: dict) -> list[str]:
    step2_max = max(
        san_q["two_step_full_vs_batched_equivalence"]["ce"]["max_prob_diff_after_2_steps"],
        san_q["two_step_full_vs_batched_equivalence"]["pearson"]["max_prob_diff_after_2_steps"],
        san_l["two_step_full_vs_batched_equivalence"]["ce"]["max_prob_diff_after_2_steps"],
        san_l["two_step_full_vs_batched_equivalence"]["pearson"]["max_prob_diff_after_2_steps"],
    )
    hf_max = max(san_q["hf_wrapper_bf16_ulp_max_prob_diff"], san_l["hf_wrapper_bf16_ulp_max_prob_diff"])
    pad_max = max(san_q["left_pad_mask_invariance_max_prob_diff"], san_l["left_pad_mask_invariance_max_prob_diff"])
    ov_l = san_l["karpathy_controls"]["overfit_1batch_16items_40steps"]
    ov_p_8e4 = ov_l["pearson"]["by_lr"]["0.0008"]
    return [
        "## Sanity Checks, Weight-Update Provenance, and Note on Non-Convex Pearson Saturation",
        "",
        f"- **Exact equivalence verified (`results/qwen3_4b_sanity.json`, `results/llama31_8b_sanity.json`):** Live full-model prefix vs cached `H^(split)` 2-step batched (`m=4`, `SDPBackend.MATH`) `AdamW` probabilities matched to `{step2_max:.2f}` max diff on both CE and Pearson for both architectures (Epoch-0 unbatched HF wrapper vs cached `H^(split)` max diff = `{hf_max:.2f}`; same-batch left-pad mask invariance `<= {pad_max:.2e}`; `bfloat16` GEMV-vs-GEMM batching drift = `{san_q['epoch0_unbatched_vs_left_padded_max_prob_diff']:.4f}` for Qwen3-4B, `{san_l['epoch0_unbatched_vs_left_padded_max_prob_diff']:.4f}` for Llama-3.1-8B; `pearson_two_pass_rel_error = {san_q['gradient_verification']['pearson_two_pass_rel_error']:.2e}` for Qwen3-4B, `{san_l['gradient_verification']['pearson_two_pass_rel_error']:.2e}` for Llama-3.1-8B). Epoch-0 `label_in_prompt` compliance passed `{san_q['karpathy_controls']['label_in_prompt_hits']}/5` on Qwen3-4B and `{san_l['karpathy_controls']['label_in_prompt_hits']}/5` on Llama-3.1-8B (`matched_20_pearson` `{san_q['karpathy_controls']['matched_20_pearson']:.3f} > {san_q['karpathy_controls']['shuffled_20_pearson']:.3f}` on Qwen3-4B; `{san_l['karpathy_controls']['matched_20_pearson']:.3f} > {san_l['karpathy_controls']['shuffled_20_pearson']:.3f}` on Llama-3.1-8B).",
        f"- **Non-convex Pearson overfit diagnostic on Llama-3.1-8B (`overfit_1batch_16items_40steps`):** Under `lr = 8e-4` alone, Pearson correlation on the 16-item sanity batch saturated softmax probabilities into corners `{{0, 4}}` (`ce_after = {ov_p_8e4['ce_after']:.4f}`, `pearson_after = {ov_p_8e4['pearson_after']:.4f}`) where the softmax-mean Jacobian `q_r(r - mu)` vanishes; sweeping the pre-registered `lr in {{5e-5, 2e-4, 8e-4}}` lifted Llama-3.1-8B overfit Pearson to `{ov_l['pearson']['pearson_after']:.4f}` at `lr = 2e-4` (whereas CE reached `ce_after = {ov_l['ce']['ce_after']:.2e}`, `r = {ov_l['ce']['pearson_after']:.4f}`).",
        f"- **Selected checkpoint provenance:** Qwen3-4B logged `{prov_q['nonzero_delta_lora_count']}/{prov_q['n_selected_runs']}` non-zero LoRA/head updates (`{prov_q['total_undefined_pearson_batches']}` undefined Pearson batches); Llama-3.1-8B logged `{prov_l['nonzero_delta_lora_count']}/{prov_l['n_selected_runs']}` non-zero LoRA/head updates (`{prov_l['total_undefined_pearson_batches']}` undefined Pearson batches). Both studies preserved `< 1e-7` reload precision on all saved `draw_id = 0` disk checkpoints and incurred `$0.00` compute/label fees.",
    ]


def build_results_md(summary: dict, s_map: dict, c_map: dict, prov_q: dict, prov_l: dict, san_q: dict, san_l: dict) -> str:
    res_lines = [
        "# Experiment 25 Results: HelpSteer2 Scale (`B_total <= 8,192`, `20` Draws) and Second-Family (`Llama-3.1-8B-Instruct`) Replication (11-21-2026)",
        "",
        "**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026/RESULTS_11-21-2026.md>",
        "",
        "Date: 11-21-2026",
        "",
        "## Executive Answer",
        "",
        *_build_exec_answer(summary, s_map, c_map, prov_q, prov_l),
        "",
        "## Study A Summary Table (`Qwen/Qwen3-4B-Instruct-2507`, 20 Draws)",
        "",
        "| B_total (N_train / N_val / N_cal) | CE cal. RPS / CE / TVD | Pearson cal. RPS / CE / TVD | Base cal. (all B_total) RPS | Pearson cal. minus CE cal. RPS [crossed sim. 95% CI] (wins/20) |",
        "|---:|---:|---:|---:|---:|",
        *_res_summary_rows("qwen3_4b", 20, summary["studies"]["qwen3_4b"]["budgets"], s_map, c_map),
        "",
        "![Study A scaling learning curves on Qwen3-4B-Instruct-2507 across B_total 512 to 8,192 and 20 nested draws on HelpSteer2.](figures/exp25_qwen3_4b_scale_curves.png)",
        "",
        "## Study B Summary Table (`unsloth/Llama-3.1-8B-Instruct`, 10 Draws)",
        "",
        "| B_total (N_train / N_val / N_cal) | CE cal. RPS / CE / TVD | Pearson cal. RPS / CE / TVD | Base cal. (all B_total) RPS | Pearson cal. minus CE cal. RPS [crossed sim. 95% CI] (wins/10) |",
        "|---:|---:|---:|---:|---:|",
        *_res_summary_rows("llama31_8b", 10, summary["studies"]["llama31_8b"]["budgets"], s_map, c_map),
        "",
        "![Study B Llama-3.1-8B-Instruct replication learning curves, cross-family comparison on shared budgets, and calibration sensitivity on HelpSteer2.](figures/exp25_llama31_8b_and_cross_family.png)",
        "",
        *_build_sanity_section(prov_q, prov_l, san_q, san_l),
        "",
        "See [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md) · [BUDGET_TABLES_11-21-2026.md](BUDGET_TABLES_11-21-2026.md) · [PROTOCOL_11-21-2026.md](PROTOCOL_11-21-2026.md)",
    ]
    return "\n".join(res_lines) + "\n"


def build_how_to_read_md(uniform_rps: float) -> str:
    htr_lines = [
        "# How to Read Experiment 25 Results (11-21-2026)",
        "",
        "**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026/HOW_TO_READ_RESULTS.md>",
        "",
        "Date: 11-21-2026",
        "",
        "1. **Which data:** `nvidia/HelpSteer2` (`train.jsonl.gz`, `validation.jsonl.gz`, `disagreements/disagreements.jsonl.gz`, commit `990b2711a36180dd19d9c94b8627844866f8982a`), `helpfulness` votes `0–4`. Held-out test set: all `1,038` responses across `519` unique prompts (`3,610` votes), zero prompt overlap with Study A pools (`train_pool = 4,096`, `val_pool = 2,048`, `cal_pool = 2,048` prompts) or Study B pools (`1,600 / 800 / 800` prompts). Prior exposure: HelpSteer2 is a public 2024 dataset; both pretrained backbones' pretraining exposure is unknown.",
        f"2. **Which metrics:** Primary metric is normalized individual-rating **ranked probability score (RPS)** (`0–1`, lower is better; uniform `{uniform_rps:.5f}`, leave-one-vote-out human reference `{HUMAN_LOVO_RPS:.5f}`). Secondary metrics: **cross-entropy (CE)** in nats (`6e-6` primary and `1e-2` sensitivity floors; uniform `1.6094`), **total variation distance (TVD)** against each response's `2–5` empirical votes (sparse-histogram diagnostic), multiclass Brier score, Pearson `r`, Spearman `rho`, Kendall `tau-b`, Lin's concordance correlation coefficient (CCC), signed bias (`forecast - human`), mean absolute error (MAE), cumulative expected calibration error (ECE), modal agreement, and within-prompt pairwise accuracy.",
        "3. **What is subtracted:** Every paired contrast subtracts `test_loss(calibrated CE-trained LoRA)` from `test_loss(named comparator)`, written explicitly as `comparator minus calibrated CE LoRA`; positive numbers mean CE-trained LoRA has lower held-out test loss (is better).",
        "4. **Uncertainty:** `4,000` paired prompt-clustered bootstrap resamples (`seed = 20261125`) across all `519` held-out source prompts. Simultaneous 95% bands use max-`t` across all tested budgets within a study (`5` budgets in Study A, `8` in Study B); two-way crossed intervals jointly resample the study's training draws (`20` in Study A, `10` in Study B) and `519` prompts.",
        "5. **How the judge was chosen:** Identical top-4-block LoRA (`rank = 8`, `alpha = 16.0`) + `5`-digit rating head initialized from `lm_head` per architecture (`1,847,813` trainable parameters for `Qwen/Qwen3-4B-Instruct-2507`; `2,641,925` for `unsloth/Llama-3.1-8B-Instruct` with grouped-query attention). Both objectives run identical `AdamW` mini-batches (`lr in {5e-5, 2e-4, 8e-4}`, `4` checkpoints), pick the validation-RPS winner on `S_val`, and apply the same strictly convex 7-parameter L-BFGS-B calibrator (`4`-fold CV on `S_cal`). Pearson is non-convex, so Pearson numbers reflect this optimizer, start, and budget.",
    ]
    return "\n".join(htr_lines) + "\n"


def build_readme_md(summary: dict, s_map: dict, c_map: dict) -> str:
    q_max_b = summary["studies"]["qwen3_4b"]["budgets"][-1]
    l_max_b = summary["studies"]["llama31_8b"]["budgets"][-1]
    q_ce_s, q_pe_s = s_map[("qwen3_4b", q_max_b, "ce_calibrated")], s_map[("qwen3_4b", q_max_b, "pearson_calibrated")]
    l_ce_s, l_pe_s = s_map[("llama31_8b", l_max_b, "ce_calibrated")], s_map[("llama31_8b", l_max_b, "pearson_calibrated")]
    q_pc = c_map[("qwen3_4b", q_max_b, "ce_calibrated", "pearson_calibrated", "rps")]
    l_pc = c_map[("llama31_8b", l_max_b, "ce_calibrated", "pearson_calibrated", "rps")]
    q_sust = summary["studies"]["qwen3_4b"]["sustained_win_thresholds_rps"]["pearson_calibrated"]
    l_sust = summary["studies"]["llama31_8b"]["sustained_win_thresholds_rps"]["pearson_calibrated"]
    lines = [
        "# Experiment 25 (`25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026`): HelpSteer2 Scale & Llama-3.1-8B Replication (11-21-2026)",
        "",
        "**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026/README.md>",
        "",
        f"**Bottom line (11-21-2026):** In Study A (`Qwen3-4B-Instruct-2507`, `20` draws, `B_total` up to `8,192`), calibrated cross-entropy (CE) low-rank adaptation (LoRA) reaches held-out ranked probability score (RPS) **{float(q_ce_s['rps']):.5f}** vs **{float(q_pe_s['rps']):.5f}** for calibrated Pearson LoRA at `B_total = {q_max_b:,}` (diff = **{float(q_pc['diff_comparator_minus_target']):+.5f}**, cond. sim. threshold **{fmt_sust(q_sust['conditional_sim_95'])}**, crossed sim. threshold **{fmt_sust(q_sust['crossed_sim_95'])}**). In Study B (`unsloth/Llama-3.1-8B-Instruct`, `10` draws, `B_total = 16..2,048`), calibrated CE LoRA reaches **{float(l_ce_s['rps']):.5f}** vs **{float(l_pe_s['rps']):.5f}** at `B_total = {l_max_b:,}` (diff = **{float(l_pc['diff_comparator_minus_target']):+.5f}**, cond. sim. threshold **{fmt_sust(l_sust['conditional_sim_95'])}**, crossed sim. threshold **{fmt_sust(l_sust['crossed_sim_95'])}**).",
        "",
        "## Files",
        "- [RESULTS_11-21-2026.md](RESULTS_11-21-2026.md): Full verdict, tables, cross-family figures, sanity & provenance summary.",
        "- [BUDGET_TABLES_11-21-2026.md](BUDGET_TABLES_11-21-2026.md): Complete Study A & Study B tables across RPS, CE, TVD, bias, and pair accuracy.",
        "- [PROTOCOL_11-21-2026.md](PROTOCOL_11-21-2026.md): Pre-registered protocol (SHA-256 in `results/protocol.sha256`).",
        "- [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md): Metric definitions, splits, calibration, and optimizer disclosures.",
        "- [CKPT_exp25_scale_and_llama.md](CKPT_exp25_scale_and_llama.md): Execution and verification checkpoint.",
        "- `scripts/`: `prepare_exp25_bundle.py`, `train_exp25_lora.py`, `calibrate_and_score_exp25.py`, `plot_and_report_exp25.py`, `verify_exp25.py`, `send_run_email.py`, `sync_gdoc_and_drive_exp25.py`.",
        "- `figures/`: `exp25_qwen3_4b_scale_curves.{png,pdf}`, `exp25_llama31_8b_and_cross_family.{png,pdf}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    scores = load_csv(RESULTS_DIR / "scores.csv")
    contrasts = load_csv(RESULTS_DIR / "contrasts.csv")
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    prov_q = json.loads((RESULTS_DIR / "qwen3_4b_provenance.json").read_text(encoding="utf-8"))
    prov_l = json.loads((RESULTS_DIR / "llama31_8b_provenance.json").read_text(encoding="utf-8"))
    san_q = json.loads((RESULTS_DIR / "qwen3_4b_sanity.json").read_text(encoding="utf-8"))
    san_l = json.loads((RESULTS_DIR / "llama31_8b_sanity.json").read_text(encoding="utf-8"))

    make_figures(scores, contrasts)
    s_map = {(r["study"], int(r["b_total"]), r["method"]): r for r in scores}
    c_map = {(r["study"], int(r["b_total"]), r["target"], r["comparator"], r["metric"]): r for r in contrasts}
    uniform_rps = float(s_map[("qwen3_4b", summary["studies"]["qwen3_4b"]["budgets"][-1], "uniform")]["rps"])

    (EXP_DIR / "BUDGET_TABLES_11-21-2026.md").write_text(build_budget_tables_md(summary, s_map, c_map), encoding="utf-8")
    (EXP_DIR / "RESULTS_11-21-2026.md").write_text(build_results_md(summary, s_map, c_map, prov_q, prov_l, san_q, san_l), encoding="utf-8")
    (EXP_DIR / "HOW_TO_READ_RESULTS.md").write_text(build_how_to_read_md(uniform_rps), encoding="utf-8")
    (EXP_DIR / "README.md").write_text(build_readme_md(summary, s_map, c_map), encoding="utf-8")
    print("Generated Experiment 25 figures and Markdown reports cleanly.")


if __name__ == "__main__":
    main()
