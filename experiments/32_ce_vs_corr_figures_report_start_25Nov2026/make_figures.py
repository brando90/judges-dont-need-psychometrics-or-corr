#!/usr/bin/env python3
"""Figures for the CE-fit vs Pearson-fit summary report (11-25-2026).

Every number is read from committed result files in sibling experiment folders;
nothing is typed in by hand except the exact rational counterexample of
experiment 27, which is recomputed here from its definition.
Run from anywhere:  python3 make_figures.py   (needs numpy, pandas, matplotlib)
"""
from fractions import Fraction as F
from math import log
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
E26 = EXP / "26_five_new_frontier_judges_on_fresh_homework_and_independent_recheck_start_23Nov2026"
E24 = EXP / "24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026"
E25 = EXP / "25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026"
E28 = EXP / "28_public_rating_sets_chaosnli_dices_frontier_judges_ce_vs_pearson_start_23Nov2026"
E29 = EXP / "29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026"

CE_C, PE_C, CAL_C, UNI_C, RAW_C, HIST_C = "#1f77b4", "#d62728", "#ff7f0e", "#7f7f7f", "#2ca02c", "#9467bd"
# The salt stays "exp30": it fixes the SVG element ids, so changing it would change every committed figure.
# This folder was created as experiment 30 and renumbered to 32 on 11-25-2026 (see REPORT.md).
plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "exp30", "font.size": 10, "axes.grid": True, "grid.alpha": 0.35, "figure.dpi": 150})


def log2x(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], rotation=0)
    ax.minorticks_off()


def save(fig, name):
    # SVG is what the report embeds (text, renders on GitHub); PNG is a same-content raster copy.
    for ext in ("svg", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight", metadata={"Date": None} if ext == "svg" else None)
        print("wrote", p, p.stat().st_size, "bytes")
    plt.close(fig)


# ---------------------------------------------------------------- fig 1: exp 27 counterexample
def fig1():
    P = [(F(3, 5), F(3, 10), F(1, 10)), (F(1, 5), F(3, 5), F(1, 5)), (F(1, 10), F(3, 10), F(3, 5))]
    Q = [(F(13, 20), F(1, 5), F(3, 20)), (F(2, 5), F(1, 5), F(2, 5)), (F(3, 20), F(1, 5), F(13, 20))]
    mean = lambda p: sum(F(k) * v for k, v in enumerate(p))
    tvd = np.mean([float(sum(abs(a - b) for a, b in zip(p, q)) / 2) for p, q in zip(P, Q)])
    ce = lambda p, q: -sum(float(a) * log(float(b)) for a, b in zip(p, q))
    ce_true = np.mean([ce(p, p) for p in P]); ce_fc = np.mean([ce(p, q) for p, q in zip(P, Q)])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    x = np.arange(3)
    for i, ax in enumerate(axes):
        ax.bar(x - 0.18, [float(v) for v in P[i]], 0.36, color=HIST_C, label="human distribution")
        ax.bar(x + 0.18, [float(v) for v in Q[i]], 0.36, color=PE_C, alpha=0.8, label="judge forecast")
        ax.set_xticks(x); ax.set_xticklabels(["0", "1", "2"]); ax.set_xlabel("rating")
        ax.set_title(f"item {i+1}: both means = {mean(P[i])}")
    axes[0].set_ylabel("probability"); axes[0].legend(fontsize=8, loc="upper center")
    fig.suptitle(f"Exp 27: Pearson(judge means, human means) = 1, yet the distributions differ\n"
                 f"mean TVD = {tvd:.2f}; CE = {ce_fc:.3f} nats vs {ce_true:.3f} for the true distribution (lower is better)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig1_exp27_same_means_different_distributions")


# ---------------------------------------------------------------- fig 2: exp 28 RPS vs ratings
SETS = [("chaosnli_snli", "ChaosNLI SNLI"), ("chaosnli_mnli", "ChaosNLI MNLI-m"),
        ("chaosnli_alphanli", "alphaNLI (K=2 control)"), ("dices350", "DICES-350"), ("dices990", "DICES-990")]


def fig2():
    s = pd.read_csv(E28 / "results/scores_absolute.csv")
    s = s[s.d == 1]
    raw = pd.read_csv(E28 / "results/raw_judge_scores.csv").groupby("dataset").rps.mean()
    m = s.groupby(["dataset", "method", "B"]).rps.mean()
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, (ds, title) in zip(axes, SETS):
        for meth, lab, c, ls in [("ce", "CE-fit", CE_C, "-"), ("pearson", "Pearson-fit", PE_C, "-"),
                                 ("pcal", "Pearson-fit + calibration", CAL_C, "--"),
                                 ("uniform", "uniform", UNI_C, ":")]:
            y = m.loc[ds, meth]
            ax.plot(y.index, y.values, ls, marker="o", ms=3.5, color=c, label=lab)
        ax.axhline(raw[ds], color=RAW_C, ls="-.", lw=1.3, label="raw judge (no fitting)")
        log2x(ax, [4, 16, 64, 256]); ax.set_title(title); ax.set_xlabel("human ratings used to fit (1 per item)")
    axes[0].set_ylabel("held-out RPS (lower is better)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Exp 28: public multi-rater sets, mean over 5 frontier judges", fontsize=11)
    fig.tight_layout()
    save(fig, "fig2_exp28_public_sets_rps_vs_ratings")


def fig3():
    s = pd.read_csv(E28 / "results/scores_absolute.csv")
    s = s[(s.d == 1) & (s.B == 256) & s.method.isin(["ce", "pearson"])]
    m = s.groupby(["dataset", "method"])[["rps", "ce", "tvd"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    x = np.arange(len(SETS))
    for ax, (metric, lab) in zip(axes, [("rps", "held-out RPS"), ("ce", "held-out CE (nats)"), ("tvd", "held-out TVD")]):
        ax.bar(x - 0.2, [m.loc[(d, "pearson"), metric] for d, _ in SETS], 0.4, color=PE_C, label="Pearson-fit")
        ax.bar(x + 0.2, [m.loc[(d, "ce"), metric] for d, _ in SETS], 0.4, color=CE_C, label="CE-fit")
        ax.set_xticks(x); ax.set_xticklabels([t.replace(" (K=2 control)", "\n(K=2)").replace("ChaosNLI ", "") for _, t in SETS], fontsize=8)
        ax.set_title(lab + " (lower is better)"); ax.grid(True, axis="y"); ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle("Exp 28 at 256 ratings (1 per item), mean over 5 judges: CE-fit is lower on all three proper metrics", fontsize=11)
    fig.tight_layout()
    save(fig, "fig3_exp28_three_metrics_at_256")


# ---------------------------------------------------------------- fig 4: exp 26 per judge
JUDGES = [("gpt-6-astra", "GPT-6 Astra"), ("claude-opus-5-5", "Claude Opus 5.5"), ("kimi-k3", "Kimi K3"),
          ("glm-5.3", "GLM 5.3"), ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash")]


def fig4():
    s = pd.read_csv(E26 / "results/new5/scores.csv")
    s = s[(s.ratings_per_answer == 2) & (s.metric == "rps")]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.5), sharey=True)
    for ax, (j, title) in zip(axes, JUDGES):
        g = s[s.judge == j]
        for meth, lab, c, ls in [("ce_fit", "CE-fit", CE_C, "-"), ("pearson_fit", "Pearson-fit", PE_C, "-"),
                                 ("identity", "uncalibrated judge", RAW_C, "-."), ("train_hist", "training histogram", HIST_C, ":")]:
            h = g[g.method == meth].sort_values("training_ratings")
            ax.plot(h.training_ratings, h["mean"], ls, marker="o", ms=3.5, color=c, label=lab)
            if meth in ("ce_fit", "pearson_fit"):
                ax.fill_between(h.training_ratings, h.lower95, h.upper95, color=c, alpha=0.12, lw=0)
        log2x(ax, [4, 16, 64, 256]); ax.set_title(title); ax.set_xlabel("training ratings (2 per answer)")
    uni = s[s.method == "uniform"]["mean"].iloc[0]
    axes[0].set_ylabel("held-out RPS (lower is better)")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"Exp 26: 160 homework answers, 5 frontier judges (uniform = {uni:.4f}, off the chart; shading = 95% interval)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig4_exp26_homework_rps_vs_ratings_per_judge")


# ---------------------------------------------------------------- fig 5: exp 24/25 LoRA HelpSteer2
def fig5():
    a = pd.read_csv(E24 / "results/scores.csv"); a = a[a.depth == 1]
    b = pd.read_csv(E25 / "results/scores.csv")
    panels = [(a, "Exp 24: Qwen3-4B LoRA, 10 draws", [16, 64, 256, 1024]),
              (b[b.study == "qwen3_4b"], "Exp 25A: Qwen3-4B LoRA, 20 draws", [512, 1024, 2048, 4096, 8192]),
              (b[b.study == "llama31_8b"], "Exp 25B: Llama-3.1-8B LoRA, 10 draws", [16, 64, 256, 1024])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, (d, title, ticks) in zip(axes, panels):
        for meth, lab, c, ls in [("ce_raw", "CE LoRA, raw", CE_C, "--"), ("pearson_raw", "Pearson LoRA, raw", PE_C, "--"),
                                 ("ce_calibrated", "CE LoRA + calibration", CE_C, "-"),
                                 ("pearson_calibrated", "Pearson LoRA + calibration", PE_C, "-"),
                                 ("uniform", "uniform", UNI_C, ":")]:
            h = d[d.method == meth].sort_values("b_total")
            ax.plot(h.b_total, h.rps, ls, marker="o", ms=3.5, color=c, label=lab)
        log2x(ax, ticks); ax.set_title(title); ax.set_xlabel("total human ratings B_total (train + val + cal)")
    axes[0].set_ylabel("held-out RPS (lower is better)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("HelpSteer2 (K=5): raw CE LoRA beats raw Pearson LoRA; after calibration the two are close", fontsize=11)
    fig.tight_layout()
    save(fig, "fig5_exp24_25_helpsteer2_lora_rps_vs_budget")


# ---------------------------------------------------------------- fig 6: exp 29 four objectives
STUDIES29 = [("chaos3_qwen", 1, "ChaosNLI, Qwen3-4B"), ("chaos3_llama", 1, "ChaosNLI, Llama-3.1-8B"),
             ("chaos3_qwen", 5, "ChaosNLI d=5, Qwen3-4B"), ("alpha2_qwen", 1, "alphaNLI (K=2), Qwen3-4B"),
             ("dices3_qwen", 1, "DICES, Qwen3-4B"), ("dices3_llama", 1, "DICES, Llama-3.1-8B"),
             ("hs2_qwen", 1, "HelpSteer2 (K=5), Qwen3-4B*")]


def fig6():
    s = pd.read_csv(E29 / "results/scores.csv")
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.2))
    axes = axes.ravel()
    for ax, (st, dep, title) in zip(axes, STUDIES29):
        g = s[(s.study == st) & (s.depth == dep)]
        for meth, lab, c, ls in [("ce_raw", "CE", CE_C, "-"), ("rps_raw", "RPS", "#17becf", "-"),
                                 ("mse_raw", "MSE", CAL_C, "--"), ("pearson_raw", "Pearson", PE_C, "-"),
                                 ("uniform", "uniform", UNI_C, ":")]:
            h = g[g.method == meth].sort_values("b_total")
            ax.plot(h.b_total, h.rps, ls, marker="o", ms=3.5, color=c, label=lab)
        ticks = sorted(g.b_total.unique())
        log2x(ax, ticks if len(ticks) <= 4 else ticks[::2]); ax.set_title(title, fontsize=10)
        ax.set_xlabel("total human ratings B_total")
    for i in (0, 4):
        axes[i].set_ylabel("held-out RPS, no calibration\n(lower is better)")
    axes[0].legend(fontsize=8, title="LoRA trained on", title_fontsize=8)
    axes[7].axis("off")
    axes[7].text(0.02, 0.5, "*HelpSteer2 CE and Pearson curves reuse\nexperiment 24 predictions; RPS and MSE\nare new experiment 29 runs.\n\nMeans over training draws.\nSee exp 29 RESULTS for crossed\nsimultaneous 95% intervals.",
                 fontsize=9, va="center")
    fig.suptitle("Exp 29: same LoRA judge trained on four objectives (raw, uncalibrated forecasts)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig6_exp29_four_objectives_rps_vs_budget")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
