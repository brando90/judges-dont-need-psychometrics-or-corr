#!/usr/bin/env python3
"""Figures and tables for the 11-25-2026 ICML-format technical memo.

Every number is read from committed result files in the sibling experiment folders.
Values computed here from a definition: the exact rational three-item counterexample of
experiment 27; the appendix-C bootstrap means, paired differences and crossovers, recomputed
from the per-seed raw.jsonl files of experiments 13, 06, 07 and 08 (10,000 percentile-bootstrap
draws over seeds, fixed seed); and one refit of experiment 13's toy for the distribution figure
(experiment 13's own toy.py, same seed as its make_report.py).  Output: figures/*.pdf, tables/*.tex, assets_log.txt
(the log prints the numbers that the memo text quotes, so each can be traced).

Run (any directory):  python3 -I make_memo_assets.py      (numpy, pandas, matplotlib)
"""
from fractions import Fraction as F
from math import log
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments"
FIG = HERE / "figures"
TAB = HERE / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)


def exp(prefix):
    hits = sorted(p for p in EXP.glob(prefix + "_*") if p.is_dir())
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


E24, E25, E26, E28, E29, E31 = (exp(n) for n in ("24", "25", "26", "28", "29", "31"))
PAPER = ROOT / "paper_latex_and_notes" / "AAAI_2027_EvalAIScientists_JudgesKL"

CE_C, PE_C, CAL_C, UNI_C, RAW_C, HIST_C, RPS_C = "#1f77b4", "#d62728", "#ff7f0e", "#7f7f7f", "#2ca02c", "#9467bd", "#17becf"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3,
    "pdf.fonttype": 42, "axes.grid": True, "grid.linewidth": 0.4, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
LOG = []


def log_line(s):
    LOG.append(s)
    print(s)


def finish(ax):
    """Every axis of every figure carries a grid (Brando, 11-21-2026); log axes also get faint minor lines."""
    ax.grid(True, which="major", alpha=0.35)
    if ax.get_xscale() == "log" or ax.get_yscale() == "log":
        ax.grid(True, which="minor", alpha=0.15)
    ax.set_axisbelow(True)


def log2x(ax, budgets):
    """Base-2 log x axis: every other budget labelled, the rest as faint minor grid lines."""
    budgets = sorted(set(int(b) for b in budgets))
    ax.set_xscale("log", base=2)
    major = budgets[::2]
    minor = [b for b in budgets if b not in major]
    ax.set_xticks(major)
    ax.set_xticklabels([str(b) for b in major])
    ax.set_xticks(minor, minor=True)
    ax.set_xticklabels([], minor=True)
    ax.set_xlim(min(budgets) / 1.2, max(budgets) * 1.2)


def save(fig, name):
    p = FIG / f"{name}.pdf"
    fig.savefig(p)
    plt.close(fig)
    log_line(f"wrote figures/{p.name}")


# ----------------------------------------------------------------------------- figure: exact counterexample (exp 27)
def fig_counterexample():
    P = [(F(3, 5), F(3, 10), F(1, 10)), (F(1, 5), F(3, 5), F(1, 5)), (F(1, 10), F(3, 10), F(3, 5))]
    Q = [(F(13, 20), F(1, 5), F(3, 20)), (F(2, 5), F(1, 5), F(2, 5)), (F(3, 20), F(1, 5), F(13, 20))]
    mean = lambda p: sum(F(k) * v for k, v in enumerate(p))
    assert [mean(p) for p in P] == [mean(q) for q in Q], "means must agree"
    tvd = float(np.mean([float(sum(abs(a - b) for a, b in zip(p, q)) / 2) for p, q in zip(P, Q)]))
    ce = lambda p, q: -sum(float(a) * log(float(b)) for a, b in zip(p, q))
    ce_true = float(np.mean([ce(p, p) for p in P]))
    ce_fc = float(np.mean([ce(p, q) for p, q in zip(P, Q)]))
    # excess normalized RPS (K=3): mean over items of (1/(K-1)) sum_k (Q_k - P_k)^2
    cdf = lambda p: np.cumsum([float(v) for v in p])[:-1]
    ex = float(np.mean([np.sum((cdf(q) - cdf(p)) ** 2) / 2 for p, q in zip(P, Q)]))
    log_line(f"counterexample: mean TVD {tvd:.4f}, CE true {ce_true:.6f}, CE forecast {ce_fc:.6f}, excess RPS {ex:.6f} (= 3/200 = {3/200})")
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 1.65), sharey=True)
    x = np.arange(3)
    for i, ax in enumerate(axes):
        ax.bar(x - 0.19, [float(v) for v in P[i]], 0.38, color=HIST_C, label="human distribution $p$")
        ax.bar(x + 0.19, [float(v) for v in Q[i]], 0.38, color=PE_C, alpha=0.85, label="judge forecast $q$")
        ax.set_xticks(x)
        ax.set_xticklabels(["0", "1", "2"])
        ax.set_xlabel("rating")
        ax.set_title(f"item {i + 1}: both means equal {mean(P[i])}")
        ax.set_ylim(0, 0.75)
        finish(ax)
    axes[0].set_ylabel("probability")
    axes[0].legend(loc="upper right")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_counterexample")


# ----------------------------------------------------------------------------- experiment 28
SETS28 = [("chaosnli_snli", "ChaosNLI SNLI"), ("chaosnli_mnli", "ChaosNLI MNLI-m"), ("chaosnli_alphanli", "alphaNLI (two classes)"),
          ("dices350", "DICES-350"), ("dices990", "DICES-990")]


def e28_data():
    s = pd.read_csv(E28 / "results/scores_absolute.csv")
    raw = pd.read_csv(E28 / "results/raw_judge_scores.csv").groupby("dataset").rps.mean()
    be = pd.read_csv(E28 / "results/breakeven_summary.csv")
    return s, raw, be


def fig_exp28(d, name, budgets_label):
    s, raw, _ = e28_data()
    s = s[s.d == d]
    m = s.groupby(["dataset", "method", "B"]).rps.mean()
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.5))
    axes = axes.ravel()
    handles = None
    for ax, (ds, title) in zip(axes, SETS28):
        for meth, lab, c, ls, mk in [("ce", "CE-fit", CE_C, "-", "o"), ("pearson", "Pearson-fit", PE_C, "-", "s"),
                                     ("pcal", "Pearson-fit + calibration", CAL_C, "--", "^"), ("uniform", "uniform forecast", UNI_C, ":", None)]:
            y = m.loc[ds, meth]
            ax.plot(y.index, y.values, ls, marker=mk, color=c, label=lab)
        ax.axhline(raw[ds], color=RAW_C, ls="-.", lw=1.1, label="raw judge, no fitting")
        log2x(ax, sorted(m.loc[ds, "ce"].index))
        ax.set_title(title)
        ax.set_xlabel(f"Human ratings used to fit ({budgets_label})")
        finish(ax)
    for i in (0, 3):
        axes[i].set_ylabel("Held-out test ranked\nprobability score (RPS);\nlower is better")
    h, l = axes[0].get_legend_handles_labels()
    axes[5].axis("off")
    axes[5].legend(h, l, loc="center", frameon=False, title="Mean over five frontier judges")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, name)


def tab_exp28():
    s, raw, be = e28_data()
    rows = []
    s1 = s[(s.d == 1) & (s.B == 256)].groupby(["dataset", "method"]).rps.mean()
    for ds, title in SETS28:
        b1 = be[(be.dataset == ds) & (be.d == 1)].N_both
        b5 = be[(be.dataset == ds) & (be.d == 5)].N_both
        def rng(v):
            lo, hi = int(v.min()), int(v.max())
            return f"{lo}" if lo == hi else f"{lo}--{hi}"
        rows.append((title, s1[(ds, "uniform")], raw[ds], s1[(ds, "ce")], s1[(ds, "pearson")], s1[(ds, "pcal")],
                     rng(b1), rng(b5)))
        log_line(f"exp28 {ds}: break-even N_both d=1 {int(b1.min())}-{int(b1.max())}, d=5 {int(b5.min())}-{int(b5.max())}")
    lines = [r"\begin{tabular}{lrrrrrcc}", r"\toprule",
             r"Rating set & Unif. & Raw & CE-fit & Pearson-fit & Pearson-fit+cal. & $N^\ast$ (1/item) & $N^\ast$ (5/item) \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r[0]} & {r[1]:.3f} & {r[2]:.3f} & \\textbf{{{r[3]:.3f}}} & {r[4]:.3f} & {r[5]:.3f} & {r[6]} & {r[7]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_exp28.tex").write_text("\n".join(lines) + "\n")
    # counts quoted in the text
    piv = s.pivot_table(index=["dataset", "judge", "d", "B"], columns="method", values="rps")
    lower = (piv["pearson"] < piv["ce"])
    log_line(f"exp28: comparison cells {len(piv)}; Pearson-fit has lower mean RPS than CE-fit in {int(lower.sum())} "
             f"(by d: {lower.groupby(level='d').sum().to_dict()}; by B: {lower[lower].groupby(level='B').size().to_dict()})")
    nb = be.groupby("d").N_both.agg(["min", "max"])
    log_line(f"exp28 break-even N_both over all 25 dataset-judge pairs, by d: {nb.to_dict('index')}; pairs with a value: {be.groupby('d').N_both.count().to_dict()}")
    # CE-fit lower RPS than Pearson-fit at the largest budget, per cell
    big = piv.reset_index()
    big = big[((big.d == 1) & (big.B == 256)) | ((big.d == 5) & (big.B == 320))]
    log_line(f"exp28 at largest budget: CE-fit lower mean RPS than Pearson-fit in {int((big.ce < big.pearson).sum())} of {len(big)} cells")
    # all three metrics at 256 (d=1), mean over judges, per set
    m3 = s[(s.d == 1) & (s.B == 256) & s.method.isin(["ce", "pearson"])].groupby(["dataset", "method"])[["rps", "ce", "tvd"]].mean()
    for ds, title in SETS28:
        log_line(f"exp28 {title} @256 d=1: CE-fit RPS/CE/TVD {m3.loc[(ds, 'ce')].round(3).tolist()}  Pearson-fit {m3.loc[(ds, 'pearson')].round(3).tolist()}")


# ----------------------------------------------------------------------------- experiment 26
JUDGES26 = [("gpt-6-astra", "GPT-6 Astra"), ("claude-opus-5-5", "Claude Opus 5.5"), ("kimi-k3", "Kimi K3"),
            ("glm-5.3", "GLM 5.3"), ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash")]


def tab_exp26():
    s = pd.read_csv(E26 / "results/new5/scores.csv")
    s = s[(s.ratings_per_answer == 2) & (s.metric == "rps")]
    dec = pd.read_csv(E26 / "results/new5/decisions.csv")
    lines = [r"\begin{tabular}{lrrrrrl}", r"\toprule",
             r"Judge & Uncal. & Train hist. & CE-fit & Pearson-fit & Diff. & Test passes down to \\", r"\midrule"]
    pooled = []
    for j, title in JUDGES26:
        g = s[(s.judge == j) & (s.training_ratings == 256)].set_index("method")["mean"]
        d = dec[dec.judge == j].sort_values("training_ratings", ascending=False)
        passed = []
        for _, r in d.iterrows():
            if r.formal_status == "pass":
                passed.append(int(r.training_ratings))
            else:
                break
        down = f"{min(passed)}" if passed else "none (fails at 256)"
        diff = g["pearson_fit"] - g["ce_fit"]
        lines.append(f"{title} & {g['identity']:.4f} & {g['train_hist']:.4f} & \\textbf{{{g['ce_fit']:.4f}}} & {g['pearson_fit']:.4f} & +{diff:.4f} & {down} \\\\")
        log_line(f"exp26 {title}: @256 CE-fit {g['ce_fit']:.4f} Pearson-fit {g['pearson_fit']:.4f} diff {diff:.4f}; uncal {g['identity']:.4f}; hist {g['train_hist']:.4f}; descending test passes down to {down}")
        allb = s[(s.judge == j) & s.method.isin(["ce_fit", "pearson_fit"])].pivot_table(index="training_ratings", columns="method", values="mean")
        pooled.append(int((allb.ce_fit < allb.pearson_fit).sum()))
    uni = s[s.method == "uniform"]["mean"].iloc[0]
    lines += [r"\midrule", f"Uniform over six grades & \\multicolumn{{6}}{{l}}{{{uni:.4f}}} \\\\", r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_exp26.tex").write_text("\n".join(lines) + "\n")
    log_line(f"exp26: CE-fit has the lower mean RPS in {sum(pooled)} of {5 * 7} judge-by-budget cells (per judge {pooled})")


def fig_exp26():
    s = pd.read_csv(E26 / "results/new5/scores.csv")
    s = s[(s.ratings_per_answer == 2) & (s.metric == "rps")]
    fig, axes = plt.subplots(1, 5, figsize=(6.75, 1.9), sharey=True)
    for ax, (j, title) in zip(axes, JUDGES26):
        g = s[s.judge == j]
        for meth, lab, c, ls, mk in [("ce_fit", "CE-fit", CE_C, "-", "o"), ("pearson_fit", "Pearson-fit", PE_C, "-", "s"),
                                     ("identity", "uncalibrated judge", RAW_C, "-.", None), ("train_hist", "training histogram", HIST_C, ":", None)]:
            h = g[g.method == meth].sort_values("training_ratings")
            ax.plot(h.training_ratings, h["mean"], ls, marker=mk, color=c, label=lab)
            if meth in ("ce_fit", "pearson_fit"):
                ax.fill_between(h.training_ratings, h.lower95, h.upper95, color=c, alpha=0.12, lw=0)
        log2x(ax, [4, 8, 16, 32, 64, 128, 256])
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("Training ratings", fontsize=7)
        finish(ax)
    axes[0].set_ylabel("Held-out test ranked\nprobability score (RPS)")
    axes[0].legend(loc="upper right", fontsize=5.5)
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig_exp26_per_judge")


# ----------------------------------------------------------------------------- experiment 29
STUDIES29 = [("chaos3_qwen", 1, "ChaosNLI\nQwen3-4B"), ("chaos3_llama", 1, "ChaosNLI\nLlama-3.1-8B"),
             ("chaos3_qwen", 5, "ChaosNLI, 5/item\nQwen3-4B"), ("alpha2_qwen", 1, "alphaNLI (K=2)\nQwen3-4B"),
             ("dices3_qwen", 1, "DICES\nQwen3-4B"), ("dices3_llama", 1, "DICES\nLlama-3.1-8B"),
             ("hs2_qwen", 1, "HelpSteer2\nQwen3-4B*")]


def e29_data():
    return pd.read_csv(E29 / "results/scores.csv"), pd.read_csv(E29 / "results/contrasts.csv")


def fig_exp29_bars():
    s, _ = e29_data()
    fig, ax = plt.subplots(figsize=(6.75, 2.35))
    w = 0.19
    x = np.arange(len(STUDIES29))
    for k, (meth, lab, c) in enumerate([("ce_raw", "CE", CE_C), ("rps_raw", "RPS", RPS_C), ("mse_raw", "MSE", CAL_C), ("pearson_raw", "Pearson", PE_C)]):
        ys = []
        for st, dep, _ in STUDIES29:
            g = s[(s.study == st) & (s.depth == dep)]
            b = g.b_total.max()
            ys.append(float(g[(g.b_total == b) & (g.method == meth)].rps.iloc[0]))
        ax.bar(x + (k - 1.5) * w, ys, w, color=c, label=lab + "-trained")
    for i, (st, dep, _) in enumerate(STUDIES29):
        g = s[(s.study == st) & (s.depth == dep)]
        b = g.b_total.max()
        u = float(g[(g.b_total == b) & (g.method == "uniform")].rps.iloc[0])
        ax.plot([i - 2 * w, i + 2 * w], [u, u], color="k", ls=":", lw=1.0, label="uniform forecast" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([lab + f"\n(B={int(s[(s.study == st) & (s.depth == dep)].b_total.max())})" for st, dep, lab in STUDIES29], fontsize=6.5)
    ax.set_ylabel("Held-out test ranked probability\nscore (RPS), raw forecast;\nlower is better")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    finish(ax)
    ax.grid(True, axis="x", alpha=0.15)
    fig.tight_layout()
    save(fig, "fig_exp29_four_objectives")


def fig_exp29_curves():
    s, _ = e29_data()
    fig, axes = plt.subplots(2, 4, figsize=(6.75, 3.6))
    axes = axes.ravel()
    for ax, (st, dep, title) in zip(axes, STUDIES29):
        g = s[(s.study == st) & (s.depth == dep)]
        for meth, lab, c, ls, mk in [("ce_raw", "CE", CE_C, "-", "o"), ("rps_raw", "RPS", RPS_C, "-", "v"), ("mse_raw", "MSE", CAL_C, "--", "D"),
                                     ("pearson_raw", "Pearson", PE_C, "-", "s"), ("uniform", "uniform", UNI_C, ":", None)]:
            h = g[g.method == meth].sort_values("b_total")
            ax.plot(h.b_total, h.rps, ls, marker=mk, color=c, label=lab)
        log2x(ax, sorted(g.b_total.unique()))
        ax.set_title(title.replace("\n", ", "), fontsize=6.5)
        ax.set_xlabel("Total human ratings $B$", fontsize=7)
        finish(ax)
    for i in (0, 4):
        axes[i].set_ylabel("Held-out test ranked\nprobability score (RPS)")
    h, l = axes[0].get_legend_handles_labels()
    axes[7].axis("off")
    axes[7].legend(h, l, loc="center", frameon=False, title="Low-rank adaptation\n(LoRA) trained with")
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, "fig_exp29_curves")


def tab_exp29():
    s, c = e29_data()
    lines = [r"\begin{tabular}{lrrrrrrl}", r"\toprule",
             r"LoRA setup & $B$ & CE & RPS & MSE & Pearson & Pearson$-$CE & Crossed 95\% band \\", r"\midrule"]
    plain = {"chaos3_qwen": "Qwen3-4B", "chaos3_llama": "Llama-3.1-8B", "alpha2_qwen": "Qwen3-4B", "dices3_qwen": "Qwen3-4B",
             "dices3_llama": "Llama-3.1-8B", "hs2_qwen": "Qwen3-4B"}
    names = {"chaos3_qwen": "ChaosNLI", "chaos3_llama": "ChaosNLI", "alpha2_qwen": "alphaNLI ($K{=}2$)", "dices3_qwen": "DICES",
             "dices3_llama": "DICES", "hs2_qwen": "HelpSteer2$^{\\ast}$"}
    allpos = True
    for st, dep, _ in STUDIES29:
        g = s[(s.study == st) & (s.depth == dep)]
        b = int(g.b_total.max())
        v = {m: float(g[(g.b_total == b) & (g.method == m)].rps.iloc[0]) for m in ("ce_raw", "rps_raw", "mse_raw", "pearson_raw")}
        cc = c[(c.study == st) & (c.depth == dep) & (c.b_total == b) & (c.metric == "rps") & (c.target == "ce_raw") & (c.comparator == "pearson_raw")].iloc[0]
        lo, hi = cc.cross_sim_lo_95, cc.cross_sim_hi_95
        allpos &= bool(lo > 0)
        label = f"{names[st]}{', 5/item' if dep == 5 else ''}, {plain[st]}"
        lines.append(f"{label} & {b} & \\textbf{{{v['ce_raw']:.4f}}} & {v['rps_raw']:.4f} & {v['mse_raw']:.4f} & {v['pearson_raw']:.4f} & "
                     f"+{cc.diff_comparator_minus_target:.4f} & [{lo:+.4f}, {hi:+.4f}] \\\\")
        log_line(f"exp29 {st} d={dep} B={b}: raw RPS CE {v['ce_raw']:.4f} RPS {v['rps_raw']:.4f} MSE {v['mse_raw']:.4f} Pearson {v['pearson_raw']:.4f}; "
                 f"Pearson-minus-CE {cc.diff_comparator_minus_target:+.4f} crossed sim 95% [{lo:+.4f}, {hi:+.4f}]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_exp29.tex").write_text("\n".join(lines) + "\n")
    log_line(f"exp29: crossed simultaneous lower bound above zero for Pearson minus CE in all 7 setups at the largest budget: {allpos}")
    # MSE vs CE at the largest budget, crossed band
    rows = []
    for st, dep, _ in STUDIES29:
        g = s[(s.study == st) & (s.depth == dep)]
        b = int(g.b_total.max())
        cc = c[(c.study == st) & (c.depth == dep) & (c.b_total == b) & (c.metric == "rps") & (c.target == "ce_raw") & (c.comparator == "mse_raw")].iloc[0]
        rows.append((st, dep, cc.diff_comparator_minus_target, cc.cross_sim_lo_95, cc.cross_sim_hi_95))
        log_line(f"exp29 MSE-minus-CE {st} d={dep}: {cc.diff_comparator_minus_target:+.5f} crossed sim 95% [{cc.cross_sim_lo_95:+.5f}, {cc.cross_sim_hi_95:+.5f}]")


# ----------------------------------------------------------------------------- experiments 24 and 25 (appendix figure)
def fig_lora_helpsteer():
    a = pd.read_csv(E24 / "results/scores.csv")
    a = a[a.depth == 1]
    b = pd.read_csv(E25 / "results/scores.csv")
    panels = [(a, "Exp. 24: Qwen3-4B, 10 draws"), (b[b.study == "qwen3_4b"], "Exp. 25A: Qwen3-4B, 20 draws"), (b[b.study == "llama31_8b"], "Exp. 25B: Llama-3.1-8B, 10 draws")]
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.0), sharey=True)
    for ax, (d, title) in zip(axes, panels):
        for meth, lab, c, ls, mk in [("ce_raw", "CE LoRA, raw", CE_C, "--", "o"), ("pearson_raw", "Pearson LoRA, raw", PE_C, "--", "s"),
                                     ("ce_calibrated", "CE LoRA + calibration", CE_C, "-", "o"), ("pearson_calibrated", "Pearson LoRA + calibration", PE_C, "-", "s"),
                                     ("uniform", "uniform", UNI_C, ":", None)]:
            h = d[d.method == meth].sort_values("b_total")
            ax.plot(h.b_total, h.rps, ls, marker=mk, color=c, label=lab)
        log2x(ax, sorted(d.b_total.unique()))
        ax.set_title(title)
        ax.set_xlabel("Total human ratings $B_{\\rm total}$")
        finish(ax)
    axes[0].set_ylabel("Held-out test ranked\nprobability score (RPS);\nlower is better")
    axes[0].legend(loc="upper right", fontsize=5.5)
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig_lora_helpsteer2")


# ----------------------------------------------------------------------------- experiment 31: scaling laws
def e31_fits():
    return pd.read_csv(E31 / "results/fits_11-25-2026.csv")


def curve(row, n):
    return row["floor"] + row["a_at_Nmin"] * (n / row["N_min"]) ** (-row["b"])


def fig_scaling():
    f = e31_fits()
    s28, raw28, _ = e28_data()
    s29, _ = e29_data()
    panels = [("28_dices990_d1", "dices990", "Exp. 28, DICES-990 (1/item)", 28), ("28_chaosnli_mnli_d1", "chaosnli_mnli", "Exp. 28, ChaosNLI MNLI-m (1/item)", 28),
              ("29_chaos3_qwen_d1", ("chaos3_qwen", 1), "Exp. 29, ChaosNLI, Qwen3-4B LoRA", 29), ("29_hs2_qwen_d1", ("hs2_qwen", 1), "Exp. 29, HelpSteer2, Qwen3-4B LoRA", 29)]
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 1.95))
    for ax, (pid, key, title, which) in zip(axes, panels):
        fr = f[(f.panel_id == pid) & (f.metric == "rps")]
        ce, pe = fr[fr.objective == "CE"].iloc[0], fr[fr.objective == "Pearson"].iloc[0]
        if which == 28:
            g = s28[(s28.d == 1) & (s28.dataset == key)].groupby(["method", "B"]).rps.mean()
            xs = sorted(g["ce"].index)
            cey, pey = g["ce"].values, g["pearson"].values
            uni, raw = float(g["uniform"].iloc[0]), float(raw28[key])
        else:
            g = s29[(s29.study == key[0]) & (s29.depth == key[1])]
            xs = sorted(g.b_total.unique())
            cey = g[g.method == "ce_raw"].sort_values("b_total").rps.values
            pey = g[g.method == "pearson_raw"].sort_values("b_total").rps.values
            uni = float(g[g.method == "uniform"].rps.iloc[0])
            raw = float(g[g.method == "base_raw"].rps.iloc[0]) if "base_raw" in set(g.method) else None
        n = np.geomspace(min(xs), max(xs), 100)
        ne = np.geomspace(max(xs), max(xs) * 4, 40)
        ax.plot(xs, cey, "o", color=CE_C, label="CE-fit")
        ax.plot(n, [curve(ce, v) for v in n], "-", color=CE_C, lw=1.0, label="fitted floor + $aN^{-b}$")
        ax.plot(ne, [curve(ce, v) for v in ne], ":", color=CE_C, lw=1.0)
        ax.axhline(ce["floor"], color=CE_C, ls=(0, (1, 3)), lw=0.8)
        ax.plot(xs, pey, "s--", color=PE_C, label="Pearson-fit")
        ax.axhline(uni, color=UNI_C, ls=":", lw=1.0, label="uniform forecast")
        if raw is not None:
            ax.axhline(raw, color=RAW_C, ls="-.", lw=1.0, label="raw model, no fitting")
        ax.set_xscale("log", base=2)
        ax.set_xlim(min(xs) / 1.2, max(xs) * 4.8)
        ticks = [int(v) for v in xs][::2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xticks([int(v) for v in xs if int(v) not in ticks], minor=True)
        ax.set_xticklabels([], minor=True)
        ax.set_title(title, fontsize=6.3)
        ax.set_xlabel("Human ratings $N$ used to fit", fontsize=7)
        finish(ax)
        log_line(f"scaling panel {pid}: CE floor {ce['floor']:.3f} [{ce.floor_lo:.3f}, {ce.floor_hi:.3f}], b {ce.b:.2f} [{ce.b_lo:.2f}, {ce.b_hi:.2f}]; Pearson b {pe.b:.2f}")
    axes[0].set_ylabel("Held-out test ranked\nprobability score (RPS)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.09), frameon=False, fontsize=6.3)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_scaling")


def tab_scaling():
    f = e31_fits()
    names28 = dict(SETS28)
    rows = []
    for ds, title in SETS28:
        rows.append((f"Exp.\\ 28, {title}", f"28_{ds}_d1", 256))
    for st, title in [("chaos3_qwen", "ChaosNLI, Qwen3-4B"), ("chaos3_llama", "ChaosNLI, Llama-3.1-8B"), ("alpha2_qwen", "alphaNLI, Qwen3-4B"),
                      ("dices3_qwen", "DICES, Qwen3-4B"), ("dices3_llama", "DICES, Llama-3.1-8B"), ("hs2_qwen", "HelpSteer2, Qwen3-4B")]:
        rows.append((f"Exp.\\ 29, {title}", f"29_{st}_d1", None))
    lines = [r"\begin{tabular}{lrrrccl}", r"\toprule",
             r"Panel (held-out test RPS) & $N_{\max}$ & CE-fit at $N_{\max}$ & Pearson-fit at $N_{\max}$ & CE-fit floor [95\%] & CE-fit $b$ [95\%] & Pearson-fit curve (experiment 31 verdict) \\", r"\midrule"]
    for title, pid, _ in rows:
        fr = f[(f.panel_id == pid) & (f.metric == "rps")]
        ce, pe = fr[fr.objective == "CE"].iloc[0], fr[fr.objective == "Pearson"].iloc[0]
        pv = str(pe.verdict)
        if pv.startswith("flat"):
            short = "flat"
        elif "b <= 0.2" in pv:
            short = r"slow decline, floor not identified"
        elif pv.startswith("poor"):
            short = "poor power-law fit"
        elif pv.startswith("non"):
            short = "non-monotone"
        else:
            short = pv
        lines.append(f"{title} & {int(ce.N_max)} & {ce['last']:.3f} & {pe['last']:.3f} & {ce.floor:.3f} [{ce.floor_lo:.3f}, {ce.floor_hi:.3f}] & "
                     f"{ce.b:.2f} [{ce.b_lo:.2f}, {ce.b_hi:.2f}] & {short} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_scaling.tex").write_text("\n".join(lines) + "\n")
    # medians quoted in the text (CE-fit, judge-by-set cells of experiment 28, 1 rating per item, floor identified)
    c = f[(f.exp == 28) & (f.level == "cell") & (f.objective == "CE") & (f.panel_id.str.endswith("_d1"))]
    for metric in ("rps", "ce", "tvd"):
        cm = c[(c.metric == metric) & (c.floor_identified.astype(str) == "True")]
        log_line(f"scaling median b, CE-fit, exp28 cells d=1 with identified floor, {metric}: {cm.b.median():.2f} over {len(cm)} cells (all CE cells {len(c[c.metric == metric])})")
    for st in ("chaos3_qwen", "hs2_qwen", "chaos3_llama"):
        r = f[(f.panel_id == f"29_{st}_d1") & (f.metric == "rps") & (f.objective == "CE")].iloc[0]
        log_line(f"scaling exp29 {st} CE-LoRA b {r.b:.2f} [{r.b_lo:.2f}, {r.b_hi:.2f}]")
    log_line(f"scaling: removed share per doubling at b=0.66: {1 - 2 ** -0.66:.3f}")


# ----------------------------------------------------------------------------- appendix C: synthetic studies (experiments 13, 06, 07, 08)
import importlib.util
import json

ARCH = EXP / "archive"
E13 = sorted(ARCH.glob("13_*"))[0]
E06, E07, E08 = (exp(n) for n in ("06", "07", "08"))
BOOT = 10000
SP_C, KE_C, MX_C = "#9467bd", "#8c564b", "#e377c2"
RPS_AX = "Held-out test ranked\nprobability score (RPS) excess\nover the true distribution\n(0 is best; lower is better)"
KL_AX = "Held-out test cross-entropy (CE) excess\n= Kullback\u2013Leibler (KL) divergence\nKL(true $\\Vert$ judge), nats\n(0 is best; lower is better)"


def load_raw(folder):
    return [json.loads(line) for line in open(folder / "results" / "raw.jsonl")]


def per_seed(rows, getter, allow_nan=False):
    """{(scenario, n): np.array over seeds (ascending seed order)}."""
    cells = {}
    for r in rows:
        cells.setdefault((r["scenario"], r["n"]), {})[r["seed"]] = getter(r)
    out = {}
    for k, v in cells.items():
        arr = np.array([v[sd] for sd in sorted(v)], float)
        assert allow_nan or not np.isnan(arr).any(), k
        out[k] = arr
    return out


def boot_ci(v, rng):
    """(mean, 2.5th, 97.5th percentile of the bootstrap mean) over seeds."""
    v = np.asarray(v, float)
    m = v[rng.integers(0, len(v), size=(BOOT, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sustained_from(ns, flags):
    """Smallest n from which flags[n] is True at every larger tested n (None if never)."""
    for i in range(len(ns)):
        if all(flags[n] for n in ns[i:]):
            return ns[i]
    return None


def paired_summary(a, b, ns, sc, rng):
    """a, b: {(sc, n): array}.  Returns {n: (mean, lo, hi)} of a - b, paired by seed."""
    out = {}
    for n in ns:
        assert a[(sc, n)].shape == b[(sc, n)].shape
        out[n] = boot_ci(a[(sc, n)] - b[(sc, n)], rng)
    return out


def ratings(n):
    return 3 * n


FIT13 = [("ce", "CE-trained", CE_C, "-"), ("rps", "RPS-trained", RPS_C, "-"),
         ("pearson", "Pearson-trained", PE_C, "-"), ("pearson_head", "Pearson-trained + calibrated head", CAL_C, "--")]


def fig_exp13():
    rows = load_raw(E13)
    rng = np.random.default_rng(20261125)
    ns = sorted({r["n"] for r in rows})
    assert len(rows) == 2200 and {r["seed"] for r in rows} == set(range(100))
    data = {m: {f: per_seed(rows, lambda r, f=f, m=m: r[f][m]) for f, _, _, _ in FIT13} for m in ("rps_excess", "ce_excess")}
    fig, axs = plt.subplots(2, 2, figsize=(6.9, 4.3), sharex=True)
    short_rps = "Held-out test ranked\nprobability score (RPS)\nexcess (lower is better)"
    short_kl = "Held-out test cross-entropy\n(CE) excess = Kullback\u2013Leibler\n(KL) divergence KL(true $\\Vert$ judge),\nnats (lower is better)"
    summary = {}
    for i, sc in enumerate("AB"):
        for j, (m, ylab) in enumerate([("rps_excess", short_rps), ("ce_excess", short_kl)]):
            ax = axs[i, j]
            for f, lab, col, ls in FIT13:
                st = np.array([boot_ci(data[m][f][(sc, n)], rng) for n in ns])
                x = [ratings(n) for n in ns]
                ax.plot(x, st[:, 0], ls, color=col, label=lab)
                ax.fill_between(x, st[:, 1], st[:, 2], color=col, alpha=0.18, linewidth=0)
                summary[(m, sc, f)] = st
            d = paired_summary(data[m]["pearson"], data[m]["ce"], ns, sc, rng)
            cross = sustained_from(ns, {n: d[n][1] > 0 for n in ns})
            summary[(m, sc, "diff")] = d
            summary[(m, sc, "cross")] = cross
            log_line(f"exp13 {sc} {m}: CE-trained significantly better than Pearson-trained from "
                     + (f"{cross} tasks ({ratings(cross)} ratings)" if cross else "never") + " (paired 95% bootstrap lower bound above 0 at every larger n)")
            if cross:
                ax.axvline(ratings(cross), color="k", ls=":", lw=0.9)
            ax.set_yscale("log")
            log2x(ax, [ratings(n) for n in ns])
            ax.set_title(f"Scenario {sc}: " + ("mean nearly constant" if sc == "A" else "informative mean"), loc="left")
            if j == 0:
                ax.set_ylabel(ylab)
            else:
                ax.set_ylabel(ylab)
            finish(ax)
    for ax in axs[1]:
        ax.set_xlabel("Training ratings (three per task)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(h_pad=0.9, w_pad=0.8)
    save(fig, "fig_exp13_curves")
    for sc in "AB":
        for f, lab, _, _ in FIT13:
            for m in ("rps_excess", "ce_excess"):
                mm = summary[(m, sc, f)][-1, 0]
                log_line(f"exp13 {sc} 2048 tasks (6144 ratings) {lab} {m} mean {mm:.4f}")
    # ratios at the largest budget
    for sc in "AB":
        ce = summary[("rps_excess", sc, "ce")][-1, 0]
        pe = summary[("rps_excess", sc, "pearson")][-1, 0]
        log_line(f"exp13 {sc} 2048 tasks: Pearson-trained RPS excess / CE-trained RPS excess = {pe / ce:.1f}")
    return summary, ns, data


def tab_exp13(summary, ns):
    sel = [4, 8, 16, 32, 64, 128, 2048]
    lines = [r"\begin{tabular}{@{}rrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Scenario A (mean nearly constant)} & \multicolumn{3}{c}{Scenario B (informative mean)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Tasks (ratings) & CE & Pearson & Pearson $-$ CE [95\% CI] & CE & Pearson & Pearson $-$ CE [95\% CI] \\", r"\midrule"]
    for n in sel:
        k = ns.index(n)
        cells = []
        for sc in "AB":
            ce = summary[("rps_excess", sc, "ce")][k, 0]
            pe = summary[("rps_excess", sc, "pearson")][k, 0]
            dm, dlo, dhi = summary[("rps_excess", sc, "diff")][n]
            cells += [f"{ce:.4f}", f"{pe:.4f}", f"{dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]"]
            log_line(f"exp13 table {sc} n={n}: CE {ce:.4f} Pearson {pe:.4f} diff {dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]")
        lines.append(f"{n} ({ratings(n):,}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_exp13.tex").write_text("\n".join(lines).replace(",", "{,}", 0) + "\n")
    log_line("wrote tables/tab_exp13.tex")


def fig_exp13_distributions():
    spec = importlib.util.spec_from_file_location("toy13", E13 / "scripts" / "toy.py")
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    from scipy.special import softmax
    n = 1024
    r0 = np.random.default_rng(5)
    x = r0.uniform(-1, 1, n)
    Pt = toy.truth(x, "A")
    P = np.array([r0.multinomial(3, p) for p in Pt]) / 3
    Phi = toy.phi(x)
    W0 = toy.marginal_W0(Phi, P, 0)
    Wc, Wp = toy.fit(Phi, P, "ce", n, W0), toy.fit(Phi, P, "pearson", n, W0)
    s = softmax(Phi @ Wp, 1) @ toy.GRID
    m, sd = s.mean(), s.std()
    Hh = np.stack([np.ones(n), (s - m) / sd], 1)
    Wh = toy.fit(Hh, P, "ce", n, toy.marginal_W0(Hh, P, 0))
    xs = np.array([-0.7, 0.0, 0.5])
    st = softmax(toy.phi(xs) @ Wp, 1) @ toy.GRID
    Qh = softmax(np.stack([np.ones(3), (st - m) / sd], 1) @ Wh, 1)
    Qc = softmax(toy.phi(xs) @ Wc, 1)
    Tp = toy.truth(xs, "A")
    Qp = softmax(toy.phi(xs) @ Wp, 1)
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 1.9), sharey=True)
    w = 0.27
    for a, xv, tp, qc, qh in zip(axs, xs, Tp, Qc, Qh):
        a.bar(toy.GRID - w, tp, w, color="k", label="true human distribution")
        a.bar(toy.GRID, qc, w, color=CE_C, label="CE-trained judge")
        a.bar(toy.GRID + w, qh, w, color=CAL_C, label="Pearson-trained + calibrated head")
        a.set_title(f"$x={xv:g}$: true mean rating {tp @ toy.GRID:.2f}")
        a.set_xlabel("Rating")
        finish(a)
        tv_c, tv_h = 0.5 * np.abs(qc - tp).sum(), 0.5 * np.abs(qh - tp).sum()
        log_line(f"exp13 distributions x={xv:g}: true mean {tp @ toy.GRID:.2f}, true P(0)+P(5) {tp[0] + tp[5]:.2f}, CE-trained P(0)+P(5) {qc[0] + qc[5]:.2f}, "
                 f"head P(0)+P(5) {qh[0] + qh[5]:.2f}; TVD to truth: CE-trained {tv_c:.2f}, head {tv_h:.2f}")
    axs[0].set_ylabel("Probability")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_exp13_distributions")


FIT06 = [("ce", "CE-trained", CE_C, "-"), ("pearson", "Pearson-trained", PE_C, "-"), ("spearman", "Spearman-trained", SP_C, "-"),
         ("kendall", "Kendall-trained", KE_C, "-"), ("pearson_spearman", "½ Pearson + ½ Spearman-trained", MX_C, "--")]


def fig_exp06(data13):
    rows6, rows13 = load_raw(E06), load_raw(E13)
    rng = np.random.default_rng(20261126)
    ns = sorted({r["n"] for r in rows6})
    d6 = {f: per_seed(rows6, lambda r, f=f: r[f]["rps_excess"]) for f in ("spearman", "kendall", "pearson_spearman")}
    d13 = {f: per_seed(rows13, lambda r, f=f: r[f]["rps_excess"]) for f in ("ce", "pearson")}
    saved13 = {(r["scenario"], r["n"], r["seed"]): r["ce"]["rps_excess"] for r in rows13}
    chk = [abs(r["ce_check"]["rps_excess"] - saved13[(r["scenario"], r["n"], r["seed"])]) for r in rows6 if "ce_check" in r]
    log_line(f"exp06 check: a CE refit on the regenerated tasks (seed 0 in each of {len(chk)} scenario-by-size cells) differs from experiment 13's saved CE fit by at most {max(chk):.1e} in RPS excess")
    series = {"ce": d13["ce"], "pearson": d13["pearson"], "spearman": d6["spearman"], "kendall": d6["kendall"], "pearson_spearman": d6["pearson_spearman"]}
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=False)
    out = {}
    for i, sc in enumerate("AB"):
        ax = axs[i]
        for f, lab, col, ls in FIT06:
            st = np.array([boot_ci(series[f][(sc, n)], rng) for n in ns])
            x = [ratings(n) for n in ns]
            ax.plot(x, st[:, 0], ls, color=col, label=lab)
            ax.fill_between(x, st[:, 1], st[:, 2], color=col, alpha=0.12, linewidth=0)
            out[(sc, f)] = st
        for f in ("pearson", "spearman", "kendall", "pearson_spearman"):
            d = paired_summary(series[f], series["ce"], ns, sc, rng)
            cross = sustained_from(ns, {n: d[n][1] > 0 for n in ns})
            out[(sc, f, "cross")] = cross
            log_line(f"exp06 {sc}: CE-trained significantly better than {dict((a, b) for a, b, _, _ in FIT06)[f]} on RPS excess from "
                     + (f"{cross} tasks ({ratings(cross)} ratings)" if cross else "never"))
        for f in ("pearson", "spearman", "kendall", "pearson_spearman"):
            log_line(f"exp06 {sc} 2048 tasks: {f} RPS excess {out[(sc, f)][-1, 0]:.4f}, CE {out[(sc, 'ce')][-1, 0]:.4f}, ratio {out[(sc, f)][-1, 0] / out[(sc, 'ce')][-1, 0]:.1f}")
        ax.set_yscale("log")
        log2x(ax, [ratings(n) for n in ns])
        ax.set_title(f"Scenario {sc}: " + ("mean nearly constant" if sc == "A" else "informative mean"), loc="left")
        ax.set_xlabel("Training ratings (three per task)")
        ax.set_ylabel(RPS_AX)
        finish(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.10), columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_exp06_rank")
    return out


def fig_exp07():
    rows = load_raw(E07)
    rng = np.random.default_rng(20261127)
    ns = sorted({r["n"] for r in rows})
    assert len(rows) == 540 and {r["seed"] for r in rows} == set(range(30))
    def g(fit, cond, key="rps_excess"):
        return per_seed(rows, lambda r: r["fits"][fit][cond][key])
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.5))
    for i, sc in enumerate("AB"):
        ax = axs[i]
        spec = [("ce", "long", "CE-trained (2,000 iterations)", CE_C, "-"),
                ("pearson", "orig", "Pearson-trained, original budget", PE_C, ":"),
                ("pearson", "best", "Pearson-trained, best of 5 starts", PE_C, "-"),
                ("spearman", "best", "Spearman-trained, best of 5 starts", SP_C, "-"),
                ("pearson_spearman", "best", "½ Pearson + ½ Spearman, best of 5 starts", MX_C, "--")]
        for fit, cond, lab, col, ls in spec:
            d = g(fit, cond)
            st = np.array([boot_ci(d[(sc, n)], rng) for n in ns])
            x = [ratings(n) for n in ns]
            ax.plot(x, st[:, 0], ls, color=col, label=lab)
            ax.fill_between(x, st[:, 1], st[:, 2], color=col, alpha=0.10, linewidth=0)
        ax.set_yscale("log")
        log2x(ax, [ratings(n) for n in ns])
        ax.set_title(f"Scenario {sc}: " + ("mean nearly constant" if sc == "A" else "informative mean"), loc="left")
        ax.set_xlabel("Training ratings (three per task)")
        ax.set_ylabel(RPS_AX)
        finish(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.15), columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_exp07_restarts")
    # crossovers, convergence and restart gains
    for sc in "AB":
        ce = g("ce", "long")
        for fit in ("pearson", "spearman", "pearson_spearman"):
            for cond in ("orig", "best"):
                d = paired_summary(g(fit, cond), ce, ns, sc, rng)
                cross = sustained_from(ns, {n: d[n][1] > 0 for n in ns})
                log_line(f"exp07 {sc} {fit} {cond}: CE-trained (long) significantly better on RPS excess from "
                         + (f"{cross} tasks ({ratings(cross)} ratings)" if cross else "never"))
    big = [r for r in rows if r["n"] >= 32]
    for fit in ("ce", "rps", "pearson", "pearson_head", "spearman", "kendall", "pearson_spearman"):
        orig = np.mean([r["fits"][fit]["orig"]["success"] for r in big])
        hit = np.mean([r["fits"][fit]["long"]["budget_hit"] for r in big])
        log_line(f"exp07 {fit}, 32 or more tasks: original fit converged in {100 * orig:.0f}% of fits; long fit hit its 2000-iteration budget in {100 * hit:.0f}% of fits")
    # experiment 07's own definition: best of the five restarts beats the first (original-start) long fit by more than 1e-6, per objective and size, scenarios A and B pooled
    shares = {}
    for fit in ("pearson", "spearman", "pearson_spearman"):
        for n in ns:
            sel = [r for r in rows if r["n"] == n]
            shares[(fit, n)] = 100 * np.mean([min(r["fits"][fit]["restart_train_obj"]) < r["fits"][fit]["restart_train_obj"][0] - 1e-6 for r in sel])
    lo_k, hi_k = min(shares, key=shares.get), max(shares, key=shares.get)
    log_line(f"exp07: share of fits where the best of 5 starts lowers the penalized training objective by more than 1e-6 relative to the first long fit, "
             f"over the three correlation objectives and nine sizes: {shares[lo_k]:.0f}% to {shares[hi_k]:.0f}%")
    for sc in "AB":
        d_o, d_b, d_c = (per_seed(rows, lambda r, c=c: r["fits"][f_][c]["rps_excess"]) for f_, c in (("pearson", "orig"), ("pearson", "best"), ("ce", "long")))
        log_line(f"exp07 {sc} 1024 tasks: Pearson-trained RPS excess {d_o[(sc, 1024)].mean():.4f} (original) -> {d_b[(sc, 1024)].mean():.4f} (best of 5); CE-trained {d_c[(sc, 1024)].mean():.4f}")


def tab_exp08():
    rows = load_raw(E08)
    rng = np.random.default_rng(20261128)
    ns = sorted({r["n"] for r in rows})
    assert len(rows) == 4400
    g = lambda fit: per_seed(rows, lambda r: r["fits"][fit]["rps_excess"])
    gc = lambda fit: per_seed(rows, lambda r: r["fits"][fit]["pearson"], allow_nan=True)
    ce_f, pe_f, ce_v, pe_v, cst = g("ce_fixed"), g("pearson_fixed"), g("ce_valrps"), g("pearson_valcorr"), g("constant")
    lines = [r"\begin{tabular}{@{}p{0.27\columnwidth}p{0.17\columnwidth}p{0.17\columnwidth}p{0.17\columnwidth}@{}}", r"\toprule",
             r"Scenario & Pearson-trained better at & CE-trained better from & Tuned CE better from \\", r"\midrule"]
    names = {"A": "A: mean nearly constant, spread varies", "B": "B: informative mean, spread varies",
             "C": "C: nonlinear mean, constant spread", "D": "D: linear mean, constant spread"}
    fmt = lambda c: f"{ratings(c)} ratings" if c else "never"
    for sc in "ABCD":
        d_f = paired_summary(pe_f, ce_f, ns, sc, rng)           # Pearson-trained minus CE-trained, fixed ridge
        d_v = paired_summary(pe_v, ce_v, ns, sc, rng)           # tuned Pearson (1) minus tuned CE (3)
        cross_f = sustained_from(ns, {n: d_f[n][1] > 0 for n in ns})
        cross_v = sustained_from(ns, {n: d_v[n][1] > 0 for n in ns})
        pe_better = [n for n in ns if d_f[n][2] < 0]
        pe_txt = "never" if not pe_better else (f"{ratings(pe_better[0])} to {ratings(pe_better[-1])} ratings" if len(pe_better) > 1 else f"{ratings(pe_better[0])} ratings")
        lines.append(f"{names[sc]} & {pe_txt} & {fmt(cross_f)} & {fmt(cross_v)} \\\\")
        log_line(f"exp08 {sc}: Pearson-trained significantly better at tasks {pe_better}; CE-trained (fixed ridge) better from {cross_f} tasks; tuned CE better than tuned Pearson from {cross_v} tasks")
        d_c = paired_summary(pe_f, cst, ns, sc, rng)             # Pearson-trained (fixed ridge) minus the constant forecast
        sus = sustained_from(ns, {n: d_c[n][1] > 0 for n in ns})
        log_line(f"exp08 {sc}: Pearson-trained judge (fixed ridge) has significantly higher test RPS excess than the constant forecast (paired 95% lower bound above 0) at every tested n from {sus} tasks ({ratings(sus)} ratings); "
                 f"at 2048 tasks the paired difference is {d_c[2048][0]:+.3f} [{d_c[2048][1]:+.3f}, {d_c[2048][2]:+.3f}]")
        cor = gc("pearson_fixed")
        log_line(f"exp08 {sc} 2048 tasks: Pearson-trained held-out Pearson correlation with the true mean {np.nanmean(cor[(sc, 2048)]):.2f}")
        d_cor = paired_summary(cor, gc("ce_fixed"), ns, sc, rng)    # held-out Pearson of Pearson-trained minus that of CE-trained
        cross_cor = sustained_from(ns, {n: d_cor[n][1] > 0 for n in ns})
        log_line(f"exp08 {sc}: Pearson-trained judge (fixed ridge) has significantly higher held-out test Pearson correlation than the CE-trained judge from "
                 + (f"{cross_cor} tasks ({ratings(cross_cor)} ratings)" if cross_cor else "never"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_exp08.tex").write_text("\n".join(lines) + "\n")
    log_line("wrote tables/tab_exp08.tex")
    gcheck = json.load(open(E08 / "results" / "ce_gradient_check.json"))
    e = gcheck["exp08"]
    log_line(f"exp08 gradient check: ce_fixed rows {e['ce_fixed']['rows']} with metric difference above 1e-9: {e['ce_fixed']['rows_with_a_metric_difference_above_1e-9']}; "
             f"ce_valrps rows {e['ce_valrps']['rows']} with such a difference: {e['ce_valrps']['rows_with_a_metric_difference_above_1e-9']}, "
             f"largest test RPS excess change {e['ce_valrps']['max_abs_difference']['rps_excess']:.5f}; exact CE fits {e['exact_ce_fits']} converged {e['exact_ce_fits_scipy_converged']}")



# ----------------------------------------------------------------------------- appendix C: retrospective theorem ratings (experiments 02, 12, 14)
import re


def parse_ci(cell):
    m = re.match(r"\s*([+-]?\d+\.\d+) \[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\]", cell)
    assert m, cell
    return tuple(float(x) for x in m.groups())


def table_cells(text, *label_prefixes):
    """Cells of the first markdown table row whose first cell starts with any of label_prefixes."""
    for ln in text.splitlines():
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if any(cells[0].startswith(q) for q in label_prefixes):
                return cells
    raise KeyError(label_prefixes)


def pool_rows(text):
    """(mixture difference CI, single difference CI, a, b, a_single, b_single) for one pool section; a = Pearson-trained, b = CE-trained test RPS.
    Experiments 12 and 14 word the row labels slightly differently, so both wordings are accepted."""
    mix = parse_ci(table_cells(text, "② Pearson mixture minus ④ CE mixture", "② Pearson mixture vs ④ CE mixture")[1])
    single = parse_ci(table_cells(text, "② best single minus ④ best single", "② best single judge vs ④ best single judge")[1])
    a = float(table_cells(text, "② mixture maximizing train Pearson")[1])
    b = float(table_cells(text, "④ mixture minimizing train CE")[1])
    a1 = float(table_cells(text, "② best single by train Pearson", "② best single judge by train Pearson")[1])
    b1 = float(table_cells(text, "④ best single by train CE", "④ best single judge by train CE")[1])
    return mix, single, a, b, a1, b1


def fig_forest():
    e12 = sorted(ARCH.glob("12_*"))[0]
    e14 = sorted(ARCH.glob("14_*"))[0]
    t14 = (e14 / "results.md").read_text()
    secs = {k: t14[t14.index(h):] for k, h in (("old", "### Old pool"), ("open", "### Open pool"), ("comb", "### Combined pool"))}
    for k in secs:                                                   # cut each section at the next header
        nxt = re.search(r"\n#{2,3} ", secs[k][5:])
        secs[k] = secs[k][: nxt.start() + 5] if nxt else secs[k]
    r12 = pool_rows((e12 / "results.md").read_text())
    r_old, r_open, r_comb = (pool_rows(secs[k]) for k in ("old", "open", "comb"))
    assert abs(r12[0][0] - r_old[0][0]) < 1e-9 and abs(r12[2] - r_old[2]) < 1e-9, "experiments 12 and 14 disagree on the shared 5-judge pool"
    t02 = (exp("02") / "RESULTS_comparison.md").read_text()
    c = table_cells(t02, "rps")
    ce02, pe02 = float(c[1].split()[0]), float(c[2].split()[0])
    d, lo, hi = parse_ci(c[3])
    row02 = ((-d, -hi, -lo), pe02, ce02)                              # Pearson-trained minus CE-trained: negate and swap the CE-minus-Pearson interval
    rows = [("Mixture over judges", None),
            ("Exp. 02: 5 frontier judges (5-fold, by task family)", row02),
            ("Exp. 12: 5 frontier judges (1,000 splits)", (r12[0], r12[2], r12[3])),
            ("Exp. 14: 39 open-weight judges", (r_open[0], r_open[2], r_open[3])),
            ("Exp. 14: 44 judges, frontier and open-weight", (r_comb[0], r_comb[2], r_comb[3])),
            ("Best single judge", None),
            ("Exp. 12: 5 frontier judges", (r12[1], r12[4], r12[5])),
            ("Exp. 14: 39 open-weight judges", (r_open[1], r_open[4], r_open[5])),
            ("Exp. 14: 44 judges, frontier and open-weight", (r_comb[1], r_comb[4], r_comb[5]))]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ypos = list(range(len(rows) - 1, -1, -1))
    heads = []
    for k, (y, (lab, v)) in enumerate(zip(ypos, rows)):
        if v is None:
            heads.append(k)
            continue
        (dm, dlo, dhi), a, b = v
        col = CE_C if dlo > 0 else (PE_C if dhi < 0 else UNI_C)
        ax.errorbar(dm, y, xerr=[[dm - dlo], [dhi - dm]], fmt="o", color=col, capsize=2, lw=1.4, ms=3.5)
        ax.annotate(f"{a:.3f} / {b:.3f}", xy=(1.02, y), xycoords=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, family="monospace", annotation_clip=False)
        log_line(f"forest {lab}: Pearson-trained minus CE-trained test RPS {dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]; mean test RPS Pearson-trained {a:.4f}, CE-trained {b:.4f}")
    ax.annotate("mean test RPS:\nPearson-trained / CE-trained", xy=(1.02, len(rows) - 0.45), xycoords=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=6.5, color="0.25", annotation_clip=False)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=7)
    for k in heads:
        ax.get_yticklabels()[k].set_fontweight("bold")
        ax.get_yticklabels()[k].set_color("0.3")
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_xlim(-0.075, 0.10)
    ax.set_ylim(-0.7, len(rows) + 0.2)
    ax.set_xticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10])
    ax.set_xlabel("Held-out test ranked probability score (RPS) of the Pearson-trained judge\nminus that of the cross-entropy (CE)-trained judge\n(positive: the CE-trained judge is better; bars: 95% intervals)")
    finish(ax)
    ax.grid(True, axis="y", alpha=0.15)
    fig.tight_layout()
    save(fig, "fig_forest")


# ----------------------------------------------------------------------------- appendix D: the RPS staircase example
def fig_rps_staircase():
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    q = np.array([0, 0, 0.1, 0.3, 0.4, 0.2])
    r = 4
    Q = np.cumsum(q)[:5]
    obs = (r <= np.arange(5)).astype(float)
    gap2 = (Q - obs) ** 2
    log_line(f"rps staircase figure: r = {r}, forecast {q.tolist()}, cumulative probabilities {np.round(Q, 2).tolist()}, squared gaps {np.round(gap2, 2).tolist()}, sum {gap2.sum():.2f}, RPS {gap2.mean():.3f}")
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    j = np.arange(5)
    ax.bar(j, Q, width=0.56, color=CE_C, alpha=0.45)
    for jj in j:
        lo, hi = sorted((Q[jj], obs[jj]))
        if hi - lo > 1e-9:
            ax.bar(jj, hi - lo, bottom=lo, width=0.56, color=PE_C, alpha=0.6)
            ax.text(jj, hi + 0.03, f"{gap2[jj]:.2f}", ha="center", va="bottom", fontsize=7, color=PE_C)
    ax.step(np.r_[j - 0.5, 4.5], np.r_[obs, obs[-1]], where="post", color="k", lw=1.6)
    ax.set_xticks(j)
    ax.set_xlabel("Threshold $j$ (the question: is the rating at most $j$?)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.32)
    handles = [Patch(color=CE_C, alpha=0.45), Line2D([0], [0], color="k", lw=1.6), Patch(color=PE_C, alpha=0.6)]
    labels = [r"judge: $Q_j=\Pr[\mathrm{rating}\leq j]$", r"observed rating 4: $\mathbf{1}\{4\leq j\}$", "squared gap (value printed above)"]
    ax.legend(handles, labels, loc="upper left", fontsize=6.5, frameon=False)
    finish(ax)
    fig.tight_layout()
    save(fig, "fig_rps_staircase")


# ----------------------------------------------------------------------------- appendix C: design ranges, population limits, correlation facts, RPS examples
import subprocess
import sys


def log_design_and_facts():
    spec = importlib.util.spec_from_file_location("toy13", E13 / "scripts" / "toy.py")
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    xg = np.linspace(-1, 1, 4001)
    kap = np.exp(1.5 + 2.2 * np.cos(1.5 * np.pi * xg))
    log_line(f"exp13 design: kappa(x) ranges from {kap.min():.2f} to {kap.max():.1f}; the judge has {13 * 6} weights (13 features times 6 logits)")
    for sc in "AB":
        P = toy.truth(xg, sc)
        mean = P @ toy.GRID
        var = P @ toy.GRID ** 2 - mean ** 2
        log_line(f"exp13 design scenario {sc}: true mean rating ranges from {mean.min():.1f} to {mean.max():.1f}; true variance from {var.min():.2f} to {var.max():.2f}")
    pop_all = json.load(open(E13 / "results" / "population.json"))
    for sc in "AB":
        for k, nm in (("ce", "CE-trained"), ("rps", "RPS-trained"), ("pearson", "Pearson-trained"), ("pearson_head", "Pearson-trained + calibrated head")):
            v = pop_all[sc][k]
            log_line(f"exp13 population limit scenario {sc} {nm}: RPS excess {v['rps_excess']:.4f}, CE excess (KL) {v['ce_excess']:.4f} nats, TVD {v['tvd']:.3f}, test Pearson with the true mean {v['pearson']:.4f}, test Spearman {v['spearman']:.4f}")
    rows = load_raw(E13)
    for sc in "AB":
        floor = np.mean([r["ce"]["rps"] - r["ce"]["rps_excess"] for r in rows if r["scenario"] == sc])
        log_line(f"exp13 scenario {sc}: test RPS of the true distribution (the floor that RPS excess is measured from) {floor:.3f}")
    pop = json.load(open(E13 / "results" / "population.json"))
    for sc in "AB":
        ph, ce = pop[sc]["pearson_head"], pop[sc]["ce"]
        log_line(f"exp13 population limit (fit on a 4000-point grid of true distributions, near-zero ridge) scenario {sc}: Pearson-trained + calibrated head has test Pearson {ph['pearson']:.4f} and RPS excess {ph['rps_excess']:.4f}; CE-trained RPS excess {ce['rps_excess']:.4f}")
    # head versus raw Pearson and versus CE, and held-out correlations
    rng = np.random.default_rng(20261129)
    ns = sorted({r["n"] for r in rows})
    h = per_seed(rows, lambda r: r["pearson_head"]["rps_excess"])
    pe = per_seed(rows, lambda r: r["pearson"]["rps_excess"])
    ce = per_seed(rows, lambda r: r["ce"]["rps_excess"])
    for sc in "AB":
        d = paired_summary(pe, h, ns, sc, rng)
        cross = sustained_from(ns, {n: d[n][1] > 0 for n in ns})
        better_than_ce = [n for n in ns if h[(sc, n)].mean() < ce[(sc, n)].mean()]
        log_line(f"exp13 {sc}: the calibrated head beats the raw Pearson-trained judge (paired 95% lower bound above 0 at every larger size) from "
                 + (f"{cross} tasks ({ratings(cross)} ratings)" if cross else "never") + f"; its mean RPS excess is below the CE-trained judge's at sizes {better_than_ce if better_than_ce else 'none'}")
    rows6 = load_raw(E06)
    cor13 = {f: per_seed(rows, lambda r, f=f: r[f]["pearson"], allow_nan=True) for f in ("ce", "pearson")}
    cor6 = {f: per_seed(rows6, lambda r, f=f: r[f]["pearson"], allow_nan=True) for f in ("spearman", "kendall", "pearson_spearman")}
    sp13 = {f: per_seed(rows, lambda r, f=f: r[f]["spearman"], allow_nan=True) for f in ("ce",)}
    sp6 = {f: per_seed(rows6, lambda r, f=f: r[f]["spearman"], allow_nan=True) for f in ("spearman",)}
    allc = {**cor13, **cor6}
    for sc, n in (("A", 2048), ("B", 16)):
        vals = {f: np.nanmean(allc[f][(sc, n)]) for f in allc}
        corr_trained = [vals[f] for f in ("pearson", "spearman", "kendall", "pearson_spearman")]
        log_line(f"exp06/13 scenario {sc}, {n} tasks ({ratings(n)} ratings): held-out test Pearson correlation with the true mean: CE-trained {vals['ce']:.2f}; "
                 f"correlation-trained judges {min(corr_trained):.2f} to {max(corr_trained):.2f}")
    log_line(f"exp06/13 scenario B, 2048 tasks ({ratings(2048)} ratings): test Spearman of the Spearman-trained judge {np.nanmean(sp6['spearman'][('B', 2048)]):.3f} against {np.nanmean(sp13['ce'][('B', 2048)]):.3f} for the CE-trained judge")
    # exp 01: the same-mean mixtures q_lambda = (1-lambda) Bin(5,t) + lambda [(1-t) delta_0 + t delta_5]
    from math import comb
    worst_m = worst_v = 0.0
    for t in np.linspace(0.1, 0.9, 17):
        binom = np.array([comb(5, r) * t ** r * (1 - t) ** (5 - r) for r in range(6)])
        edge = np.zeros(6)
        edge[0], edge[5] = 1 - t, t
        for lam in (0, 0.25, 0.5, 0.75, 0.95):
            q = (1 - lam) * binom + lam * edge
            mean = q @ np.arange(6)
            var = q @ np.arange(6) ** 2 - mean ** 2
            worst_m = max(worst_m, abs(mean - 5 * t))
            worst_v = max(worst_v, abs(var - 5 * t * (1 - t) * (1 + 4 * lam)))
    log_line(f"exp01 mixtures q_lambda with lambda in 0, 0.25, 0.5, 0.75, 0.95 and t from 0.1 to 0.9: mean equals 5t (max absolute error {worst_m:.1e}); variance equals 5t(1-t)(1+4 lambda) (max absolute error {worst_v:.1e})")
    # calibrated head versus CE-trained and raw Pearson-trained judges at selected sizes
    for sc in "AB":
        parts = []
        for n in (4, 16, 64, 512, 2048):
            parts.append(f"{n} tasks: head {h[(sc, n)].mean():.4f}, CE-trained {ce[(sc, n)].mean():.4f}, Pearson-trained {pe[(sc, n)].mean():.4f}")
        log_line(f"exp13 {sc} mean test RPS excess by training size: " + "; ".join(parts))
    # exp06 and exp13: sizes from which a correlation-trained judge has higher held-out correlation than the CE-trained judge (own metric)
    rngc = np.random.default_rng(20261130)
    ns13 = sorted({r["n"] for r in rows})
    own = [("pearson", rows, "pearson", "Pearson-trained", "Pearson"), ("spearman", rows6, "spearman", "Spearman-trained", "Spearman"),
           ("kendall", rows6, "spearman", "Kendall-trained", "Spearman"), ("pearson_spearman", rows6, "pearson", "1/2 Pearson + 1/2 Spearman-trained", "Pearson")]
    for sc in "AB":
        for fit, src, metric, name, mname in own:
            a = per_seed(src, lambda r, fit=fit, metric=metric: r[fit][metric], allow_nan=True)
            b = per_seed(rows, lambda r, metric=metric: r["ce"][metric], allow_nan=True)
            d = paired_summary(a, b, ns13, sc, rngc)
            cross = sustained_from(ns13, {n: d[n][1] > 0 for n in ns13})
            log_line(f"exp06/13 {sc}: {name} judge has significantly higher held-out test {mname} correlation than the CE-trained judge (paired 95% lower bound above 0 at every larger size) from "
                     + (f"{cross} tasks ({ratings(cross)} ratings)" if cross else "never"))
    # sizes at which a correlation-trained judge has a higher mean held-out correlation (own metric) than the CE-trained judge
    for sc in "AB":
        for fit, src, metric, name, mname in own:
            a = per_seed(src, lambda r, fit=fit, metric=metric: r[fit][metric], allow_nan=True)
            b = per_seed(rows, lambda r, metric=metric: r["ce"][metric], allow_nan=True)
            higher = [n for n in ns13 if np.nanmean(a[(sc, n)]) > np.nanmean(b[(sc, n)])]
            log_line(f"exp06/13 {sc}: sizes (tasks) at which the {name} judge has a higher mean held-out test {mname} correlation than the CE-trained judge: {higher if higher else 'none'}")
    # experiment 08: sizes at which the tuned CE-trained judge is significantly worse than the fixed-ridge Pearson-trained judge
    r8 = load_raw(E08)
    ns8 = sorted({r["n"] for r in r8})
    g8 = lambda fit: per_seed(r8, lambda r: r["fits"][fit]["rps_excess"])
    ce_v8, pe_f8 = g8("ce_valrps"), g8("pearson_fixed")
    rng8 = np.random.default_rng(20261201)
    for sc in "ABCD":
        d = paired_summary(ce_v8, pe_f8, ns8, sc, rng8)
        worse = [n for n in ns8 if d[n][1] > 0]
        log_line(f"exp08 {sc}: sizes (tasks) at which the validation-tuned CE-trained judge has significantly higher test RPS excess than the fixed-ridge Pearson-trained judge (paired 95% lower bound above 0): {worse if worse else 'none'}")
    # numerical check of the frozen cross-entropy gradient (experiment 08, results/ce_gradient_check.json)
    gc13 = json.load(open(E08 / "results" / "ce_gradient_check.json"))["exp13"]
    fr = gc13["frozen_rerun_vs_saved_rows"]
    log_line(f"exp13 frozen rerun on this machine versus saved rows: CE rows differing above 1e-9: {fr['ce']['rows_with_a_metric_difference_above_1e-9']} of {fr['ce']['rows']}; RPS rows: {fr['rps']['rows_with_a_metric_difference_above_1e-9']}; "
             f"Pearson rows: {fr['pearson']['rows_with_a_metric_difference_above_1e-9']} of {fr['pearson']['rows']} (largest RPS excess change in one row {fr['pearson']['max_abs_difference']['rps_excess']:.4f}); "
             f"Pearson-plus-head rows: {fr['pearson_head']['rows_with_a_metric_difference_above_1e-9']} (largest RPS excess change {fr['pearson_head']['max_abs_difference']['rps_excess']:.2f})")
    e8 = json.load(open(E08 / "results" / "ce_gradient_check.json"))["exp08"]
    rm = e8['reported_quantities_that_change']['tuned_ce_ridge_mode']['C/256']
    log_line(f"exp08 numerical check: validation-tuned CE rows that moved {e8['ce_valrps']['rows_with_a_metric_difference_above_1e-9']} of {e8['ce_valrps']['rows']}; ridge choices that changed {len(e8['ce_valrps_rows_that_moved']['ridge_changes'])}; "
             f"share choosing c = 0.01 in scenario C at 256 tasks {rm['saved']} to {rm['exact_ce']}")
    # the RPS appendix examples, printed by the AAAI paper folder's script (pure arithmetic, no data)
    out = subprocess.run([sys.executable, "-I", str(PAPER / "rps_appendix_examples.py")], capture_output=True, text=True, check=True).stdout
    for ln in out.splitlines():
        log_line("rps_appendix_examples.py: " + ln.strip())

def main():
    fig_counterexample()
    fig_exp28(1, "fig_exp28", "one per item")
    fig_exp28(5, "fig_exp28_five_per_item", "five per item")
    tab_exp28()
    fig_exp26()
    tab_exp26()
    fig_exp29_bars()
    fig_exp29_curves()
    tab_exp29()
    fig_lora_helpsteer()
    fig_scaling()
    tab_scaling()
    summary13, ns13, data13 = fig_exp13()
    tab_exp13(summary13, ns13)
    fig_exp13_distributions()
    fig_exp06(data13)
    fig_exp07()
    tab_exp08()
    fig_forest()
    fig_rps_staircase()
    log_design_and_facts()
    shutil.copy(PAPER / "figures/fig_ratings_needed.pdf", FIG / "fig_ratings_needed.pdf")
    log_line("copied figures/fig_ratings_needed.pdf from the AAAI paper folder (printed there by experiment 08's make_paper_figure.py)")
    (HERE / "assets_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
