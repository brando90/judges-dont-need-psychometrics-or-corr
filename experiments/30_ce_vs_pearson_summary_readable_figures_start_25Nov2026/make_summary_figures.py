"""Readable summary figures for CE vs Pearson judges (11-25-2026): Exp 28, Exp 26, Exp 29 and a break-even plot.
Reads only saved results files of the repo (no model calls). Usage: python3 make_summary_figures.py [REPO_ROOT]
Every axis has grid lines on; lower is better for RPS, CE and TVD (all held-out, proper scores or distance to the human rating distribution)."""
import csv, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
OUT = Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)
E26, E28, E29 = [Path(glob.glob(str(EXP / f"{n}_*"))[0]) for n in (26, 28, 29)]
COL = {"ce": "#0072B2", "pearson": "#D55E00", "uniform": "#7F7F7F", "raw": "#000000"}
LAB = {"ce": "CE-fit", "pearson": "Pearson-fit", "uniform": "uniform", "raw": "raw judge (no fit)"}
MET = {"rps": "RPS", "ce": "CE (nats)", "tvd": "TVD"}
SETS = {"chaosnli_snli": "ChaosNLI SNLI", "chaosnli_mnli": "ChaosNLI MNLI-m", "chaosnli_alphanli": "alphaNLI (2 classes)",
        "dices350": "DICES-350", "dices990": "DICES-990"}
JUDGES = {"gpt-6-astra": "GPT-6 Astra", "claude-opus-5-5": "Claude Opus 5.5", "kimi-k3": "Kimi K3", "glm-5.3": "GLM 5.3", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash"}
rd = lambda p: list(csv.DictReader(open(p)))


def style(ax, xlabel, ylabel=None, title=None):
    ax.set_xscale("log", base=2); ax.grid(True, alpha=0.35)
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if title: ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)


def draw28(ax, ds, metric, s, raw):
    for m in ("uniform", "pearson", "ce"):
        by = defaultdict(list)
        for r in s:
            if r["dataset"] == ds and r["d"] == "1" and r["method"] == m: by[int(r["B"])].append(float(r[metric]))
        xs = sorted(by); mean = np.array([np.mean(by[x]) for x in xs])
        ax.plot(xs, mean, color=COL[m], lw=2.4 if m == "ce" else 1.8, marker="o", ms=3.5, label=LAB[m])
        ax.fill_between(xs, [min(by[x]) for x in xs], [max(by[x]) for x in xs], color=COL[m], alpha=0.14, lw=0)
    v = np.mean([float(r[metric]) for r in raw if r["dataset"] == ds])
    ax.axhline(v, color=COL["raw"], ls="--", lw=1.3, label=LAB["raw"])
    style(ax, "human ratings used to fit (one per item)", f"held-out {MET[metric]}\n(lower is better)", SETS[ds])


def draw26(ax, j, metric, rows):
    for key, m in (("uniform", "uniform"), ("pearson_fit", "pearson"), ("ce_fit", "ce")):
        g = sorted([(int(r["training_ratings"]), float(r["mean"]), float(r["lower95"]), float(r["upper95"])) for r in rows if r["judge"] == j and r["method"] == key])
        xs = [a[0] for a in g]
        ax.plot(xs, [a[1] for a in g], color=COL[m], lw=2.4 if m == "ce" else 1.8, marker="o", ms=3.5, label=LAB[m])
        ax.fill_between(xs, [a[2] for a in g], [a[3] for a in g], color=COL[m], alpha=0.14, lw=0)
    g = [float(r["mean"]) for r in rows if r["judge"] == j and r["method"] == "identity"]
    if g: ax.axhline(np.mean(g), color=COL["raw"], ls="--", lw=1.3, label=LAB["raw"])
    style(ax, "human ratings used to fit (two per answer)", f"held-out {MET[metric]}\n(lower is better)", JUDGES[j])


def grid(keys, draw, title, fname, single_prefix=None):
    """2 x 3 grid (5 panels + legend), plus one single-panel file per key when single_prefix is given."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4)); axes = axes.ravel()
    for ax, k in zip(axes, keys): draw(ax, k)
    h, l = axes[0].get_legend_handles_labels()
    axes[5].axis("off"); axes[5].legend(h, l, loc="center", fontsize=12, frameon=False)
    fig.suptitle(title, fontsize=10.5); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / fname, dpi=130); plt.close(fig)
    if single_prefix:
        for k in keys:
            f, ax = plt.subplots(figsize=(6.4, 4.4)); draw(ax, k); ax.legend(fontsize=8)
            f.tight_layout(); f.savefig(OUT / f"{single_prefix}_{k}.png", dpi=130); plt.close(f)


def fig28(metric):
    s = rd(E28 / "results/scores_absolute.csv"); raw = rd(E28 / "results/raw_judge_scores.csv")
    grid(list(SETS), lambda ax, ds: draw28(ax, ds, metric, s, raw),
         f"Exp 28: held-out {MET[metric]} vs number of human ratings. Line = mean over 5 judges, band = min to max judge.\n"
         "Judges GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3, DeepSeek V4.1 Flash via the Stanford AI API Gateway; analysis Claude Sonnet 5.5 in Claude Code",
         f"fig_exp28_{metric}_per_set.png", f"single_exp28_{metric}" if metric == "rps" else None)


def fig26(metric):
    rows = [r for r in rd(E26 / "results/new5/scores.csv") if r["ratings_per_answer"] == "2" and r["metric"] == metric]
    grid(list(JUDGES), lambda ax, j: draw26(ax, j, metric, rows),
         f"Exp 26: held-out {MET[metric]} vs number of human ratings, 160 fresh homework answers, mean of 10 folds, band = 95% interval.\n"
         "Judges via the Stanford AI API Gateway; analysis in Python (numpy, scipy)",
         f"fig_exp26_{metric}_per_judge.png", f"single_exp26_{metric}" if metric == "rps" else None)


def fig_breakeven():
    b = [r for r in rd(E28 / "results/breakeven_summary.csv") if r["d"] == "1"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharey=True); axes = axes.ravel()
    jl = list(JUDGES); off = np.linspace(-0.3, 0.3, 3)
    for ax, ds in zip(axes[:5], SETS):
        for k, (key, name, col, mk) in enumerate((("N_uniform", "CE-fit beats uniform", COL["uniform"], "s"), ("N_pearson", "CE-fit beats Pearson-fit", COL["pearson"], "o"), ("N_raw", "CE-fit beats raw judge", COL["raw"], "^"))):
            for i, j in enumerate(jl):
                r = [x for x in b if x["dataset"] == ds and x["judge"] == j][0]
                if r[key]: ax.scatter(i + off[k], float(r[key]), color=col, marker=mk, s=38, label=name if i == 0 else None)
                else: ax.scatter(i + off[k], 512, color=col, marker="x", s=38, label=None)
        ax.set_yscale("log", base=2); ax.set_ylim(2.5, 700); ax.grid(True, alpha=0.35)
        ax.set_yticks([4, 8, 16, 32, 64, 128, 256, 512]); ax.set_yticklabels(["4", "8", "16", "32", "64", "128", "256", "not\nreached"], fontsize=8)
        ax.set_xticks(range(5)); ax.set_xticklabels([JUDGES[j].replace(" ", "\n", 1) for j in jl], fontsize=7)
        ax.set_title(SETS[ds], fontsize=10)
    h, l = axes[0].get_legend_handles_labels(); axes[5].axis("off"); axes[5].legend(h, l, loc="center", fontsize=12, frameon=False)
    for ax in (axes[0], axes[3]): ax.set_ylabel("human ratings needed (one per item)", fontsize=9)
    fig.suptitle("Exp 28 break-even: fewest human ratings from which CE-fit has lower held-out RPS than the comparator, and keeps it (simultaneous 97.5% bound above 0). "
                 "Lower = CE-fit pays off sooner", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "fig_exp28_breakeven.png", dpi=140); plt.close(fig)


def fig29():
    s = rd(E29 / "results/scores.csv")
    panels = [("chaos3_qwen", "1", "ChaosNLI, Qwen3-4B, d=1"), ("chaos3_qwen", "5", "ChaosNLI, Qwen3-4B, d=5"), ("alpha2_qwen", "1", "alphaNLI, Qwen3-4B"),
              ("dices3_qwen", "1", "DICES, Qwen3-4B"), ("hs2_qwen", "1", "HelpSteer2, Qwen3-4B"), ("chaos3_llama", "1", "ChaosNLI, Llama-3.1-8B"), ("dices3_llama", "1", "DICES, Llama-3.1-8B")]
    for metric in ("rps", "ce", "tvd"):
        fig, axes = plt.subplots(2, 4, figsize=(20, 8)); axes = axes.ravel()
        for ax, (st, d, name) in zip(axes, panels):
            for meth, m, lab in (("uniform", "uniform", "uniform"), ("pearson_raw", "pearson", "Pearson-trained LoRA"), ("ce_raw", "ce", "CE-trained LoRA")):
                g = sorted([(int(r["b_total"]), float(r[metric])) for r in s if r["study"] == st and r["depth"] == d and r["method"] == meth])
                if g: ax.plot([a[0] for a in g], [a[1] for a in g], color=COL[m], lw=2.4 if m == "ce" else 1.8, marker="o", ms=3.5, label=lab)
            style(ax, "human ratings (budget)", f"held-out {MET[metric]}\n(lower is better)", name)
        axes[0].legend(fontsize=8)
        axes[-1].axis("off")
        fig.suptitle(f"Exp 29: raw (uncalibrated) LoRA judges, held-out {MET[metric]} vs human ratings; mean over 10 draws. "
                     "Qwen3-4B-Instruct-2507 and Llama-3.1-8B-Instruct, LoRA rank 8, GPU box mercury2", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / f"fig_exp29_{metric}_lora_curves.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    for m in ("rps", "ce", "tvd"): fig28(m); fig26(m)
    fig_breakeven(); fig29()
    print(sorted(p.name for p in OUT.glob("*.png")))
