"""Learning-curve figures for experiment 28 (11-23-2026). Usage: python3 make_figures.py RESULTS_DIR FIG_DIR"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COL = {"ce": "#0072B2", "pearson": "#D55E00", "pcal": "#CC79A7", "hist": "#009E73", "uniform": "#7F7F7F", "raw": "#000000"}
LAB = {"ce": "CE-fit", "pearson": "Pearson-fit", "pcal": "Pearson-fit + distribution calibration", "hist": "training histogram",
       "uniform": "uniform", "raw": "raw judge forecast (no fit)"}
NAMES = {"chaosnli_snli": "ChaosNLI SNLI", "chaosnli_mnli": "ChaosNLI MNLI-m", "chaosnli_alphanli": "ChaosNLI alphaNLI (2 classes, control)",
         "dices350": "DICES-350", "dices990": "DICES-990"}


def main(res, figdir):
    res, figdir = Path(res), Path(figdir); figdir.mkdir(exist_ok=True)
    s = pd.read_csv(res / "scores_absolute.csv"); raw = pd.read_csv(res / "raw_judge_scores.csv")
    for metric in ("rps", "ce"):
        fig, axes = plt.subplots(2, 5, figsize=(21, 7.6), sharey="col")
        for ci, ds in enumerate(NAMES):
            for ri, d in enumerate((1, 5)):
                ax = axes[ri, ci]
                sub = s[(s.dataset == ds) & (s.d == d)]
                for m in ("uniform", "hist", "pearson", "pcal", "ce"):
                    g = sub[sub.method == m].groupby("B")[metric].agg(["mean", "min", "max"])
                    ax.plot(g.index, g["mean"], color=COL[m], lw=2.2 if m == "ce" else 1.6, marker="o", ms=3.5, label=LAB[m])
                    ax.fill_between(g.index, g["min"], g["max"], color=COL[m], alpha=0.13, lw=0)
                ax.axhline(raw[raw.dataset == ds][metric].mean(), color=COL["raw"], ls="--", lw=1.4, label=LAB["raw"])
                ax.set_xscale("log", base=2); ax.grid(True, alpha=0.35)
                ax.set_title(f"{NAMES[ds]}\n{'one rating per item' if d == 1 else 'five ratings per item'}", fontsize=9.5)
                ax.set_xlabel("human ratings spent on training"); 
                if ci == 0: ax.set_ylabel(f"held-out {metric.upper()} (lower is better)")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=6, fontsize=9.5, frameon=False)
        fig.suptitle(f"Held-out {metric.upper()} vs number of human ratings; line = mean over 5 judges, band = min to max judge", fontsize=12)
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(figdir / f"learning_curves_{metric}.png", dpi=150); plt.close(fig)
    # break-even gap: CE advantage over each comparator (RPS) with 95% simultaneous bounds, one panel per set, d=1
    c = pd.read_csv(res / "contrasts.csv"); c = c[(c.metric == "rps") & (c.d == 1)]
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.4), sharey=False)
    for ax, ds in zip(axes, NAMES):
        for comp, col in (("pearson", COL["pearson"]), ("uniform", COL["uniform"]), ("pcal", COL["pcal"]), ("raw", COL["raw"])):
            g = c[(c.dataset == ds) & (c.comparator == comp)].groupby("B")["mean_comparator_minus_ce"].agg(["mean", "min", "max"])
            ax.plot(g.index, g["mean"], color=col, marker="o", ms=3.5, label=f"{comp} minus CE")
            ax.fill_between(g.index, g["min"], g["max"], color=col, alpha=0.13, lw=0)
        ax.axhline(0, color="k", lw=1); ax.set_xscale("log", base=2); ax.grid(True, alpha=0.35)
        ax.set_title(NAMES[ds], fontsize=9.5); ax.set_xlabel("human ratings (one per item)")
    axes[0].set_ylabel("comparator RPS minus CE RPS\n(above zero = CE better)")
    axes[0].legend(fontsize=8.5)
    fig.suptitle("CE advantage on held-out RPS, one rating per item; band = min to max over the 5 judges", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(figdir / "ce_advantage_rps_d1.png", dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:3])
