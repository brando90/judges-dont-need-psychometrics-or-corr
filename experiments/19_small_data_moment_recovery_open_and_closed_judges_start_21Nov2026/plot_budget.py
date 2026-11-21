"""Plot the matched human-rating comparisons from the two primary judge pairs."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
PANELS = (
    ("rps", "Held-out test ranked probability\nscore (RPS) difference"),
    ("m1", "Held-out test first-moment\nsquared loss difference"),
    ("m2", "Held-out test second-moment\nsquared loss difference"),
)
PAIRS = (
    ("open_pair", "Open-weight Qwen pair", "#176d79", "o", -0.15),
    ("closed_pair", "Closed-weight OpenAI pair", "#9b482b", "s", 0.15),
)


def main():
    with (ROOT / "human_judges" / "contrasts.csv").open() as stream:
        rows = {(r["input"], int(r["n"]), int(r["k"]), r["metric"]): r for r in csv.DictReader(stream)
                if r["method"] == "pearson" and r["subtract"] == "ce"}
    budgets = [(n, k) for n in (3, 7, 15) for k in (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 5.8), sharey=True)
    for ax, (metric, label) in zip(axes, PANELS):
        for pair, _, color, marker, offset in PAIRS:
            for i, (n, k) in enumerate(budgets):
                row = rows[pair, n, k, metric]
                point, low, high = (float(row[col]) for col in ("difference", "lower_95", "upper_95"))
                ax.hlines(i + offset, low, high, color=color, linewidth=1.6)
                ax.plot(point, i + offset, marker, color=color, markersize=5)
        ax.axvline(0, color="#3c4447", linewidth=1)
        ax.set_xlabel(label + "\n(Pearson fit − cross-entropy fit)", fontsize=9)
        ax.set_axisbelow(True)
        ax.grid(True, color="#d7dede", alpha=0.8, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].set_yticks(range(len(budgets)), [f"{n} × {k}" for n, k in budgets])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Training families × ratings per artifact")
    handles = [Line2D([], [], color=color, marker=marker, label=name)
               for _, name, color, marker, _ in PAIRS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.92), ncol=2, frameon=False)
    fig.text(0.08, 0.97, "Does cross-entropy help with few human ratings?",
             fontsize=16, weight="bold", va="top")
    fig.text(0.08, 0.175, "Positive differences favor cross-entropy. Each underlying loss is on [0, 1], lower is better. "
             "Moment losses use ratings divided by five.", fontsize=9)
    fig.text(0.08, 0.125, "Both methods exhaustively fitted the same 27 calibration settings. "
             "Means use 250 matched splits, each with four held-out families.", fontsize=9)
    fig.text(0.08, 0.082, "Bars are pointwise 95% intervals from 1,000 paired family Bayesian bootstrap draws with refitting. "
             "The 195 expert ratings had prior exposure.", fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.81, bottom=0.34, wspace=0.20)
    fig.savefig(ROOT / "small_data_human_ratings.png", dpi=170, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


if __name__ == "__main__":
    main()
