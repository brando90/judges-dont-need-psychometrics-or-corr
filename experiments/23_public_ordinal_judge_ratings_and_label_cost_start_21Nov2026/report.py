"""Render the registered budget comparisons and reproducible results tables."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
TRACKS = {
    "gpt4_consistency": "SummEval factual consistency",
    "image_quality": "KonIQ-10k image quality",
}
METHODS = {
    "ce": ("CE fit", "#176479", "o"),
    "pearson": ("Pearson fit", "#a44829", "v"),
    "uniform": ("Uniform", "#6b7280", ""),
    "histogram": ("Training histogram", "#a06a13", "s"),
    "calibrated_pearson": ("Pearson + calibration", "#534098", "^"),
    "rps": ("RPS fit", "#466137", "d"),
}


def read(track, table, depth=1):
    with (ROOT / "results" / track / (table + ".csv")).open() as stream:
        rows = list(csv.DictReader(stream))
    return [r for r in rows if "depth" not in r or int(r["depth"]) == depth]


def curve(ax, xs, ys, method, **kwargs):
    label, color, marker = METHODS[method]
    ax.plot(xs, ys, label=label, color=color, marker=marker, markersize=3.8,
            linewidth=2, linestyle="--" if method == "uniform" else "-", **kwargs)


def budget_plot():
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.65), sharex="col")
    fig.subplots_adjust(left=.105, right=.986, bottom=.19, top=.85, hspace=.30, wspace=.30)
    for col, (track, title) in enumerate(TRACKS.items()):
        rows = read(track, "scores")
        budgets = sorted({int(r["budget"]) for r in rows})
        upper, lower = axes[:, col]
        for method in list(METHODS)[:5]:
            ys = [float(r["rps"]) for r in rows if r["method"] == method]
            curve(upper, budgets, ys, method)
        upper.set_title(title, fontsize=14, loc="left", fontweight="bold", pad=10)
        upper.set_ylabel("Held-out ranked probability\nscore (RPS), lower is better")
        upper.yaxis.set_major_locator(MaxNLocator(5))
        band = read(track, "contrasts")
        for method in ("pearson", "uniform"):
            selected = [r for r in band if r["comparator"] == method]
            curve(lower, budgets, [float(r["rps_gain"]) for r in selected], method)
            lower.fill_between(budgets, [float(r["simultaneous_lo"]) for r in selected],
                               [float(r["simultaneous_hi"]) for r in selected],
                               color=METHODS[method][1], alpha=.16, linewidth=0)
        lower.axhline(0, color="#282f33", lw=1)
        lower.set_ylabel("Held-out test RPS difference\n(comparator minus CE fit)")
        lower.set_xlabel("Human training ratings (one per item)")
        for ax in (upper, lower):
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(FixedLocator(sorted({*budgets[::2], budgets[1], budgets[-1]})))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
            ax.xaxis.set_minor_locator(LogLocator(base=2, subs=[2**.5]))
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            ax.grid(True, which="major", alpha=.25)
            ax.grid(True, which="minor", axis="x", alpha=.10)
    fig.suptitle("What does one more batch of human ratings buy?", x=.105, y=.982,
                 ha="left", fontsize=18, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=5, bbox_to_anchor=(.095,.936),
               frameon=False, columnspacing=1.3, handlelength=2)
    fig.text(.105,.104,
             "Positive differences favor the cross-entropy (CE) fit. Bands cover all budgets and four comparators simultaneously within each benchmark\n"
             "(97.5%, 4,000 held-out group bootstrap draws), conditional on the 64 matched training samples. RPS ranges from 0 to 1.",
             fontsize=8.5, va="top")
    fig.text(.105,.049,
             "Exhaustive training-label search over 28 GPT-4 forecast transformations including uniform (40 held-out articles) or 217 image-feature judges\n"
             "(2,015 held-out images). The calibrated comparator estimates a grade histogram in each training-score quartile using the same labels.",
             fontsize=8.5, va="top")
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
        fig.savefig(out / f"public_rating_budget.{ext}", dpi=165, facecolor="white", metadata=metadata)
    plt.close(fig)


def money(rate, budget):
    return f"${rate*budget:,.2f}"


def tables():
    lines = ["# Held-out budget tables | 11-21-2026",
             "", "As of 11-21-2026. Every budget counts individual training grades, one per item unless shown otherwise. Cross-entropy (CE) and Pearson fits use exactly the same candidate library and observed training labels. Positive differences in normalized ranked probability score (RPS) favor the CE fit. Test items have equal weight.", ""]
    for track, name in TRACKS.items():
        lines += [f"## {name}", ""]
        contrast = read(track, "contrasts")
        crossed = read(track, "crossed_sensitivity")
        cs = {(int(r["budget"]), r["comparator"]): r for r in contrast}
        cx = {(int(r["budget"]), r["comparator"]): r for r in crossed}
        budgets = sorted({b for b, _ in cs})
        lines += ["Paired held-out RPS for the named comparator minus the CE fit. Brackets are simultaneous 97.5% within-benchmark bands from 4,000 paired held-out article/image bootstrap samples across all budgets and four comparators. ‘Both’ counts matched training draws in which CE had a strictly lower test RPS than raw Pearson and uniform. ‘Crossed’ also resamples the 64 training draws and tests both simultaneous lower bounds against zero. [Design](PROTOCOL_11-21-2026.md)", "",
                  "| Training grades | Pearson − CE fit [band] | Uniform − CE fit [band] | Calibrated Pearson − CE fit [band] | Training histogram − CE fit [band] | Both / 64 | Crossed clears both |",
                  "|---:|---:|---:|---:|---:|---:|:---:|"]
        for b in budgets:
            cells = []
            for m in ("pearson", "uniform", "calibrated_pearson", "histogram"):
                row = cs[b,m]
                cells.append(f"{float(row['rps_gain']):+.4f} [{float(row['simultaneous_lo']):+.4f}, {float(row['simultaneous_hi']):+.4f}]")
            clears = all(float(cx[b,m]["crossed_lo"]) > 0 for m in ("uniform", "pearson"))
            lines.append(f"| {b:,} | " + " | ".join(cells) + f" | {cx[b,'uniform']['draws_beat_pearson_and_uniform']} | {'yes' if clears else 'no'} |")
        lines += ["", f"### Absolute losses at {max(budgets):,} training grades", "",
                  "Held-out RPS (0–1), CE (natural-log units per human grade) and total variation distance (TVD, 0–1) are lower when better. TVD compares the predicted probabilities to each item's empirical histogram. Test populations are 640 summaries from 40 held-out articles, three expert grades per summary, or 2,015 held-out photographs with 215,415 crowd votes.", "",
                  "| Method | Test RPS | Test CE (nats) | Test TVD |", "|---|---:|---:|---:|"]
        for r in read(track, "scores"):
            if int(r["budget"]) == max(budgets):
                name = METHODS[r["method"]][0] if r["method"] in METHODS else "Original judge"
                lines.append(f"| {name} | {float(r['rps']):.5f} | {float(r['ce']):.5f} | {float(r['tvd']):.5f} |")
        lines += ["", f"[Scores at every budget and depth](results/{track}/scores.csv) · [Uncertainty for both depths](results/{track}/contrasts.csv) · [Separate ranking and calibration metrics](results/{track}/diagnostics.csv) · [Crossed bands](results/{track}/crossed_sensitivity.csv)", ""]
    lines += ["## Training cost scenarios", "", "Assumed US dollars per single retained grade. Acquisition excludes recruitment, overhead and test-set grades. The original per-grade prices on these two tasks are unknown. Reuse required no additional paid annotations.", "",
              "| Training grades | At $0.10/grade | At $1/grade | At $5/grade |", "|---:|---:|---:|---:|"]
    for b in budgets:
        lines.append(f"| {b:,} | " + " | ".join(money(r,b) for r in (.1,1,5)) + " |")
    (ROOT / "BUDGET_TABLES_11-21-2026.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    tables()
    budget_plot()
