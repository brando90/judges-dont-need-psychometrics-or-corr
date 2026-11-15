"""Paper figure and numbers for the learning study (added 11-14-2026; no new fits).

Writes paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/figures/fig_ratings_needed.pdf from experiment 08's
saved results, and prints the numbers the paper's learning-study section quotes that no other report prints:
  * experiment 08 crossovers (CE-trained vs Pearson-trained, fixed ridge) in training ratings, per scenario;
  * experiment 06 ratios of held-out test RPS excess at 2,048 tasks (correlation-trained / CE-trained),
    using experiment 13's saved CE and Pearson fits and experiment 06's rank-objective fits on the same tasks.
Run from the repository root:
  python3 experiments/08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026/scripts/make_paper_figure.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
E08 = os.path.join(ROOT, "experiments", "08_toy_target_complexity_and_tuned_regularization_vs_ratings_needed_start_14Nov2026")
E06 = os.path.join(ROOT, "experiments", "06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026")
E13 = os.path.join(ROOT, "experiments", "archive", "13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026")
OUT = os.path.join(ROOT, "paper_latex_and_notes", "AAAI_2027_EvalAIScientists_JudgesKL", "figures", "fig_ratings_needed.pdf")
rng = np.random.default_rng(0)

cells = {}
for line in open(os.path.join(E08, "results", "raw.jsonl")):
    r = json.loads(line)
    cells.setdefault((r["scenario"], r["n"]), {})[r["seed"]] = r["fits"]
NS = sorted({n for (_, n) in cells})


def vals(sc, n, j):
    c = cells[(sc, n)]
    return np.array([c[s][j]["rps_excess"] for s in sorted(c)])


def ci(v, draws=10000):
    b = v[rng.integers(0, len(v), (draws, len(v)))].mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def crossover(sc):
    better = [ci(vals(sc, n, "pearson_fixed") - vals(sc, n, "ce_fixed"))[1] > 0 for n in NS]
    return next((NS[k] for k in range(len(NS)) if all(better[k:])), None)


TITLE = {"A": "A: disagreement varies,\nmean nearly constant", "B": "B: mean and\ndisagreement vary",
         "C": "C: mean varies (nonlinear),\nconstant disagreement", "D": "D: mean varies (linear),\nconstant disagreement"}
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5, "legend.fontsize": 6.2})
fig, axs = plt.subplots(2, 2, figsize=(3.35, 3.3), sharex=True)
for ax, sc in zip(axs.ravel(), "ABCD"):
    x = 3 * np.array(NS)
    for j, col, ls, lab in (("ce_fixed", "tab:blue", "-", "CE-trained"), ("pearson_fixed", "tab:red", "-", "Pearson-trained"),
                            ("constant", "0.5", "--", "constant (training histogram)")):
        m = np.array([ci(vals(sc, n, j)) for n in NS])
        ax.plot(x, m[:, 0], ls, color=col, lw=1.1, marker="o", ms=1.8, label=lab)
        ax.fill_between(x, m[:, 1], m[:, 2], color=col, alpha=0.15, lw=0)
    s = crossover(sc)
    ax.axvline(3 * s, color="black", ls=":", lw=0.9)
    ax.annotate(f"CE better from\n{3 * s} ratings", xy=(3 * s, 0.04), xycoords=("data", "axes fraction"), xytext=(2, 0),
                textcoords="offset points", fontsize=5.8, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.4, lw=0.5); ax.grid(True, which="minor", ls=":", alpha=0.2, lw=0.4)
    ax.set_title(TITLE[sc], pad=2)
for ax in axs[:, 0]:
    ax.set_ylabel("held-out test RPS excess")
for ax in axs[1]:
    ax.set_xlabel("training ratings (3 per task)")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=0.8)
fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.3, h_pad=0.4, w_pad=0.4)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
print("wrote", os.path.relpath(OUT, ROOT))
for sc in "ABCD":
    s = crossover(sc)
    print(f"Exp 08, scenario {sc}: CE-trained significantly better than Pearson-trained at every larger size from {s} tasks = {3 * s} ratings")

# Experiment 06 ratios at 2,048 tasks (same tasks as experiment 13).
e13 = {(r["scenario"], r["n"], r["seed"]): r for r in (json.loads(l) for l in open(os.path.join(E13, "results", "raw.jsonl")))}
e06 = [json.loads(l) for l in open(os.path.join(E06, "results", "raw.jsonl"))]
for sc in "AB":
    ce = np.mean([e13[(sc, 2048, s)]["ce"]["rps_excess"] for s in range(100)])
    corr = {"pearson": np.mean([e13[(sc, 2048, s)]["pearson"]["rps_excess"] for s in range(100)])}
    for k in ("spearman", "kendall", "pearson_spearman"):
        corr[k] = np.mean([r[k]["rps_excess"] for r in e06 if r["scenario"] == sc and r["n"] == 2048])
    ratios = {k: v / ce for k, v in corr.items()}
    print(f"Exp 06, scenario {sc}, 2,048 tasks: CE-trained test RPS excess {ce:.5f}; correlation-trained "
          + ", ".join(f"{k} {v:.4f} ({ratios[k]:.1f}x)" for k, v in corr.items())
          + f"; ratio range {min(ratios.values()):.1f}-{max(ratios.values()):.1f}x")
