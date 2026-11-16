"""Experiment 07 report: convergence of the original fits, effect of a larger budget and best-of-5 restarts.

Reads results/raw.jsonl (no new fits). Uses only (scenario, n) cells with all 30 seeds.
Writes fig_restarts_vs_n.png and results/tables.md, and prints the tables.
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = [json.loads(l) for l in open(os.path.join(HERE, "results", "raw.jsonl"))]
cells = {}
for r in rows:
    cells.setdefault((r["scenario"], r["n"]), {})[r["seed"]] = r["fits"]
NS = sorted({n for (sc, n), v in cells.items() if len(v) == 30 and len(cells.get(("A" if sc == "B" else "B", n), {})) == 30})
KINDS = ["ce", "rps", "pearson", "pearson_head", "spearman", "kendall", "pearson_spearman"]
NAMES = {"ce": "CE-trained (④)", "rps": "RPS-trained", "pearson": "Pearson-trained (②)", "pearson_head": "Pearson + calibrated head",
         "spearman": "Spearman-trained", "kendall": "Kendall-trained", "pearson_spearman": "½ Pearson + ½ Spearman-trained"}
COL = {"ce": "tab:blue", "rps": "tab:green", "pearson": "tab:red", "pearson_head": "tab:orange", "spearman": "tab:purple",
       "kendall": "tab:brown", "pearson_spearman": "tab:pink"}
METRICS = {"rps_excess": "Held-out test ranked probability\nscore (RPS) excess over truth", "ce_excess": "Held-out test cross-entropy (CE) excess\n= Kullback–Leibler divergence\n(true ‖ judge), nats"}
SCEN = {"A": "Scenario A: mean nearly constant, spread varies", "B": "Scenario B: mean is a monotone nonlinear function"}
rng = np.random.default_rng(0)


def vals(sc, n, kind, cond, key):
    c = cells[(sc, n)]
    return np.array([c[s][kind][cond][key] for s in sorted(c)], dtype=float)


def mean_ci(v, draws=10000):
    boot = v[rng.integers(0, len(v), (draws, len(v)))].mean(1)
    return v.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def crossover(sc, met, kind, cond):
    better = [mean_ci(vals(sc, n, kind, cond, met) - vals(sc, n, "ce", cond, met))[1] > 0 for n in NS]
    return next((NS[k] for k in range(len(NS)) if all(better[k:])), None)


L = [f"Cells with all 30 seeds: n = {NS}.\n"]
L.append("## Convergence of the original fits (400 iterations, 600 evaluations): % of fits where scipy reports success / % that hit the budget\n")
L.append("| Objective | " + " | ".join(f"n={n}" for n in NS) + " |")
L.append("|---" * (len(NS) + 1) + "|")
for kind in KINDS:
    cellstr = []
    for n in NS:
        s = np.concatenate([vals(sc, n, kind, "orig", "success") for sc in "AB"])
        h = np.concatenate([vals(sc, n, kind, "orig", "budget_hit") for sc in "AB"])
        cellstr.append(f"{100 * s.mean():.0f} / {100 * h.mean():.0f}")
    L.append(f"| {NAMES[kind]} | " + " | ".join(cellstr) + " |")
L.append("\nBoth scenarios pooled (60 fits per cell).\n")

L.append("## Did restarts find a better training objective? % of fits where the best-of-5 training objective is lower than the original start's (long budget) by more than 1e-6\n")
L.append("| Objective | " + " | ".join(f"n={n}" for n in NS) + " |")
L.append("|---" * (len(NS) + 1) + "|")
for kind in KINDS:
    if kind in ("ce", "pearson_head"):
        continue
    cellstr = []
    for n in NS:
        imp = np.concatenate([[min(cells[(sc, n)][s][kind]["restart_train_obj"]) < cells[(sc, n)][s][kind]["restart_train_obj"][0] - 1e-6
                               for s in sorted(cells[(sc, n)])] for sc in "AB"])
        cellstr.append(f"{100 * imp.mean():.0f}")
    L.append(f"| {NAMES[kind]} | " + " | ".join(cellstr) + " |")

for sc in "AB":
    for met in METRICS:
        L.append(f"\n## Scenario {sc}, held-out test {'RPS' if met == 'rps_excess' else 'CE'} excess: original vs best-of-5 (mean over 30 seeds)\n")
        L.append("| n | " + " | ".join(f"{NAMES[k]} orig → best" for k in KINDS) + " |")
        L.append("|---" * (len(KINDS) + 1) + "|")
        for n in NS:
            L.append(f"| {n} | " + " | ".join(f"{vals(sc, n, k, 'orig', met).mean():.4f} → {vals(sc, n, k, 'best', met).mean():.4f}" for k in KINDS) + " |")
        L.append("\nPaired change, best-of-5 minus original [95% CI] (negative = restarts and budget helped):\n")
        L.append("| n | " + " | ".join(NAMES[k] for k in KINDS) + " |")
        L.append("|---" * (len(KINDS) + 1) + "|")
        for n in NS:
            row = []
            for k in KINDS:
                d = mean_ci(vals(sc, n, k, "best", met) - vals(sc, n, k, "orig", met))
                row.append(f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
            L.append(f"| {n} | " + " | ".join(row) + " |")
        L.append("\nSmallest n from which the CE-trained judge is significantly better at every larger n (original → best-of-5):")
        for k in KINDS:
            if k == "ce":
                continue
            a, b = crossover(sc, met, k, "orig"), crossover(sc, met, k, "best")
            fmt = lambda v: f"{v} tasks ({3 * v} ratings)" if v else "not at every larger n"
            L.append(f"- {NAMES[k]}: {fmt(a)} → {fmt(b)}")

fig, axs = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
for i, sc in enumerate("AB"):
    for j, met in enumerate(METRICS):
        ax = axs[i, j]
        for k in KINDS:
            o = np.array([vals(sc, n, k, "orig", met).mean() for n in NS])
            b = np.array([mean_ci(vals(sc, n, k, "best", met)) for n in NS])
            ax.plot(NS, o, "-", color=COL[k], lw=1, alpha=0.45)
            ax.plot(NS, b[:, 0], "o--", color=COL[k], ms=3, label=NAMES[k])
            ax.fill_between(NS, b[:, 1], b[:, 2], color=COL[k], alpha=0.12)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.4); ax.grid(True, which="minor", ls=":", alpha=0.2)
        groups = {}
        for k, short in (("pearson", "Pearson"), ("pearson_head", "Pearson+head"), ("spearman", "Spearman"), ("kendall", "Kendall"), ("pearson_spearman", "½P+½S")):
            s = crossover(sc, met, k, "best")
            if s is not None:
                groups.setdefault(s, []).append(short)
        for q, (start, names) in enumerate(sorted(groups.items())):
            ax.axvline(start, color="black", ls="--", lw=1)
            ax.annotate(f"CE beats {', '.join(names)}\nfrom n = {start} ({3 * start} ratings)", xy=(start, 0.03 + 0.12 * q),
                        xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=7, va="bottom")
        ax.set_title(f"{SCEN[sc]}\n{'RPS' if met == 'rps_excess' else 'cross-entropy (CE)'} on 500 fresh test tasks per seed", fontsize=9.5)
        ax.set_ylabel(METRICS[met] + "\n(lower is better)", fontsize=8.5)
        ax.set_xlabel("training tasks n (3 ratings each, so 3n training ratings)", fontsize=8.5)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Experiment 07: dashed = larger budget and best of 5 (CE: one start); faint solid = original. Bands: 95% bootstrap over 30 seeds.", fontsize=9.5)
fig.text(0.5, 0.012, 'Train: 78-weight softmax, L-BFGS-B, perturbed log-histogram start, ridge (c/n)||W||². Original: 400/600; long: 2,000/4,000 iterations/evaluations; starts chosen by train objective.\nCE is convex and exact-gradient CE refits match (reading guide); other objectives have no global certificate. Intervals: 95% percentile bootstrap, 10,000 seed draws.', ha="center", fontsize=7)
fig.tight_layout(rect=(0, 0.07, 1, 0.91))
fig.savefig(os.path.join(HERE, "fig_restarts_vs_n.png"), dpi=130)
L.insert(0, '# Generated results: experiment 07\n\nSynthetic data: n training tasks with three sampled ratings each; 500 held-out test tasks per seed, scored against the true distribution. 30 seeds per scenario and size. Earlier A/B tasks are reused.\n\nRanked probability score (RPS) excess is the mean squared cumulative-probability gap to truth (0 to 1; zero optimal). Cross-entropy (CE) excess is Kullback–Leibler divergence KL(true || judge), in nats (0 upward; zero optimal). Both are lower-is-better. Pearson and Spearman compare expected ratings with the true mean on test tasks (−1 to 1; higher is better).\n\nSame 78-weight softmax family; train objectives and (c/n)||W||^2 ridge. L-BFGS-B: 400 iterations / 600 evaluations; log-histogram start plus Gaussian weight noise (sd 0.01). CE is convex, and exact-gradient refits reproduce the CE fits (HOW_TO_READ_RESULTS.md); the other objectives have no global-optimum certificate. Long condition: 2,000 iterations / 4,000 evaluations. Best condition: minimum penalized training objective over five starts; CE uses one long start, and the head is refitted once on the selected Pearson score.\n\nUncertainty: paired 95% percentile bootstrap over seeds, 10,000 draws; no p-values. Intervals are pointwise. Crossovers refer to every larger tested size, not an unobserved sample size or a simultaneous-coverage guarantee.\n\n')
open(os.path.join(HERE, "results", "tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
