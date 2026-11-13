"""Merge experiment 06 fits with experiment 13's saved fits (same tasks), draw the figure and write the tables.

Reads results/raw.jsonl (this experiment) and experiment 13's results/raw.jsonl; uses only (scenario, n) cells
where all 100 seeds are present. Writes fig_rank_objectives_vs_n.png and results/tables.md, and prints them.
Differences are paired by seed: (objective's held-out test excess) minus (CE-trained judge's), 95% bootstrap over seeds.
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP13 = os.path.join(HERE, "..", "archive", "13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026")
new = [json.loads(l) for l in open(os.path.join(HERE, "results", "raw.jsonl"))]
old = {(r["scenario"], r["n"], r["seed"]): r for r in (json.loads(l) for l in open(os.path.join(EXP13, "results", "raw.jsonl")))}

cells = defaultdict(dict)
check = []
for r in new:
    key = (r["scenario"], r["n"], r["seed"])
    merged = {f: old[key][f] for f in ("ce", "rps", "pearson")}
    merged.update({f: r[f] for f in ("spearman", "kendall", "pearson_spearman")})
    cells[(r["scenario"], r["n"])][r["seed"]] = merged
    if "ce_check" in r:
        check.append(abs(r["ce_check"]["rps_excess"] - old[key]["ce"]["rps_excess"]))
NS = sorted({n for (sc, n), v in cells.items() if len(v) == 100 and len(cells.get(("A" if sc == "B" else "B", n), {})) == 100})
FITS = ["ce", "rps", "pearson", "spearman", "kendall", "pearson_spearman"]
NAMES = {"ce": "CE-trained (④)", "rps": "RPS-trained", "pearson": "Pearson-trained (②)", "spearman": "Spearman-trained",
         "kendall": "Kendall-trained", "pearson_spearman": "½ Pearson + ½ Spearman-trained"}
COL = {"ce": "tab:blue", "rps": "tab:green", "pearson": "tab:red", "spearman": "tab:purple", "kendall": "tab:brown", "pearson_spearman": "tab:pink"}
LS = {"spearman": "--", "kendall": ":", "pearson_spearman": "-."}
METRICS = {"rps_excess": "Held-out test RPS excess\nover the true distribution", "ce_excess": "Held-out test CE excess = KL(true ‖ judge)\nover the true distribution (nats)"}
SCEN = {"A": "Scenario A: mean nearly constant, spread varies", "B": "Scenario B: mean is a monotone nonlinear function"}
rng = np.random.default_rng(0)


def vals(sc, n, fit, met):
    c = cells[(sc, n)]
    return np.array([c[s][fit][met] for s in sorted(c)])


def mean_ci(v, draws=10000):
    boot = v[rng.integers(0, len(v), (draws, len(v)))].mean(1)
    return v.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


fig, axs = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
for i, sc in enumerate("AB"):
    for j, met in enumerate(METRICS):
        ax = axs[i, j]
        for fit in FITS:
            ms = np.array([mean_ci(vals(sc, n, fit, met)) for n in NS])
            ax.plot(NS, ms[:, 0], "o" + LS.get(fit, "-"), color=COL[fit], label=NAMES[fit], ms=3)
            ax.fill_between(NS, ms[:, 1], ms[:, 2], color=COL[fit], alpha=0.12)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.4); ax.grid(True, which="minor", ls=":", alpha=0.2)
        # Dashed vertical line where the CE-trained judge becomes significantly better (paired 95% CI above 0)
        # than each correlation-trained judge at every larger n; judges crossing at the same n share a line.
        groups = {}
        for comp, short in (("pearson", "Pearson"), ("spearman", "Spearman"), ("kendall", "Kendall"), ("pearson_spearman", "½P+½S")):
            better = [mean_ci(vals(sc, n, comp, met) - vals(sc, n, "ce", met))[1] > 0 for n in NS]
            start = next((NS[k] for k in range(len(NS)) if all(better[k:])), None)
            if start is not None:
                groups.setdefault(start, []).append(short)
        for k, (start, names) in enumerate(sorted(groups.items())):
            ax.axvline(start, color="black", ls="--", lw=1)
            ax.annotate(f"CE beats {', '.join(names)}\nfrom n = {start} ({3 * start} ratings)", xy=(start, 0.03 + 0.12 * k),
                        xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=7, va="bottom")
        ax.set_title(f"{SCEN[sc]}\n{'RPS' if met == 'rps_excess' else 'cross-entropy (CE)'} on 500 fresh test tasks per seed", fontsize=9.5)
        ax.set_ylabel(METRICS[met] + " (lower is better)", fontsize=8.5)
        ax.set_xlabel("training tasks n (3 ratings each, so 3n training ratings)", fontsize=8.5)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Experiment 06 (synthetic toy, same tasks as experiment 13): six training objectives, one judge family. Bands: 95% bootstrap over 100 seeds.", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(os.path.join(HERE, "fig_rank_objectives_vs_n.png"), dpi=130)

# Second figure: the same judges evaluated with test correlations instead of proper losses.
CORR = {"pearson": "Held-out test Pearson correlation\n(judge's expected rating vs true mean rating)", "spearman": "Held-out test Spearman correlation\n(judge's expected rating vs true mean rating)"}
fig, axs = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
for i, sc in enumerate("AB"):
    for j, met in enumerate(CORR):
        ax = axs[i, j]
        for fit in FITS:
            ms = np.array([mean_ci(np.nan_to_num(vals(sc, n, fit, met), nan=0.0)) for n in NS])
            ax.plot(NS, ms[:, 0], "o" + LS.get(fit, "-"), color=COL[fit], label=NAMES[fit], ms=3)
            ax.fill_between(NS, ms[:, 1], ms[:, 2], color=COL[fit], alpha=0.12)
        ax.set_xscale("log")
        ax.grid(True, which="major", alpha=0.4); ax.grid(True, which="minor", ls=":", alpha=0.2)
        ax.set_title(f"{SCEN[sc]}\n{'Pearson' if met == 'pearson' else 'Spearman'} correlation on 500 fresh test tasks per seed", fontsize=9.5)
        ax.set_ylabel(CORR[met] + " (higher is better)", fontsize=8.5)
        ax.set_xlabel("training tasks n (3 ratings each, so 3n training ratings)", fontsize=8.5)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Experiment 06: the same six judges as the RPS/CE figure, scored instead by test correlation with the true mean rating. Bands: 95% bootstrap over 100 seeds.", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(os.path.join(HERE, "fig_rank_objectives_correlations_vs_n.png"), dpi=130)
undefined = {f: sum(np.isnan(vals(sc, n, f, "pearson")).sum() for sc in "AB" for n in NS) for f in FITS}

lines = [f"Cells with all 100 seeds: n = {NS}. Data reproduction check (seed 0 per cell, CE refit vs experiment 13 saved): max |difference| in test RPS excess = {max(check):.1e} over {len(check)} cells.\n"]
for sc in "AB":
    for met in METRICS:
        lines.append(f"\n### Scenario {sc}, held-out test {'RPS' if met == 'rps_excess' else 'CE'} excess (mean over 100 seeds)\n")
        lines.append("| n (ratings) | " + " | ".join(NAMES[f] for f in FITS) + " |")
        lines.append("|---" * (len(FITS) + 1) + "|")
        for n in NS:
            lines.append(f"| {n} ({3 * n}) | " + " | ".join(f"{vals(sc, n, f, met).mean():.4f}" for f in FITS) + " |")
        lines.append("\nPaired difference, objective minus CE-trained [95% CI] (positive = CE-trained better):\n")
        lines.append("| n | " + " | ".join(NAMES[f] for f in FITS if f != "ce") + " |")
        lines.append("|---" * len(FITS) + "|")
        first = {}
        for k, n in enumerate(NS):
            row = []
            for f in FITS:
                if f == "ce":
                    continue
                d = mean_ci(vals(sc, n, f, met) - vals(sc, n, "ce", met))
                row.append(f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
                first.setdefault(f, []).append(d[1] > 0)
            lines.append(f"| {n} | " + " | ".join(row) + " |")
        lines.append("\nSmallest n from which the CE-trained judge is significantly better at every larger n:")
        for f, better in first.items():
            start = next((NS[k] for k in range(len(NS)) if all(better[k:])), None)
            lines.append(f"- {NAMES[f]}: {start} tasks ({3 * start} ratings)" if start else f"- {NAMES[f]}: not at every larger n")
    lines.append(f"\n### Scenario {sc}, test rank agreement with the true mean rating (mean over 100 seeds)\n")
    lines.append("| n | " + " | ".join(f"{NAMES[f]} Spearman" for f in ("ce", "pearson", "spearman", "kendall", "pearson_spearman")) + " |")
    lines.append("|---" * 6 + "|")
    for n in NS:
        lines.append(f"| {n} | " + " | ".join(f"{np.nanmean(vals(sc, n, f, 'spearman')):.3f}" for f in ("ce", "pearson", "spearman", "kendall", "pearson_spearman")) + " |")
    lines.append(f"\n### Scenario {sc}, test Pearson correlation with the true mean rating (mean over 100 seeds; undefined counted as 0)\n")
    lines.append("| n | " + " | ".join(NAMES[f] for f in FITS) + " |")
    lines.append("|---" * (len(FITS) + 1) + "|")
    for n in NS:
        lines.append(f"| {n} | " + " | ".join(f"{np.nan_to_num(vals(sc, n, f, 'pearson'), nan=0.0).mean():.3f}" for f in FITS) + " |")
lines.append(f"\nUndefined test correlations (constant judge output), counted as 0 in the correlation figure and table: {undefined}")
open(os.path.join(HERE, "results", "tables.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
