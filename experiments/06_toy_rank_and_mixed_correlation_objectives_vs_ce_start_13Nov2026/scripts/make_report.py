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
METRICS = {"rps_excess": "Held-out test ranked probability\nscore (RPS) excess over truth", "ce_excess": "Held-out test cross-entropy (CE) excess\n= Kullback–Leibler divergence\n(true ‖ judge), nats"}
SCEN = {"A": "Scenario A: mean nearly constant, spread varies", "B": "Scenario B: mean is a monotone nonlinear function"}
rng = np.random.default_rng(0)


def vals(sc, n, fit, met):
    c = cells[(sc, n)]
    return np.array([c[s][fit][met] for s in sorted(c)])


rng_lines = np.random.default_rng(1)  # separate stream so drawing figure lines never shifts the table intervals


def mean_ci_lines(v, draws=10000):
    boot = v[rng_lines.integers(0, len(v), (draws, len(v)))].mean(1)
    return v.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


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
        ax.set_ylabel(METRICS[met] + "\n(lower is better)", fontsize=8.5)
        ax.set_xlabel("training tasks n (3 ratings each, so 3n training ratings)", fontsize=8.5)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Experiment 06 (synthetic toy, same tasks as experiment 13): six training objectives, one judge family. Bands: 95% bootstrap over 100 seeds.", fontsize=9.5)
fig.text(0.5, 0.012, 'Train: 78-weight softmax, L-BFGS-B, perturbed log-histogram start, ridge (c/n)||W||². c = 1; 400 iterations / 600 evaluations per fit.\nCE is convex and exact-gradient CE refits match (reading guide); other objectives have no global certificate. Intervals: 95% percentile bootstrap, 10,000 seed draws.', ha="center", fontsize=7)
fig.tight_layout(rect=(0, 0.07, 1, 0.91))
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
        # Data-needed lines on test correlation (paired 95% bootstrap over seeds, CE-trained minus each judge):
        # dashed black = CE-trained significantly higher at every larger n; dotted gray = that correlation-trained
        # judge significantly higher at every larger n. Judges sharing an n share a line.
        ce_from, corr_from = {}, {}
        for comp, short in (("pearson", "Pearson"), ("spearman", "Spearman"), ("kendall", "Kendall"), ("pearson_spearman", "½P+½S")):
            d = [mean_ci_lines(np.nan_to_num(vals(sc, n, "ce", met), nan=0.0) - np.nan_to_num(vals(sc, n, comp, met), nan=0.0)) for n in NS]
            ce_b, co_b = [x[1] > 0 for x in d], [x[2] < 0 for x in d]
            s1 = next((NS[k] for k in range(len(NS)) if all(ce_b[k:])), None)
            s2 = next((NS[k] for k in range(len(NS)) if all(co_b[k:])), None)
            if s1 is not None:
                ce_from.setdefault(s1, []).append(short)
            if s2 is not None:
                corr_from.setdefault(s2, []).append(short)
        k = 0
        for start, names in sorted(ce_from.items()):
            ax.axvline(start, color="black", ls="--", lw=1)
            ax.annotate(f"CE higher than {', '.join(names)}\nat every n from {start} ({3 * start} ratings)", xy=(start, 0.97 - 0.13 * k),
                        xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points", fontsize=7, va="top")
            k += 1
        for start, names in sorted(corr_from.items()):
            ax.axvline(start, color="gray", ls=":", lw=1.5)
            ax.annotate(f"{', '.join(names)}-trained higher\nat every n from {start} ({3 * start} ratings)", xy=(start, 0.03 + 0.13 * (k - len(ce_from))),
                        xycoords=("data", "axes fraction"), xytext=(-4, 0), textcoords="offset points", fontsize=7, va="bottom", ha="right")
            k += 1
        ax.set_title(f"{SCEN[sc]}\n{'Pearson' if met == 'pearson' else 'Spearman'} correlation on 500 fresh test tasks per seed", fontsize=9.5)
        ax.set_ylabel(CORR[met] + "\n(higher is better)", fontsize=8.5)
        ax.set_xlabel("training tasks n (3 ratings each, so 3n training ratings)", fontsize=8.5)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Experiment 06: the same six judges as the RPS/CE figure, scored instead by test correlation with the true mean rating. Bands: 95% bootstrap over 100 seeds.", fontsize=9.5)
fig.text(0.5, 0.012, 'Train: 78-weight softmax, L-BFGS-B, perturbed log-histogram start, ridge (c/n)||W||². c = 1; 400 iterations / 600 evaluations per fit.\nCE is convex and exact-gradient CE refits match (reading guide); other objectives have no global certificate. Intervals: 95% percentile bootstrap, 10,000 seed draws.', ha="center", fontsize=7)
fig.tight_layout(rect=(0, 0.07, 1, 0.91))
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
lines.insert(0, '# Generated results: experiment 06\n\nSynthetic data: n training tasks with three sampled ratings each; 500 held-out test tasks per seed, scored against the true distribution. 100 seeds per scenario and size. Earlier A/B tasks are reused.\n\nRanked probability score (RPS) excess is the mean squared cumulative-probability gap to truth (0 to 1; zero optimal). Cross-entropy (CE) excess is Kullback–Leibler divergence KL(true || judge), in nats (0 upward; zero optimal). Both are lower-is-better. Pearson and Spearman compare expected ratings with the true mean on test tasks (−1 to 1; higher is better).\n\nSame 78-weight softmax family; train objectives and (c/n)||W||^2 ridge. L-BFGS-B: 400 iterations / 600 evaluations; log-histogram start plus Gaussian weight noise (sd 0.01). CE is convex, and exact-gradient refits reproduce the CE fits (HOW_TO_READ_RESULTS.md); the other objectives have no global-optimum certificate.\n\nUncertainty: paired 95% percentile bootstrap over seeds, 10,000 draws; no p-values. Intervals are pointwise. Crossovers refer to every larger tested size, not an unobserved sample size or a simultaneous-coverage guarantee.\n\n')
open(os.path.join(HERE, "results", "tables.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
