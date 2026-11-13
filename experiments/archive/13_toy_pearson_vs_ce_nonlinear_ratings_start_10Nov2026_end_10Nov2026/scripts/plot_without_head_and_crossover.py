"""Experiment 13, added 11-13-2026: zoomed figure without the Pearson + calibrated-head curve, and crossover points.

Reads the saved per-seed results (results/raw.jsonl); runs no new fits.
Writes fig_excess_vs_n_without_head.png next to the README and prints:
  * held-out test excess per fit at small n (mean over 100 seeds),
  * the paired per-seed difference (Pearson-trained minus CE-trained) with a 95% bootstrap interval over seeds,
  * the smallest n from which the CE-trained judge is better (interval above 0) at every larger n.
Each training task has k = 3 sampled ratings, so n training tasks = 3n training ratings.
Run from the repository root: python3 experiments/archive/13_.../scripts/plot_without_head_and_crossover.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = [json.loads(line) for line in open(os.path.join(HERE, "results", "raw.jsonl"))]
NS = sorted({r["n"] for r in rows})
FITS = ["ce", "rps", "pearson"]
NAMES = {"ce": "CE-trained (④)", "rps": "RPS-trained", "pearson": "Pearson-trained (②)"}
COL = {"ce": "tab:blue", "rps": "tab:green", "pearson": "tab:red"}
METRICS = {"rps_excess": "Held-out test RPS excess\nover the true distribution", "ce_excess": "Held-out test CE excess = KL(true ‖ judge)\nover the true distribution (nats)"}
SCEN = {"A": "Scenario A: mean nearly constant, spread varies", "B": "Scenario B: mean is a monotone nonlinear function"}
rng = np.random.default_rng(0)


def values(sc, fit, met, n):
    by_seed = {r["seed"]: r[fit][met] for r in rows if r["scenario"] == sc and r["n"] == n}
    return np.array([by_seed[s] for s in sorted(by_seed)])


def mean_ci(v, draws=10000):
    boot = v[rng.integers(0, len(v), (draws, len(v)))].mean(1)
    return v.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for i, sc in enumerate("AB"):
    for j, met in enumerate(METRICS):
        ax = axs[i, j]
        for fit in FITS:
            ms = np.array([mean_ci(values(sc, fit, met, n)) for n in NS])
            ax.plot(NS, ms[:, 0], "o-", color=COL[fit], label=NAMES[fit], ms=3)
            ax.fill_between(NS, ms[:, 1], ms[:, 2], color=COL[fit], alpha=0.15)
        diffs = [mean_ci(values(sc, "pearson", met, n) - values(sc, "ce", met, n)) for n in NS]
        better = [d[1] > 0 for d in diffs]
        start = next((NS[k] for k in range(len(NS)) if all(better[k:])), None)
        if start is not None:
            ax.axvline(start, color="black", ls="--", lw=1)
            ax.annotate(f"CE-trained better from n = {start}\n({3 * start} training ratings)", xy=(start, 0.97), xycoords=("data", "axes fraction"),
                        xytext=(6, -2), textcoords="offset points", va="top", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{SCEN[sc]}\n{'RPS' if met == 'rps_excess' else 'cross-entropy (CE)'} on 500 fresh test tasks per seed", fontsize=9.5)
        ax.set_ylabel(METRICS[met] + " (lower is better)", fontsize=8.5)
        ax.set_xlabel("training tasks n (3 ratings each, so 3n training ratings)", fontsize=8.5)
axs[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle("Experiment 13 (synthetic toy): same judge family, three training objectives. Bands: 95% bootstrap over 100 seeds.", fontsize=10)
fig.tight_layout()
out = os.path.join(HERE, "fig_excess_vs_n_without_head.png")
fig.savefig(out, dpi=130)
print("wrote", os.path.relpath(out))

# Full figure (all four fits, n -> infinity limits, power-law fit, targets) with metric-named axes.
# Same content as the frozen fig_excess_vs_n.png, which labels its y-axis only "excess over true distribution".
pop = json.load(open(os.path.join(HERE, "results", "population.json")))
summary = json.load(open(os.path.join(HERE, "results", "summary.json")))
FULL = FITS + ["pearson_head"]
NAMES_FULL = dict(NAMES, pearson_head="Pearson-trained + calibrated head")
COL_FULL = dict(COL, pearson_head="tab:orange")
TARGET = {"rps_excess": 0.01, "ce_excess": 0.05}
fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for i, sc in enumerate("AB"):
    for j, met in enumerate(METRICS):
        ax = axs[i, j]
        for fit in FULL:
            ms = np.array([mean_ci(values(sc, fit, met, n)) for n in NS])
            ax.plot(NS, ms[:, 0], "o-", color=COL_FULL[fit], label=NAMES_FULL[fit], ms=3)
            ax.fill_between(NS, ms[:, 1], ms[:, 2], color=COL_FULL[fit], alpha=0.15)
            ax.axhline(pop[sc][fit][met], color=COL_FULL[fit], ls=":", lw=1)
        d = summary[f"{sc}/ce/{met}"]
        xs = np.logspace(np.log10(16), np.log10(2048), 50)
        ax.plot(xs, d["E"] + d["A"] * xs ** (-d["alpha"]), "k--", lw=1, label="power-law fit to CE-trained")
        ax.axhline(TARGET[met], color="gray", lw=1)
        ax.text(NS[0], TARGET[met] * 1.08, f"target {TARGET[met]}", color="gray", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{SCEN[sc]}\n{'RPS' if met == 'rps_excess' else 'cross-entropy (CE)'} on 500 fresh test tasks per seed", fontsize=9.5)
        ax.set_ylabel(METRICS[met] + " (lower is better)", fontsize=8.5)
        ax.set_xlabel("training tasks n (3 ratings each, so 3n training ratings)", fontsize=8.5)
axs[0, 0].legend(fontsize=7.5, loc="upper right")
fig.suptitle("Experiment 13 (synthetic toy), all four objectives. Dotted: each fit's limit as n grows. Bands: 95% bootstrap over 100 seeds.", fontsize=10)
fig.tight_layout()
out = os.path.join(HERE, "fig_excess_vs_n_with_head_labeled.png")
fig.savefig(out, dpi=130)
print("wrote", os.path.relpath(out))

for sc in "AB":
    truth_rps = np.mean([r["ce"]["rps"] - r["ce"]["rps_excess"] for r in rows if r["scenario"] == sc])
    print(f"\nScenario {sc}: test RPS of the true distribution itself (mean over all runs) = {truth_rps:.4f}")
    for met in METRICS:
        print(f"\nScenario {sc}, {met} (mean over seeds; difference = Pearson-trained minus CE-trained, 95% bootstrap CI)")
        print("    n  ratings     CE-trained    RPS-trained   Pearson-trained   difference [95% CI]")
        diffs = []
        for n in NS:
            c, r, p = (values(sc, f, met, n).mean() for f in FITS)
            d = mean_ci(values(sc, "pearson", met, n) - values(sc, "ce", met, n))
            diffs.append(d)
            print(f"{n:5d} {3 * n:7d} {c:14.4f} {r:14.4f} {p:17.4f}   {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
        better = [d[1] > 0 for d in diffs]
        worse = [d[2] < 0 for d in diffs]
        start = next((NS[k] for k in range(len(NS)) if all(better[k:])), None)
        print("  CE-trained significantly better at every n from:", start, f"({3 * start} ratings)" if start else "")
        print("  n where Pearson-trained significantly better:", [NS[k] for k in range(len(NS)) if worse[k]])
