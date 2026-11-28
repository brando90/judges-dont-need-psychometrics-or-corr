#!/usr/bin/env python3
"""Figures for Experiment 44 from results/*.csv. ax.grid(True) on every axis (project rule). Colour-blind-safe Okabe-Ito colours plus markers.
Claims in titles are computed from the CSVs, not typed."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib44 as L  # noqa: E402

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else L.ROOT / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else L.ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
COL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]  # Okabe-Ito: five judges (fig 1, 3) and five sets (fig 4)
MRK = ["o", "s", "^", "D", "v"]
TITLE = {"chaosnli_snli": "ChaosNLI SNLI (600 items)", "chaosnli_mnli": "ChaosNLI MNLI-m (600 items)",
         "chaosnli_alphanli": "ChaosNLI alphaNLI (400 items)", "dices350": "DICES-350 (350 items)", "dices990": "DICES-990 (400 items)"}
SETNAME = {k: v.split(" (")[0] for k, v in TITLE.items()}
METRIC = {"chaosnli_snli": "Brier", "chaosnli_mnli": "Brier", "chaosnli_alphanli": "Brier", "dices350": "normalized RPS", "dices990": "normalized RPS"}
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 10})


def rd(name):
    with open(RES / name, newline="") as f:
        return list(csv.DictReader(f))


def style(ax, logx=False, logy=False):
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.2)
    ax.minorticks_on()
    if logx:
        ax.xaxis.set_minor_formatter(NullFormatter())
    if logy:
        ax.yaxis.set_minor_formatter(NullFormatter())


def fig1():
    J = rd("arm_p_judges.csv")
    n_sig = sum(float(r["p_holm"]) < 0.05 for r in J)
    names = {"gender": "gender:\nMan − Woman", "age": "age:\ngen z − (millenial, gen x+)", "race": "race:\nWhite − Black/African American"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0), layout="constrained")
    for ax, ds in zip(axes, L.DICES):
        for ci, c in enumerate(L.CONTRASTS):
            for j, jn in enumerate(L.JUDGES):
                r = next(x for x in J if x["dataset"] == ds and x["contrast"] == c and x["judge"] == jn)
                y = ci + (j - 2) * 0.13
                d, lo, hi = float(r["delta_pool"]), float(r["ci_lo"]), float(r["ci_hi"])
                ax.plot([lo, hi], [y, y], color=COL[j], lw=1.8, alpha=0.9)
                ax.plot(d, y, MRK[j], color=COL[j], ms=7, label=jn if ci == 0 else None)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(3)); ax.set_yticklabels([names[c] for c in L.CONTRASTS]); ax.invert_yaxis()
        ax.set_xlabel("Risk against pool A minus risk against pool B (normalized RPS)\npositive = judge looks worse against pool A; eligible items only")
        ax.set_title(TITLE[ds])
        style(ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=5, title="frozen judge (nothing is fitted)", frameon=False)
    fig.suptitle(f"Experiment 44, Arm P: changing the rater pool shifts all five judges by a similar amount; {n_sig} of {len(J)} cells are Holm-significant.\n"
                 "Bars: 95% two-way (items × raters) bootstrap interval, 4,000 draws.", fontsize=12)
    fig.savefig(OUT / "fig1_pool_contrast.png", dpi=150)
    plt.close(fig)


def fig2():
    C = rd("arm_s_curves.csv"); G = rd("arm_s_gain.csv")
    pos = sum(float(r["delta_G"]) > 0 for r in G); excl = sum(float(r["ci_lo"]) > 0 for r in G)
    taus = (0.5, 0.75, 1.0, 1.5, 2.0)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.6), layout="constrained")
    for ax, ds in zip(axes, L.DATASETS):
        meds = {}
        for sname, color in (("consensus", "#0072B2"), ("contested", "#D55E00")):
            ys = []
            for tau in taus:
                v = [float(r["risk_primary"]) for r in C if r["dataset"] == ds and r["stratum"] == sname and float(r["tau"]) == tau]
                ys.append(np.mean(v)); ax.plot([tau] * len(v), v, ".", color=color, alpha=0.4, ms=6)
            meds[sname] = np.median([float(r["median_agreement"]) for r in C if r["dataset"] == ds and r["stratum"] == sname])
            ax.plot(taus, ys, "-o", color=color, lw=2.4, label=sname)
        ax.set_xscale("log"); ax.set_xticks(taus); ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2"])
        ax.set_xlabel("Temperature τ on the judge's forecast\n(τ < 1 sharpens, τ > 1 softens)")
        ax.set_ylabel(f"{METRIC[ds]} risk, half-B ratings\n(lower is better)")
        ax.set_title(f"{TITLE[ds]}\nagreement: consensus {meds['consensus']:.2f} / contested {meds['contested']:.2f}", fontsize=10)
        style(ax, logx=True)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, ["consensus items (top third by half-A agreement)", "contested items (bottom third)"], loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(f"Experiment 44, Arm S: sharpening costs far more on contested items than on consensus items (gain difference positive in {pos} of {len(G)} cells, interval excludes 0 in {excl}).\n"
                 "Dots: the five frozen judges; line: their mean. Top-1 label is unchanged at every τ. Agreement = median half-A top-category share; scores use the other half of the ratings.", fontsize=12)
    fig.savefig(OUT / "fig2_sharpening_by_stratum.png", dpi=150)
    plt.close(fig)


def fig3():
    J = rd("arm_j_jsb.csv"); P = rd("arm_j_permutation.csv")
    mx = max(float(r["max_abs_jsb_change"]) for r in P)
    fig = plt.figure(figsize=(22, 10.5), layout="constrained")
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05])
    for k, ds in enumerate(L.DATASETS):
        ax = fig.add_subplot(gs[0, k])
        rows = [r for r in J if r["dataset"] == ds]
        for j, r in enumerate(rows):
            x, y = float(r["jsb"]), float(r["primary_risk"])
            ax.errorbar(x, y, xerr=[[x - float(r["jsb_ci_lo"])], [float(r["jsb_ci_hi"]) - x]], fmt=MRK[j], color=COL[j], ms=8, capsize=3,
                        label=r["judge"])
        ax.set_xscale("log")
        ax.set_xlabel("Binned JSD (lower is better)")
        ax.set_ylabel(f"{METRIC[ds]} risk, all items\n(lower is better)")
        ax.set_title(f"{TITLE[ds]}\nKendall τ_b between the two rankings: {float(rows[0]['tau_b_jsb_vs_primary_risk']):+.1f}", fontsize=10)
        style(ax, logx=True)
        if k == 0:
            ax.legend(fontsize=8, loc="lower left")
    ax = fig.add_subplot(gs[1, :])
    w = 0.15
    for j, jn in enumerate(L.JUDGES):
        vals = [float(next(r for r in P if r["dataset"] == ds and r["judge"] == jn)["risk_permuted_minus_original_mean"]) for ds in L.DATASETS]
        ax.bar(np.arange(5) + (j - 2) * w, vals, w, color=COL[j], label=jn)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{TITLE[d]}\n[{METRIC[d]}]" for d in L.DATASETS])
    ax.set_ylabel("Risk after shuffling forecasts within human bins,\nminus original risk (lower is better; > 0 = worse)")
    ax.set_title(f"Bottom: the same shuffle changes binned JSD by at most {mx:.0e} in every cell, but worsens the proper-score risk in every cell (mean of 200 shuffles per bar).", fontsize=10)
    ax.legend(fontsize=9, ncol=5, loc="upper right")
    style(ax)
    fig.suptitle("Experiment 44, Arm J: binned JSD (Elangovan et al., ICLR 2025, log base 2, 95% item-bootstrap intervals) cannot see which item a forecast belongs to inside a human bin; a proper score can.\n"
                 "Frozen judges, all items, nothing fitted. Top: JSb ranking versus proper-score ranking per set.", fontsize=12)
    fig.savefig(OUT / "fig3_binned_jsd.png", dpi=150)
    plt.close(fig)


def fig4():
    M = rd("arm_m_pairs.csv")
    Ns = np.array([25, 50, 100, 200, 400, 800]); grid = np.geomspace(25, 3200, 120)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), layout="constrained")
    ax = axes[0]
    for c, ds in zip(COL, L.DATASETS):
        rows = [r for r in M if r["dataset"] == ds]
        for k, ls in ((1, "-"), (10, "--")):
            med = [np.median([float(r[f"mdd_N{n}_k{k}"]) for r in rows]) for n in Ns]
            ax.plot(Ns, med, ls, color=c, marker="o", ms=4, label=SETNAME[ds] if k == 1 else None)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Number of held-out items N (each scored with k ratings)")
    ax.set_ylabel("Minimum detectable paired difference in primary-score risk\n(two-sided 5%, 80% power; median of 10 judge pairs; lower is better)")
    ax.set_title("(a) Solid: k = 1 rating per item. Dashed: k = 10 ratings per item.\nThe detectable difference falls as 1/√N; extra ratings per item help much less than extra items.", fontsize=10)
    ax.legend(loc="lower left"); style(ax, logx=True, logy=True)
    ax = axes[1]
    for c, ds in zip(COL, L.DATASETS):
        n80 = np.array([float(r["n80_k1"]) for r in M if r["dataset"] == ds])
        ax.step(grid, [(n80 <= g).sum() for g in grid], where="post", color=c, lw=2.2, label=SETNAME[ds])
    ax.axvline(400, color="k", lw=1, ls=":"); ax.text(410, 0.15, "N = 400", fontsize=9)
    ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 200, 400, 800, 1600, 3200]); ax.set_xticklabels(["25", "50", "100", "200", "400", "800", "1600", "3200"])
    ax.set_xlabel("Number of held-out items N at one rating per item (k_test = 1)")
    ax.set_ylabel("Judge pairs (of 10) whose difference on all items\nis detectable at that N (N_80 ≤ N)")
    ax.set_title("(b) Number of the 10 frozen-judge pairs resolvable at one rating per item", fontsize=10)
    ax.legend(loc="upper left"); style(ax, logx=True)
    fig.suptitle("Experiment 44, Arm M: power for paired comparisons of frozen judges (variance components estimated from all items of each set; nothing is fitted).", fontsize=12)
    fig.savefig(OUT / "fig4_power.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        f()
    print("figures written to", OUT)
