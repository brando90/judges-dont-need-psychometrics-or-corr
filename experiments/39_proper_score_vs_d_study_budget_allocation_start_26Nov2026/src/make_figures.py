#!/usr/bin/env python3
"""Two summary figures for Studies A and B."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"

DS_LABEL = {
    "chaosnli_snli": "ChaosNLI SNLI",
    "chaosnli_mnli": "ChaosNLI MNLI-m",
    "chaosnli_alphanli": "ChaosNLI alphaNLI",
    "dices350": "DICES-350",
    "dices990": "DICES-990",
}
DS_ORDER = list(DS_LABEL)
# colorblind-friendly
C_U1 = "#0072B2"
C_U2 = "#E69F00"
C_PS = "#009E73"
C_PSM = "#CC79A7"
C_OR = "#000000"
C_EQ = "#999999"


def ci_yerr(rec):
    m = rec["mean"]
    lo, hi = rec["ci95"]
    return m, np.array([[m - lo], [hi - m]])


def _save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / f"{stem}.png"
    svg = FIG / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)
    print("wrote", svg)


def figure_a(summary):
    fig, axes = plt.subplots(2, 5, figsize=(16.2, 7.4), sharex=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.14, wspace=0.32, hspace=0.42)
    for j, name in enumerate(DS_ORDER):
        ds = summary["datasets"][name]
        budgets = [int(B) for B in ds["budgets"]]
        # top: chosen n_raters (modal) and fraction agreeing with RPS oracle
        ax = axes[0, j]
        for tag, color, marker, label in (
            ("u1", C_U1, "o", "G/D-U1 (per-item)"),
            ("u2", C_U2, "s", "G/D-U2 (grand mean)"),
            ("ps", C_PS, "D", "Proper-score (primary)"),
            ("rps_oracle", C_OR, "x", "RPS oracle"),
        ):
            ys = []
            for B in budgets:
                mode = ds["budgets"][str(B)]["selection"][tag]["mode"]
                nr = int(mode.split("x")[1])
                ys.append(nr)
            ax.plot(budgets, ys, color=color, marker=marker, lw=1.6, ms=6, label=label if j == 0 else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets], fontsize=7)
        ax.set_ylim(0, 18)
        ax.set_yticks([2, 4, 8, 16])
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)
        ax.set_title(DS_LABEL[name], fontsize=10)
        if j == 0:
            ax.set_ylabel("Modal selected $n_{\\mathrm{raters}}$ per item", fontsize=9)
        # bottom: test RPS
        ax = axes[1, j]
        for tag, color, marker, label in (
            ("u1", C_U1, "o", "G/D-U1"),
            ("u2", C_U2, "s", "G/D-U2"),
            ("ps", C_PS, "D", "Proper-score"),
            ("rps_oracle", C_OR, "x", "RPS oracle"),
        ):
            means, lo, hi = [], [], []
            for B in budgets:
                rec = ds["budgets"][str(B)]["metrics"]["rps"][tag]
                means.append(rec["mean"])
                lo.append(rec["ci95"][0])
                hi.append(rec["ci95"][1])
            means, lo, hi = map(np.array, (means, lo, hi))
            ax.plot(budgets, means, color=color, marker=marker, lw=1.6, ms=6, label=label if j == 0 else None)
            ax.fill_between(budgets, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets], fontsize=7)
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)
        ax.set_xlabel("Held-out design budget $B = n_{\\mathrm{items}} \\times n_{\\mathrm{raters}}$", fontsize=8)
        if j == 0:
            ax.set_ylabel("Held-out test ranked probability score\n(RPS; lower is better)", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Study A: allocation of a fixed rating budget, and the resulting held-out RPS", fontsize=12, y=1.02)
    fig.text(0.5, 0.01,
             "Top: modal $n_{\\mathrm{raters}}$ chosen at each budget. G/D-U1 always prefers more raters; G/D-U2 prefers more items. "
             "Bottom: mean test RPS of the deployed design with Student-$t$ 95% intervals over 150 replications. "
             "Primary predictor: 27-transform CE-fit of frozen Experiment 28 judge probabilities.",
             ha="center", fontsize=8, color="#222")
    _save(fig, "fig1_study_a_allocation_and_rps")


def figure_b(summary):
    pilots = summary["pilots"]
    tgrid = summary["t_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.18, wspace=0.28)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(pilots)))
    # left: P(match oracle)
    ax = axes[0]
    for color, pilot in zip(cmap, pilots):
        xs, ys, lo, hi = [], [], [], []
        for t in tgrid:
            cell = summary["cells"][f"t{t:.2f}_{pilot}"]
            rec = cell["chose_oracle"]
            xs.append(t)
            ys.append(rec["proportion"])
            lo.append(rec["ci95"][0])
            hi.append(rec["ci95"][1])
        ax.plot(xs, ys, color=color, marker="o", lw=1.6, ms=5, label=f"pilot {pilot}")
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.axhline(1.0, color="#444", lw=0.8, ls="--")
    ax.axvline(4/15, color="#444", lw=0.8, ls=":")
    ax.axvline(11/15, color="#444", lw=0.8, ls=":")
    ax.set_xlabel("Mixing parameter $t$  (C1 $\\to$ C2)", fontsize=10)
    ax.set_ylabel("Fraction of replications in which REML chose the oracle design", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.35)
    ax.set_title("Selection accuracy", fontsize=11)
    # right: paired held-out error vs equal 4x4 (negative = REML better)
    ax = axes[1]
    for color, pilot in zip(cmap, pilots):
        xs, ys, lo, hi = [], [], [], []
        for t in tgrid:
            cell = summary["cells"][f"t{t:.2f}_{pilot}"]
            rec = cell["paired_obs_chosen_minus_equal"]
            xs.append(t)
            ys.append(rec["mean"])
            lo.append(rec["ci95"][0])
            hi.append(rec["ci95"][1])
        ax.plot(xs, ys, color=color, marker="o", lw=1.6, ms=5, label=f"pilot {pilot}")
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0.0, color="#444", lw=0.8, ls="--")
    ax.axvline(4/15, color="#444", lw=0.8, ls=":")
    ax.axvline(11/15, color="#444", lw=0.8, ls=":")
    ax.set_xlabel("Mixing parameter $t$  (C1 $\\to$ C2)", fontsize=10)
    ax.set_ylabel("Held-out absolute error of the REML-chosen design minus equal $4\\times4$\n(negative means REML is better)", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.35)
    ax.set_title("Error gap versus equal allocation", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Study B: D-study REML allocation as the two facets become harder to tell apart", fontsize=12, y=1.04)
    fig.text(0.5, 0.01,
             "C1 ($t=0$) prefers $8\\times2$; C2 ($t=1$) prefers $2\\times8$. Dotted lines mark the analytic crossovers "
             "$t=4/15$ ($8\\times2$ ties $4\\times4$) and $t=11/15$ ($4\\times4$ ties $2\\times8$). "
             "In between, $4\\times4$ is uniquely optimal (at $t=1/2$, $v_\\Delta=0.3875$ vs $0.475$). "
             "Bands are Student-$t$ 95% intervals (left: Wilson) over 200 replications (256 at the $64\\times8$ reproduction cells).",
             ha="center", fontsize=8, color="#222")
    _save(fig, "fig2_study_b_reml_accuracy_and_regret")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    a = json.loads((RES / "study_a_summary.json").read_text())
    b = json.loads((RES / "study_b_summary.json").read_text())
    figure_a(a)
    figure_b(b)


if __name__ == "__main__":
    main()
