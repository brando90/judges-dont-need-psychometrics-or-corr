#!/usr/bin/env python3
"""Figures for the corrected (v2) Study A. Reads results/cells_v2.csv only.

Every axis has ax.grid(True). Every axis label names its metric and split.
Colors are the Okabe-Ito colorblind-safe set; identity is also carried by marker shape and a legend.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
FIG = V2 / "figures"
RES = V2 / "results"

DS_LABEL = {
    "chaosnli_snli": "ChaosNLI SNLI",
    "chaosnli_mnli": "ChaosNLI MNLI-m",
    "chaosnli_alphanli": "ChaosNLI alphaNLI",
    "dices350": "DICES-350",
    "dices990": "DICES-990",
}
DS_ORDER = list(DS_LABEL)
BUDGETS = [64, 128, 256, 512]

STYLE = {
    "rps_fix_raters": ("fix_raters (max raters per item)", "#0072B2", "o"),
    "rps_fix_items": ("fix_items (max items)", "#E69F00", "s"),
    "rps_ds_rel80": ("ds_rel80 (pilot D-study, reliability 0.80)", "#CC79A7", "^"),
    "rps_ps": ("ps (pilot proper-score selection)", "#009E73", "D"),
    "rps_or_rps": ("neutral RPS-oracle (diagnostic only)", "#000000", "x"),
}


def _save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{stem}.svg")


def figure_rps_by_selector(df):
    """Fig 1: mean held-out test RPS above the floor, per selector, per dataset."""
    fig, axes = plt.subplots(1, 5, figsize=(17.0, 4.3), sharex=True)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.30, wspace=0.34)
    for ax, ds in zip(axes, DS_ORDER):
        d = df[df.dataset == ds].set_index("B").loc[BUDGETS]
        for col, (label, color, marker) in STYLE.items():
            y = (d[col] - d["floor_rps"]).to_numpy()
            ax.plot(BUDGETS, y, marker=marker, color=color, lw=1.6, ms=5.5, label=label,
                    ls="--" if col == "rps_or_rps" else "-", mfc="none" if col == "rps_or_rps" else color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_title(DS_LABEL[ds], fontsize=10)
        ax.set_xlabel("Deployment rating budget $B=n_{items}\\times n_{raters}$\n(ratings, excluding the 1,280-rating pilot)", fontsize=8)
        ax.set_ylabel("Held-out test RPS minus floor\n(lower is better)", fontsize=8)
        ax.grid(True, which="major", alpha=0.6)
        ax.grid(True, which="minor", alpha=0.25)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Study A v2: mean held-out test ranked probability score (RPS) above its floor, by design selector\n"
        "Normalized individual-rating RPS on the frozen test items; mean over 150 replications; floor = forecast-independent part of RPS. "
        "Selectors that use the pilot are charged its cost.",
        fontsize=9, y=0.995)
    _save(fig, "fig1_v2_test_rps_by_selector")


FOREST = [
    ("F1", "F1: fix_items minus fix_raters", "Gain of fix_items over fix_raters"),
    ("F2", "F2: ps vs fix_raters", "Gain of ps over fix_raters"),
    ("F3", "F3: ps vs fix_items", "Gain of ps over fix_items"),
    ("F5", "F5: ds_rel80 vs fix_items", "Gain of ds_rel80 over fix_items"),
]


def figure_forest(df):
    """Fig 2: paired gains with two-way-bootstrap 95% intervals; filled marker = Holm-significant."""
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 8.2), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.88, bottom=0.12, wspace=0.08)
    rows = []
    for ds in DS_ORDER:
        for B in BUDGETS:
            rows.append((ds, B))
    ypos = np.arange(len(rows))[::-1]
    ylabels = [f"{DS_LABEL[ds]}, B={B}" for ds, B in rows]
    for ax, (fam, title, _) in zip(axes, FOREST):
        for (ds, B), y in zip(rows, ypos):
            r = df[(df.dataset == ds) & (df.B == B)].iloc[0]
            g, lo, hi, ph = r[f"{fam}_gain"], r[f"{fam}_ci_lo"], r[f"{fam}_ci_hi"], r[f"{fam}_p_holm"]
            sig = ph < 0.05
            color = "#009E73" if (sig and g > 0) else ("#D55E00" if (sig and g < 0) else "#555555")
            ax.plot([lo, hi], [y, y], color=color, lw=1.4)
            ax.plot([g], [y], "o", color=color, mfc=color if sig else "white", ms=5.5)
        ax.axvline(0.0, color="black", lw=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Gain in held-out test RPS\n(RPS of second minus RPS of first;\npositive = first is better)", fontsize=8)
        ax.grid(True, which="major", alpha=0.6)
        ax.grid(True, which="minor", alpha=0.25)
        ax.minorticks_on()
        ax.tick_params(labelsize=8)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(ylabels, fontsize=7.5)
    axes[0].set_ylabel("Dataset and deployment budget B (ratings)", fontsize=8)
    for ax in axes[1:]:
        ax.set_yticks(ypos)
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    fig.suptitle(
        "Study A v2: paired differences in mean held-out test RPS between design selectors (20 dataset-budget cells)\n"
        "Bars: two-way bootstrap 95% interval over 150 replications x test items (10,000 draws). "
        "Filled marker: significant after Holm-Bonferroni within the family; hollow: not. Green = first selector better, red = second better, gray = not significant.",
        fontsize=9, y=0.985)
    _save(fig, "fig2_v2_paired_gains_holm")


def main():
    df = pd.read_csv(RES / "cells_v2.csv")
    df = df[df.B.isin(BUDGETS)]
    assert len(df) == 20, len(df)
    figure_rps_by_selector(df)
    figure_forest(df)


if __name__ == "__main__":
    main()
