"""Figures for the held-out real-data audit. Reads results.json. Does not refit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def _style(ax) -> None:
    ax.grid(True, which="major", color="0.75", linewidth=0.8)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="0.9", linewidth=0.4)


def _num(v):
    if isinstance(v, str) or v is None:
        return np.nan
    return float(v)


def main() -> None:
    report = json.loads((ROOT / "results.json").read_text())
    names = list(report["datasets"])
    short = [n.replace("summeval_", "SE ").replace("chaosnli_", "Chaos ").replace("veribench_fsc_exp09", "VB FSC").replace("_", " ") for n in names]

    fig, ax = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    x = np.arange(len(names))
    w = 0.36
    pear = [_num(report["datasets"][n]["moment_test"]["pearson_one_pair"]) for n in names]
    spear = [_num(report["datasets"][n]["moment_test"]["spearman_one_pair"]) for n in names]
    ax.bar(x - w / 2, pear, width=w, label="Pearson r", color="#1f4e79")
    ax.bar(x + w / 2, spear, width=w, label="Spearman rho", color="#c47b2b")
    ax.set_xticks(x, short, rotation=20, ha="right")
    ax.set_ylabel("Held-out test association of one seeded rating pair per item")
    ax.set_xlabel("Dataset, held-out test items (prompt-level split, seed 37)")
    ax.set_title("Pearson and Spearman on the same held-out human rating pairs. Higher means more agreement.")
    ax.legend()
    _style(ax)
    fig.savefig(ROOT / "figures" / "real_pearson_vs_spearman.png", dpi=160)
    plt.close(fig)

    # CCA flat, proper scores up, for reversal versus original. Use CE and Brier ratios or raw values.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
    ce0, ce1, br0, br1 = [], [], [], []
    for n in names:
        panel = report["datasets"][n]["judge"]["panel"]
        rev = report["datasets"][n]["judge"]["attacks"]["reversal"]["scores"]
        ce0.append(_num(panel["ce"]["estimate"]))
        ce1.append(_num(rev["ce"]["estimate"]))
        br0.append(_num(panel["brier"]["estimate"]))
        br1.append(_num(rev["brier"]["estimate"]))
    axes[0].bar(x - w / 2, ce0, width=w, label="Original CE judge", color="#2e7d32")
    axes[0].bar(x + w / 2, ce1, width=w, label="Class-reversed probabilities", color="#c62828")
    axes[0].set_ylabel("Held-out test cross-entropy (nats, lower is better)")
    axes[1].bar(x - w / 2, br0, width=w, label="Original CE judge", color="#2e7d32")
    axes[1].bar(x + w / 2, br1, width=w, label="Class-reversed probabilities", color="#c62828")
    axes[1].set_ylabel("Held-out test Brier score (lower is better)")
    for ax in axes:
        ax.set_xticks(x, short, rotation=20, ha="right")
        ax.set_xlabel("Held-out test items")
        ax.legend(fontsize=8)
        _style(ax)
    fig.suptitle("Reversing the class axis leaves every canonical correlation unchanged and raises the proper scores.\nCross-entropy softmax, 8 train-only principal components of a 64-d token hash, L2 0.01, L-BFGS from zero, ≤200 iterations.")
    fig.savefig(ROOT / "figures" / "real_reversal_proper_scores.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    diffs = []
    for n in names:
        # prefer held-out half split, else held-out leave-one-out
        half = report["datasets"][n]["collapse_half_test"]
        loro = report["datasets"][n]["collapse_loro_test"]
        chosen = half if half.get("available") else loro
        diffs.append(_num(chosen.get("pearson_collapsed_minus_spread")) if chosen.get("available") else np.nan)
    ax.bar(x, diffs, color="#1f4e79")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, short, rotation=20, ha="right")
    ax.set_ylabel("Held-out Pearson of collapsed mean minus Pearson of raw ratings")
    ax.set_xlabel("Within-prompt split on held-out test items (half-split if at least 4 ratings, else leave-one-rater-out)")
    ax.set_title("Positive means replacing each rating by its within-prompt mean raises Pearson correlation.")
    _style(ax)
    fig.savefig(ROOT / "figures" / "real_collapse_pearson.png", dpi=160)
    plt.close(fig)
    print("wrote real figures")


if __name__ == "__main__":
    main()
