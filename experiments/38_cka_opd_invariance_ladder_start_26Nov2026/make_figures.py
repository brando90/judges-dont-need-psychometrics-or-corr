"""Figures for the held-out invariance ladder. Reads results.json; does not refit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SHORT = {
    "summeval_expert_coherence": "SummEval coherence",
    "summeval_expert_consistency": "SummEval consistency",
    "summeval_expert_fluency": "SummEval fluency",
    "summeval_expert_relevance": "SummEval relevance",
    "chaosNLI_snli": "ChaosNLI SNLI",
    "chaosNLI_mnli_m": "ChaosNLI MNLI-m",
    "chaosnli_alpha": "ChaosNLI alphaNLI",
    "dices990": "DICES-990",
    "veribench_fsc_exp09": "VeriBench FSC",
}


def _grid(ax):
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)


def main():
    payload = json.loads((ROOT / "results.json").read_text())
    names = list(payload["datasets"])
    labels = [SHORT.get(n, n) for n in names]
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Absolute change in held-out linear CKA. Sign, reversal, and orthogonal
    # sit on machine precision; shear does not.
    keys = [
        ("sign_flip", "sign flip"),
        ("reversal_permutation", "reversal"),
        ("orthogonal", "orthogonal"),
        ("shear", "shear"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(names))
    width = 0.18
    for i, (key, label) in enumerate(keys):
        vals = []
        for name in names:
            base = payload["datasets"][name]["transforms"]["identity"]["linear_cka"]
            vals.append(abs(payload["datasets"][name]["transforms"][key]["linear_cka"] - base))
        # Exact zeros cannot sit on a log axis. 1e-18 is a display floor, below float noise.
        ax.bar(x + (i - 1.5) * width, np.maximum(vals, 1e-18), width, label=label)
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Absolute change in held-out test linear CKA (1e-18 display floor)")
    ax.set_xlabel("Held-out test prompts")
    ax.legend(frameon=False, ncol=4)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "cka_abs_change_heldout.png", dpi=160)
    plt.close(fig)

    # Cross-entropy of the transformed softmax minus the identity softmax.
    # Positive means the transformed judge is worse. Point-mass CE is infinite
    # and is not on this axis.
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    keys = [
        ("sign_flip", "sign flip"),
        ("reversal_permutation", "reversal"),
        ("orthogonal", "orthogonal"),
        ("shear", "shear"),
    ]
    for i, (key, label) in enumerate(keys):
        vals = []
        for name in names:
            base = payload["datasets"][name]["transforms"]["identity"]["proper"]["ce"]["mean"]
            other = payload["datasets"][name]["transforms"][key]["proper"]["ce"]["mean"]
            vals.append(other - base)
        ax.bar(x + (i - 1.5) * width, vals, width, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Held-out test CE (nats), transformed minus identity")
    ax.set_xlabel("Held-out test prompts")
    ax.legend(frameon=False, ncol=4)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "ce_gap_heldout.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    delta = []
    for name in names:
        c = payload["datasets"][name]["collapse"]
        delta.append(c["linear_cka_collapsed"] - c["linear_cka_uncollapsed"])
    ax.bar(x, delta, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Held-out linear CKA, collapsed residual minus orthogonal residual")
    ax.set_xlabel("Held-out test ratings, one row per rating")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "collapse_cka_increase_heldout.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
