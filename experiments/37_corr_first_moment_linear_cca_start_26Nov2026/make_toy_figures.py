"""Toy figures for Pearson versus a linear second moment, and for CCA's invariances.

Every numeric axis is labeled with the metric. `ax.grid(True)` is on every data axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cca_tools import canonical_correlations, pearson

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"


def _style(ax) -> None:
    ax.grid(True, which="major", color="0.75", linewidth=0.8)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="0.9", linewidth=0.4)


def figure_shapes() -> None:
    x = np.linspace(-2.0, 2.0, 401)
    panels = [
        (x, "Linear Y = X", "Pearson r = 1 on this grid"),
        (x ** 2, "Quadratic Y = X^2", f"Pearson r = {pearson(x, x ** 2):.2e}"),
        (np.tanh(1.5 * x), "S-curve Y = tanh(1.5 X)", f"Pearson r = {pearson(x, np.tanh(1.5 * x)):.3f}"),
        (np.where(x >= 0, 1.0, -1.0), "Step Y = sign(X)", f"Pearson r = {pearson(x, np.where(x >= 0, 1.0, -1.0)):.3f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), constrained_layout=True)
    for ax, (y, title, note) in zip(axes.ravel(), panels):
        ax.plot(x, y, color="#1f4e79", linewidth=2.0)
        ax.set_xlabel("Synthetic score X (no split; population curve)")
        ax.set_ylabel("Synthetic score Y (no split; population curve)")
        ax.set_title(f"{title}\n{note}", fontsize=10)
        _style(ax)
    fig.suptitle("Pearson correlation reads a linear association. A deterministic curve can still have r = 0.", fontsize=11)
    fig.savefig(FIG / "toy_nonlinear_vs_linear.png", dpi=160)
    plt.close(fig)


def figure_onehot() -> None:
    p0 = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
    p1 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    pmf = np.vstack([p0, p0, p1, p1])
    perm = np.array([1, 2, 3, 4, 0])
    scrambled = pmf[:, perm]
    rho = canonical_correlations(pmf, pmf)
    rho_p = canonical_correlations(scrambled, pmf)
    # Proper scores of the true law versus the label-permuted law, equal mixture of the two prompts.
    # Cross-entropy and Brier are expectations under the true conditional law.
    def expected_ce(q_rows):
        # prompts 0 and 1, weights 1/2. q_rows has the forecast for prompt 0 then prompt 1.
        ce0 = -(0.5 * np.log(max(q_rows[0, 0], 1e-15)) + 0.5 * np.log(max(q_rows[0, 4], 1e-15)))
        ce1 = -np.log(max(q_rows[1, 2], 1e-15))
        return 0.5 * (ce0 + ce1)

    def expected_brier(q_rows):
        def one(q, p):
            return float(np.sum((q - p) ** 2))
        # Expected one-hot Brier equals ||q||^2 + 1 - 2 q·p, which is ||q-p||^2 when p is the mean of the one-hot.
        return 0.5 * (one(q_rows[0], p0) + one(q_rows[1], p1))

    q_true = np.vstack([p0, p1])
    q_bad = q_true[:, perm]
    labels = ["CCA top\ntrue PMF", "CCA top\npermuted labels", "Brier\ntrue PMF", "Brier\npermuted labels"]
    vals = [rho[0], rho_p[0], expected_brier(q_true), expected_brier(q_bad)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    colors = ["#1f4e79", "#1f4e79", "#2e7d32", "#c62828"]
    ax.bar(np.arange(4), vals, color=colors)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("Canonical correlation (left two) or Brier score (right two), synthetic two-prompt law")
    ax.set_xlabel("Same two prompts. Scalar conditional means are both 2, so scalar Pearson of the means is undefined.")
    _style(ax)
    ax.set_title(
        f"One-hot CCA stays {rho[0]:.3f} after a label permutation. Brier moves from {expected_brier(q_true):.3f} to {expected_brier(q_bad):.3f}.\n"
        f"Cross-entropy of the true forecast is {expected_ce(q_true):.3f} nats; the permuted forecast hits a zero and is infinite."
    )
    fig.savefig(FIG / "toy_onehot_cca_vs_proper.png", dpi=160)
    plt.close(fig)


def figure_collapse() -> None:
    # Population values from the Lean witness, aspect 1, equal conditional variance and a mean gap.
    # Squared correlation 1/16 versus 1/4. Signed correlation is the positive square root: 1/4 versus 1/2.
    names = ["Pearson |r|\nspread judge", "Pearson |r|\ncollapsed mean", "Brier\ntrue law", "Brier\npoint mass", "RPS\ntrue law", "RPS\npoint mass"]
    vals = [0.25, 0.5, 3 / 8, 0.5, 3 / 16, 0.25]
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    colors = ["#1f4e79", "#1f4e79", "#2e7d32", "#c62828", "#2e7d32", "#c62828"]
    ax.bar(np.arange(6), vals, color=colors)
    ax.set_xticks(np.arange(6), names)
    ax.set_ylabel("Population value on the two-prompt binary witness (not human data)")
    ax.set_xlabel("Aspect 1 of the Lean witness: conditional means 1/4 and 3/4, conditional variance 3/16")
    _style(ax)
    ax.set_title("Collapsing the conditional spread raises |correlation| from 1/4 to 1/2 and worsens Brier and RPS.")
    fig.savefig(FIG / "toy_collapse_inflates_corr.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    figure_shapes()
    figure_onehot()
    figure_collapse()
    print("wrote toy figures")


if __name__ == "__main__":
    main()
