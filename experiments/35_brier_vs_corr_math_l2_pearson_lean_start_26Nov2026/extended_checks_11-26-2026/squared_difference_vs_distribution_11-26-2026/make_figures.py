#!/usr/bin/env python3
"""Two figures for SQUARED_DIFFERENCE_VS_DISTRIBUTION_11-26-2026.md.  Deterministic, no randomness.

fig1: equal mean, different laws (K = 6 bars; K = 3 simplex with iso-mean lines coloured by Brier distance).
fig2: (a) Y = 2X has Z_Y = Z_X but a different law; (b) squared difference between independent draws.
Every axis has a grid (agents-config plotting rule).  Run:  python3 -I make_figures.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
C_P, C_Q = "#1f77b4", "#d95f02"


def bars(ax, xs, p, q, lp, lq):
    w = 0.38
    ax.bar(xs - w / 2, p, w, color=C_P, label=lp)
    ax.bar(xs + w / 2, q, w, color=C_Q, label=lq)
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=8, ncol=2)


# ------------------------------------------------------------------ figure 1
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
xs = np.arange(6)
p = np.array([0.5, 0, 0, 0, 0, 0.5])
q = np.array([0, 0, 0.5, 0.5, 0, 0])
bars(ax, xs, p, q, "human rating law $p$", "other law $q$")
ax.axvline(2.5, color="k", ls="--", lw=1)
ax.text(2.55, 0.36, "both means = 2.5", fontsize=8)
ax.set_ylim(0, 0.72)
ax.set_xlabel("Rating value $r$ on the 0 to 5 scale")
ax.set_ylabel("Probability of the rating value (a distribution, not a score)")
ax.set_title("(a) Mean difference is 0, yet total variation = 1\nand Brier distance $\\sum_r (p-q)^2$ = 1", fontsize=9)

ax = axes[1]
# simplex (q0, q1, q2); mean m = q1 + 2 q2; plot in 2-D with corners (0,0), (1,0), (1/2, sqrt(3)/2)
corner = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def xy(v):
    v = np.asarray(v, float)
    return v @ corner


tri = np.vstack([corner, corner[0]])
ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1)
p3 = np.array([0.5, 0.0, 0.5])  # mean 1
# colour a fine grid of the simplex by Brier distance from p3
n = 160
pts, cols = [], []
for i in range(n + 1):
    for j in range(n + 1 - i):
        v = np.array([i, j, n - i - j]) / n
        pts.append(xy(v))
        cols.append(np.sum((v - p3) ** 2))
pts = np.array(pts)
sc = ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=3, cmap="viridis", vmin=0, vmax=2)
# iso-mean lines: mean m in {0.5, 1, 1.5}
for m, ls in [(0.5, ":"), (1.0, "-"), (1.5, ":")]:
    # q1 + 2 q2 = m  with q0 = 1 - q1 - q2 >= 0: endpoints on the boundary
    ends = []
    for q2 in np.linspace(0, 1, 2001):
        q1 = m - 2 * q2
        q0 = 1 - q1 - q2
        if q1 >= -1e-12 and q0 >= -1e-12:
            ends.append(xy([q0, q1, q2]))
    ends = np.array(ends)
    ax.plot(ends[:, 0], ends[:, 1], color="white" if m == 1.0 else "0.8", ls=ls, lw=2.2 if m == 1.0 else 1.2)
ax.plot(*xy(p3), "o", color="red", ms=8)
ax.annotate("$p=(1/2,0,1/2)$", xy(p3), xytext=(0.0, 0.66), fontsize=8, color="red", ha="left",
            arrowprops=dict(arrowstyle="->", color="red", lw=1))
ax.text(*xy([0, 0, 1]) + np.array([0, 0.015]), "all mass on 2", fontsize=8, va="bottom", ha="center")
ax.text(*xy([1, 0, 0]) + np.array([0.02, -0.02]), "all mass on 0", fontsize=8, va="top", ha="left")
ax.text(*xy([0, 1, 0]) + np.array([0.02, -0.02]), "all mass on 1", fontsize=8, va="top", ha="right")
ax.set_xlabel("simplex coordinate for $q=(q(0),q(1),q(2))$ (arbitrary units)")
ax.set_ylabel("simplex coordinate (arbitrary units)")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.08, 0.95)
ax.set_aspect("equal")
ax.set_axisbelow(True)
ax.grid(True)
cb = plt.colorbar(sc, ax=ax, shrink=0.8)
cb.set_label("Brier distance from $p$: $\\sum_r (p(r)-q(r))^2$ (lower = closer)", fontsize=8)
ax.set_title("(b) $K=3$: the mean-1 line (white) holds a continuum of laws\n(dimension $K-2=1$) that share one mean", fontsize=9)
fig.suptitle("Equal means do not mean equal distributions (exact pmfs; no human ratings)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_equal_mean_different_laws.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ figure 2
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
xs = np.arange(5)
px = np.array([0.25, 0.5, 0.25, 0, 0])
py = np.array([0.25, 0, 0.5, 0, 0.25])
bars(ax, xs, px, py, "$X$ with law $p_X$", "$Y=2X$ with law $p_Y$")
ax.set_ylim(0, 0.65)
ax.set_xlabel("Value of the random variable (0 to 4)")
ax.set_ylabel("Probability of the value (a distribution)")
ax.set_title(
    "(a) Standardized difference $E[(Z_X-Z_Y)^2]=2(1-\\mathrm{corr})=0$,\nyet Brier distance 3/8 and total variation 1/2", fontsize=9
)

ax = axes[1]
vals = [0.5 * y**2 + 0.5 * (5 - y) ** 2 for y in range(6)] + [25 / 2]
labels = [f"point\nmass {y}" for y in range(6)] + ["matched\n$q=p$"]
colors = [C_Q] * 6 + [C_P]
ax.bar(np.arange(7), vals, color=colors)
for i, v in enumerate(vals):
    ax.text(i, v + 0.2, f"{v:g}", ha="center", fontsize=8)
ax.set_xticks(np.arange(7))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 15)
ax.set_xlabel("Judge's predicted law $q$\n(the judge draws $Y\\sim q$, independent of the human rating $X\\sim p$)")
ax.set_ylabel("Expected squared difference $E[(X-Y)^2]$ (lower is better)")
ax.set_axisbelow(True)
ax.grid(True)
ax.set_title("(b) $p=\\frac{1}{2}\\delta_0+\\frac{1}{2}\\delta_5$: the truthful $q=p$ scores 12.5;\na point mass at 2 or 3 scores 6.5", fontsize=9)
fig.suptitle("Squared difference of random variables: it cannot match a distribution (exact pmfs; no human ratings)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_standardized_and_independent_draws.png"), dpi=150)
plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
