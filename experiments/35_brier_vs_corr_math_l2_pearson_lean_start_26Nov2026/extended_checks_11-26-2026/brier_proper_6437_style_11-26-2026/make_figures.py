"""Figures for BRIER_PROPER_6437_STYLE_11-26-2026.md. Analytic curves and seeded random
reports only; no human ratings. Writes figures/fig1..fig3 (PNG, dpi 150).
Run:  python3 -I make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ---- Figure 1: K = 2, true P(rating = 1) = 0.7
p = 0.7
q = np.linspace(0.01, 0.99, 981)
brier = (q - p) ** 2 * 2 + 2 * p * (1 - p)            # sum over both coordinates
ll = -(p * np.log(q) + (1 - p) * np.log(1 - q))
H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
floor_b = 2 * p * (1 - p)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(q, brier, label="expected Brier score")
ax[0].axhline(floor_b, ls="--", c="gray", label=f"floor 1 - ||p||^2 = {floor_b:.2f}")
ax[0].fill_between(q, floor_b, brier, alpha=0.2, label="excess = ||q - p||^2")
ax[0].axvline(p, c="k", lw=0.8)
ax[0].set_ylim(0.35, 1.0)
ax[0].set_xlabel("Reported probability q(1) of rating 1 (true p(1) = 0.7)")
ax[0].set_ylabel("Expected Brier score under the true rating distribution")
ax[0].set_title("Brier: symmetric bowl, minimum at q = p")
ax[0].legend(fontsize=8)
ax[1].plot(q, ll, c="C1", label="expected log-loss (nats)")
ax[1].axhline(H, ls="--", c="gray", label=f"floor = entropy H(p) = {H:.4f}")
ax[1].fill_between(q, H, ll, alpha=0.2, color="C1", label="excess = KL(p || q)")
ax[1].axvline(p, c="k", lw=0.8)
ax[1].set_ylim(0.55, 1.4)
ax[1].set_xlabel("Reported probability q(1) of rating 1 (true p(1) = 0.7)")
ax[1].set_ylabel("Expected log-loss (nats) under the true rating distribution")
ax[1].set_title("Log-loss: lopsided bowl, minimum at q = p")
ax[1].legend(fontsize=8)
for a in ax:
    a.grid(True)
fig.tight_layout()
fig.savefig(OUT / "fig1_k2_brier_vs_logloss.png", dpi=150)
plt.close(fig)

# ---- Figure 2: K = 3, excess expected Brier = ||q - p||^2 over the simplex slice (q0, q1)
p3 = np.array([0.2, 0.5, 0.3])
g = np.linspace(0, 1, 401)
Q0, Q1 = np.meshgrid(g, g)
Q2 = 1 - Q0 - Q1
excess = (Q0 - p3[0]) ** 2 + (Q1 - p3[1]) ** 2 + (Q2 - p3[2]) ** 2
excess = np.where(Q2 >= -1e-12, excess, np.nan)
fig, ax = plt.subplots(figsize=(5.6, 4.8))
cs = ax.contourf(Q0, Q1, excess, levels=20, cmap="viridis")
ax.contour(Q0, Q1, excess, levels=[0.01, 0.05, 0.1, 0.2, 0.4], colors="w", linewidths=0.7)
ax.plot(p3[0], p3[1], "r*", ms=12, label="true p = (0.2, 0.5, 0.3)")
cb = fig.colorbar(cs, ax=ax)
cb.set_label("Excess expected Brier score = ||q - p||^2")
ax.set_xlabel("Reported q(0) (q(2) = 1 - q(0) - q(1))")
ax.set_ylabel("Reported q(1) (q(2) = 1 - q(0) - q(1))")
ax.set_title("K = 3: excess expected Brier over the simplex")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True)
fig.tight_layout()
fig.savefig(OUT / "fig2_k3_brier_excess_simplex.png", dpi=150)
plt.close(fig)

# ---- Figure 3: locality. Cost against q(x) for random reports
rng = np.random.default_rng(7)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
q3 = rng.dirichlet(np.ones(3), size=3000)
c3 = ((q3 - np.eye(3)[0]) ** 2).sum(1)
ax[0].scatter(q3[:, 0], c3, s=3, alpha=0.4, label="Brier, 3,000 random reports")
nu = np.linspace(0, 1, 201)
ax[0].plot(nu, 1.5 * (1 - nu) ** 2, c="k", lw=1, label="lower edge 1.5 (1 - q(x))^2")
ax[0].plot(nu, (1 - nu) ** 2 + (1 - nu) ** 2, c="r", lw=1, label="upper edge 2 (1 - q(x))^2")
ax[0].set_xlabel("Probability q(x) the report gives the observed rating x (K = 3)")
ax[0].set_ylabel("Brier score of the report on observed rating x")
ax[0].set_title("K = 3: a cloud, so Brier is not local")
ax[0].legend(fontsize=8)
q2 = rng.dirichlet(np.ones(2), size=3000)
c2 = ((q2 - np.eye(2)[0]) ** 2).sum(1)
ax[1].scatter(q2[:, 0], c2, s=3, alpha=0.4, label="Brier, 3,000 random reports")
ax[1].plot(nu, 2 * (1 - nu) ** 2, c="r", lw=1, label="2 (1 - q(x))^2")
ax[1].set_xlabel("Probability q(x) the report gives the observed rating x (K = 2)")
ax[1].set_ylabel("Brier score of the report on observed rating x")
ax[1].set_title("K = 2: one curve, so Brier is local")
ax[1].legend(fontsize=8)
for a in ax:
    a.grid(True)
fig.tight_layout()
fig.savefig(OUT / "fig3_locality_k3_vs_k2.png", dpi=150)
plt.close(fig)
print("wrote", *sorted(x.name for x in OUT.glob("*.png")))
