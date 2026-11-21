"""Known-mean conditional variance risk; inputs are only the published aggregate CSV."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
with (root / "population_risk.csv").open() as stream:
    data = [r for r in csv.DictReader(stream) if r["known_information"] == "mean_known"
            and r["tie_policy"] == "low_first" and int(r["total_train_ratings"]) >= 3]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True, layout="constrained")
sizes = sorted({int(r["total_train_ratings"]) for r in data})
for ax, shape, title in zip(axes, ["low", "high"], ["True spread is low", "True spread is high"]):
    for selector, label, color, marker in (("ce", "Cross-entropy-selected", "#16615e", "o"),
                                          ("pearson", "Pearson-selected, low-first tie", "#95432f", "s")):
        means = [np.mean([float(r["variance_mse"]) for r in data if r["true_law"].endswith(shape)
                          and r["selector"] == selector and int(r["total_train_ratings"]) == n]) for n in sizes]
        ax.plot(sizes, means, marker=marker, ms=4, color=color, label=label)
    ax.set_title(title)
    ax.set_xlabel("Total training ratings across three types")
    ax.set_xticks([3,6,12,18,24,36])
    ax.grid(True, axis="both", color="#dcdcd7", lw=0.65)
    ax.set_axisbelow(True)
    for edge in ("top","right"):
        ax.spines[edge].set_visible(False)
axes[0].set_ylabel("Population-test conditional variance\nmean squared error (rating points⁴)")
axes[1].legend(loc="center right", fontsize=8, frameon=False)
fig.suptitle("With the mean known, the Pearson tie rule determines spread recovery", fontweight="bold")
fig.savefig(root / "known_mean_variance_risk.png", dpi=180)
