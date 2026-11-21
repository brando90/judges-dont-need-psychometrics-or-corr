"""Render the historical-budget comparison from public, aggregate CSVs."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
with (ROOT / "paired_rps.csv").open() as stream:
    rows = [r for r in csv.DictReader(stream) if r["subtrahend_selector"] == "ce"]

figure, axis = plt.subplots(figsize=(8, 4.7), layout="constrained")
for i, r in enumerate(rows):
    x = float(r["pearson_selected_minus_other_test_rps"])
    lo, hi = float(r["family_bayes_95_lo"]), float(r["family_bayes_95_hi"])
    axis.errorbar(x, i, xerr=[[x-lo], [hi-x]], fmt="o", color="#186357", capsize=4, lw=1.8)
axis.axvline(0, color="#575454", lw=1.1)
axis.set_yticks(np.arange(len(rows)), [f'{r["train_families"]} families, {r["train_ratings_per_artifact"]} '
                                       + ("rating" if r["train_ratings_per_artifact"] == "1" else "ratings")
                                       for r in rows])
axis.invert_yaxis()
axis.set_ylabel("Training families and\nratings per artifact")
axis.set_xlabel("Held-out normalized ranked probability score (RPS)\n"
                "Pearson-selected judge minus cross-entropy-selected judge")
axis.grid(axis="both", color="#ddddda", lw=0.65)
axis.set_axisbelow(True)
axis.set_title("Single-judge selection on historical human ratings", loc="left", fontweight="bold")
for name in ("top", "right"):
    axis.spines[name].set_visible(False)
figure.savefig(ROOT / "historical_rps.png", dpi=180)
