"""Exp 18 figure: test RPS excess (over the true-distribution RPS) vs n, sweep A, k=3, J=3, 4 scenarios. Date 11-21-2026."""
import csv, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = list(csv.DictReader(open(sys.argv[1])))
arms = [("constant", "constant", "#888888"), ("pearson_raw", "Pearson-trained", "#d95f02"), ("pearson_head_tuned", "Pearson score + calibrated head", "#e6ab02"),
        ("mse_tuned", "MSE-trained", "#7570b3"), ("ce_tuned", "CE-trained", "#1b9e77"), ("rps_tuned", "RPS-trained", "#1f78b4")]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for a_, s in zip(ax, "ABCD"):
    ns = sorted({int(r["n"]) for r in R})
    for arm, lab, col in arms:
        y = [np.mean([float(r[f"{arm}.rps"]) - float(r["true_rps"]) for r in R if r["scenario"] == s and int(r["n"]) == n and r["k"] == "3" and r["J"] == "3"]) for n in ns]
        a_.plot(ns, y, marker="o", ms=3, lw=1.6, label=lab, color=col)
    a_.set_xscale("log", base=2); a_.set_yscale("log"); a_.set_title(f"scenario {s} (k=3, J=3)"); a_.set_xlabel("n training items")
ax[0].set_ylabel("test RPS minus true RPS (lower is better)"); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130)
