"""Experiment 51 summary: predictions L1-L3 and per-(view, lambda) tables from results/fixed_ridge_rows.json.
Usage: python summarize_grid.py RESULTS_DIR FIG_DIR
paired_ci entries are [mean, lo, hi] of (first - second) test normalized RPS. Writes grid_tables.md, grid_summary.json, figures to FIG_DIR."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

res, figd = Path(sys.argv[1]), Path(sys.argv[2]); figd.mkdir(parents=True, exist_ok=True)
rows = json.loads((res / "fixed_ridge_rows.json").read_text())
cell = defaultdict(list)
for r in rows:
    cell[(r["view"], r["lam"])].append(r)
views = list(dict.fromkeys(r["view"] for r in rows)); lams = sorted(set(r["lam"] for r in rows))
sig = {r["view"]: r["signal"] for r in rows}
def m(rs, key): return float(np.mean([r[key][0] for r in rs]))
def rng(rs, key): v = [r[key][0] for r in rs]; return float(min(v)), float(max(v))
def n_above(rs, key): return int(sum(r[key][1] > 0 for r in rs))
def n_below(rs, key): return int(sum(r[key][2] < 0 for r in rs))

summ = {"lams": lams, "cells": []}
for v in views:
    for lam in lams:
        rs = cell[(v, lam)]
        if not rs: continue
        summ["cells"].append({"view": v, "lam": lam, "signal": sig[v], "n_seeds": len(rs),
            "raw_gap_mean": m(rs, "raw_gap"), "raw_gap_range": rng(rs, "raw_gap"), "raw_gap_seeds_ci_above": n_above(rs, "raw_gap"), "raw_gap_seeds_ci_below": n_below(rs, "raw_gap"),
            "matched_gap_mean": m(rs, "matched_gap"), "matched_gap_range": rng(rs, "matched_gap"),
            "matched_gap_seeds_ci_above": n_above(rs, "matched_gap"), "matched_gap_seeds_ci_below": n_below(rs, "matched_gap"),
            "matched_mse_minus_ce_mean": m(rs, "matched_mse_minus_ce"),
            "ce_raw_minus_const_mean": m(rs, "ce_raw_minus_const"), "ce_cal_minus_const_mean": m(rs, "ce_cal_minus_const")})
C = summ["cells"]
l1_fail = [c for c in C if abs(c["matched_gap_mean"]) > 0.003]
l2_fail = [c for c in C if c["signal"] and not c["raw_gap_mean"] > 0]
best_gain = {}
for v in views:
    cs = [c for c in C if c["view"] == v]
    b = min(cs, key=lambda c: c["ce_raw_minus_const_mean"])
    best_gain[v] = {"lam": b["lam"], "gain_over_constant": -b["ce_raw_minus_const_mean"]}
l3_fail = [v for v, b in best_gain.items() if b["gain_over_constant"] > 0.02]
summ["L1"] = {"cells": len(C), "fail_cells": [(c["view"], c["lam"], c["matched_gap_mean"]) for c in l1_fail]}
summ["L2"] = {"cells_with_signal": sum(c["signal"] for c in C), "fail_cells": [(c["view"], c["lam"], c["raw_gap_mean"]) for c in l2_fail]}
summ["L3"] = {"best_lambda_gain_over_constant": best_gain, "fail_views": l3_fail}
out = [f"## Seed-mean gaps by penalty (test normalized RPS; positive = CE-trained head better). Lambdas: {', '.join(f'{x:g}' for x in lams)}\n"]
for key, title in (("raw_gap_mean", "raw gap (Pearson-trained raw minus CE-trained raw)"), ("matched_gap_mean", "matched gap (both calibrated)"),
                   ("ce_raw_minus_const_mean", "CE-trained raw minus constant (negative = beats constant)")):
    out.append(f"\n### {title}\n\n| view | " + " | ".join(f"{x:g}" for x in lams) + " |\n|---|" + "---|" * len(lams))
    for v in views:
        d_ = {c["lam"]: c[key] for c in C if c["view"] == v}
        out.append(f"| {v}{'' if sig[v] else ' (no signal)'} | " + " | ".join(f"{d_[x]:+.4f}" if x in d_ else "" for x in lams) + " |")
out.append(f"\nL1 (|matched gap| <= 0.003 everywhere): {len(C) - len(l1_fail)} of {len(C)} cells pass; failing: " + ("; ".join(f"{a} lam={b:g} {g:+.4f}" for a, b, g in summ['L1']['fail_cells']) or "none"))
out.append(f"L2 (raw gap > 0 on views with signal): {summ['L2']['cells_with_signal'] - len(l2_fail)} of {summ['L2']['cells_with_signal']} cells pass; failing: " + ("; ".join(f"{a} lam={b:g} {g:+.4f}" for a, b, g in summ['L2']['fail_cells']) or "none"))
out.append("L3 (best-lambda CE-trained gain over constant <= 0.02): " + "; ".join(f"{v} {b['gain_over_constant']:+.4f} at {b['lam']:g}" for v, b in best_gain.items()) + f". Failing views: {', '.join(l3_fail) or 'none'}")
(res / "grid_tables.md").write_text("\n".join(out) + "\n")
(res / "grid_summary.json").write_text(json.dumps(summ, indent=1))
print("\n".join(out))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, t in ((axs[0], "raw_gap_mean", "raw gap"), (axs[1], "matched_gap_mean", "matched gap")):
        for v in views:
            xs = [c["lam"] for c in C if c["view"] == v]; ys = [c[key] for c in C if c["view"] == v]
            ax.plot(xs, ys, marker="o", ms=3, lw=1, label=v, ls="-" if sig[v] else ":")
        ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.8); ax.grid(True)
        ax.set_xlabel("ridge penalty lambda (fixed, all heads)"); ax.set_ylabel(f"test nRPS {t}, seed mean (Pearson-trained minus CE-trained)"); ax.set_title(t)
    axs[1].axhspan(-0.003, 0.003, color="grey", alpha=0.2)
    axs[1].legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(figd / "fixed_ridge_gaps.png", dpi=150)
except Exception as e:
    print("figure skipped:", e)
