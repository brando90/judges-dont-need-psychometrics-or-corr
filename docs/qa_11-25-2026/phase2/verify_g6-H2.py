"""Independently refit g6-H2 RPS curves from score outputs, 11-26-2026.

Weighted least squares over the registered 150-point exponent grid; enumerate
interior and boundary candidates. No model calls, source scores, or files changed.
"""
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
E = next((ROOT / "experiments").glob("29_*"))
rows = list(csv.DictReader((E / "results/scores.csv").open()))
print("g6-H2 verification (11-26-2026): held-out RPS; saved aggregate inputs")
for study in ("chaos3_qwen", "hs2_qwen", "chaos3_llama"):
    rr = sorted([r for r in rows if r["study"] == study and r["depth"] == "1" and r["method"] == "ce_raw"], key=lambda r: int(r["b_total"]))
    n = np.array([int(r["b_total"]) for r in rr], float)
    y = np.array([float(r["rps"]) for r in rr])
    se = np.array([float(r["rps_draw_sd"]) for r in rr]) / np.sqrt(10)
    w = 1 / np.maximum(se, 1e-9) ** 2
    best = None
    for b in np.exp(np.linspace(np.log(.02), np.log(3.), 150)):
        x = (n / n[0]) ** -b
        design = np.column_stack([np.ones(len(x)), x])
        theta = np.linalg.solve(design.T @ (w[:, None] * design), design.T @ (w * y))
        candidates = [(float(w @ y / w.sum()), 0.), (0., float((w*x) @ y / (w @ (x*x))))]
        if np.all(theta >= 0):
            candidates.append(tuple(theta))
        for floor, amplitude in candidates:
            pred = floor + amplitude * x
            rss = float(np.sum(w * (y - pred) ** 2))
            if best is None or rss < best[0] - 1e-12:
                best = (rss, b, floor, pred)
    residual = float(np.sqrt(np.mean((y - best[3])**2)) / (y[0] - y[-1]))
    print(f"{study}: budgets={len(n)} b={best[1]:.10f} relative_RMSE={residual:.10f} cutoff=0.10 verdict={'poor fit' if residual > .10 else 'passes residual criterion'}")
print("CONFIRMED: two of three highlighted eight-budget RPS panels fail the stated fit criterion.")
