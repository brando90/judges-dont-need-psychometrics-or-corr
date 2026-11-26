"""Verify g6-H1: registered across-budget H1, 11-26-2026.

Inputs are saved score/contrast outputs, not the absent human-label bundle.
No training or resampling is rerun; stored crossed bounds are counted.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E = next((ROOT / "experiments").glob("29_*"))
scores = list(csv.DictReader((E / "results/scores.csv").open()))
contrasts = list(csv.DictReader((E / "results/contrasts.csv").open()))
lookup = {(r["study"], r["depth"], r["b_total"], r["method"]): r for r in scores}
print("g6-H1 verification (11-26-2026)")
for target in ("ce_raw", "rps_raw"):
    rows = [r for r in contrasts if r["metric"] == "rps" and r["target"] == target
            and r["comparator"] == "pearson_raw" and r["study"].startswith(("chaos3", "dices3"))]
    differences = []
    for r in rows:
        key = (r["study"], r["depth"], r["b_total"])
        diff = float(lookup[(*key, "pearson_raw")]["rps"]) - float(lookup[(*key, target)]["rps"])
        assert abs(diff - float(r["diff_comparator_minus_target"])) < 1e-12
        differences.append(diff)
    print(target, "K=3 RPS point wins", sum(x > 0 for x in differences), "/", len(rows),
          "crossed-band wins", sum(float(r["cross_sim_lo_95"]) > 0 for r in rows), "/", len(rows))
    max_rows = {}
    for r in contrasts:
        if r["metric"] == "rps" and r["target"] == target and r["comparator"] == "pearson_raw":
            key = (r["study"], r["depth"])
            if key not in max_rows or int(r["b_total"]) > int(max_rows[key]["b_total"]):
                max_rows[key] = r
    print(target, "all-panel largest-budget crossed wins", sum(float(r["cross_sim_lo_95"]) > 0 for r in max_rows.values()), "/", len(max_rows))
r = next(r for r in contrasts if r["study"] == "chaos3_llama" and r["b_total"] == "32"
         and r["metric"] == "rps" and r["target"] == "ce_raw" and r["comparator"] == "pearson_raw")
print("ChaosNLI/Llama B=32 RPS comparator-minus-target and saved crossed 95% band:",
      *(f"{float(r[k]):.10f}" for k in ("diff_comparator_minus_target", "cross_sim_lo_95", "cross_sim_hi_95")))
assert "across budgets" in (E / "PROTOCOL_11-23-2026.md").read_text()
print("CONFIRMED: max-budget evidence does not confirm the full registered across-budget hypothesis.")
