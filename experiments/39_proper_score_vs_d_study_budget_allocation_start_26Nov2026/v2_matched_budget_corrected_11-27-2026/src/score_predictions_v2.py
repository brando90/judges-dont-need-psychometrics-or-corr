#!/usr/bin/env python3
"""Mechanically score the pre-declared predictions 1-6 and decision rules D1-D4 of PROTOCOL_V2_11-27-2026.md.

Reads results/analysis_v2.json, results/cells_v2.csv and results/verification_v2.json (all written by
earlier scripts) and writes results/prediction_scorecard_v2.json. A prediction that fails is reported as FAIL;
nothing here is tuned after seeing the data (the thresholds are copied from the protocol).
Exit code is 0 even when a prediction fails, because a failed prediction is a result, not an error.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

V2 = Path(__file__).resolve().parents[1]
RES = V2 / "results"

a = json.load(open(RES / "analysis_v2.json"))
cells = pd.read_csv(RES / "cells_v2.csv")
cells = cells[cells.B.isin(a["comparable_budgets"])].copy()
ver = json.load(open(RES / "verification_v2.json"))
assert len(cells) == 20


def fam(name, metric="rps"):
    return a["families"][name]["metrics"][metric]


def n_mean_first_better(name):
    return int(fam(name)["mean_gain_positive_cells"])


out = {}

# ---- Prediction 1 (F1)
f1 = fam("F1")
gain_pct = []
for ds_name, ds in a["datasets"].items():
    for B, cell in ds["cells"].items():
        gain_pct.append(cell["families"]["F1"]["rps"]["pct_of_total_rps"])
max_gain_pct = max(gain_pct)
p1 = {
    "claim": "fix_items lower mean RPS in a majority of 20 cells; Holm-significant in at most 6; gains below 3% of total RPS",
    "n_mean_better": n_mean_first_better("F1"),
    "n_holm_beats": int(f1["first_beats_second_holm"]),
    "max_gain_pct_of_total_rps": round(max_gain_pct, 3),
}
p1["holds"] = p1["n_mean_better"] >= 11 and p1["n_holm_beats"] <= 6 and p1["max_gain_pct_of_total_rps"] < 3.0
out["prediction_1_F1"] = p1

# ---- Prediction 2 (F3)
f3 = fam("F3")
p2 = {
    "claim": "ps does not significantly beat fix_items in any cell after Holm",
    "n_holm_ps_beats_fix_items": int(f3["first_beats_second_holm"]),
    "n_holm_fix_items_beats_ps": int(f3["second_beats_first_holm"]),
    "n_mean_ps_better": n_mean_first_better("F3"),
}
p2["holds"] = p2["n_holm_ps_beats_fix_items"] == 0
out["prediction_2_F3"] = p2

# ---- Prediction 3 (F6)
f6 = fam("F6")
p3 = {
    "claim": "ps does not beat ds_rel80 in at least 11 of 20 cells (beat = lower mean and Holm-significant)",
    "n_holm_ps_beats_ds_rel80": int(f6["first_beats_second_holm"]),
    "n_holm_ds_rel80_beats_ps": int(f6["second_beats_first_holm"]),
    "n_mean_ps_better": n_mean_first_better("F6"),
}
p3["holds"] = p3["n_holm_ps_beats_ds_rel80"] < 11
out["prediction_3_F6"] = p3

# ---- Prediction 4 (adaptivity)
per_budget = {}
for B in a["comparable_budgets"]:
    modes = []
    for ds_name in ["chaosnli_snli", "chaosnli_mnli", "chaosnli_alphanli", "dices350", "dices990"]:
        sel = a["datasets"][ds_name]["cells"][str(B)]["selection"]["ds_rel80"]
        modes.append(sel["mode"])
    per_budget[str(B)] = {"modes": modes, "n_distinct": len(set(modes))}
n_budgets_adaptive = sum(1 for v in per_budget.values() if v["n_distinct"] >= 2)
p4 = {
    "claim": "ds_rel80 has >= 2 distinct modal designs across the five datasets at >= 2 of the 4 comparable budgets",
    "per_budget": per_budget,
    "n_budgets_with_two_or_more_modes": n_budgets_adaptive,
}
p4["holds"] = n_budgets_adaptive >= 2
out["prediction_4_adaptivity"] = p4

# ---- Prediction 5 (ties)
uf = a["checks"]["mean_unique_best_fraction_or_rps"]
p5 = {"claim": "fraction of replications with a unique best RPS design (or_rps) is below 0.5 overall",
      "mean_unique_best_fraction": round(float(uf), 4)}
p5["holds"] = uf < 0.5
out["prediction_5_ties"] = p5

# ---- Prediction 6 (v1 regression)
n_failed = sum(1 for c in ver if not c["pass"])
p6 = {
    "claim": "ds_u1 == fix_raters and ds_u2 == fix_items in 100% of replications; deployed scores equal v1 (v1 summary means; per-replication v1 file was never committed)",
    "ds_u1_equals_fix_raters_all_cells": bool(a["checks"]["ds_u1_equals_fix_raters_all_cells"]),
    "ds_u2_equals_fix_items_all_cells": bool(a["checks"]["ds_u2_equals_fix_items_all_cells"]),
    "verify_v2_failed_checks": n_failed,
    "verify_v2_n_checks": len(ver),
}
p6["holds"] = p6["ds_u1_equals_fix_raters_all_cells"] and p6["ds_u2_equals_fix_items_all_cells"] and n_failed == 0
out["prediction_6_v1_regression"] = p6

# ---- Decision rules
f2 = fam("F2")
f4 = fam("F4")
f5 = fam("F5")
d = {
    "D1_count_fix_items_beats_fix_raters_holm": int(f1["first_beats_second_holm"]),
    "D1_count_fix_items_better_by_mean": n_mean_first_better("F1"),
    "D2_ps_beats_fix_items_in_at_least_one_cell_holm": int(f3["first_beats_second_holm"]) >= 1,
    "D3_ds_rel80_beats_both_fixed_rules_in_at_least_one_cell_holm": None,
    "D4_ps_beats_ds_rel80_in_at_least_11_cells_holm": int(f6["first_beats_second_holm"]) >= 11,
}
# D3: a cell where ds_rel80 beats BOTH fixed rules after Holm (needs the per-cell verdicts of F4 and F5)
d3_cells = []
for ds_name, ds in a["datasets"].items():
    for B, cell in ds["cells"].items():
        r4 = cell["families"]["F4"]["rps"]
        r5 = cell["families"]["F5"]["rps"]
        if r4["first_beats_second_holm"] and r5["first_beats_second_holm"]:
            d3_cells.append((ds_name, B))
d["D3_ds_rel80_beats_both_fixed_rules_in_at_least_one_cell_holm"] = len(d3_cells) >= 1
d["D3_cells"] = d3_cells
d["extra_counts"] = {
    "F2_ps_beats_fix_raters_holm": int(f2["first_beats_second_holm"]),
    "F4_ds_rel80_beats_fix_raters_holm": int(f4["first_beats_second_holm"]),
    "F5_ds_rel80_beats_fix_items_holm": int(f5["first_beats_second_holm"]),
    "F5_fix_items_beats_ds_rel80_holm": int(f5["second_beats_first_holm"]),
}
out["decision_rules"] = d

json.dump(out, open(RES / "prediction_scorecard_v2.json", "w"), indent=2)
for k, v in out.items():
    if k.startswith("prediction"):
        print(("HOLDS " if v["holds"] else "FAILS ") + k, {kk: vv for kk, vv in v.items() if kk not in ("claim", "holds", "per_budget")})
print("DECISION RULES", json.dumps(d))
