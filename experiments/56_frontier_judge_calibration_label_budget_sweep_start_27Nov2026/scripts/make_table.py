#!/usr/bin/env python3
"""Markdown tables for Experiment 56 from results/label_budget_frontier.json (and the labelled deviation run)."""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
d = json.loads((R / "results/label_budget_frontier.json").read_text())
dev = json.loads((R / "results_deviation_all_pool_items/label_budget_frontier.json").read_text())
devm = {(r["set"], r["judge"]): r["SS_n"]["9999"] for r in dev["rows"]}
N = ["10", "25", "50", "100", "200"]
out = ["| set | judge | raw skill | " + " | ".join(f"n = {n}" for n in N) + " | all pool items, 1 rating (deviation) | full calibration | n* |", "|" + "---|" * 10]
for r in d["rows"]:
    out.append(f"| {r['set']} | {r['judge']} | {r['SS_raw']:+.3f} | " + " | ".join(f"{r['SS_n'][n]:+.3f} ({r['share_hurt'][n]:.0%})" for n in N)
               + f" | {devm[(r['set'], r['judge'])]:+.3f} | {r['SS_full']:+.3f} | {r['n_star'] or '>200'} |")
out += ["", "Cell = mean skill over 50 draws (share of draws in which calibrating on n single ratings left the judge below its uncalibrated skill).", "",
        "| n | cases with mean calibrated skill below raw (of 25) | cases with mean calibrated skill below 0 | sets where every judge's mean calibrated skill is above 0 |", "|---|---|---|---|"]
sets = sorted({r["set"] for r in d["rows"]})
for n in N:
    rows = d["rows"]
    out.append(f"| {n} | {sum(r['SS_n'][n] < r['SS_raw'] for r in rows)} | {sum(r['SS_n'][n] < 0 for r in rows)} | {sum(all(r['SS_n'][n] > 0 for r in rows if r['set'] == s) for s in sets)} of 5 |")
out += ["", "Share of draws in which calibrated GPT-6 Astra still has lower skill than calibrated GLM 5.3 (the Experiment 46 reversal pair):", "",
        "| set | " + " | ".join(f"n = {n}" for n in N) + " |", "|---|" + "---|" * 5]
for s in ("chaosnli_snli", "chaosnli_mnli"):
    out.append(f"| {s} | " + " | ".join(f"{d['astra_below_glm_share'][f'{s}|{n}']:.0%}" for n in N) + " |")
(R / "results/tables.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
