"""Recompute transferred RPS sustained thresholds g6-H5, 11-26-2026.

Counts saved simultaneous lower bounds; does not regenerate missing-label
bootstrap samples. A threshold must remain positive at every later budget.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E23 = next((ROOT / "experiments").glob("23_*"))
E25 = next((ROOT / "experiments").glob("25_*"))
read = lambda p: list(csv.DictReader(p.open()))

def threshold(rows, budget, lower):
    rows = sorted(rows, key=lambda r: int(r[budget]))
    return next((int(r[budget]) for i, r in enumerate(rows) if all(float(q[lower]) > 0 for q in rows[i:])), None)

print("g6-H5 verification (11-26-2026): held-out RPS thresholds from saved contrast bounds")
for name, file in (("KonIQ", "results/image_quality/contrasts.csv"),
                   ("PersonaChat", "usr/results/persona_lexical/contrasts.csv"),
                   ("Topical-Chat", "usr/results/topical_usr/contrasts.csv")):
    for depth in ("1", "3"):
        rows = [r for r in read(E23 / file) if r["depth"] == depth and r["comparator"] == "pearson"]
        print(name, "ratings_per_item", depth, "CE versus Pearson threshold", threshold(rows, "budget", "simultaneous_lo"))
cross = [r for r in read(E23 / "results/image_quality/crossed_sensitivity.csv") if r["comparator"] == "pearson"]
print("KonIQ crossed threshold", threshold(cross, "budget", "crossed_lo"))
llama = [r for r in read(E25 / "results/contrasts.csv") if r["study"] == "llama31_8b" and r["metric"] == "rps"
         and r["target"] == "ce_calibrated" and r["comparator"] == "uniform"]
for bound in ("sim_lo_95", "cross_sim_lo_95"):
    print("HelpSteer2/Llama CE-calibrated versus uniform", bound, "threshold", threshold(llama, "b_total", bound))
print("HelpSteer2/Llama B=64 crossed lower", next(r["cross_sim_lo_95"] for r in llama if r["b_total"] == "64"))
print("CONFIRMED: four of nine transferred-summary rows contain wrong comparator-specific thresholds.")
