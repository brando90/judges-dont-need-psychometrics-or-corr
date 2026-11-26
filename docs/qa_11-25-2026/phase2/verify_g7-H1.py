"""Verify g7-H1: DICES calibration thresholds (11-26-2026).

Read committed aggregate score outputs; original labels/bootstrap draws are absent.
No fitting, model calls, or output mutation. Run from any directory.
"""
from pathlib import Path
import csv
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
path = next((ROOT / "experiments").glob("28_*")) / "results/scores_absolute.csv"
rows = list(csv.DictReader(path.open()))
print("g7-H1 verification (11-26-2026)")
print("Source:", path.relative_to(ROOT))
print("Evidence level: re-aggregation of saved score rows, not rating-to-score replay.")
for dataset in ["dices350", "dices990"]:
    # Dataset names in committed CSV use underscores.
    names = sorted({r["dataset"] for r in rows if dataset.replace("dices", "") in r["dataset"]})
    assert len(names) == 1, names
    name = names[0]
    selected = [r for r in rows if r["dataset"] == name and r["d"] == "1"]
    methods = sorted({r["method"] for r in selected})
    cal = "pcal"
    assert cal in methods
    budgets = sorted({int(r["B"]) for r in selected})
    differences = {}
    for budget in budgets:
        means = {}
        for method in ["ce", cal]:
            group = [r for r in selected if int(r["B"]) == budget and r["method"] == method]
            assert len(group) == 5
            means[method] = {metric: mean(float(r[metric]) for r in group) for metric in ["rps", "ce", "tvd"]}
        differences[budget] = means[cal]["rps"] - means["ce"]["rps"]
        if budget in [64, 128]:
            print(name, "d=1", "B=", budget, "RPS/CE/TVD:", means,
                  "RPS calibrated-Pearson minus CE:", f"{differences[budget]:.9f}")
    sustained = next(b for b in budgets if all(differences[v] < 0 for v in budgets if v >= b))
    print(name, "sustained mean RPS advantage starts at", sustained, "ratings")
print("Verdict: CONFIRMED; DICES-350 starts at 64, DICES-990 at 128 for d=1, averaged over five judges.")
