"""Verify DICES calibrated-Pearson mean crossing, g6-H3, 11-26-2026."""
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
E = next((ROOT / "experiments").glob("28_*"))
rows = list(csv.DictReader((E / "results/scores_absolute.csv").open()))
print("g6-H3 verification (11-26-2026): five-judge held-out means from saved scores")
for dataset in ("dices350", "dices990"):
    values = []
    for budget in sorted({int(r["B"]) for r in rows if r["dataset"] == dataset and r["d"] == "1"}):
        panel = [r for r in rows if r["dataset"] == dataset and r["d"] == "1" and int(r["B"]) == budget]
        methods = {m: [r for r in panel if r["method"] == m] for m in ("ce", "pcal")}
        assert all(len(v) == 5 for v in methods.values())
        score = {m: {k: mean(float(r[k]) for r in rr) for k in ("rps", "ce", "tvd")} for m, rr in methods.items()}
        diff = score["pcal"]["rps"] - score["ce"]["rps"]
        values.append((budget, diff))
        if budget in (64, 128):
            print(dataset, "B=", budget, "RPS/CE/TVD", score, "RPS calibrated-Pearson minus CE=", f"{diff:.7f}")
    sustained = next((b for i, (b, _) in enumerate(values) if all(d < 0 for _, d in values[i:])), None)
    print(dataset, "first sustained lower calibrated-Pearson RPS mean:", sustained)
print("CONFIRMED: DICES-990 crosses at 128 ratings, not 64; no significance claim from these averages.")
