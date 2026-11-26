"""Verify illustrative cost boundary g6-H4, 11-26-2026."""
import csv
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E = next((ROOT / "experiments").glob("28_*"))
rows = list(csv.DictReader((E / "results/breakeven_summary.csv").open()))
print("g6-H4 verification (11-26-2026)")
print("Source summary columns:", list(rows[0]))
thresholds = [Decimal(r["N_both"]) for r in rows if r["d"] == "5"]
print("Saved d=5 joint thresholds:", sorted(set(thresholds)), "pairs:", len(thresholds))
lower, upper, price = min(thresholds), max(thresholds), Decimal(5)
print(f"10–20 ratings at illustrative USD 5/rating: USD {lower*price:.2f}–{upper*price:.2f}")
assert upper * price == 100
print("CONFIRMED: at most USD 100, not under USD 100; this is illustrative arithmetic, not spending.")
