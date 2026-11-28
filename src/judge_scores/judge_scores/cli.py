"""Command line: ``judge-scores report FORECASTS.csv RATINGS.csv --labels K [--ordered] [--calibrate]``.

FORECASTS.csv has the header ``item_id,p0,...,p{K-1}`` (one row per item); RATINGS.csv has ``item_id,rating`` with one row per
human rating (labels 0..K-1). Prints the report as JSON."""
from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

from .core import ratings_to_counts
from .report import agreement_vs_skill_report


def _read(forecasts_path, ratings_path, K):
    with open(forecasts_path, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["item_id"] for r in rows]
    q = np.array([[float(r[f"p{k}"]) for k in range(K)] for r in rows])
    by_item = {i: [] for i in ids}
    with open(ratings_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["item_id"] not in by_item:
                raise ValueError(f"rating for unknown item {r['item_id']}")
            by_item[r["item_id"]].append(int(r["rating"]))
    return q, ratings_to_counts([by_item[i] for i in ids], K)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="judge-scores")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="agreement and correlation beside the matched proper score and skill")
    rp.add_argument("forecasts")
    rp.add_argument("ratings")
    rp.add_argument("--labels", type=int, required=True, help="number of labels K")
    rp.add_argument("--ordered", action="store_true", help="labels are ordered 0<1<...<K-1")
    rp.add_argument("--rule", choices=["brier", "ce", "nrps"], default=None)
    rp.add_argument("--draws", type=int, default=4000)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--calibrate", action="store_true", help="also split the risk into calibration error and resolution")
    a = ap.parse_args(argv)
    q, counts = _read(a.forecasts, a.ratings, a.labels)
    rep = agreement_vs_skill_report(q, counts, ordered=a.ordered, rule=a.rule, B=a.draws, seed=a.seed, calibrate=a.calibrate)
    json.dump(rep, sys.stdout, indent=1, default=float)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
