"""Matched RPS-fit versus CE-fit pairs on the sister repository's committed validation grid (experiment 29, commit f76d502).
Usage: python3 -I my_matched_pairs_from_committed_grid.py PATH/TO/validation_grid_11-27-2026.csv
A pair is two rows that agree on family, feature set, penalty alpha, mixture, constant mode and temperature and differ only in the training objective.
Reported per family and penalty: the number of pairs, the number in which the RPS-fit head has LOWER validation RPS than the CE-fit head, and the median of
(validation RPS of the RPS-fit head minus validation RPS of the CE-fit head); positive means the CE-fit head is better. Then the loss-scale ratio behind the penalty confound."""
import csv, sys, collections, statistics as st

rows = list(csv.DictReader(open(sys.argv[1])))
key = lambda r: (r["family"], r["feature_set"], r["alpha"], r["mix"], r["const_mode"], r["temperature"])
by = collections.defaultdict(dict)
for r in rows:
    by[key(r)][r["objective"]] = r
print("grid rows:", len(rows), "| matched keys with both objectives:", sum(1 for d in by.values() if len(d) == 2))
print("Validation RPS of the RPS-fit head minus validation RPS of the CE-fit head, matched pairs (positive: the CE-fit head is better).")
for fam in sorted({r["family"] for r in rows}):
    for a in sorted({r["alpha"] for r in rows}, key=float):
        diffs = [float(d["rps"]["val_rps"]) - float(d["ce"]["val_rps"]) for k, d in by.items() if k[0] == fam and k[2] == a and len(d) == 2]
        if diffs:
            print("  family %-16s alpha %-5s pairs %3d | RPS-fit head lower in %3d | median %+.4f" % (fam, a, len(diffs), sum(x < 0 for x in diffs), st.median(diffs)))
own_rps = [float(r["val_rps"]) for r in rows if r["objective"] == "rps"]
own_ce = [float(r["val_ce"]) for r in rows if r["objective"] == "ce"]
print("Loss scale: median validation RPS of RPS-fit rows %.4f, median validation CE of CE-fit rows %.4f, ratio %.1f" % (st.median(own_rps), st.median(own_ce), st.median(own_ce) / st.median(own_rps)))
print("The L2 term is added to the mean loss (lib29.py line 169: score.mean() + 0.5 * alpha * (w @ w)), so one alpha weighs about that many times more against the RPS loss than against CE.")
