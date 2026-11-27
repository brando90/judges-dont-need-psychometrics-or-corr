#!/usr/bin/env python3
"""Count the frozen, anonymized VeriBench formal-spec-coverage (FSC) ratings used by the sister repo's Experiment 28.

Claude, 11-26-2026. Read-only. Prints facts about the 11-26-2026 anonymized files and checks that their
`exp27_split` equals the sealed 11-24-2026 split. It reads no rater names or rater ids: the input files hold only
sorted rating lists with the rater identity removed.

Usage:
    python3 -I fsc_counts_11_26.py <sister repo root>

The sister repo is https://github.com/brando90/trustworthy-theorem-judges, checked out at or after commit 7855746.
Exit status is nonzero when a check fails.
"""
import csv
import collections
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
exp28 = os.path.join(root, "experiments", "28_multi_family_distributional_judge_and_open_weight_lora_on_new_ratings_start_26Nov2026", "data_11-26-2026")
exp27 = os.path.join(root, "experiments", "27_build_a_judge_that_beats_the_constant_forecast_on_held_out_tasks_old_new_ratings_start_24Nov2026", "data_11-24-2026")
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""))
    if not ok:
        fails += 1


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


old = rows(os.path.join(exp28, "human_ratings_old_11-26-2026.csv"))
new = rows(os.path.join(exp28, "human_ratings_new_11-26-2026.csv"))
tasks = rows(os.path.join(exp28, "task_splits_11-26-2026.csv"))


def ratings_of(r):
    return [int(x) for x in r["ratings_sorted_0to5_rater_identity_removed"].split()]


def summarize(label, rs):
    n_items = len(rs)
    n_rat = sum(len(ratings_of(r)) for r in rs)
    per = collections.Counter(len(ratings_of(r)) for r in rs)
    vals = collections.Counter(v for r in rs for v in ratings_of(r))
    n_tasks = len({r["task"] for r in rs})
    by_split_items = collections.Counter(r["exp27_split"] for r in rs)
    by_split_tasks = collections.Counter(t["exp27_split"] for t in tasks if t["set"] == label)
    print(f"FACT  {label}: items {n_items} | tasks {n_tasks} | ratings {n_rat}")
    print(f"FACT  {label}: items by number of ratings {dict(sorted(per.items()))}")
    print(f"FACT  {label}: rating value counts {dict(sorted(vals.items()))} | share of fives {vals[5] / n_rat:.3f}")
    print(f"FACT  {label}: items by exp27_split {dict(sorted(by_split_items.items()))} | tasks by exp27_split {dict(sorted(by_split_tasks.items()))}")
    folds = collections.Counter(r["cv5_fold"] for r in rs)
    print(f"FACT  {label}: items by cv5_fold {dict(sorted(folds.items()))}")
    check(f"{label}: n_ratings column equals the number of listed ratings", all(int(r["n_ratings"]) == len(ratings_of(r)) for r in rs))
    check(f"{label}: all ratings are integers 0..5", all(0 <= v <= 5 for r in rs for v in ratings_of(r)))
    check(f"{label}: a task never crosses an exp27_split", all(len({r["exp27_split"] for r in rs if r["task"] == t}) == 1 for t in {r["task"] for r in rs}))
    return n_items, n_rat, n_tasks


oi, orr, ot = summarize("old", old)
ni, nr, nt = summarize("new", new)
print(f"FACT  pooled: items {oi + ni} | tasks {ot + nt} | ratings {orr + nr}")
check("pooled counts are 175 items, 50 tasks, 445 ratings", (oi + ni, ot + nt, orr + nr) == (175, 50, 445))
check("old and new share no task", not ({r["task"] for r in old} & {r["task"] for r in new}))
check("old and new share no item", not ({r["item"] for r in old} & {r["item"] for r in new}))

# the cross-validation fold is per task too
check("cv5_fold is constant within a task", all(len({r["cv5_fold"] for r in old + new if r["task"] == t}) == 1 for t in {r["task"] for r in old + new}))

# the 11-26 split must equal the sealed 11-24 split on every shared task
sealed = rows(os.path.join(exp27, "task_splits_11-24-2026.csv"))
seal = {(t["set"], t["task"]): t for t in sealed}
cols = sealed[0].keys()
print(f"FACT  sealed 11-24 task_splits columns: {list(cols)}")
split_col = "split" if "split" in cols else [c for c in cols if "split" in c][0]
shared = [(t["set"], t["task"]) for t in tasks if (t["set"], t["task"]) in seal]
agree = sum(1 for k in shared if seal[k][split_col] == next(t["exp27_split"] for t in tasks if (t["set"], t["task"]) == k))
print(f"FACT  tasks present in both the 11-24 sealed split and the 11-26 file: {len(shared)} of {len(tasks)}; same split in both: {agree}")
check("every 11-26 task appears in the sealed 11-24 split with the same split", len(shared) == len(tasks) and agree == len(shared))

print()
print("RESULT: ALL FSC COUNT CHECKS PASS" if fails == 0 else f"RESULT: {fails} FSC COUNT CHECK(S) FAILED")
sys.exit(1 if fails else 0)
