#!/usr/bin/env python3
"""Check the POPQUORN claims of Experiment 36 (commit fcd8db9) against the pinned raw files.

Claude, 11-26-2026. Read-only. It prints facts about the two raw_data.csv files and checks the SHA-256 values and the
row, item and rater counts that Experiment 36's qa_mega_11-26-2026/COMMON_11-26-2026.md and qa3_stage3_11-26-2026.md state.
It counts distinct user ids; it prints none of them.

Fetch the two files first (pinned commit dbb118c29b3d9bcce5aa172d8249ded9ece4df84 of Jiaxin-Pei/Potato-Prolific-Dataset):
    R=dbb118c29b3d9bcce5aa172d8249ded9ece4df84
    B=https://raw.githubusercontent.com/Jiaxin-Pei/Potato-Prolific-Dataset/$R/dataset
    mkdir -p DIR && curl -sS -o DIR/offensiveness_raw.csv $B/offensiveness/raw_data.csv
    curl -sS -o DIR/politeness_rating_raw.csv $B/politeness_rating/raw_data.csv

Usage:
    python3 -I popquorn_check.py DIR

Exit status is nonzero when a check fails.
"""
import collections
import csv
import hashlib
import statistics as st
import sys

d = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "."
CLAIMS = {
    "offensiveness": {"sha": "fb0589093d45da76bf4ef4d0abff679b540bdd5b17af5764913b3ccd0a069e1e", "rows": 13036, "items": 1500, "raters": 262},
    "politeness_rating": {"sha": "beb7c68c6366ab27819cdf40fec7eb76676f5903c99dad95c13d4bd2bc28365f", "rows": 25042, "items": 3718, "raters": 506},
}
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""))
    if not ok:
        fails += 1


for name, claim in CLAIMS.items():
    path = f"{d}/{name}_raw.csv"
    raw = open(path, "rb").read()
    check(f"{name}: SHA-256 equals the pinned value", hashlib.sha256(raw).hexdigest() == claim["sha"])
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    col = "offensiveness" if name == "offensiveness" else "politeness"
    items = collections.defaultdict(list)
    pairs = collections.Counter()
    users = set()
    for r in rows:
        items[r["instance_id"]].append(float(r[col]))
        users.add(r["user_id"])
        pairs[(r["instance_id"], r["user_id"])] += 1
    values = collections.Counter(r[col] for r in rows)
    check(f"{name}: rows", len(rows) == claim["rows"], f"{len(rows)}")
    check(f"{name}: items", len(items) == claim["items"], f"{len(items)}")
    check(f"{name}: raters", len(users) == claim["raters"], f"{len(users)}")
    check(f"{name}: no duplicate (item, rater) pair", all(c == 1 for c in pairs.values()))
    check(f"{name}: values are exactly 1.0 to 5.0", sorted(values) == ["1.0", "2.0", "3.0", "4.0", "5.0"])
    ks = sorted(len(v) for v in items.values())
    kc = collections.Counter(ks)
    print(f"FACT  {name}: ratings per item min {ks[0]}, median {st.median(ks)}, mean {st.mean(ks):.2f}, max {ks[-1]}; histogram " + ", ".join(f"{k}:{c}" for k, c in sorted(kc.items())))
    allsame = sum(1 for v in items.values() if len(set(v)) == 1)
    print(f"FACT  {name}: items on which every rater agrees exactly {allsame} of {len(items)} ({allsame / len(items):.3f})")
    shares = ", ".join(f"{k}:{c / len(rows):.3f}" for k, c in sorted(values.items()))
    print(f"FACT  {name}: share of ratings by value {shares}")
    bias = []
    for v in items.values():
        c = collections.Counter(v)
        k = len(v)
        bias.append((1 - sum((n / k) ** 2 for n in c.values())) / k)
    print(f"FACT  {name}: mean over items of (1 - ||p_hat||^2) / k = {st.mean(bias):.4f} (the plug-in Brier bias of an empirical histogram of k ratings)")
print("\n" + ("ALL CHECKS PASS" if fails == 0 else f"{fails} CHECK(S) FAILED"))
sys.exit(1 if fails else 0)
