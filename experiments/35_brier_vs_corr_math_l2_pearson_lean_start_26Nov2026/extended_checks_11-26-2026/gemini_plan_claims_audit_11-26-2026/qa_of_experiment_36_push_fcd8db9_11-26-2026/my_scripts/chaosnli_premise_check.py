#!/usr/bin/env python3
"""How many held-out ChaosNLI test items in Experiment 36 (commit fcd8db9) share an exact premise sentence with a training or validation item?

Claude, 11-26-2026. Read-only. Experiment 36 makes every ChaosNLI item (uid) its own split group (data_io.py, "Each uid is its own split group"), so two
hypotheses written for the same premise can land in different splits. Its own leakage assertion checks group names (uids), which cannot see this.
This script rebuilds the SNLI+MNLI 3-class set with Experiment 36's loader, takes Experiment 36's seed-0 group split, extracts the premise string from each
item's text ("premise: <premise> hypothesis: <hypothesis>") and counts test items whose premise string also occurs in train or validation.
Exact string match only; a near-duplicate premise would not be counted, so the number is a lower bound on premise-level overlap.

Usage:
    python3 -I chaosnli_premise_check.py EXPERIMENT_36_FOLDER CACHE_DIR
Exit status is nonzero when the split sizes differ from the committed 1,867 / 622 / 624.
"""
import collections
import os
import sys

sys.dont_write_bytecode = True
e36 = os.path.abspath(sys.argv[1])
os.environ["SANMI36_DATA"] = os.path.abspath(sys.argv[2])
sys.path.insert(0, e36)
import data_io as d  # noqa: E402
import run_all as ra  # noqa: E402

data = d.load_chaos_3way()
prep = ra.prepare(data)
tr, va, te = prep["train"], prep["val"], prep["test"]


def premise(text):
    head, _, _ = text.partition(" hypothesis: ")
    return head[len("premise: "):] if head.startswith("premise: ") else head


prem = [premise(t) for t in data.texts]
ok = (len(tr), len(va), len(te)) == (1867, 622, 624)
print(("PASS  " if ok else "FAIL  ") + f"split sizes train/validation/test {len(tr)}/{len(va)}/{len(te)} equal the committed 1,867/622/624")
seen_trval = set(prem[i] for i in tr) | set(prem[i] for i in va)
seen_tr = set(prem[i] for i in tr)
leak_trval = [i for i in te if prem[i] in seen_trval]
leak_tr = [i for i in te if prem[i] in seen_tr]
print(f"FACT  test items: {len(te)}; distinct premise strings among them: {len(set(prem[i] for i in te))}")
print(f"FACT  test items whose exact premise string also occurs in a training or validation item: {len(leak_trval)} of {len(te)} ({len(leak_trval) / len(te):.3f}); in a training item alone: {len(leak_tr)}")
allp = collections.Counter(prem)
multi = sum(1 for c in allp.values() if c > 1)
print(f"FACT  all {len(prem)} items: {len(allp)} distinct premises, {multi} premises used by more than one item (the pooled count is the ceiling for what a uid split can leak)")
split_of = {}
for nm, idx in (("train", tr), ("val", va), ("test", te)):
    for i in idx:
        split_of.setdefault(prem[i], set()).add(nm)
straddle = sum(1 for s in split_of.values() if len(s) > 1)
print(f"FACT  premises whose items lie in more than one of the three splits: {straddle} of {len(allp)}")
print("\n" + ("ALL CHECKS PASS" if ok else "SPLIT SIZES DIFFER FROM THE COMMITTED ONES"))
sys.exit(0 if ok else 1)
