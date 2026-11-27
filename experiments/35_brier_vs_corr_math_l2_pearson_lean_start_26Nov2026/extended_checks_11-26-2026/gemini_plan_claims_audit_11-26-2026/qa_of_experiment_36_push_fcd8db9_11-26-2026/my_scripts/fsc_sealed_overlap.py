#!/usr/bin/env python3
"""How does Experiment 36's FSC split (commit fcd8db9) overlap the sealed split of the sister repo?

Claude, 11-26-2026. Read-only. It (1) runs Experiment 36's own native_main loader against a checkout of the sister repo
(`git show` of the pinned commit, so the bytes are the pinned ones) and checks the counts that the experiment states;
(2) applies Experiment 36's own `split_by_group(groups, seed=0)` to the task names and reports, for each of its
train/validation/test splits, how many tasks the sealed `exp27_split` assigns to train, validation and test.
It prints counts only: no rater name or id.

The sealed split comes from the sister repo's frozen file
experiments/28_multi_family_distributional_judge_and_open_weight_lora_on_new_ratings_start_26Nov2026/data_11-26-2026/task_splits_11-26-2026.csv.
The R_v1 source (brando90/veribench) was not readable from the sandbox, so for R_v1 and the pooled set the script ASSUMES that
the R_v1 task names equal the frozen "old" task names; the pooled split depends only on the sorted set of group strings
("rv1::<task>" and "native::<task>"), so a different naming would change the result. The native_main part has no such assumption.

Usage:
    python3 fsc_sealed_overlap.py SISTER_REPO_ROOT EXPERIMENT_36_FOLDER
where SISTER_REPO_ROOT is a clone of https://github.com/brando90/trustworthy-theorem-judges that contains commit
460466dd10b735c5ff25f9e9cb21f0441fcf960b and the frozen file above (commit 7855746 or later).
Exit status is nonzero when a check fails.
"""
import collections
import csv
import os
import sys
import tempfile

sys.dont_write_bytecode = True
sister = os.path.abspath(sys.argv[1])
e36 = os.path.abspath(sys.argv[2])
os.environ["TTJ_ROOT"] = sister
os.environ["SANMI36_DATA"] = tempfile.mkdtemp(prefix="e36_cache_")
sys.path.insert(0, e36)
import data_io as d  # noqa: E402  (Experiment 36's own loader)

fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""))
    if not ok:
        fails += 1


split_csv = os.path.join(
    sister, "experiments", "28_multi_family_distributional_judge_and_open_weight_lora_on_new_ratings_start_26Nov2026",
    "data_11-26-2026", "task_splits_11-26-2026.csv")
rows = list(csv.DictReader(open(split_csv, newline="")))
sealed = {(r["set"], r["task"]): r["exp27_split"] for r in rows}
for s in ("old", "new"):
    c = collections.Counter(v for (ss, _), v in sealed.items() if ss == s)
    print(f"FACT  sealed split, {s} tasks: " + ", ".join(f"{k} {n}" for k, n in sorted(c.items())))
check("sealed split has 20 test tasks (10 old and 10 new)", sum(1 for v in sealed.values() if v == "test") == 20)

nat = d.load_veribench_fsc_native_main()
n_rat = sum(r.size for r in nat.ratings)
vals = collections.Counter(int(v) for r in nat.ratings for v in r)
ks = collections.Counter(r.size for r in nat.ratings)
n_active = len({x for rr in nat.rater_ids for x in rr})
check("native_main loader: 100 items", nat.n_items == 100, str(nat.n_items))
check("native_main loader: 25 tasks", len(set(nat.groups)) == 25, str(len(set(nat.groups))))
check("native_main loader: 250 ratings", n_rat == 250, str(n_rat))
check("native_main loader: 6 active raters", n_active == 6, str(n_active))
check("native_main loader: share of fives is 0.740", abs(vals[5] / n_rat - 0.740) < 5e-4, f"{vals[5] / n_rat:.3f}")
print("FACT  native_main ratings per item: " + ", ".join(f"{k}: {c}" for k, c in sorted(ks.items())) + " | rating value counts: " + ", ".join(f"{k}: {c}" for k, c in sorted(vals.items())))
new_tasks = sorted({t for (s, t) in sealed if s == "new"})
old_tasks = sorted({t for (s, t) in sealed if s == "old"})
check("native_main task names equal the frozen new task names", set(nat.groups) == set(new_tasks))


def report(label, groups, mapping):
    tr, va, te = d.split_by_group(groups, seed=0)
    print(f"FACT  {label}: Experiment 36's seed-0 group split, tasks per split and the sealed (exp27) membership of those tasks")
    out = {}
    for nm, idx in (("train", tr), ("val", va), ("test", te)):
        gs = sorted({groups[i] for i in idx})
        c = collections.Counter(mapping[g] for g in gs)
        out[nm] = c
        print(f"FACT    {nm:5s}: {len(gs):2d} tasks; sealed train {c.get('train', 0)}, sealed val {c.get('val', 0)}, sealed test {c.get('test', 0)}")
    return out


nat_out = report("native_main only (25 tasks)", new_tasks, {t: sealed[("new", t)] for t in new_tasks})
groups = [f"rv1::{t}" for t in old_tasks] + [f"native::{t}" for t in new_tasks]
mp = {f"rv1::{t}": sealed[("old", t)] for t in old_tasks}
mp.update({f"native::{t}": sealed[("new", t)] for t in new_tasks})
pool_out = report("pooled (50 tasks; ASSUMES the R_v1 task names equal the frozen old task names)", groups, mp)
check("pooled split is 30/10/10 tasks as Experiment 36 states", [sum(pool_out[k].values()) for k in ("train", "val", "test")] == [30, 10, 10])
used = pool_out["train"].get("test", 0) + pool_out["val"].get("test", 0)
print(f"FACT  pooled: {used} of the 20 sealed-test tasks are Experiment 36 training or validation tasks (train {pool_out['train'].get('test', 0)}, validation {pool_out['val'].get('test', 0)})")
used_n = nat_out["train"].get("test", 0) + nat_out["val"].get("test", 0)
print(f"FACT  native_main only: {used_n} of the 10 sealed-test new tasks are Experiment 36 training or validation tasks")
# Item-level counts. native_main is exact (read from the loader). For the pooled set the exact count uses the stated 3 items per old task
# (75 items / 25 tasks; Experiment 36's PROTOCOL says all 3 candidate samples of a task stay together) and the ASSUMPTION that the R_v1 task names
# equal the frozen "old" task names; the pigeonhole bound below needs neither the names nor anything but the item totals.
OLD_ITEMS_PER_TASK = 3
tr_i, va_i, te_i = d.split_by_group(nat.groups, seed=0)
sealed_test_items_new = sum(1 for g in nat.groups if sealed[("new", g)] == "test")
leaked_new = sum(1 for i in list(tr_i) + list(va_i) if sealed[("new", nat.groups[i])] == "test")
print(f"FACT  native_main only, in items: {leaked_new} of the {sealed_test_items_new} sealed-test native_main items are in Experiment 36's training or validation items (its own test split has {len(te_i)} items)")
pool_item_groups = [f"rv1::{t}" for t in old_tasks for _ in range(OLD_ITEMS_PER_TASK)] + [f"native::{g}" for g in nat.groups]
p_tr, p_va, p_te = d.split_by_group(pool_item_groups, seed=0)
check("pooled item-level split is 107 / 34 / 34 items as Experiment 36 states", [len(p_tr), len(p_va), len(p_te)] == [107, 34, 34], f"{len(p_tr)}/{len(p_va)}/{len(p_te)}")
sealed_of_item = [mp[g] for g in pool_item_groups]
sealed_test_items_pool = sum(1 for x in sealed_of_item if x == "test")
leaked_pool = sum(1 for i in list(p_tr) + list(p_va) if sealed_of_item[i] == "test")
print(f"FACT  pooled, in items (assumes the R_v1 task names): {leaked_pool} of the {sealed_test_items_pool} sealed-test items are in Experiment 36's training or validation items (train {sum(1 for i in p_tr if sealed_of_item[i] == 'test')}, validation {sum(1 for i in p_va if sealed_of_item[i] == 'test')}); sealed-test items in its own test split: {sum(1 for i in p_te if sealed_of_item[i] == 'test')} of {len(p_te)}")
print(f"FACT  pooled, pigeonhole bound (no task names needed): the sealed split holds {sealed_test_items_pool} test items and Experiment 36's pooled test split holds {len(p_te)} items, so at least {sealed_test_items_pool - len(p_te)} sealed-test items are in its training or validation items")
print("\n" + ("ALL CHECKS PASS" if fails == 0 else f"{fails} CHECK(S) FAILED"))
sys.exit(1 if fails else 0)
