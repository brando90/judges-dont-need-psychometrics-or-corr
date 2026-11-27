"""Audit helper (Claude, 11-26-2026).
Usage: python3 compare_results_json.py <git repo> <commit> <path of results.json inside the repo> <reproduced results.json>
Compares every numeric leaf of the committed file with a reproduced file. Prints leaf counts, the maximum
absolute difference, how many leaves differ after rounding to four decimals (the precision of the printed
tables), and the metric names involved."""
import collections, json, math, subprocess, sys
repo, commit, rel, new_path = sys.argv[1:5]
old = json.loads(subprocess.check_output(["git", "-C", repo, "show", f"{commit}:{rel}"]))
new = json.load(open(new_path))
rows, struct = [], []
def walk(a, b, p=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b: struct.append(("key", p + "/" + k)); continue
            walk(a[k], b[k], p + "/" + k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): struct.append(("len", p)); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{p}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b): rows.append((0.0, p, a, b))
        else: rows.append((abs(a - b), p, a, b))
    elif a != b: struct.append(("val", p, str(a)[:60], str(b)[:60]))
walk(old, new)
diff = [r for r in rows if r[0] > 0]
chg4 = [r for r in diff if round(r[2], 4) != round(r[3], 4)]
print(f"numeric leaves compared: {len(rows)} | differing at all: {len(diff)} | differing after rounding to 4 decimals: {len(chg4)}")
print(f"maximum absolute difference: {max([r[0] for r in rows] + [0.0]):.3e}")
print(f"string/structural differences: {len(struct)} {struct[:3]}")
print("metric names among 4-decimal changes:", dict(collections.Counter(r[1].split('/')[-1] for r in chg4)))
for name in ("mean", "lo", "hi", "linear_cka", "normalized_opd", "pearson_expected_rating", "rbf_cka", "cca_sq", "validation_ce", "fun"):
    rr = [r for r in diff if r[1].endswith("/" + name)]
    if rr: print(f"  {name}: {len(rr)} leaves differ, max abs difference {max(r[0] for r in rr):.3e}")
