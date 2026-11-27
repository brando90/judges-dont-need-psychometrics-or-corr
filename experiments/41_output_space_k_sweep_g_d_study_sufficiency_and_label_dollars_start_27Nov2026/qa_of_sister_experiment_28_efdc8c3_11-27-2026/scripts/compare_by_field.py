"""Group rerun-vs-committed differences by leaf field name (point metrics, CI bounds, p-values)."""
import json, sys, collections
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
agg = collections.defaultdict(float); ex = {}
def walk(x, y, path, last):
    if isinstance(x, dict):
        for k in x:
            if k == "eval_seconds": continue
            walk(x[k], y[k], path + "/" + k, k)
    elif isinstance(x, list):
        for i, (u, v) in enumerate(zip(x, y)): walk(u, v, path, last + "[ci]" if isinstance(u, float) else last)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        if x != x or y != y: return
        d = abs(x - y)
        key = ("point:" if "/point/" in path else "ci:" if "[ci]" in last else "") + last
        if d > agg[key]: agg[key] = d; ex[key] = path
walk(a, b, "", "")
for k in sorted(agg, key=lambda k: -agg[k])[:25]: print(f"{k:28s} {agg[k]:.2e}  {ex[k]}")
