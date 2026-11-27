"""Compare two final_results JSON files numerically (ignores eval_seconds); prints max abs diff and count of leaves."""
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
diffs = []; n = [0]; mism = []
def walk(x, y, path):
    if isinstance(x, dict):
        for k in set(x) | set(y if isinstance(y, dict) else {}):
            if k == "eval_seconds": continue
            if not isinstance(y, dict) or k not in x or k not in y: mism.append(path+"/"+str(k)); continue
            walk(x[k], y[k], path+"/"+str(k))
    elif isinstance(x, list):
        if not isinstance(y, list) or len(x) != len(y): mism.append(path); return
        for i, (u, v) in enumerate(zip(x, y)): walk(u, v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        n[0] += 1
        if x != x and y != y: return
        diffs.append((abs(x - y), path))
    else:
        n[0] += 1
        if x != y: mism.append(path)
walk(a, b, "")
diffs.sort(reverse=True); cnt=[d for d in diffs if d[1].split("/")[-1].startswith("n_")]; val=[d for d in diffs if not d[1].split("/")[-1].startswith("n_")]; print("count-field max", cnt[0] if cnt else None, "n count fields differing", sum(1 for d in cnt if d[0]>0)); print("value-field max", val[0] if val else None, "n>1e-9", sum(1 for d in val if d[0]>1e-9))
print("numeric leaves", n[0], "max abs diff", diffs[0] if diffs else None, "structural/string mismatches", len(mism), mism[:5])
