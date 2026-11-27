"""Audit H4 jsonl files: statuses, repeats per item, parse failures, missing items, fields present (no rater info printed)."""
import json, sys, collections
from pathlib import Path
E = Path(sys.argv[1])
for lane in ("codex_astra", "agy_argon", "agy_opus"):
    p = E / "results" / f"h4_{lane}_11-26-2026.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    st = collections.Counter(r.get("status") for r in rows)
    items = collections.Counter(r["id"].split("#r")[0] for r in rows if r.get("status") == "ok")
    allitems = {r["id"].split("#r")[0] for r in rows}
    keys = collections.Counter(k for r in rows for k in r.keys())
    models = collections.Counter(str(r.get("model") or r.get("lane") or r.get("cli")) for r in rows)
    sums = [abs(sum(r["parsed"]["probs"]) - 1) for r in rows if r.get("status") == "ok"]
    print(lane, "rows", len(rows), "status", dict(st), "ok_items", len(items), "all_items", len(allitems),
          "repeats_dist", dict(collections.Counter(items.values())), "max|sum-1|", max(sums))
    print("   keys", dict(keys)); print("   model field", dict(models))
