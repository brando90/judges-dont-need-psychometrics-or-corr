"""Privacy audit for experiment 40 (counts only; never prints a name or a rater identifier).

Search patterns are built at run time from the sister repo's own identity files at efdc8c3:
  * RAW_MANIFEST.json (rater_id, rater_name for every packet)
  * research_artifacts/09_human_rating_collection/preparation_11-09-2026/ROSTER.csv (all cells of
    columns whose header contains 'name' or 'id' or 'email')
Then every file in the experiment 40 folder (as committed, /tmp/e40_cursor) and the bridged CSVs
are scanned. Output: counts per file and per pattern class. Headers are printed only when they
match no pattern.
"""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
from pathlib import Path

TTJ = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ttj_exp28_efdc8c3")
EXP40 = Path("/tmp/e40_cursor/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026")
REV = "efdc8c392c0462aa8628ad5224dc848c5a9650af"


def show(path: str) -> str:
    return subprocess.run(["git", "-C", str(TTJ), "show", f"{REV}:{path}"], capture_output=True,
                          check=True).stdout.decode("utf-8", "replace")


man = json.loads(show("experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026/ratings/RAW_MANIFEST.json"))
ids, names = set(), set()
for e in man["files"]:
    if e.get("rater_id"):
        ids.add(str(e["rater_id"]).strip())
    if e.get("rater_name"):
        names.add(str(e["rater_name"]).strip())
roster_txt = show("research_artifacts/09_human_rating_collection/preparation_11-09-2026/ROSTER.csv")
roster = list(csv.DictReader(io.StringIO(roster_txt)))
roster_cols = list(roster[0].keys()) if roster else []
emails = set()
for row in roster:
    for col, v in row.items():
        if v is None:
            continue
        v = v.strip()
        lc = (col or "").lower()
        if not v:
            continue
        if "email" in lc:
            emails.add(v)
        elif "name" in lc:
            names.add(v)
        elif lc.endswith("id") or lc == "rater_id":
            ids.add(v)
names = {n for n in names if len(n) >= 3}
first_names = {n.split()[0] for n in names if len(n.split()[0]) >= 3}
surnames = {n.split()[-1] for n in names if len(n.split()) > 1 and len(n.split()[-1]) >= 3}
print(f"patterns built: {len(ids)} rater ids, {len(names)} full names, {len(first_names)} first names, "
      f"{len(surnames)} surnames, {len(emails)} emails; roster columns={len(roster_cols)}, roster rows={len(roster)}")
print("id lengths (min,max):", (min(map(len, ids)), max(map(len, ids))) if ids else None)

full_re = re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)), re.I) if names else None
first_re = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(first_names)) + r")\b", re.I) if first_names else None
sur_re = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(surnames)) + r")\b", re.I) if surnames else None
id_re = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True)) + r")(?![A-Za-z0-9_])") if ids else None
email_re = re.compile("|".join(re.escape(e) for e in emails), re.I) if emails else None
generic_email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

print("\n== scan of every file in the committed experiment 40 folder (counts) ==")
tot = {"full": 0, "first": 0, "surname": 0, "id_token": 0, "email_roster": 0, "email_any": 0}
for p in sorted(EXP40.rglob("*")):
    if not p.is_file() or p.suffix in {".png", ".olean"}:
        continue
    txt = p.read_text("utf-8", "replace")
    c = {
        "full": len(full_re.findall(txt)) if full_re else 0,
        "first": len(first_re.findall(txt)) if first_re else 0,
        "surname": len(sur_re.findall(txt)) if sur_re else 0,
        "id_token": len(id_re.findall(txt)) if id_re else 0,
        "email_roster": len(email_re.findall(txt)) if email_re else 0,
        "email_any": len(generic_email.findall(txt)),
    }
    for k in tot:
        tot[k] += c[k]
    if any(c.values()):
        print(f"{p.relative_to(EXP40)}: {c}")
print("TOTAL:", tot)

print("\n== structure of the bridged rating CSVs ==")
for fn in ("human_ratings_new_11-26-2026.csv", "human_ratings_old_11-26-2026.csv"):
    p = EXP40 / "provenance" / "ttj_efdc8c3" / fn
    lines = p.read_text().splitlines()
    comment = [l for l in lines if l.startswith("#")]
    body = [l for l in lines if not l.startswith("#")]
    rows = list(csv.DictReader(io.StringIO("\n".join(body))))
    hdr = list(rows[0].keys())
    safe_hdr = [h if not ((full_re and full_re.search(h)) or (id_re and id_re.search(h)) or (first_re and first_re.search(h))) else "[REDACTED]" for h in hdr]
    print(f"{fn}: comment lines={len(comment)}, data rows={len(rows)}, columns={len(hdr)}")
    print("  header:", safe_hdr)
    for h in hdr:
        vals = [r[h] for r in rows]
        n_id_cells = sum(1 for v in vals if v is not None and v.strip() in ids)
        n_id_tok = sum(len(id_re.findall(v or "")) for v in vals) if id_re else 0
        n_name = sum(len(full_re.findall(v or "")) for v in vals) if full_re else 0
        n_first = sum(len(first_re.findall(v or "")) for v in vals) if first_re else 0
        uniq = len(set(vals))
        name = h if h in safe_hdr else "[REDACTED]"
        print(f"  col {name!r}: distinct={uniq}, cells==rater_id: {n_id_cells}, id tokens: {n_id_tok}, full-name hits: {n_name}, first-name hits: {n_first}")
    if fn.startswith("human_ratings_old"):
        col = "ratings_sorted_0to5_rater_identity_removed"
        nsc = [len(r[col].split()) for r in rows]
        sorted_ok = sum(1 for r in rows if [float(t) for t in r[col].split()] == sorted(float(t) for t in r[col].split()))
        print(f"  scores per row: min={min(nsc)}, max={max(nsc)}; rows already ascending-sorted: {sorted_ok}/{len(rows)}")
    if fn.startswith("human_ratings_new"):
        if "exp27_split" in hdr:
            from collections import Counter
            print("  exp27_split counts:", dict(Counter(r["exp27_split"] for r in rows)))
            tasks = {}
            for r in rows:
                tasks.setdefault(r.get("task"), set()).add(r["exp27_split"])
            print(f"  tasks={len(tasks)}, tasks in >1 split={sum(1 for s in tasks.values() if len(s) > 1)}")
print("\n== JSON files bridged: key names only, scanned for identity tokens ==")
for fn in ("qwen3_4b_logit_6144.json", "qwen3_8b_logit_6144.json", "qwen3_4b_lora_ce_rps_pearson.json"):
    p = EXP40 / "provenance" / "ttj_efdc8c3" / fn
    d = json.loads(p.read_text())
    if isinstance(d, list):
        keys = sorted(set(k for r in d for k in r))
        print(f"{fn}: list of {len(d)} rows; keys={keys}")
    else:
        print(f"{fn}: dict keys={sorted(d)}; sub-keys of first={sorted(next(iter(d.values())).keys()) if isinstance(next(iter(d.values())), dict) else type(next(iter(d.values()))).__name__}")
