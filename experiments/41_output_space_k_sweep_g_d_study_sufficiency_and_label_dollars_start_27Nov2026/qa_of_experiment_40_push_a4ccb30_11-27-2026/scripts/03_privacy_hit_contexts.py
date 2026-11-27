"""Classify the name-pattern hits of 02_privacy_counts.py without printing a name.

For each hit: file, line, match length, whether the pattern is a multi-token name, whether the
hit is case-exact and whole-word, and the surrounding text with the match masked as <HIT>.
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
names = {str(e["rater_name"]).strip() for e in man["files"] if e.get("rater_name")}
for row in csv.DictReader(io.StringIO(show("research_artifacts/09_human_rating_collection/preparation_11-09-2026/ROSTER.csv"))):
    for col, v in row.items():
        if v and "name" in (col or "").lower() and "email" not in (col or "").lower():
            names.add(v.strip())
names = {n for n in names if len(n) >= 3}
multi = {n for n in names if len(n.split()) > 1}
firsts = {n.split()[0] for n in names if len(n.split()[0]) >= 3}
pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names | firsts, key=len, reverse=True)) + r")\b", re.I)
summary = {"multi_token_exact_case": 0, "single_token_exact_case": 0, "case_insensitive_only": 0}
for p in sorted(EXP40.rglob("*")):
    if not p.is_file() or p.suffix in {".png"}:
        continue
    for ln, line in enumerate(p.read_text("utf-8", "replace").splitlines(), 1):
        for m in pat.finditer(line):
            s = m.group(0)
            exact = s in names or s in firsts
            kind = "multi" if s in multi else "single"
            if exact and kind == "multi":
                summary["multi_token_exact_case"] += 1
            elif exact:
                summary["single_token_exact_case"] += 1
            else:
                summary["case_insensitive_only"] += 1
            ctx = line[max(0, m.start() - 40):m.start()] + "<HIT>" + line[m.end():m.end() + 40]
            if exact:
                print(f"{p.relative_to(EXP40)}:{ln} len={len(s)} {kind} exact-case :: {ctx!r}")
print("summary:", summary)
