#!/usr/bin/env python3
"""Check the coverage matrix against the repository (existence checks only, plus six pinned headline numbers).

    python3 -I verify_coverage_paths.py [--matrix PATH] [--map OLDPREFIX=NEWPREFIX ...]

What it checks (prints PASS or FAIL for each group; the exit code is nonzero on any FAIL):
  1. Every backticked token that starts with `experiments/` or `paper_latex_and_notes/` (or with `scripts/`, `logs/`, `results/`,
     `lean/`, `figures/`, which are resolved inside the experiment-41 folder) names a file or folder that exists in the working tree,
     or, failing that, in `origin/main` (Cursor's experiment 40 is on origin/main until this branch is rebased).
  2. Every backticked identifier with an underscore (for example `plugin_DB_bias`) occurs as a whole word in at least one file of the
     repository (working tree, or origin/main). It reports which identifiers occur ONLY outside .lean files, so a person can check that
     the ones the text calls Lean theorems are in .lean files. This catches a misspelled or invented name; it does not prove the
     sentence that cites it.
  3. The items are numbered 1..N without a gap, and the `Status counts:` line equals the count of `Status:` tags in items 1 to 27.
  4. Six pinned numbers in the text equal what the result files say (maximum absolute matched gaps, the number of views, the
     system-level Spearman pair in experiment 23, and the two minutes-per-rating figures from the sister repository).

`--map` rewrites a path prefix before the lookup; use it only while the experiment folder still has its old number, for example
`--map experiments/41_output_space=experiments/40_output_space`. The final run has no map.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.dirname(HERE)  # the experiment folder that holds this scripts/ directory


def repo_root() -> str:
    d = HERE
    while d != "/" and not os.path.isdir(os.path.join(d, ".git")):
        d = os.path.dirname(d)
    if d == "/":
        raise SystemExit("no .git found above " + HERE)
    return d


ROOT = repo_root()
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def git(*args: str) -> str:
    r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


ORIGIN_FILES = set(git("ls-tree", "-r", "--name-only", "origin/main").splitlines())
ORIGIN_DIRS = set()
for f in ORIGIN_FILES:
    parts = f.split("/")
    for i in range(1, len(parts)):
        ORIGIN_DIRS.add("/".join(parts[:i]))


def exists(rel: str) -> tuple[bool, str]:
    rel = rel.rstrip("/")
    if os.path.exists(os.path.join(ROOT, rel)):
        return True, "working tree"
    if rel in ORIGIN_FILES or rel in ORIGIN_DIRS:
        return True, "origin/main only"
    return False, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=os.path.join(FOLDER, "COVERAGE_MATRIX_11-27-2026.md"))
    ap.add_argument("--map", action="append", default=[])
    a = ap.parse_args()
    maps = [m.split("=", 1) for m in a.map]
    text = open(a.matrix, encoding="utf-8").read()
    folder_rel = os.path.relpath(FOLDER, ROOT)

    # ---- 1. paths -------------------------------------------------------------------------------------------------------------
    tokens = sorted(set(re.findall(r"`([^`\n]+)`", text)))
    path_tokens = []
    for t in tokens:
        if " " in t:
            continue
        if t.startswith(("experiments/", "paper_latex_and_notes/")):
            path_tokens.append(t)
        elif t.startswith(("scripts/", "logs/", "results/", "lean/", "figures/")):
            path_tokens.append(os.path.join(folder_rel, t))
    bad, only_origin = [], []
    for t in path_tokens:
        u = t
        for old, new in maps:
            if u.startswith(old):
                u = new + u[len(old):]
        ok, where = exists(u)
        if not ok:
            bad.append(t)
        elif where == "origin/main only":
            only_origin.append(t)
    check("every cited repository path exists", not bad, f"{len(path_tokens)} paths; missing: {bad}" if bad else f"{len(path_tokens)} paths")
    if only_origin:
        print(f"note  {len(only_origin)} cited paths exist only on origin/main (not yet in this working tree)")

    # ---- 2. identifiers -------------------------------------------------------------------------------------------------------
    idents = sorted({t for t in tokens if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9]+)+", t) and not t.startswith(("experiments", "paper_latex"))})
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(ROOT) for f in fn
             if f.endswith((".lean", ".py", ".json", ".md", ".tex")) and "/.git/" not in dp and "/data_cache" not in dp and "/_build" not in dp]
    blobs: dict[str, str] = {}
    for f in files:
        try:
            blobs[f] = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            pass
    origin_text = git("grep", "-h", "-I", "-E", "-e", "theorem|lemma|def ", "origin/main", "--", "*.lean")
    in_lean, elsewhere, missing = [], [], []
    for name in idents:
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
        # exclude the matrix itself so the document cannot vouch for its own spelling
        hits = [f for f, t in blobs.items() if pat.search(t) and os.path.abspath(f) != os.path.abspath(a.matrix)]
        lean_hit = any(f.endswith(".lean") for f in hits) or bool(pat.search(origin_text))
        if lean_hit:
            in_lean.append(name)
        elif hits:
            elsewhere.append(name)
        else:
            if bool(pat.search(git("grep", "-h", "-I", "-w", name, "origin/main"))):
                elsewhere.append(name)
            else:
                missing.append(name)
    check("every cited identifier occurs in the repository", not missing, f"{len(idents)} identifiers; missing: {missing}" if missing else f"{len(idents)} identifiers")
    print(f"note  identifiers found in a .lean file: {len(in_lean)}; found only in non-Lean files: {elsewhere}")

    # ---- 3. numbering and status counts ---------------------------------------------------------------------------------------
    body = text[text.index("\n## A."):]  # the numbered list in 'How this was built' is not an item
    nums = [int(m) for m in re.findall(r"^(\d+)\. \*\*", body, flags=re.M)]
    check("items are numbered 1..N without a gap", nums == list(range(1, len(nums) + 1)), f"N = {len(nums)}")
    status = re.findall(r"^(\d+)\. \*\*.*?Status: (done|partial|stated-only|not run|conceded)", body, flags=re.M)
    counts = {k: 0 for k in ("done", "partial", "stated-only", "not run", "conceded")}
    for n, s in status:
        if int(n) <= 27:
            counts[s] += 1
    m = re.search(r"Status counts: done (\d+), partial (\d+), stated-only (\d+), not run (\d+), conceded (\d+)", text)
    declared = dict(zip(counts.keys(), map(int, m.groups()))) if m else None
    check("Status counts line equals the tags in items 1 to 27", declared == counts, f"tags {counts}; declared {declared}")
    check("every item from 1 to 27 carries a Status tag", len([n for n, _ in status if int(n) <= 27]) == 27)

    # ---- 4. pinned numbers ----------------------------------------------------------------------------------------------------
    summ = json.load(open(os.path.join(FOLDER, "results", "k_sweep_l2_sensitivity_summary.json")))["summary"]
    views = list(summ)
    mp = max(abs(summ[v]["matched_pearson_minus_ce"]["seed_mean"]) for v in views)
    mm = max(abs(summ[v]["matched_mse_minus_ce"]["seed_mean"]) for v in views)
    check("16 views in the L2-sensitivity summary", len(views) == 16, str(len(views)))
    check("max |matched Pearson-trained minus CE-trained| rounds to 0.0022", round(mp, 4) == 0.0022, f"{mp:.6f}")
    check("max |matched MSE-trained minus CE-trained| rounds to 0.0004", round(mm, 4) == 0.0004, f"{mm:.6f}")
    e23 = open(os.path.join(ROOT, "experiments", "23_public_ordinal_judge_ratings_and_label_cost_start_21Nov2026", "RESULTS_11-21-2026.md"), encoding="utf-8").read()
    check("experiment 23 states system-level Spearman 0.877 versus 0.835 across 16 systems",
          bool(re.search(r"0\.877 versus 0\.835[^.]*16 systems", e23)))
    s15 = json.load(open(os.path.join(FOLDER, "results", "sister_exp15_aggregates_pinned.json")))
    check("sister measured median 4.0 minutes per rating and assumed 37.34",
          s15["measured_minutes_per_rating"]["current_main"]["median_minutes"] == 4.0 and round(s15["assumed_minutes_if_time_scaled_with_length"], 2) == 37.34)
    s36 = open(os.path.join(ROOT, "experiments", "36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026", "SANMI_PITCH.md"), encoding="utf-8").read()
    check("experiment 36 states DICES-990 CE power 0.96 at 100 test prompts and Pearson r power 0.62 at 198",
          "N_test = 100` (power 0.96" in s36 and "at `N_test = 198`, power 0.62" in s36)

    print("\nRESULT:", "FAIL " + str(FAILS) if FAILS else "all checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
