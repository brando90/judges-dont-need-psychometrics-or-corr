"""Check that the numbers of each memo line occur together in one repository report.

Usage (from the repository root):
    python3 -I paper_latex_and_notes/ICML_format_tech_memo_11-25-2026/check_numbers.py [--verbose]

The memo sources hold one sentence per line.  For each line the script collects its decimals and its integers of 100 or
more, and looks for a single human-readable file (.md, .txt, .tex under experiments/ and paper_latex_and_notes/, plus the memo's
assets_log.txt; the generated tables/*.tex and the memo's own sources are excluded) that contains ALL of them, either as the same digits or as a number that rounds to the token
at the token's own precision.  A line passes if one file covers all its numbers.  Other lines are listed with the best file
and the numbers that file lacks, for a manual check against the experiment folder.
It checks that the numbers exist together in one report, not that the sentence draws the right conclusion from them.
"""
import bisect
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
VERBOSE = "--verbose" in sys.argv
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def line_tokens(line):
    line = re.sub(r"(?<!\\)%.*$", "", line)
    line = line.replace("{,}", "").replace("\\%", " ").replace("$", " ").replace("\u2212", "-")
    line = re.sub(r"\\(label|ref|eqref|Cref|cref|appref|expref|cite[tp]?|input|includegraphics)(\[[^\]]*\])?\{[^}]*\}", " ", line)
    line = re.sub(r"\\texttt\{[^}]*\}|\\url\{[^}]*\}|\\href\{[^}]*\}", " ", line)
    toks = []
    for m in re.finditer(r"(?<![A-Za-z_\\\d.])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)", line):
        plain = m.group(0).replace(",", "")
        if "." in plain or int(plain) >= 100:
            toks.append(plain)
    return line, sorted(set(toks))


files = []
for r in (os.path.join(ROOT, "experiments"), os.path.join(ROOT, "paper_latex_and_notes")):
    for dp, dn, fn in os.walk(r):
        if os.path.abspath(dp).startswith(HERE):
            continue
        for f in fn:
            if f.endswith((".md", ".txt", ".tex")):
                files.append(os.path.join(dp, f))
files.append(os.path.join(HERE, "assets_log.txt"))

index = {}
for p in files:
    try:
        text = open(p, encoding="utf-8", errors="ignore").read()
    except OSError:
        continue
    exact, fl = set(), []
    for m in NUM.finditer(text):
        t = m.group(0).replace(",", "").lstrip("-").rstrip(".")
        if not t:
            continue
        exact.add(t)
        if "." in t:
            exact.add(t.rstrip("0").rstrip("."))
            try:
                fl.append(float(t))
            except ValueError:
                pass
    index[p] = (exact, np.unique(np.array(fl, dtype=float)))
print(f"indexed {len(index)} report files")


def has(p, tok):
    exact, fl = index[p]
    if tok in exact:
        return True
    if "." in tok and tok.rstrip("0").rstrip(".") in exact:  # strip trailing zeros of decimals only, never of integers
        return True
    if "." not in tok or len(fl) == 0:
        return False
    d = len(tok.split(".")[1])
    x = float(tok)
    half = 0.5 * 10 ** (-d) + 1e-12
    i = bisect.bisect_left(fl, x - half)
    return i < len(fl) and fl[i] <= x + half


sources = [os.path.join(HERE, "memo_11-25-2026.tex")]
sources += [os.path.join(HERE, "sections", f) for f in sorted(os.listdir(os.path.join(HERE, "sections"))) if f.endswith(".tex")]
flagged = 0
checked = 0
for src in sources:
    for ln, raw in enumerate(open(src, encoding="utf-8"), 1):
        line, toks = line_tokens(raw)
        if not toks:
            continue
        checked += 1
        best, best_n = None, -1
        for p in index:
            n = sum(has(p, t) for t in toks)
            if n > best_n:
                best, best_n = p, n
                if n == len(toks):
                    break
        if best_n == len(toks):
            if VERBOSE:
                print(f"ok  {os.path.relpath(src, HERE)}:{ln}  {toks}  <- {os.path.relpath(best, ROOT)}")
            continue
        flagged += 1
        lack = [t for t in toks if not has(best, t)]
        where = {}
        for t in lack:
            where[t] = [os.path.relpath(p, ROOT) for p in index if has(p, t)][:2]
        print(f"\nCHECK {os.path.relpath(src, HERE)}:{ln}  tokens {toks}")
        shown = re.sub(r"\s+", " ", line).strip()[:230]
        print(f"   text: {shown}")
        print(f"   best file ({best_n}/{len(toks)}): {os.path.relpath(best, ROOT)}; it lacks {lack}")
        for t, w in where.items():
            print(f"   {t}: {w if w else 'no report file contains it'}")
print(f"\nchecked {checked} lines with numbers; {flagged} need a manual check")
sys.exit(1 if flagged else 0)
