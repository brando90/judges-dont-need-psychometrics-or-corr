"""Compare the re-run outputs of experiment 40 with the committed ones.

results.json: walks both trees; reports structural differences and the largest absolute numeric
difference (ignoring the wall-clock 'seconds' field). Text outputs: byte-identical or unified diff
line count. PNGs: byte-identical or a pixel comparison (same size, max channel difference).
"""
import difflib
import json
from pathlib import Path

import numpy as np

A = Path("/tmp/opus_review_exp40c/committed/exp40")
B = Path("/tmp/opus_review_exp40c/repo/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026")


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k == "seconds":
                continue
            if k not in a or k not in b:
                out["struct"].append(f"{path}/{k} missing on one side")
                continue
            walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["struct"].append(f"{path} length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = abs(float(a) - float(b))
        out["n_num"] += 1
        if d > out["max"]:
            out["max"], out["argmax"] = d, path
        if d > 1e-6:
            out["over_1e-6"].append((path, a, b))
    else:
        if a != b:
            out["struct"].append(f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}")


ja = json.loads((A / "results.json").read_text())
jb = json.loads((B / "results.json").read_text())
out = {"struct": [], "n_num": 0, "max": 0.0, "argmax": None, "over_1e-6": []}
walk(ja, jb, "", out)
print(f"results.json: numeric leaves compared={out['n_num']}, max |diff|={out['max']:.3e} at {out['argmax']}")
print(f"  leaves differing by >1e-6: {len(out['over_1e-6'])}")
for p, x, y in out["over_1e-6"][:15]:
    print("   ", p, x, y)
print(f"  structural/non-numeric differences: {len(out['struct'])}")
for s in out["struct"][:15]:
    print("   ", s)
for rel in ["REPORT.md", "HOW_TO_READ_RESULTS.md", "results.md", "tables/allocation.md", "tables/dollar_thresholds.md",
            "tables/gstudy.md", "tables/ksweep.md", "tables/paper_master_results.md", "tables/paper_master_results.tex"]:
    ta, tb = (A / rel).read_text(), (B / rel).read_text()
    if ta == tb:
        print(f"{rel}: byte-identical")
    else:
        diff = [l for l in difflib.unified_diff(ta.splitlines(), tb.splitlines(), lineterm="", n=0)
                if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---"))]
        print(f"{rel}: DIFFERS ({len(diff)} changed lines)")
        for l in diff[:6]:
            print("   ", l[:200])
try:
    from PIL import Image
except ImportError:
    Image = None
for png in sorted((A / "figures").glob("*.png")):
    pa, pb = png.read_bytes(), (B / "figures" / png.name).read_bytes()
    if pa == pb:
        print(f"figures/{png.name}: byte-identical")
    elif Image is not None:
        ia, ib = np.asarray(Image.open(png).convert("RGB"), int), np.asarray(Image.open(B / "figures" / png.name).convert("RGB"), int)
        if ia.shape != ib.shape:
            print(f"figures/{png.name}: size {ia.shape} vs {ib.shape}")
        else:
            print(f"figures/{png.name}: bytes differ; max pixel channel diff={np.abs(ia-ib).max()}, "
                  f"pixels differing={int((np.abs(ia-ib).sum(2)>0).sum())}")
    else:
        print(f"figures/{png.name}: bytes differ (PIL unavailable)")
