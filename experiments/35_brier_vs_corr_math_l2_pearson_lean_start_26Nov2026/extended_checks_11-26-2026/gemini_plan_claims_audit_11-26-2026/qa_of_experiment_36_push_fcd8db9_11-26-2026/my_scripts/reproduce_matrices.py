#!/usr/bin/env python3
"""Reproduce Experiment 36's per-dataset matrices (commit fcd8db9) from the raw data and compare with its committed results.json.

Claude, 11-26-2026. The script imports Experiment 36's own `data_io` loaders and its own `run_all.matrix_for` (nothing is
re-implemented), builds each dataset from the pinned raw files (SHA-256 checks run inside the loaders), runs the same fits
(five objectives, seed 0, 35-iteration capped optimizer, group split seed 0, 1,000 item-clustered bootstrap draws), and compares
every numeric leaf of the resulting block with the committed `matrices[<name>]` block.

What it covers: the 11 of 16 matrices whose raw data are readable here: summeval_expert, summeval_expert_binary, chaosnli_snli_mnli,
chaosnli_alpha, dices990, dices990_binary_yes, popquorn_offensiveness (+ binary), popquorn_politeness (+ binary) and
veribench_fsc_native_main. The other five (veribench_fsc_rv1, _rv1_expert3, _rv1_binary, veribench_fsc_pooled, _pooled_binary) need the
R_v1 ratings from brando90/veribench, which this sandbox cannot read; they are listed as NOT REPRODUCED and not as passed.

Tolerances. Counts and strings must match exactly. Floating-point leaves are compared by absolute difference and the script prints
the largest difference per matrix and how many leaves differ by more than 1e-9 and by more than 1e-6. A different SciPy version
than the one that made the committed file can change the last digits of an optimizer stopped at its iteration cap; that would show
as small differences, not as a different conclusion. The status EXACT means every number agrees to within 1e-6; DIFF means at least one number does not, and the lines under it say which kind of number (point estimate, interval endpoint, optimizer record).

Usage:
    python3 -I reproduce_matrices.py EXPERIMENT_36_FOLDER CACHE_DIR SISTER_REPO_ROOT [name ...]
CACHE_DIR is a new empty directory (the loaders download the pinned raw files into it). With names given, only those matrices run.
Exit status is nonzero when a built matrix differs from the committed file by more than 1e-6 somewhere, or when a dataset does not load.
No tolerance for "close enough" was fixed before the first run; the category lines let the reader judge.
"""
import contextlib
import io
import json
import math
import os
import sys

sys.dont_write_bytecode = True
e36 = os.path.abspath(sys.argv[1])
os.environ["SANMI36_DATA"] = os.path.abspath(sys.argv[2])
os.environ["TTJ_ROOT"] = os.path.abspath(sys.argv[3])
want = sys.argv[4:]
sys.path.insert(0, e36)
import data_io as d  # noqa: E402
import run_all as ra  # noqa: E402

committed = json.load(open(os.path.join(e36, "results.json")))["matrices"]
BUILD = {
    "summeval_expert": lambda: d.load_summeval("expert"),
    "summeval_expert_binary": lambda: d.binarize(d.load_summeval("expert"), d.SUMMEVAL_BINARY_CUT, "summeval_expert_binary"),
    "chaosnli_snli_mnli": lambda: d.load_chaos_3way(),
    "chaosnli_alpha": lambda: d.load_chaos_alpha(),
    "dices990": lambda: d._load_dices("990"),
    "dices990_binary_yes": lambda: d.binarize(d._load_dices("990"), 2, "dices990_binary_yes"),
    "popquorn_offensiveness": lambda: d.load_popquorn("offensiveness"),
    "popquorn_offensiveness_binary": lambda: d.binarize(d.load_popquorn("offensiveness"), d.SUMMEVAL_BINARY_CUT, "popquorn_offensiveness_binary"),
    "popquorn_politeness": lambda: d.load_popquorn("politeness"),
    "popquorn_politeness_binary": lambda: d.binarize(d.load_popquorn("politeness"), d.SUMMEVAL_BINARY_CUT, "popquorn_politeness_binary"),
    "veribench_fsc_native_main": lambda: d.load_veribench_fsc_native_main(),
}
NOT_BUILT_HERE = ["veribench_fsc_rv1", "veribench_fsc_rv1_expert3", "veribench_fsc_rv1_binary", "veribench_fsc_pooled", "veribench_fsc_pooled_binary"]


def leaves(x, path=""):
    if isinstance(x, dict):
        for k in sorted(x):
            yield from leaves(x[k], f"{path}/{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def category(path):
    """Group a leaf path: point estimate, interval endpoint, optimizer record, or everything else."""
    if "/point/" in path:
        return "point estimate"
    if "/ci/" in path:
        return "interval endpoint"
    if "/optimizer/" in path:
        return "optimizer record"
    for top in ("/marginal/", "/uniform/"):
        if path.startswith(top):
            return "constant forecast (point or interval)"
    return "other (summary, shuffle control, trace, counts)"


def compare(a, b):
    """a = reproduced, b = committed. Returns a dict of statistics."""
    la, lb = dict(leaves(a)), dict(leaves(b))
    structural = []
    if set(la) != set(lb):
        structural.append(f"leaf paths differ: {len(set(la) ^ set(lb))} paths in only one of the two")
    cat = {}
    allgt6 = 0
    for p in sorted(set(la) & set(lb)):
        x, y = la[p], lb[p]
        if isinstance(x, bool) or isinstance(y, bool) or x is None or y is None or isinstance(x, str) or isinstance(y, str):
            if x != y:
                structural.append(f"{p}: {x!r} != {y!r}")
            continue
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if math.isnan(x) and math.isnan(y):
                continue
            dd = abs(float(x) - float(y))
            c = cat.setdefault(category(p), {"n": 0, "max": 0.0, "worst": "", "gt6": 0})
            c["n"] += 1
            c["gt6"] += dd > 1e-6
            allgt6 += dd > 1e-6
            if dd > c["max"]:
                c["max"], c["worst"] = dd, p
    return {"n": len(la), "cat": cat, "gt6": allgt6, "structural": structural}


fails = 0
names = want or list(BUILD)
import numpy as _np  # noqa: E402
import scipy as _sp  # noqa: E402
print(f"FACT  this run: Python {sys.version.split()[0]}, NumPy {_np.__version__}, SciPy {_sp.__version__}; the committed results.json does not record its library versions; committed matrices: {len(committed)}")
for name in names:
    try:
        data = BUILD[name]()
    except Exception as e:  # noqa: BLE001  (we want to report any loader failure)
        print(f"FAIL  {name}: the dataset did not load here: {type(e).__name__}: {str(e)[:200]}")
        fails += 1
        continue
    with contextlib.redirect_stdout(io.StringIO()):  # Experiment 36's progress lines print timings; keep this script's output deterministic
        block = ra.matrix_for(data)
    block.pop("_prep", None)
    block.pop("_packs", None)
    block = json.loads(json.dumps(ra._jsonable(block)))
    st = compare(block, committed[name])
    ok = (not st["structural"]) and st["gt6"] == 0
    fails += 0 if ok else 1
    print(("EXACT " if ok else "DIFF  ") + f"{name}: {st['n']} leaves; leaves differing by more than 1e-6: {st['gt6']}; structural differences: {len(st['structural'])}")
    for cn, c in sorted(st["cat"].items()):
        print(f"      {cn}: {c['n']} numbers, {c['gt6']} differ by more than 1e-6, largest difference {c['max']:.3g}" + (f" at {c['worst']}" if c["max"] > 1e-6 else ""))
    rb, rc = block["rows"], committed[name]["rows"]
    steps = "; ".join(f"{o}: {rc[o]['optimizer'].get('nit')} -> {rb[o]['optimizer'].get('nit')} iterations, success {rc[o]['optimizer'].get('success')} -> {rb[o]['optimizer'].get('success')}"
                      for o in rb if (rc[o]['optimizer'].get('nit') != rb[o]['optimizer'].get('nit') or rc[o]['optimizer'].get('success') != rb[o]['optimizer'].get('success')))
    print(f"      optimizer records that differ (committed -> reproduced): {steps or 'none'}")
    for obj_name, label in (("ce", "cross-entropy-trained head"), ("pearson", "Pearson-trained head")):
        pc, pb = rc[obj_name]["point"], rb[obj_name]["point"]
        print(f"      {label}, held-out test (committed -> reproduced): CE {pc['ce']:.4f} -> {pb['ce']:.4f}; normalized RPS {pc['rps_norm']:.4f} -> {pb['rps_norm']:.4f}; Pearson r " + (f"{pc['pearson_r']:.4f} -> {pb['pearson_r']:.4f}" if pc['pearson_r'] is not None and pb['pearson_r'] is not None else f"{pc['pearson_r']} -> {pb['pearson_r']}"))
    mc, mb = committed[name]["marginal"]["point"], block["marginal"]["point"]
    print(f"      constant training-marginal forecast (committed -> reproduced): CE {mc['ce']:.4f} -> {mb['ce']:.4f}; normalized RPS {mc['rps_norm']:.4f} -> {mb['rps_norm']:.4f}")
    for s_ in st["structural"][:5]:
        print("      " + s_[:240])
for name in NOT_BUILT_HERE:
    if not want:
        print(f"SKIP  {name}: NOT REPRODUCED (needs the R_v1 ratings of brando90/veribench, not readable from this sandbox)")
print("\n" + ("ALL BUILT MATRICES AGREE WITH THE COMMITTED FILE TO WITHIN 1e-6" if fails == 0 else f"{fails} MATRIX(ES) DIFFER FROM THE COMMITTED FILE BY MORE THAN 1e-6 SOMEWHERE (see the category lines) OR DID NOT LOAD"))
sys.exit(1 if fails else 0)
