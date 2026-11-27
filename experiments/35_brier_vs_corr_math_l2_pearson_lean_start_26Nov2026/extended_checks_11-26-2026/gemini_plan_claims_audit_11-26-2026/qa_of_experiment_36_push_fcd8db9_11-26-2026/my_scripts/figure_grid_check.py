#!/usr/bin/env python3
"""Regenerate Experiment 36's `make_all` figures in a scratch directory and test every axis for gridlines.

Claude, 11-26-2026. Read-only on the repo. The project rule is `ax.grid(True)` on every axis of every figure. Static
reading of `make_figures.py` cannot show that (a helper `_style(ax)` applies the grid), so this script wraps
`Figure.savefig`, and for each saved figure inspects every visible axis that holds content: are the x and y gridlines
visible, and does it have an x label and a y label? Twin axes (a second y axis that shares the x axis) have no x label of
their own; they are counted as content axes without an x label and listed, not failed.
It covers the eight figures made by `make_all`. The two `eval_power_mdd_*.png` figures come from `eval_power_mdd.py`, which
calls its own `_style(ax)` inside the plotting loop; they are not instrumented here.

Usage:
    python3 figure_grid_check.py EXPERIMENT_36_FOLDER SCRATCH_DIR
Reads EXPERIMENT_36_FOLDER/results.json and writes PNGs only into SCRATCH_DIR. Exit status is nonzero when a content axis
lacks a gridline. Colorbar axes (matplotlib labels them "<colorbar>") are colour scales, not data plots; they carry no gridlines
in these figures and are counted separately and not failed. Whether the project's "every axis" wording should be read to include a
colorbar is Brando's call; a grid drawn across a colour scale adds no information.
"""
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
e36 = os.path.abspath(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.figure as mf  # noqa: E402

sys.path.insert(0, e36)
import make_figures as mk  # noqa: E402

orig = mf.Figure.savefig
seen = []


def patched(self, path, *a, **k):
    name = os.path.basename(str(path))
    info = []
    for ax in self.get_axes():
        if not ax.get_visible():
            continue
        gx = any(line.get_visible() for line in ax.get_xgridlines())
        gy = any(line.get_visible() for line in ax.get_ygridlines())
        content = bool(ax.lines or ax.collections or ax.patches or ax.images)
        # A colorbar is a colour scale, not a data plot: matplotlib labels its axes "<colorbar>" and keeps a `_colorbar` handle on it.
        # (A first version of this script looked for an image on an unlabelled axis, missed the colorbars, and reported them as failures.)
        colorbar = ax.get_label() == "<colorbar>" or hasattr(ax, "_colorbar")
        info.append((gx, gy, content, bool(ax.get_xlabel()), bool(ax.get_ylabel()), colorbar))
    seen.append((name, info))
    return orig(self, str(out / name), *a, **k)


mf.Figure.savefig = patched
payload = json.load(open(os.path.join(e36, "results.json")))
mk.make_all(payload, out)
fails = 0
print(f"FACT  figures written by make_all: {len(seen)}")
for name, info in seen:
    no_grid = sum(1 for (gx, gy, c, xl, yl, cb) in info if c and not (gx and gy) and not cb)
    no_x = sum(1 for (gx, gy, c, xl, yl, cb) in info if c and not xl and not cb)
    no_y = sum(1 for (gx, gy, c, xl, yl, cb) in info if c and not yl and not cb)
    n_cb = sum(1 for t in info if t[5])
    n_data = sum(1 for (gx, gy, c, xl, yl, cb) in info if c and not cb)
    ok = no_grid == 0
    fails += 0 if ok else 1
    print(("PASS  " if ok else "FAIL  ") + f"{name}: {len(info)} axes = {n_data} data axes + {n_cb} colorbar axes (colorbars have no gridlines and are not counted); data axes without both gridlines {no_grid}; without an x label {no_x} (twin axes allowed); without a y label {no_y}")
print("\n" + ("ALL CHECKS PASS" if fails == 0 else f"{fails} FIGURE(S) HAVE AN AXIS WITHOUT GRIDLINES"))
sys.exit(1 if fails else 0)
