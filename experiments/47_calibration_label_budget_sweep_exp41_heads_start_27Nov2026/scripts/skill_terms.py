#!/usr/bin/env python3
"""Experiment 47, amendment A1 (reporting only, after the advisor asked for skill-score differences; no new fitting):
re-express G(n) and E(n) as skill-score differences against the constant train-marginal forecast.
  S_const = test nRPS of the constant train-marginal forecast (per view, seed); SS(h) = 1 - nRPS(h)/S_const.
  dSS(n) = SS(Pearson-trained head + map on n labels) - SS(CE-trained head raw) = -G(n)/S_const   (negative = CE-trained raw better)
  SS_cal(n) = SS(Pearson-trained head + map on n labels) = -E(n)/S_const;  SS_ce = SS(CE-trained head raw).
Intervals: per-seed paired item-bootstrap CI of G (2,000 draws, from label_budget_rows.json) divided by S_const (base held fixed),
and the 2.5/97.5 percentiles over the 100 seed-draw cells.
Usage: python skill_terms.py RESULTS_DIR   (reads label_budget_rows.json, else the committed label_budget_rows_rounded.json; writes skill_terms.json, skill_terms_tables.md)"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heads41 import build_views, heads, item_nrps_ce  # noqa: E402

res = Path(sys.argv[1])
rf = res / "label_budget_rows.json"
rows = json.loads((rf if rf.exists() else res / "label_budget_rows_rounded.json").read_text())  # committed copy is rounded to 1e-6
base = {}
for vname, b, k, data in build_views():
    for seed in range(5):
        H = heads(vname, data, seed)
        const, _ = item_nrps_ce(np.repeat(H["const"].reshape(1, -1), len(H["test"]), 0), data.ratings, H["test"], k)
        ce, _ = item_nrps_ce(H["ce_pt"], data.ratings, H["test"], k)
        base[(vname, seed)] = (float(const.mean()), float(1 - ce.mean() / const.mean()))
cells = defaultdict(list)
for r in rows:
    cells[(r["design"], r["view"], str(r["n_requested"]))].append(r)
out_rows = []
for (design, view, n), rs in cells.items():
    dss_draws, ss_cal, ss_ce, ci_lo, ci_hi = [], [], [], [], []
    for r in rs:
        sc, sce = base[(view, r["seed"])]
        dss_draws += [-g / sc for g in r["G_draws"]]
        ss_cal.append(-r["E_mean"] / sc); ss_ce.append(sce)
        ci_lo.append(-r["G_ci"][1] / sc); ci_hi.append(-r["G_ci"][0] / sc)
    d = np.array(dss_draws)
    out_rows.append({"design": design, "view": view, "n": n, "signal": rs[0]["signal"], "dSS_mean": float(d.mean()),
                     "dSS_cell_p2.5": float(np.percentile(d, 2.5)), "dSS_cell_p97.5": float(np.percentile(d, 97.5)),
                     "dSS_seed_ci_lo_min": float(min(ci_lo)), "dSS_seed_ci_hi_max": float(max(ci_hi)),
                     "seeds_dSS_ci_below_0": int(sum(h < 0 for h in ci_hi)), "SS_cal_pearson_mean": float(np.mean(ss_cal)),
                     "SS_ce_raw_mean": float(np.mean(ss_ce)), "S_const_mean": float(np.mean([base[(view, s)][0] for s in range(5)]))})
NS = ["25", "50", "100", "200", "all"]
md = ["Skill-score terms (A1). dSS = SS(Pearson-trained + map on n labels) - SS(CE-trained raw); negative = CE-trained raw better. "
      "Cell: seed-draw mean [2.5%, 97.5% over 100 seed-draw cells]. SS_ce = the CE-trained raw head's own skill over the constant.\n"]
for design in ("R1", "RA"):
    md.append(f"\n### Design {design}\n\n| view | SS_ce (raw) | " + " | ".join(f"dSS n={n}" for n in NS) + " | SS of calibrated Pearson head at n=25 |\n|---|---|" + "---|" * len(NS) + "---|")
    for view in dict.fromkeys(r["view"] for r in out_rows):
        g = {r["n"]: r for r in out_rows if r["design"] == design and r["view"] == view}
        md.append(f"| {view} | {g['25']['SS_ce_raw_mean']:+.3f} | " + " | ".join(f"{g[n]['dSS_mean']:+.3f} [{g[n]['dSS_cell_p2.5']:+.3f}, {g[n]['dSS_cell_p97.5']:+.3f}]" for n in NS)
                  + f" | {g['25']['SS_cal_pearson_mean']:+.3f} |")
(res / "skill_terms.json").write_text(json.dumps(out_rows, indent=1))
(res / "skill_terms_tables.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
