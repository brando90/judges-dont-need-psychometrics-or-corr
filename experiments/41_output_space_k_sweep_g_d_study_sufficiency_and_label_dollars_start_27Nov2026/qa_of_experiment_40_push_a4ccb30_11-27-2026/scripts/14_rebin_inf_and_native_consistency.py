"""(1) Count real-rebin rows whose MSE-head cross-entropy is infinite, and rows where the point is
infinite but a finite interval was printed (percentile_interval drops non-finite replicates).
(2) Check that the VeriBench native_main ratings loaded from trustworthy-theorem-judges@460466d
agree, item by item, with the sorted scores in the bridged efdc8c3 CSV whose exp27_split defines
the split (counts only; no identifiers printed).
Usage: python3 14_rebin_inf_and_native_consistency.py <exp40_dir>
"""
import csv
import json
import sys
from pathlib import Path

EXP40 = Path(sys.argv[1])
res = json.loads(Path("/tmp/e40_cursor/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/results.json").read_text())
rows = res["rebin"]
inf_point = [r for r in rows if r["mse"]["ce"] is None]
inf_with_finite_ci = [r for r in inf_point if r["mse"]["ce_ci"]["lo"] is not None]
print(f"(1) real rebin rows: {len(rows)}; MSE-head CE infinite (stored as null): {len(inf_point)}; "
      f"of which a finite 95% interval was printed anyway: {len(inf_with_finite_ci)}")
for r in inf_with_finite_ci:
    print(f"    {r['dataset']} {r['mode']} k={r['k_ratings']}: interval computed from {r['mse']['ce_ci']['n_finite']} of 1000 finite replicates")
dup = sum(1 for r in rows if r["dataset"] == "dices990" and r["mode"] == "native")
print(f"    DICES native rows duplicating the K3 rows (native K = 3): {dup}")

sys.path.insert(0, str(EXP40))
import run_all as ra  # noqa: E402
data = ra.load_veribench_fsc_native_main()
prov = EXP40 / "provenance" / "ttj_efdc8c3" / "human_ratings_new_11-26-2026.csv"
csv_rows = {r["item"]: r for r in csv.DictReader(prov.open())}
# The loader keeps the text and task folder per artifact; match artifacts on (task, sorted scores multiset)
# is weak, so first try item-id alignment through the group/task and rating counts.
by_task_loader = {}
for g, ys in zip(data.groups, data.ratings):
    by_task_loader.setdefault(g, []).append(tuple(sorted(int(y) for y in ys)))
by_task_csv = {}
for r in csv_rows.values():
    by_task_csv.setdefault(r["task"], []).append(tuple(sorted(int(float(t)) for t in r["ratings_sorted_0to5_rater_identity_removed"].split())))
tasks = set(by_task_loader) | set(by_task_csv)
same_multiset = sum(1 for t in tasks if sorted(by_task_loader.get(t, [])) == sorted(by_task_csv.get(t, [])))
print(f"(2) native_main: loader items={data.n_items}, ratings={sum(r.size for r in data.ratings)}; "
      f"CSV items={len(csv_rows)}, ratings={sum(int(r['n_ratings']) for r in csv_rows.values())}")
print(f"    tasks={len(tasks)}; tasks whose per-artifact sorted-score multisets are identical in both sources: {same_multiset}")
