"""Audit the committed results.json of experiment 40 (no recomputation; cheap).

Checks: budget binding (ratings_used vs B), D-study feasibility per budget, same-k counts,
how many paired D-study-minus-oracle intervals exclude 0, validation-chosen k, hard-coded REPORT
numbers (17, 10, Phi(10)=0.7875, VeriBench B=64 row, dollar hits), and that the oracle difference
is >= 0 by construction.
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/tmp/e40_cursor/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/results.json")
d = json.loads(path.read_text())
print("file:", path)
n_cells_total = n_diff = n_excl0 = n_pos_excl0 = 0
n_same = n_rows = n_dinfeasible = 0
n_unbound = 0
for name, b in d["datasets"].items():
    g = b["gstudy_train"]
    rec = g.get("recommendation", {})
    print(f"\n=== {name}: n_train={b['n_train']} n_val={b['n_val']} n_test={b['n_test']} facet={b['facet']}")
    print(f"  train G-study: status={g.get('status')} s_c2_raw={g.get('sigma_c2_raw')} s_r2_raw={g.get('sigma_r2_raw')} "
          f"s_res2={g.get('sigma_res2')} median ratings/item={g.get('ratings_per_item_median')}")
    print(f"  D-study rec: k={rec.get('k')} Phi={rec.get('Phi')} hit={rec.get('target_hit')} reason={rec.get('reason')}")
    cells = b["allocation"]["cells"]
    print("  fitted (k,N,ratings_used):", [(c["k"], c["N"], c["ratings_used"]) for c in cells])
    for row in b["allocation"]["budgets"]:
        n_rows += 1
        B = row["B"]
        opts = [c for c in cells if c["N"] == min(c["N"], B // c["k"])]  # informational only
        dcell, o, v = row["dstudy"], row["oracle_ce"], row["validation_chosen"]
        diff = row["dstudy_minus_oracle_ce"]
        if dcell is None:
            n_dinfeasible += 1
        if row["same_k"]:
            n_same += 1
        s = ""
        if diff:
            n_diff += 1
            excl = diff["lo"] > 0 or diff["hi"] < 0
            n_excl0 += excl
            n_pos_excl0 += diff["lo"] > 0
            assert diff["point"] >= -1e-15, "oracle difference should be >= 0 by construction"
            s = f"diff={diff['point']:.4f} [{diff['lo']:.4f},{diff['hi']:.4f}]{' EXCLUDES 0' if excl else ''}"
        print(f"  B={B:5d}: D k={None if not dcell else dcell['k']} (N={None if not dcell else dcell['N']}, used={None if not dcell else dcell['ratings_used']}) "
              f"| oracle k={o['k']} (N={o['N']}, used={o['ratings_used']}) | val k={v['k']} (N={v['N']}, used={v['ratings_used']}) | {s}")
    # budget binding: for each budget, list options whose ratings_used < B/2
    for row in b["allocation"]["budgets"]:
        B = row["B"]
        unbound = [(c["k"], c["N"], c["ratings_used"]) for c in cells
                   if (c["N"] == min(max(cc["N"] for cc in cells if cc["k"] == c["k"]), B // c["k"])) and c["ratings_used"] < 0.75 * B]
        if unbound:
            n_unbound += 1
            print(f"    budget not binding at B={B}: cells using < 75% of B (k,N,used) = {unbound}")
print("\n=== totals")
print(f"budget rows={n_rows}; D-study infeasible rows={n_dinfeasible}; same-k rows={n_same}; paired diffs={n_diff}; "
      f"intervals excluding 0={n_excl0} (positive side {n_pos_excl0}); rows with non-binding budget for some k={n_unbound}")
# Hard-coded executive numbers.
pq = d["datasets"]["popquorn_offensiveness"]["gstudy_train"]
curve = {r["k"]: r["Phi"] for r in pq["curve"]}
print("POPQUORN offensiveness train curve stored k:", sorted(curve), "-> Phi(10) stored?", 10 in curve)
print("POPQUORN off recommendation Phi:", pq["recommendation"].get("Phi"))
print("DICES rec k:", d["datasets"]["dices990"]["gstudy_train"]["recommendation"]["k"])
vb = d["datasets"]["veribench_fsc_native_main"]["allocation"]["budgets"]
for row in vb:
    print("VeriBench B", row["B"], "D", row["dstudy"]["k"] if row["dstudy"] else None, "oracle", row["oracle_ce"]["k"],
          "diff", row["dstudy_minus_oracle_ce"])
print("\n=== dollar hits")
for name, b in d["datasets"].items():
    h = b["dollars"]
    print(f"{name}: B_ce={h.get('B_ce')} B_sqrt={h.get('B_sqrt_rps')} full CE={h.get('full_train_ce'):.4f} "
          f"first budget={b['allocation']['budgets'][0]['B']} frontier CE at first budget={b['allocation']['budgets'][0]['frontier_ce']:.4f} target={h.get('ce_target'):.4f}")
