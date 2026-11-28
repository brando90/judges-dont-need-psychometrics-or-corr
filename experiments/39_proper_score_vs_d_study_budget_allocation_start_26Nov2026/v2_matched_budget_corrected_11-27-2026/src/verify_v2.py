#!/usr/bin/env python3
"""Hard checks for Study A v2 (PROTOCOL_V2_11-27-2026.md section 9). Exit 1 if any check fails."""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_v2 as AN  # noqa: E402
import study_a_v2 as V  # noqa: E402

A = V.A
RES = AN.RES
ROOT39 = V.ROOT39
checks = []


def rec(name, ok, detail=""):
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(("PASS" if ok else "FAIL"), name, detail)


def main():
    bundle = A.load_bundle()
    # 1. splits partition, pilots inside the dev pool (rep 0 and rep 149)
    for name, d in bundle.items():
        te, dv = set(map(int, d["test_idx"])), set(map(int, d["dev_idx"]))
        rec(f"split_partition_{name}", te.isdisjoint(dv) and te | dv == set(range(len(d["ids"]))), f"test {len(te)} dev {len(dv)}")
        k = A.DATASET_K[name]
        for rep in (0, 149):
            pil = set(map(int, A.rng_ss(A.MASTER, 2, k, rep).choice(d["dev_idx"], size=A.N_PILOT_ITEMS, replace=False)))
            rec(f"pilot_in_dev_{name}_rep{rep}", pil <= dv and pil.isdisjoint(te), f"{len(pil)} pilot items")
    # 2. information rule: the proper-score validation is structurally pilot-table-only
    sig = list(inspect.signature(V.ps_validation).parameters)
    src = inspect.getsource(V.ps_validation) + inspect.getsource(V.pilot_counts)
    banned = ["n_avail", "sample_counts", "h_full", "counts_full", "ratings[", "bundle", "data["]
    rec("ps_validation_signature_has_no_rating_pool", sig == ["Q", "K", "pilot_items", "X", "k", "rep"], str(sig))
    rec("ps_validation_source_has_no_rating_pool_access", not any(b in src for b in banned), f"banned tokens absent: {banned}")
    d0 = bundle["chaosnli_snli"]
    Q0 = A.transforms(d0["p_mix"])
    pil0 = A.rng_ss(A.MASTER, 2, 0, 0).choice(d0["dev_idx"], size=A.N_PILOT_ITEMS, replace=False)
    X0 = A.build_pilot_table(d0["ratings"], d0["n_avail"], pil0, A.N_PILOT_RATERS, A.rng_ss(A.MASTER, 3, 0, 0)).astype(int)
    ref = V.ps_validation(Q0, 3, pil0, X0, 0, 0)
    orig = A.sample_counts
    A.sample_counts = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("full rating pool accessed"))
    try:
        again = V.ps_validation(Q0, 3, pil0, X0, 0, 0)
        ok = ref == again
    except RuntimeError as e:  # pragma: no cover
        ok = False
    finally:
        A.sample_counts = orig
    rec("ps_validation_runs_without_full_pool_and_is_deterministic", ok)
    # 3. neutral tie rule: balanced and deterministic
    picks = [V.pick_min_neutral({(16, 4): 0.1, (32, 2): 0.1}, [(16, 4), (32, 2)], A.rng_ss(1, 2, i))[0] for i in range(4000)]
    share = np.mean([p == (32, 2) for p in picks])
    rec("neutral_tie_rule_balanced", 0.46 < share < 0.54, f"share of '32x2' among 4000 seeds: {share:.4f}")
    a1 = V.pick_min_neutral({(16, 4): 0.1, (32, 2): 0.1}, [(16, 4), (32, 2)], A.rng_ss(7, 7, 7))
    a2 = V.pick_min_neutral({(16, 4): 0.1, (32, 2): 0.1}, [(16, 4), (32, 2)], A.rng_ss(7, 7, 7))
    rec("neutral_tie_rule_deterministic", a1 == a2)
    rec("strict_minimum_never_tied", V.pick_min_neutral({(16, 4): 0.1, (32, 2): 0.2}, [(16, 4), (32, 2)], A.rng_ss(3))[0] == (16, 4))
    # 4. Holm against a hand computation: p = [.01,.04,.03,.005] -> [.03,.06,.06,.02]
    rec("holm_hand_example", np.allclose(AN.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02]), str(AN.holm([0.01, 0.04, 0.03, 0.005])))
    rec("holm_monotone_cap", np.allclose(AN.holm([0.5, 0.9]), [1.0, 1.0]) and np.allclose(AN.holm([0.001, 0.9]), [0.002, 0.9]), str(AN.holm([0.5, 0.9])))
    # 5. full-run artifacts
    csv = RES / "study_a_v2_replications.csv"
    if not csv.exists():
        rec("replications_csv_present", False, str(csv))
    else:
        df = pd.read_csv(csv)
        rec("study_a_v2_row_count", len(df) == 150 * 5, f"rows={len(df)}")
        an = json.loads((RES / "analysis_v2.json").read_text())
        rec("rebuild_error_below_1e-10", an["checks"]["max_abs_error_rebuilding_replication_scores_from_per_item_matrices"] < 1e-10,
            str(an["checks"]["max_abs_error_rebuilding_replication_scores_from_per_item_matrices"]))
        rec("ds_u1_equals_fix_raters_100pct", an["checks"]["ds_u1_equals_fix_raters_all_cells"])
        rec("ds_u2_equals_fix_items_100pct", an["checks"]["ds_u2_equals_fix_items_all_cells"])
        rec("pilot_cost_1280", an["pilot_cost_ratings"] == 1280)
        # 6. v1 regression: deployed designs reproduce the committed v1 candidate means exactly
        v1 = json.loads((ROOT39 / "results" / "study_a_summary.json").read_text())
        worst = 0.0
        for name in A.DATASET_ORDER:
            sub = df[df.dataset == name]
            for B in AN.COMP_B:
                for cand, sc in v1["datasets"][name]["budgets"][str(B)]["candidates_scores"].items():
                    for m in ("rps", "ce", "tvd", "rps_mix", "ce_mix", "tvd_mix", "item_mse", "grand_mse"):
                        worst = max(worst, abs(float(sub[f"cand_{cand}_{m}"].mean()) - sc[m]["mean"]))
        rec("deployed_scores_match_v1_summary_means", worst < 1e-12, f"max abs difference {worst:.3e}")
        # 7. reliability-target rule re-derived from the stored pilot components
        bad = 0
        for name in A.DATASET_ORDER:
            sub = df[df.dataset == name]
            for B in AN.COMP_B:
                cands = A.designs_for_budget(B)
                for tgt, tag in ((0.8, "ds_rel80"), (0.9, "ds_rel90")):
                    for _, r in sub.iterrows():
                        d = V.pick_reliability_target({"v_u": r["v_u"], "v_e": r["v_e"]}, cands, tgt)
                        bad += int(f"{d[0]}x{d[1]}" != r[f"B{B}_{tag}"])
        rec("reliability_target_rule_rederived", bad == 0, f"mismatches={bad}")
        # 8. every pilot-using selector picks a feasible design of the right budget
        feas = True
        for B in AN.COMP_B:
            ok_set = {f"{a}x{b}" for a, b in A.designs_for_budget(B)}
            for tag in V_selectors():
                feas &= set(df[f"B{B}_{tag}"]).issubset(ok_set)
        rec("all_selectors_feasible", feas)
    # 9. determinism: rerun two replications twice
    d1 = pd.DataFrame(V.run_dataset("dices350", bundle["dices350"], n_reps=2))
    d2 = pd.DataFrame(V.run_dataset("dices350", bundle["dices350"], n_reps=2))
    rec("rerun_identical", d1.equals(d2))
    (RES / "verification_v2.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    nfail = sum(not c["pass"] for c in checks)
    print(f"{nfail} failed of {len(checks)}")
    return nfail


def V_selectors():
    return ["fix_items", "fix_raters", "ds_u1", "ds_u2", "ds_rel80", "ds_rel90", "ps", "ps_mix", "or_rps", "or_item", "or_grand", "or_rps_mix"]


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
