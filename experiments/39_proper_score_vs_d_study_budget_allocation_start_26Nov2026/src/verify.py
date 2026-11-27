#!/usr/bin/env python3
"""Sanity and reproducibility checks. Exit 1 if a hard check fails."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_a as A
import study_b as B

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
checks = []


def rec(name, ok, detail):
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(("PASS" if ok else "FAIL"), name, detail)


def main():
    bundle = A.load_bundle()
    # 1. splits are disjoint and cover all items
    for name, d in bundle.items():
        te, dv = set(d["test_idx"]), set(d["dev_idx"])
        n = len(d["ids"])
        rec(f"split_partition_{name}", te.isdisjoint(dv) and te | dv == set(range(n)),
            f"test {len(te)} dev {len(dv)} n {n}")
        rec(f"enough_dev_for_pilot_{name}", len(dv) >= 80,
            f"dev={len(dv)}")
        rec(f"enough_ratings_{name}", int(d["n_avail"].min()) >= 16,
            f"min ratings {int(d["n_avail"].min())}")

    # 2. no leakage: test ids never in a deployment train draw for one seeded rep
    name = "chaosnli_snli"
    d = bundle[name]
    k = A.DATASET_K[name]
    test = set(map(int, d["test_idx"]))
    dev = np.asarray(d["dev_idx"])
    leak = 0
    for d_id in range(len(A.DESIGNS)):
        rng = A.rng_ss(A.MASTER, 5, k, 0, d_id)
        n_i = A.DESIGNS[d_id][0]
        tr = set(map(int, rng.choice(dev, size=n_i, replace=False)))
        leak += len(tr & test)
    rec("deploy_train_avoids_test_rep0_snli", leak == 0, f"overlap={leak}")

    # 3. seed determinism: two calls of anova on the same pilot table
    rng = A.rng_ss(A.MASTER, 2, 0, 0)
    items = rng.choice(bundle["chaosnli_snli"]["dev_idx"], size=80, replace=False)
    rng1 = A.rng_ss(A.MASTER, 3, 0, 0)
    rng2 = A.rng_ss(A.MASTER, 3, 0, 0)
    X1 = A.build_pilot_table(d["ratings"], d["n_avail"], items, 16, rng1)
    X2 = A.build_pilot_table(d["ratings"], d["n_avail"], items, 16, rng2)
    rec("pilot_table_deterministic", np.array_equal(X1, X2), "two draws with the same seed")

    # 4. U1 always selects max n_r (analytic property when v_e>0)
    a = json.loads((RES / "study_a_summary.json").read_text())
    u1_ok = True
    detail = []
    for name in A.DATASET_ORDER:
        for B, cell in a["datasets"][name]["budgets"].items():
            mode = cell["selection"]["u1"]["mode"]
            nr = int(mode.split("x")[1])
            max_nr = max(int(c.split("x")[1]) for c in cell["candidates"])
            if nr != max_nr or cell["selection"]["u1"]["counts"].get(mode, 0) != a["n_reps"]:
                u1_ok = False
                detail.append(f"{name} B={B} mode={mode} counts={cell['selection']['u1']['counts']}")
    rec("u1_always_max_raters", u1_ok, "; ".join(detail) if detail else "all 150/150 cells")

    # 5. Study A CSV row count
    df = pd.read_csv(RES / "study_a_replications.csv")
    rec("study_a_row_count", len(df) == 150 * 5, f"rows={len(df)}")

    # 6. Study B reproduction oracles
    b = json.loads((RES / "study_b_summary.json").read_text())
    rec("repro_C1_oracle", b["reproduction_vs_exp04"]["C1"]["oracle_matches"],
        b["reproduction_vs_exp04"]["C1"]["observed"])
    rec("repro_C2_oracle", b["reproduction_vs_exp04"]["C2"]["oracle_matches"],
        b["reproduction_vs_exp04"]["C2"]["observed"])
    rec("repro_C1_n256", b["reproduction_vs_exp04"]["C1"]["observed"]["n"] == 256, "")
    rec("repro_C2_n256", b["reproduction_vs_exp04"]["C2"]["observed"]["n"] == 256, "")

    # 7. At t=0.5, 4x4 is uniquely optimal (v_Delta 0.3875 vs 0.475 for 8x2 and 2x8).
    # The protocol's "equal at t=1/2" claim was incorrect; the true crossovers are t=4/15 and 11/15.
    mid_ok, mid_detail = True, []
    for pilot in b["pilots"]:
        cell = b["cells"][f"t{0.50:.2f}_{pilot}"]
        vd = cell["population_vDelta"]
        if cell["oracle_design"] != "4x4" or abs(vd["4x4"] - 0.3875) > 1e-12 or abs(vd["8x2"] - 0.475) > 1e-12 or abs(vd["2x8"] - 0.475) > 1e-12:
            mid_ok = False
            mid_detail.append((pilot, cell["oracle_design"], vd))
    rec("t0.5_4x4_unique_oracle", mid_ok, mid_detail or "v_Delta 0.3875 vs 0.475")
    rec("crossovers_oracles",
        b["cells"]["t0.15_64x8"]["oracle_design"] == "8x2"
        and b["cells"]["t0.30_64x8"]["oracle_design"] == "4x4"
        and b["cells"]["t0.70_64x8"]["oracle_design"] == "4x4"
        and b["cells"]["t0.85_64x8"]["oracle_design"] == "2x8",
        "8x2 at 0.15, 4x4 at 0.30-0.70, 2x8 at 0.85")

    (RES / "verification.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    nfail = sum(not c["pass"] for c in checks)
    print(f"{nfail} failed of {len(checks)}")
    return nfail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
