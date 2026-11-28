#!/usr/bin/env python3
"""Study A v2: corrected, matched-information re-run of Experiment 39 Study A.

Fixes the four defects in the independent review of commit 1cd4fc6 (see PROTOCOL_V2_11-27-2026.md):
  (a) oracle and selector ties are broken by a declared, seeded, neutral rule (not "more items");
  (b) a D-study selector that really depends on the pilot variance components (reliability target);
  (c) every pilot-using selector sees only the 80 x 16 pilot table (no hidden ratings);
  (d) statistics (two-way bootstrap, Holm) are in analyze_v2.py.

CPU only. No API calls. Reuses the v1 helpers unchanged (src/study_a.py) so that the deployment of every
candidate design is bit-identical to v1 (same seeds), which gives a regression check (verify_v2.py).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT39 = V2.parent
sys.path.insert(0, str(ROOT39 / "src"))
import study_a as A  # noqa: E402  (v1 helpers: transforms, ce_fit_and_score, sample_counts, anova_one_facet, ...)

OUT = Path(os.environ.get("V2_RES_DIR", V2 / "results"))  # override only for smoke tests
MASTER = A.MASTER
N_REPS = int(os.environ.get("V2_NREPS", A.N_REPS))  # protocol value: 150
TOL = A.TOL
DESIGNS = A.DESIGNS
BUDGETS = A.BUDGETS
RHO_TARGETS = (0.80, 0.90)
METRICS = ("rps", "ce", "tvd", "rps_mix", "ce_mix", "tvd_mix", "item_mse", "grand_mse")
ROLE = {"ps": 0, "ps_mix": 1, "or_rps": 2, "or_rps_mix": 3, "or_item": 4, "or_grand": 5}


def dname(d):
    return f"{d[0]}x{d[1]}"


def tie_rng(k, rep, B, role):
    return A.rng_ss(MASTER, 8, k, rep, B, ROLE[role])


def pick_min_neutral(score, designs, rng):
    """Lowest score; exact ties (within TOL) broken uniformly at random with a seeded generator,
    over the tied set sorted by (n_items, n_raters). Returns (design, tied_set)."""
    s = {d: score[d] for d in designs}
    best = min(s.values())
    tied = sorted(d for d in designs if s[d] <= best + TOL)
    return tied[int(rng.integers(len(tied)))], tied


def rho_hat(v_u, v_e, n_r):
    return v_u / (v_u + v_e / n_r)


def pick_reliability_target(vc, cands, target):
    """Smallest n_r among the candidates whose estimated G coefficient reaches the target;
    if none does, the largest n_r. Depends on the pilot estimates (v_u, v_e)."""
    ordered = sorted(cands, key=lambda d: d[1])
    for d in ordered:
        if rho_hat(vc["v_u"], vc["v_e"], d[1]) >= target:
            return d
    return ordered[-1]


def pilot_counts(Xrows, n_r, rng, K):
    """Training counts from the PILOT TABLE ONLY: n_r of the 16 pilot ratings of each selected item."""
    n, width = Xrows.shape
    cc = np.zeros((n, K), dtype=int)
    for t in range(n):
        choose = rng.choice(width, size=n_r, replace=False)
        cc[t] = np.bincount(Xrows[t, choose], minlength=K)
    return cc


def ps_validation(Q, K, pilot_items, X, k, rep):
    """Proper-score selector validation using ONLY the pilot table X [80,16] (ints 0..K-1).
    Returns mean validation RPS per design for the primary head and for the A1 mixture."""
    fold = np.arange(A.N_PILOT_ITEMS) % A.N_FOLDS
    A.rng_ss(MASTER, 6, k, rep).shuffle(fold)
    val = {d: [] for d in DESIGNS}
    val_mix = {d: [] for d in DESIGNS}
    for f in range(A.N_FOLDS):
        val_rows = np.where(fold == f)[0]
        train_rows = np.where(fold != f)[0]
        val_items = pilot_items[val_rows]
        val_h = np.stack([np.bincount(X[r], minlength=K) / X.shape[1] for r in val_rows])
        for d_id, (n_i, n_r) in enumerate(DESIGNS):
            if n_i > len(train_rows):
                continue
            for m in range(A.N_INNER):
                rng = A.rng_ss(MASTER, 4, k, rep, f, d_id, m)
                tr_rows = rng.choice(train_rows, size=n_i, replace=False)
                cc = pilot_counts(X[tr_rows], n_r, rng, K)
                tr_items = pilot_items[tr_rows]
                rps, _, _, _ = A.ce_fit_and_score(Q, tr_items, cc, val_items, val_h)
                val[(n_i, n_r)].append(rps)
                rps_m, _, _, _, _ = A.ce_mix_fit_and_score(Q, tr_items, cc, val_items, val_h)
                val_mix[(n_i, n_r)].append(rps_m)
    mv = {d: (float(np.mean(v)) if v else np.inf) for d, v in val.items()}
    mvm = {d: (float(np.mean(v)) if v else np.inf) for d, v in val_mix.items()}
    return mv, mvm


def run_dataset(name, data, n_reps=N_REPS):
    k = A.DATASET_K[name]
    K = int(data["K"])
    ratings = data["ratings"]
    n_avail = data["n_avail"]
    h_full = data["counts"] / data["counts"].sum(1, keepdims=True)
    universe_mean = np.array([ratings[i, : n_avail[i]].mean() for i in range(len(n_avail))])
    Q = A.transforms(data["p_mix"])
    test_idx = np.asarray(data["test_idx"], int)
    dev_idx = np.asarray(data["dev_idx"], int)
    test_h = h_full[test_idx]
    dev_universe = universe_mean[dev_idx].mean()
    rows = []
    t0 = time.time()
    for rep in range(n_reps):
        pilot_items = A.rng_ss(MASTER, 2, k, rep).choice(dev_idx, size=A.N_PILOT_ITEMS, replace=False)
        X = A.build_pilot_table(ratings, n_avail, pilot_items, A.N_PILOT_RATERS, A.rng_ss(MASTER, 3, k, rep))
        vc = A.anova_one_facet(X)
        Xi = X.astype(int)  # the ONLY rating data any pilot-using selector may touch
        mean_val, mean_val_mix = ps_validation(Q, K, pilot_items, Xi, k, rep)

        deploy = {}
        for d_id, (n_i, n_r) in enumerate(DESIGNS):
            rng = A.rng_ss(MASTER, 5, k, rep, d_id)
            tr = rng.choice(dev_idx, size=n_i, replace=False)
            cc, mean_s, mean_ho, ok = A.sample_counts(ratings, n_avail, tr, n_r, rng, K)
            assert ok.all()
            rps, ce, tvd, ix = A.ce_fit_and_score(Q, tr, cc, test_idx, test_h)
            rps_m, ce_m, tvd_m, ix_m, a_m = A.ce_mix_fit_and_score(Q, tr, cc, test_idx, test_h)
            deploy[(n_i, n_r)] = {
                "rps": rps, "ce": ce, "tvd": tvd, "rps_mix": rps_m, "ce_mix": ce_m, "tvd_mix": tvd_m,
                "item_mse": float(np.mean((mean_s - mean_ho) ** 2)),
                "grand_mse": float((mean_s.mean() - dev_universe) ** 2),
                "transform": ix, "transform_mix": ix_m, "alpha_mix": float(a_m),
            }

        rec = {"dataset": name, "rep": rep, "v_u": vc["v_u"], "v_e": vc["v_e"],
               "ms_u": vc["ms_u"], "ms_e": vc["ms_e"]}
        for d in DESIGNS:
            key = f"cand_{dname(d)}"
            for m in METRICS:
                rec[f"{key}_{m}"] = deploy[d][m]
            rec[f"{key}_transform"] = deploy[d]["transform"]
            rec[f"{key}_alpha_mix"] = deploy[d]["alpha_mix"]
            rec[f"val_{dname(d)}"] = mean_val[d]
            rec[f"valmix_{dname(d)}"] = mean_val_mix[d]
        for B in BUDGETS:
            cands = A.designs_for_budget(B)
            fix_items = max(cands, key=lambda d: d[0])
            fix_raters = max(cands, key=lambda d: d[1])
            u1 = min(cands, key=lambda d: (vc["v_e"] / d[1], -d[0]))  # v1 rule: max raters
            u2 = min(cands, key=lambda d: (vc["v_u"] / d[0] + vc["v_e"] / B, -d[0]))  # v1 rule: max items
            sel = {"fix_items": fix_items, "fix_raters": fix_raters, "ds_u1": u1, "ds_u2": u2}
            for tgt in RHO_TARGETS:
                sel[f"ds_rel{int(round(tgt * 100))}"] = pick_reliability_target(vc, cands, tgt)
            ps, tied_ps = pick_min_neutral(mean_val, cands, tie_rng(k, rep, B, "ps"))
            psm, tied_psm = pick_min_neutral(mean_val_mix, cands, tie_rng(k, rep, B, "ps_mix"))
            orp, tied_orp = pick_min_neutral({d: deploy[d]["rps"] for d in cands}, cands, tie_rng(k, rep, B, "or_rps"))
            orm, tied_orm = pick_min_neutral({d: deploy[d]["rps_mix"] for d in cands}, cands, tie_rng(k, rep, B, "or_rps_mix"))
            ori, tied_ori = pick_min_neutral({d: deploy[d]["item_mse"] for d in cands}, cands, tie_rng(k, rep, B, "or_item"))
            org, tied_org = pick_min_neutral({d: deploy[d]["grand_mse"] for d in cands}, cands, tie_rng(k, rep, B, "or_grand"))
            sel.update({"ps": ps, "ps_mix": psm, "or_rps": orp, "or_rps_mix": orm, "or_item": ori, "or_grand": org})
            for tag, d in sel.items():
                rec[f"B{B}_{tag}"] = dname(d)
            rec[f"B{B}_ntie_ps"] = len(tied_ps)
            rec[f"B{B}_ntie_ps_mix"] = len(tied_psm)
            rec[f"B{B}_ntie_or_rps"] = len(tied_orp)
            rec[f"B{B}_ntie_or_rps_mix"] = len(tied_orm)
            rec[f"B{B}_ntie_or_item"] = len(tied_ori)
            rec[f"B{B}_ntie_or_grand"] = len(tied_org)
            rec[f"B{B}_set_or_rps"] = "|".join(dname(d) for d in tied_orp)
            rec[f"B{B}_set_or_item"] = "|".join(dname(d) for d in tied_ori)
        rows.append(rec)
        if (rep + 1) % 25 == 0 or rep == 0:
            print(f"  {name} rep {rep + 1}/{n_reps}  elapsed {time.time() - t0:.1f}s", flush=True)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = A.load_bundle()
    t0 = time.time()
    rows = []
    for name in A.DATASET_ORDER:
        print(f"== {name} ==", flush=True)
        rows.extend(run_dataset(name, bundle[name], N_REPS))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "study_a_v2_replications.csv", index=False)
    meta = {
        "n_rows": int(len(df)), "runtime_seconds": round(time.time() - t0, 2), "n_reps": N_REPS,
        "master_seed": MASTER, "pilot": {"n_items": A.N_PILOT_ITEMS, "n_raters": A.N_PILOT_RATERS,
                                         "ratings": A.N_PILOT_ITEMS * A.N_PILOT_RATERS},
        "rho_targets": list(RHO_TARGETS),
        "software": {"numpy": np.__version__, "scipy": __import__("scipy").__version__, "pandas": pd.__version__},
    }
    (OUT / "study_a_v2_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("done study A v2 in", meta["runtime_seconds"], "s")


if __name__ == "__main__":
    main()
