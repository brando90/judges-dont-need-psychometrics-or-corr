#!/usr/bin/env python3
"""Study B: harder / weaker-signal Phase B of Experiment 04.

CPU only. Writes results/study_b_summary.json, study_b_replications.csv.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gtheory_lib as L

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
MASTER = 20261126

C1 = {"u": 1.0, "a": 0.6, "b": 0.1, "ua": 0.6, "ub": 0.1, "ab": 0.1, "e": 0.5}
C2 = {"u": 1.0, "a": 0.1, "b": 0.6, "ua": 0.1, "ub": 0.6, "ab": 0.1, "e": 0.5}
T_GRID = (0.0, 0.15, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85, 1.0)
PILOTS = ((16, 4, 4), (32, 4, 4), (32, 8, 4), (64, 4, 4), (64, 8, 8))
DESIGNS = ((8, 2), (4, 4), (2, 8))
N_EVAL = 256
N_REPS = 200
N_REPS_REPRO = 256  # t in {0,1} and pilot 64x8x8


def mix_components(t):
    return {c: (1 - t) * C1[c] + t * C2[c] for c in L.COMPS}


def dkey(d):
    return f"{d[0]}x{d[1]}"


def rng_for(*keys):
    return np.random.default_rng(np.random.SeedSequence([int(x) for x in keys]))


def gen(rng, v, n_u, n_a, n_b):
    """Copied from Experiment 04 run_d_study.gen (independent facets, no extras)."""
    sd = {c: math.sqrt(v[c]) for c in L.COMPS}
    U = rng.normal(0, sd["u"], n_u)
    A = rng.normal(0, sd["a"], n_a)
    B = rng.normal(0, sd["b"], n_b)
    UA = rng.normal(0, sd["ua"], (n_u, n_a))
    UB = rng.normal(0, sd["ub"], (n_u, n_b))
    AB = rng.normal(0, sd["ab"], (n_a, n_b))
    E = rng.normal(0, sd["e"], (n_u, n_a, n_b))
    X = (U[:, None, None] + A[None, :, None] + B[None, None, :]
         + UA[:, :, None] + UB[:, None, :] + AB[None, :, :] + E)
    return X, U


def eval_errors(rng, v, design, n_obj=N_EVAL):
    X, U = gen(rng, v, n_obj, design[0], design[1])
    d = X.mean((1, 2)) - U
    return float(np.mean(d ** 2))


def choose(v, cands, key="v_Delta"):
    vals = [L.d_study(v, *d)[key] for d in cands]
    return cands[int(np.argmin(vals))]


def tci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = float(x.mean()) if x.size else float("nan")
    if x.size < 2:
        return {"mean": m, "ci95": [None, None], "se": None, "n": int(x.size)}
    se = float(x.std(ddof=1) / math.sqrt(x.size))
    tcrit = float(stats.t.ppf(0.975, x.size - 1))
    return {"mean": m, "ci95": [m - tcrit * se, m + tcrit * se], "se": se, "n": int(x.size)}


def wilson(k, n):
    if n == 0:
        return {"proportion": None, "k": 0, "n": 0, "ci95": [None, None]}
    z, p = 1.959963984540054, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"proportion": float(p), "k": int(k), "n": int(n), "ci95": [float(c - h), float(c + h)]}


def n_reps_for(t, pilot):
    if t in (0.0, 1.0) and pilot == (64, 8, 8):
        return N_REPS_REPRO
    return N_REPS


def run_cell(t, t_index, pilot, pilot_index):
    v = mix_components(t)
    n_u, n_a, n_b = pilot
    R = n_reps_for(t, pilot)
    oracle = choose(v, DESIGNS)
    pop = {dkey(d): L.d_study(v, *d) for d in DESIGNS}
    rows = []
    n_reml_ok = 0
    for rep in range(R):
        Xc, _ = gen(rng_for(MASTER, 20, t_index, pilot_index, rep, 0), v, n_u, n_a, n_b)
        an = L.anova3(Xc)
        rm = L.reml3(an["ss"], an["df"], n_u, n_a, n_b)
        n_reml_ok += int(rm["success"])
        ch = choose(rm["est"], DESIGNS)
        rng_ev = rng_for(MASTER, 20, t_index, pilot_index, rep, 1)
        obs = {}
        for d in DESIGNS:
            obs[d] = eval_errors(rng_ev, v, d)
        rows.append({
            "t": t, "pilot": dkey(pilot), "n_u": n_u, "n_a": n_a, "n_b": n_b, "rep": rep,
            "reml_success": int(rm["success"]),
            "chosen": dkey(ch),
            "oracle": dkey(oracle),
            "match_oracle": int(ch == oracle),
            "obs_chosen": obs[ch],
            "obs_oracle": obs[oracle],
            "obs_equal": obs[(4, 4)],
            "obs_8x2": obs[(8, 2)],
            "obs_4x4": obs[(4, 4)],
            "obs_2x8": obs[(2, 8)],
            "true_vDelta_chosen": L.d_study(v, *ch)["v_Delta"],
            "true_vDelta_oracle": pop[dkey(oracle)]["v_Delta"],
            "true_vDelta_equal": pop["4x4"]["v_Delta"],
            "regret_true": L.d_study(v, *ch)["v_Delta"] - pop[dkey(oracle)]["v_Delta"],
        })
    return rows, pop, dkey(oracle), n_reml_ok, R


def summarize(all_rows, cell_meta):
    import pandas as pd
    df = pd.DataFrame(all_rows)
    out = {"t_grid": list(T_GRID), "pilots": [dkey(p) for p in PILOTS],
           "designs": [dkey(d) for d in DESIGNS], "cells": {}}
    for (t, pilot), meta in cell_meta.items():
        sub = df[(df.t == t) & (df.pilot == dkey(pilot))]
        key = f"t{t:.2f}_{dkey(pilot)}"
        paired_vs_oracle = sub["obs_chosen"] - sub["obs_oracle"]
        paired_vs_equal = sub["obs_chosen"] - sub["obs_equal"]
        rec = {
            "t": t,
            "pilot": dkey(pilot),
            "n_reps": int(len(sub)),
            "oracle_design": meta["oracle"],
            "population_vDelta": meta["pop_vDelta"],
            "reml_converged": meta["n_reml_ok"],
            "chose_oracle": wilson(int(sub["match_oracle"].sum()), len(sub)),
            "choice_counts": {k: int(v) for k, v in sub["chosen"].value_counts().to_dict().items()},
            "obs_err_chosen": tci(sub["obs_chosen"]),
            "obs_err_oracle": tci(sub["obs_oracle"]),
            "obs_err_equal": tci(sub["obs_equal"]),
            "paired_obs_chosen_minus_oracle": tci(paired_vs_oracle),
            "paired_obs_chosen_minus_equal": tci(paired_vs_equal),
            "true_regret": tci(sub["regret_true"]),
        }
        rec["paired_obs_chosen_minus_oracle"]["ci_excludes_0"] = not (
            rec["paired_obs_chosen_minus_oracle"]["ci95"][0] <= 0 <= rec["paired_obs_chosen_minus_oracle"]["ci95"][1]
        )
        rec["paired_obs_chosen_minus_equal"]["ci_excludes_0"] = not (
            rec["paired_obs_chosen_minus_equal"]["ci95"][0] <= 0 <= rec["paired_obs_chosen_minus_equal"]["ci95"][1]
        )
        if len(sub) > 1:
            rec["paired_obs_chosen_minus_oracle"]["paired_t_pvalue"] = float(
                stats.ttest_1samp(paired_vs_oracle, 0.0).pvalue
            )
            rec["paired_obs_chosen_minus_equal"]["paired_t_pvalue"] = float(
                stats.ttest_1samp(paired_vs_equal, 0.0).pvalue
            )
        out["cells"][key] = rec
    return out, df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    meta = {}
    for ti, t in enumerate(T_GRID):
        for pi, pilot in enumerate(PILOTS):
            print(f"== t={t:.2f} pilot={dkey(pilot)} reps={n_reps_for(t, pilot)} ==", flush=True)
            cell_rows, pop, oracle, n_ok, R = run_cell(t, ti, pilot, pi)
            rows.extend(cell_rows)
            meta[(t, pilot)] = {
                "oracle": oracle,
                "n_reml_ok": n_ok,
                "R": R,
                "pop_vDelta": {k: v["v_Delta"] for k, v in pop.items()},
                "pop_Phi": {k: v["Phi"] for k, v in pop.items()},
            }
    summary, df = summarize(rows, meta)
    summary["runtime_seconds"] = round(time.time() - t0, 2)
    summary["software"] = {"numpy": np.__version__, "scipy": __import__("scipy").__version__, "master_seed": MASTER}
    # reproduction check against Experiment 04 published numbers (C1/C2, 64x8x8, budget 16)
    repro = {}
    for t, label, published in (
        (0.0, "C1", {"oracle": "8x2", "chose_oracle": 256, "n": 256,
                     "paired_vs_equal_mean": -0.107, "paired_vs_equal_ci": [-0.149, -0.065]}),
        (1.0, "C2", {"oracle": "2x8", "chose_oracle": 256, "n": 256,
                     "paired_vs_equal_mean": -0.072, "paired_vs_equal_ci": [-0.108, -0.035]}),
    ):
        key = f"t{t:.2f}_64x8"
        cell = summary["cells"][key]
        repro[label] = {
            "published": published,
            "observed": {
                "oracle": cell["oracle_design"],
                "chose_oracle_k": cell["chose_oracle"]["k"],
                "n": cell["chose_oracle"]["n"],
                "paired_vs_equal": cell["paired_obs_chosen_minus_equal"],
            },
            "oracle_matches": cell["oracle_design"] == published["oracle"],
        }
    summary["reproduction_vs_exp04"] = repro
    df.to_csv(OUT / "study_b_replications.csv", index=False)
    (OUT / "study_b_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("done study B in", summary["runtime_seconds"], "s")


if __name__ == "__main__":
    main()
