#!/usr/bin/env python3
"""Experiment 48, amendment A1 (reporting only; decision rules unchanged): the shift results as skill-score differences.
Arm A: SS = 1 - S/S_base with S_base the target's cross-fitted marginal (Experiment 46 base), same plan, same bootstrap seeds
(SeedSequence([20261127, 48, ti]), 4,000 group draws) as calibration_shift.py, base resampled with the items.
  dSS_shift_vs_in = SS(shifted map) - SS(in-domain cross-fitted map)   (negative = gap reopened)
  dSS_shift_vs_raw = SS(shifted map) - SS(raw judge)                   (negative = calibration made the judge worse)
Arm B: SS = 1 - nRPS/S_const (constant train-marginal forecast, per view and seed); dSS = -G/S_const per seed with the
per-seed paired item-bootstrap CI of G divided by S_const (base held fixed).
Usage: python skill_terms.py LABELS_JSON SUMMEVAL_INPUT_DIR RESULTS_DIR"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_shift as CS  # noqa: E402
A = CS.A

def arm_a(labels_json, se_dir):
    items, counts, groups, fc = A.load_all(labels_json, se_dir, "/nonexistent")
    def mats(s, judge):
        ids = items[s]
        return ids, np.stack([counts[i] for i in ids]), A.floor(np.stack([fc[judge][i][0] for i in ids])), [groups[i] for i in ids]
    plan = [(j, src, tgt) for j in A.FRONTIER for src in CS.K3 for tgt in CS.K3 if src != tgt]
    plan += [(j, "pooled_other_three", tgt) for j in A.FRONTIER for tgt in CS.K3]
    plan += [("g-eval-gpt-4|sampled20", "summeval_relevance", "summeval_consistency"), ("g-eval-gpt-4|sampled20", "summeval_consistency", "summeval_relevance")]
    rows = []
    for ti, (judge, src, tgt) in enumerate(plan):
        ids, C, Q, g = mats(tgt, judge); H = C / C.sum(1, keepdims=True); prim = A.PRIMARY[tgt]
        fold = A.folds_for(g, A.SETS.index(tgt)); qin = np.zeros_like(Q); base = np.zeros_like(H)
        for f in range(5):
            tr, te = fold != f, fold == f
            qin[te] = A.apply_cal(A.fit_cal(Q[tr], C[tr]), Q[te]); m = C[tr].sum(0); base[te] = A.floor(m / m.sum())
        if src == "pooled_other_three":
            srcs = [s for s in CS.K3 if s != tgt]
            Qs = np.concatenate([mats(s, judge)[2] for s in srcs]); Cs = np.concatenate([mats(s, judge)[1] for s in srcs])
        else:
            _, Cs, Qs, _ = mats(src, judge)
        qsh = A.apply_cal(A.fit_cal(Qs, Cs), Q)
        sc = {k: A.item_scores(v, H)[prim] for k, v in (("raw", Q), ("in", qin), ("shift", qsh), ("base", base))}
        ug, ginv = np.unique(g, return_inverse=True)
        rng = np.random.default_rng(np.random.SeedSequence([CS.SEED, 48, ti]))
        W = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[ginv] for _ in range(CS.BOOT)]).astype(float)
        bm = lambda x: (W @ x) / W.sum(1)
        b = bm(sc["base"]); ss = {k: 1 - bm(sc[k]) / b for k in ("raw", "in", "shift")}
        pt = {k: float(1 - sc[k].mean() / sc["base"].mean()) for k in ("raw", "in", "shift")}
        d1, d2 = ss["shift"] - ss["in"], ss["shift"] - ss["raw"]
        rows.append({"judge": judge, "source": src, "target": tgt, "primary": prim, "S_base": float(sc["base"].mean()),
                     "SS_raw": pt["raw"], "SS_in": pt["in"], "SS_shift": pt["shift"],
                     "dSS_shift_vs_in": pt["shift"] - pt["in"], "dSS_shift_vs_in_ci": [float(np.percentile(d1, 2.5)), float(np.percentile(d1, 97.5))],
                     "dSS_shift_vs_raw": pt["shift"] - pt["raw"], "dSS_shift_vs_raw_ci": [float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))]})
    return rows

def arm_b(res):
    from heads41 import build_views, heads, item_nrps_ce
    const = {}
    for vname, _, k, data in build_views():
        for seed in range(5):
            H = heads(vname, data, seed)
            c, _ = item_nrps_ce(np.repeat(H["const"].reshape(1, -1), len(H["test"]), 0), data.ratings, H["test"], k)
            const[(vname, seed)] = float(c.mean())
    B = json.loads((res / "arm_b_exp41_heads.json").read_text())["rows"]
    grp = defaultdict(list)
    for r in B:
        sc = const[(r["target"], r["seed"])]
        grp[(r["arm"], r["source"], r["target"])].append({"seed": r["seed"], "S_const": sc,
            "dSS_in": -r["G_in"][0] / sc, "dSS_shift": -r["G_shift"][0] / sc, "dSS_shift_ci": [-r["G_shift"][2] / sc, -r["G_shift"][1] / sc],
            "dSS_shift_minus_in": -r["shift_minus_in"][0] / sc, "dSS_shift_minus_in_ci": [-r["shift_minus_in"][2] / sc, -r["shift_minus_in"][1] / sc]})
    out = []
    for (arm, src, tgt), rs in grp.items():
        out.append({"arm": arm, "source": src, "target": tgt, "seeds": rs,
                    "dSS_in_mean": float(np.mean([x["dSS_in"] for x in rs])), "dSS_shift_mean": float(np.mean([x["dSS_shift"] for x in rs])),
                    "dSS_shift_minus_in_mean": float(np.mean([x["dSS_shift_minus_in"] for x in rs])),
                    "seeds_shift_minus_in_ci_below_0": int(sum(x["dSS_shift_minus_in_ci"][1] < 0 for x in rs)),
                    "dSS_shift_minus_in_seed_ci_range": [float(min(x["dSS_shift_minus_in_ci"][0] for x in rs)), float(max(x["dSS_shift_minus_in_ci"][1] for x in rs))]})
    return out

if __name__ == "__main__":
    labels, se_dir, res = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    a = arm_a(labels, se_dir); b = arm_b(res)
    (res / "skill_terms.json").write_text(json.dumps({"arm_a": a, "arm_b": b}, indent=1))
    S = {"chaosnli_snli": "SNLI", "chaosnli_mnli": "MNLI", "dices350": "DICES-350", "dices990": "DICES-990", "pooled_other_three": "other three",
         "summeval_relevance": "SE relevance", "summeval_consistency": "SE consistency"}
    md = ["## Arm A (A1): skill-score differences, frontier judges and G-Eval. dSS = SS(shifted map) - SS(in-domain map) and SS(shifted map) - SS(raw); 95% item-cluster bootstrap CI, 4,000 draws, base resampled.\n",
          "| judge | source -> target | SS raw | SS in-domain | SS shifted | dSS shift - in [CI] | dSS shift - raw [CI] |", "|---|---|---|---|---|---|---|"]
    for r in a:
        md.append(f"| {r['judge']} | {S.get(r['source'], r['source'])} -> {S.get(r['target'], r['target'])} | {r['SS_raw']:+.3f} | {r['SS_in']:+.3f} | {r['SS_shift']:+.3f} | "
                  f"{r['dSS_shift_vs_in']:+.3f} [{r['dSS_shift_vs_in_ci'][0]:+.3f}, {r['dSS_shift_vs_in_ci'][1]:+.3f}] | {r['dSS_shift_vs_raw']:+.3f} [{r['dSS_shift_vs_raw_ci'][0]:+.3f}, {r['dSS_shift_vs_raw_ci'][1]:+.3f}] |")
    md += ["\n## Arm B (A1): Experiment 41 heads. dSS = SS(Pearson-trained + map) - SS(CE-trained raw); seed means; shift - in with per-seed CIs.\n",
           "| arm | source -> target | dSS in-domain | dSS shifted | dSS shift - in | seeds with CI < 0 | per-seed CI range |", "|---|---|---|---|---|---|---|"]
    for r in b:
        md.append(f"| {r['arm']} | {r['source']} -> {r['target']} | {r['dSS_in_mean']:+.3f} | {r['dSS_shift_mean']:+.3f} | {r['dSS_shift_minus_in_mean']:+.3f} | {r['seeds_shift_minus_in_ci_below_0']}/5 | "
                  f"[{r['dSS_shift_minus_in_seed_ci_range'][0]:+.3f}, {r['dSS_shift_minus_in_seed_ci_range'][1]:+.3f}] |")
    (res / "skill_terms_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
