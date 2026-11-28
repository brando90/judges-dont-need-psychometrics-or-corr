#!/usr/bin/env python3
"""Experiment 55: risk, base-rate skill, judge ranking, human-agreement strata and paired-comparison power under all three
strictly proper rules (Brier, cross-entropy, normalized RPS) on the five frozen frontier judges of Experiment 28.
PROTOCOL.md was committed first (commit 4daeb50). CPU only, no model calls, deterministic given the seeds.

Usage: python run_all_rules.py [OUT_DIR]      (default: ../results)
"""
from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXPS = ROOT.parent
_spec = importlib.util.spec_from_file_location("lib44", next(EXPS.glob("44_*/src/lib44.py")))
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

SEED0 = 20261127
RULES = ("brier", "ce", "nrps")
DATASETS = L.DATASETS
JUDGES = L.JUDGES
B = 4000
NFOLD = 5
NQ = 5
MATCH = {
    "chaosnli_snli": {"brier": "matched", "ce": "matched", "nrps": "not matched (unordered labels), sensitivity only"},
    "chaosnli_mnli": {"brier": "matched", "ce": "matched", "nrps": "not matched (unordered labels), sensitivity only"},
    "chaosnli_alphanli": {"brier": "matched", "ce": "matched", "nrps": "matched (binary: equals Brier / 2)"},
    "dices350": {"brier": "proper, ignores order", "ce": "proper, ignores order", "nrps": "matched (ordinal)"},
    "dices990": {"brier": "proper, ignores order", "ce": "proper, ignores order", "nrps": "matched (ordinal)"},
}


def write_csv(path, rows):
    rows = list(rows)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def floor_q(q):
    q = np.asarray(q, float)
    K = q.shape[-1]
    return (1.0 - K * L.FLOOR) * q + L.FLOOR


def crossfit_base(counts, ds_idx):
    """[N, K] floored base forecasts: each item's base is the histogram of the other four folds' ratings (random item folds)."""
    N, K = counts.shape
    rng = np.random.default_rng(np.random.SeedSequence([SEED0, 55, 0, ds_idx]))
    perm = rng.permutation(N)
    fold = np.empty(N, int)
    fold[perm] = np.arange(N) % NFOLD
    base = np.zeros((N, K))
    for f in range(NFOLD):
        h = counts[fold != f].sum(0)
        base[fold == f] = floor_q(h / h.sum())
    return base, fold


def skill_arm(dname, ds_idx, d, P, counts):
    """Per judge and rule: risk, base risk, skill, item-bootstrap CI and bootstrap p. Returns rows and the risk matrices."""
    N = len(d["ids"])
    base, _ = crossfit_base(counts, ds_idx)
    rng = np.random.default_rng(np.random.SeedSequence([SEED0, 55, 2, ds_idx]))
    W = L.boot_weights(rng, N, B)  # [B, N] shared across rules and judges so rule comparisons are paired
    rows, risks = [], {}
    for rule in RULES:
        R = L.risk_items(P, counts, rule)  # [N, J]
        Rb = L.risk_items(base, counts, rule)  # [N]
        risks[rule] = (R, Rb)
        den = (W @ Rb) / N
        for j, jn in enumerate(JUDGES):
            num = (W @ R[:, j]) / N
            draws = 1.0 - num / den
            lo, hi = L.pct_ci(draws)
            rows.append(dict(dataset=dname, rule=rule, match=MATCH[dname][rule], judge=jn, n_items=N,
                             risk=float(R[:, j].mean()), base_risk=float(Rb.mean()),
                             skill=float(1.0 - R[:, j].mean() / Rb.mean()), ci_lo=lo, ci_hi=hi, p_raw=L.boot_p(draws)))
    return rows, risks


def rank_arm(skill_rows):
    out = []
    for dname in DATASETS:
        sk = {r: np.array([next(x["skill"] for x in skill_rows if x["dataset"] == dname and x["rule"] == r and x["judge"] == j)
                           for j in JUDGES]) for r in RULES}
        best = {r: JUDGES[int(np.argmax(sk[r]))] for r in RULES}
        for a, b in itertools.combinations(RULES, 2):
            tau = float(kendalltau(sk[a], sk[b]).statistic)
            flips = sum(1 for i, j in itertools.combinations(range(len(JUDGES)), 2)
                        if (sk[a][i] - sk[a][j]) * (sk[b][i] - sk[b][j]) < 0)
            out.append(dict(dataset=dname, rule_a=a, rule_b=b, kendall_tau_b=tau, pairs_flipped_of_10=flips,
                            best_a=best[a], best_b=best[b], same_best=bool(best[a] == best[b])))
    return out


def strata_arm(dname, ds_idx, d, P, counts):
    N, K = counts.shape
    base, _ = crossfit_base(counts, ds_idx)
    HB = np.zeros((N, K))
    ent = np.zeros(N)
    for i in range(N):
        n = int(d["n_avail"][i])
        ys = np.asarray(d["ratings"][i, :n], int)
        if ys.size >= 4:
            perm = np.random.default_rng(np.random.SeedSequence([SEED0, 55, 1, ds_idx, i])).permutation(ys.size)
            h = ys.size // 2
            a, b = ys[perm[:h]], ys[perm[h:]]
        else:
            a = b = ys
        ca = np.bincount(a, minlength=K) / a.size
        ca = ca[ca > 0]
        ent[i] = float(-(ca * np.log(ca)).sum())
        HB[i] = np.bincount(b, minlength=K) / b.size
    jitter = np.random.default_rng(SEED0).random(N) * 1e-9
    order = np.argsort(ent + jitter)
    qb = np.empty(N, int)
    for q, chunk in enumerate(np.array_split(order, NQ)):
        qb[chunk] = q
    rows = []
    for rule in RULES:
        S = L.risk_items(P, HB * 1000.0, rule)  # counts-like weights: HB * 1000 gives the same per-item mean as HB
        Sb = L.risk_items(base, HB * 1000.0, rule)
        for q in range(NQ):
            m = qb == q
            for j, jn in enumerate(JUDGES):
                rows.append(dict(dataset=dname, rule=rule, match=MATCH[dname][rule], quintile=q + 1, n=int(m.sum()), judge=jn,
                                 entropy_lo=float(ent[m].min()), entropy_hi=float(ent[m].max()),
                                 skill=float(1.0 - S[m, j].mean() / Sb[m].mean())))
    return rows


def mdd_arm(dname, d, P, counts):
    rows = []
    for rule in RULES:
        vals = []
        for a, b in itertools.combinations(range(len(JUDGES)), 2):
            D, s2, n, _ = L.pair_components(counts, P[:, a], P[:, b], rule)
            sb2, sw2 = L.variance_components(D, s2, n)
            vals.append(L.mdd(sb2, sw2, 400, 1))
        rows.append(dict(dataset=dname, rule=rule, match=MATCH[dname][rule], mdd_N400_k1_median=float(np.median(vals)),
                         mdd_N400_k1_min=float(np.min(vals)), mdd_N400_k1_max=float(np.max(vals)), n_pairs=len(vals)))
    return rows


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle, sha = L.load_bundle()
    skill_rows, strata_rows, mdd_rows = [], [], []
    for ds_idx, dname in enumerate(DATASETS):
        d = bundle[dname]
        P, _ = L.forecast_array(d)
        counts = np.asarray(d["counts"], float)
        s_rows, _ = skill_arm(dname, ds_idx, d, P, counts)
        skill_rows += s_rows
        strata_rows += strata_arm(dname, ds_idx, d, P, counts)
        mdd_rows += mdd_arm(dname, d, P, counts)
        print(dname, "done", flush=True)
    # Holm over all 75 declared cells
    adj = L.holm([r["p_raw"] for r in skill_rows])
    for r, a in zip(skill_rows, adj):
        r["p_holm"] = float(a)
        r["significant_below_base"] = bool(r["skill"] < 0 and a < 0.05)
        r["significant_above_base"] = bool(r["skill"] > 0 and a < 0.05)
    rank_rows = rank_arm(skill_rows)
    write_csv(out / "skill_all_rules.csv", skill_rows)
    write_csv(out / "rank_agreement.csv", rank_rows)
    write_csv(out / "strata_all_rules.csv", strata_rows)
    write_csv(out / "mdd_all_rules.csv", mdd_rows)
    # hypotheses
    h1 = {rule: int(sum(r["significant_below_base"] for r in skill_rows if r["dataset"] == "dices350" and r["rule"] == rule))
          for rule in RULES}
    h2_cells = []
    for dname in DATASETS:
        for rule in RULES:
            sk = {j: {q: next(x["skill"] for x in strata_rows if x["dataset"] == dname and x["rule"] == rule and x["judge"] == j and x["quintile"] == q)
                      for q in (1, NQ)} for j in JUDGES}
            pooled = {r["judge"]: r["skill"] for r in skill_rows if r["dataset"] == dname and r["rule"] == rule}
            best = max(pooled, key=pooled.get)
            h2_cells.append(dict(dataset=dname, rule=rule, best_judge=best, skill_Q1=sk[best][1], skill_Q5=sk[best][NQ],
                                 holds=bool(sk[best][1] > sk[best][NQ])))
    write_csv(out / "h2_cells.csv", h2_cells)
    summary = dict(bundle_sha256=sha, B=B, folds=NFOLD, quintiles=NQ,
                   h1_dices350_n_significantly_below_base_by_rule=h1, h1_holds=bool(all(v >= 3 for v in h1.values())),
                   h2_cells_holding=int(sum(c["holds"] for c in h2_cells)), h2_cells_total=len(h2_cells),
                   h2_holds=bool(all(c["holds"] for c in h2_cells)),
                   n_cells_significantly_below_base_by_rule={rule: int(sum(r["significant_below_base"] for r in skill_rows if r["rule"] == rule)) for rule in RULES})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results")
