#!/usr/bin/env python3
"""Independent re-computation of Experiment 44 numbers with plain per-rating loops (no shared vectorised helpers).

Compares against results/*.csv and results/summary.json; prints PASS/FAIL per check, exits nonzero on any failure.
Usage: python3 src/verify.py [--results results]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib44 as L  # only for data loading, constants and the protocol's seeds  # noqa: E402

FAILS = []
TOL = 1e-9


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def floor_q(p, K):
    return [(1 - K * 1e-6) * x + 1e-6 for x in p]


def score(q, y, K, kind):
    if kind == "brier":
        return sum((q[k] - (1.0 if k == y else 0.0)) ** 2 for k in range(K))
    if kind == "ce":
        return -math.log(q[y])
    if kind == "nrps":
        tot, c = 0.0, 0.0
        for m in range(K - 1):
            c += q[m]
            tot += (c - (1.0 if y <= m else 0.0)) ** 2
        return tot / (K - 1)
    raise ValueError(kind)


def forecasts(d, i, j, K):
    if not bool(d["judge_valid"][i, j]):
        return [1.0 / K] * K
    return floor_q([float(x) for x in d["p_judges"][i, j]], K)


def item_risk(d, i, j, K, kind, q=None, ratings=None):
    q = q if q is not None else forecasts(d, i, j, K)
    ys = ratings if ratings is not None else [int(x) for x in d["ratings"][i, : int(d["n_avail"][i])]]
    return sum(score(q, y, K, kind) for y in ys) / len(ys)


def jsd(p, q):
    tot = 0.0
    for a, b in zip(p, q):
        m = 0.5 * (a + b)
        if a > 0:
            tot += 0.5 * a * math.log2(a / m)
        if b > 0:
            tot += 0.5 * b * math.log2(b / m)
    return tot


def lower_median_bin(ys, ordinal):
    if ordinal:
        s = sorted(ys)
        return s[(len(s) - 1) // 2]
    K = max(ys) + 1
    cnt = [ys.count(k) for k in range(max(max(ys) + 1, 1))]
    return cnt.index(max(cnt))


def main(res):
    bundle, _ = L.load_bundle()
    S = json.loads((res / "summary.json").read_text())
    jrows = {(r["dataset"], r["judge"]): r for r in read(res / "arm_j_jsb.csv")}
    # ---- (a) primary risk and binned JSD, every dataset x judge, per-rating loops
    for dname in L.DATASETS:
        d = bundle[dname]; K = int(d["K"]); N = len(d["ids"]); kind = L.PRIMARY[dname]
        ordinal = dname in L.DICES
        for j, jn in enumerate(L.JUDGES):
            risk = sum(item_risk(d, i, j, K, kind) for i in range(N)) / N
            r = jrows[(dname, jn)]
            check(f"primary risk {dname}/{jn}", abs(risk - float(r["primary_risk"])) < TOL, f"{risk:.9f}")
            bins = {}
            for i in range(N):
                ys = [int(x) for x in d["ratings"][i, : int(d["n_avail"][i])]]
                b = lower_median_bin(ys, ordinal)
                h, m = bins.setdefault(b, [[0.0] * K, [0.0] * K, 0])[0], bins[b][1]
                for y in ys:
                    h[y] += 1.0
                q = forecasts(d, i, j, K)
                for k in range(K):
                    m[k] += q[k]
                bins[b][2] += 1
            js = 0.0
            for b, (h, m, c) in bins.items():
                hs, ms = sum(h), sum(m)
                js += c / N * jsd([x / hs for x in h], [x / ms for x in m])
            check(f"binned JSD {dname}/{jn}", abs(js - float(r["jsb"])) < TOL, f"{js:.9f}")
    # ---- (b) pool arm: TVD and pool risks via raw CSV rows
    prow = {(r["dataset"], r["contrast"], r["judge"]): r for r in read(res / "arm_p_judges.csv")}
    trow = {(r["dataset"], r["contrast"]): r for r in read(res / "arm_p_tvd.csv")}
    for dname in L.DICES:
        d = bundle[dname]; K = int(d["K"]); N = len(d["ids"]); kind = L.PRIMARY[dname]
        attrs = {}
        with open(L.EXP39 / "data" / "raw" / L.DICES_CSV[dname], newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                a = attrs.setdefault(row["rater_id"], (set(), set(), set()))
                for s_, col in zip(a, ("rater_gender", "rater_age", "rater_race")):
                    s_.add(row[col])
        attrs = {r: tuple(next(iter(s)) if len(s) == 1 else None for s in v) for r, v in attrs.items()}
        for cname, (la, lb, fn) in L.CONTRASTS.items():
            nA = [[0] * K for _ in range(N)]; nB = [[0] * K for _ in range(N)]
            lossA = [[[] for _ in L.JUDGES] for _ in range(N)]; lossB = [[[] for _ in L.JUDGES] for _ in range(N)]
            cntA = [0] * N; cntB = [0] * N
            for i in range(N):
                qs = [forecasts(d, i, j, K) for j in range(len(L.JUDGES))]
                for t in range(int(d["n_avail"][i])):
                    y = int(d["ratings"][i, t]); lab = fn(*attrs[d["rater_ids"][i, t]])
                    if lab == 0:
                        nA[i][y] += 1; cntA[i] += 1
                        for j in range(len(L.JUDGES)):
                            lossA[i][j].append(score(qs[j], y, K, kind))
                    elif lab == 1:
                        nB[i][y] += 1; cntB[i] += 1
                        for j in range(len(L.JUDGES)):
                            lossB[i][j].append(score(qs[j], y, K, kind))
            elig = [i for i in range(N) if cntA[i] >= 5 and cntB[i] >= 5]
            tvd = sum(0.5 * sum(abs(nA[i][k] / cntA[i] - nB[i][k] / cntB[i]) for k in range(K)) for i in elig) / len(elig)
            tr = trow[(dname, cname)]
            check(f"TVD {dname}/{cname}", abs(tvd - float(tr["tvd_observed"])) < TOL and len(elig) == int(tr["n_eligible_items"]),
                  f"{tvd:.9f} eligible {len(elig)}")
            for j, jn in enumerate(L.JUDGES):
                RA = sum(sum(lossA[i][j]) / cntA[i] for i in elig) / len(elig)
                RB = sum(sum(lossB[i][j]) / cntB[i] for i in elig) / len(elig)
                pr = prow[(dname, cname, jn)]
                ok = abs(RA - float(pr["risk_pool_A"])) < TOL and abs(RB - float(pr["risk_pool_B"])) < TOL \
                    and abs((RA - RB) - float(pr["delta_pool"])) < TOL
                check(f"pool risks {dname}/{cname}/{jn}", ok, f"A {RA:.6f} B {RB:.6f}")
    # ---- (c) consensus arm: stratum risks on the temperature grid, three judges per dataset
    crow = {(r["dataset"], r["stratum"], float(r["tau"]), r["judge"]): r for r in read(res / "arm_s_curves.csv")}
    for ds_idx, dname in enumerate(L.DATASETS):
        d = bundle[dname]; K = int(d["K"]); N = len(d["ids"]); kind = L.PRIMARY[dname]
        agree, half_b = [], []
        for i in range(N):
            rng = np.random.default_rng(np.random.SeedSequence([L.SEED0, 44, 2, ds_idx, i]))
            n = int(d["n_avail"][i]); ys = [int(x) for x in d["ratings"][i, :n]]
            perm = rng.permutation(n)
            a = [ys[p] for p in perm[: n // 2]]; b = [ys[p] for p in perm[n // 2:]]
            agree.append(max(a.count(k) for k in range(K)) / len(a)); half_b.append(b)
        order = sorted(range(N), key=lambda i: (agree[i], i))
        m = N // 3
        strata = {"contested": order[:m], "consensus": order[-m:]}
        for jn in (L.JUDGES[0], L.JUDGES[2], L.JUDGES[4]):
            j = L.JUDGES.index(jn)
            for tau in (0.5, 1.0, 2.0):
                for sname, idx in strata.items():
                    tot = 0.0
                    for i in idx:
                        q0 = forecasts(d, i, j, K)
                        z = [math.log(x) / tau for x in q0]; zm = max(z)
                        e = [math.exp(x - zm) for x in z]; s = sum(e)
                        q = [(1 - K * 1e-6) * x / s + 1e-6 for x in e]
                        tot += item_risk(d, i, j, K, kind, q=q, ratings=half_b[i])
                    r = crow[(dname, sname, tau, jn)]
                    check(f"arm S {dname}/{sname}/tau={tau}/{jn}", abs(tot / m - float(r["risk_primary"])) < TOL, f"{tot / m:.9f}")
    # ---- (d) power arm: variance components and MDD for one pair per dataset
    mrow = {(r["dataset"], r["judge_a"], r["judge_b"]): r for r in read(res / "arm_m_pairs.csv")}
    for dname in L.DATASETS:
        d = bundle[dname]; K = int(d["K"]); N = len(d["ids"]); kind = L.PRIMARY[dname]
        ja, jb = 0, 1
        Dm, s2, ns = [], [], []
        for i in range(N):
            ys = [int(x) for x in d["ratings"][i, : int(d["n_avail"][i])]]
            qa, qb = forecasts(d, i, ja, K), forecasts(d, i, jb, K)
            dl = [score(qa, y, K, kind) - score(qb, y, K, kind) for y in ys]
            n = len(dl); mu = sum(dl) / n
            Dm.append(mu); ns.append(n); s2.append(sum((x - mu) ** 2 for x in dl) / (n - 1))
        mD = sum(Dm) / N
        varD = sum((x - mD) ** 2 for x in Dm) / (N - 1)
        sb2 = max(0.0, varD - sum(s / n for s, n in zip(s2, ns)) / N); sw2 = sum(s2) / N
        r = mrow[(dname, L.JUDGES[ja], L.JUDGES[jb])]
        ok = abs(sb2 - float(r["sigma_b2"])) < TOL and abs(sw2 - float(r["sigma_w2"])) < TOL and abs(mD - float(r["observed_diff"])) < TOL
        mdd100 = (1.959963984540054 + 0.8416212335729143) * math.sqrt((sb2 + sw2 / 1) / 100)
        ok = ok and abs(mdd100 - float(r["mdd_N100_k1"])) < TOL
        check(f"power components {dname}", ok, f"sb2 {sb2:.6f} sw2 {sw2:.6f} mdd(100,1) {mdd100:.6f}")
    # ---- (e) summary consistency with the CSVs
    perm = read(res / "arm_j_permutation.csv")
    check("H-J1 max change below 1e-12", max(float(r["max_abs_jsb_change"]) for r in perm) <= 1e-12,
          f"{max(float(r['max_abs_jsb_change']) for r in perm):.3e}")
    check("summary H-J1 matches", S["H_J1"]["supported"] == (max(float(r["max_abs_jsb_change"]) for r in perm) <= 1e-12))
    gain = read(res / "arm_s_gain.csv")
    check("summary H-S1 counts", S["H_S1"]["delta_G_positive"] == sum(float(r["delta_G"]) > 0 for r in gain)
          and S["H_S1"]["ci_excludes_0_above"] == sum(float(r["ci_lo"]) > 0 for r in gain))
    pairs = read(res / "arm_p_pairs.csv")
    check("summary P3 counts", S["P3_holm_significant_interactions"] == sum(float(r["p_holm"]) < 0.05 for r in pairs)
          and S["P3_point_reversals"] == sum(r["point_reversal"] == "True" for r in pairs))
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS[:5]}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
    sys.exit(main(Path(ap.parse_args().results)))
