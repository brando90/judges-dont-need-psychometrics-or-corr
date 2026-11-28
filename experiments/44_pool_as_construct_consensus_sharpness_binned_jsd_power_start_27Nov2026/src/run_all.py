#!/usr/bin/env python3
"""Experiment 44: arms P (rater pool), S (consensus sharpness), J (binned JSD), M (power at k_test = 1).

Frozen design: PROTOCOL.md (commit 13d1177) and AMENDMENT_A1.md (commit 3cba685). No model calls; CPU only.
Usage: python3 src/run_all.py [--out results] [--boot 4000] [--perm 500] [--drop-invalid]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib44 as L  # noqa: E402

TAUS = [0.5, 0.75, 1.0, 1.5, 2.0]
N_GRID = [25, 50, 100, 200, 400, 800]
K_GRID = [1, 2, 5, 10]
PROTOCOL_COMMIT, AMENDMENT_COMMIT = "13d1177", "3cba685"


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(v)) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


# ------------------------------------------------------------------ Arm P
def arm_p(dname, d, P, ds_idx, B, nperm, chunk=500):
    K, N, J = int(d["K"]), len(d["ids"]), len(L.JUDGES)
    kind = L.PRIMARY[dname]
    item, rater, y, rid_list = L.rating_table(d)
    attrs = L.rater_attrs(dname)
    R = len(rid_list)
    loss = L.score_matrix(P, kind)[item, :, y]  # [rows, J]
    judge_rows, pair_rows, tvd_rows = [], [], []
    for ci, (cname, (la, lb, fn)) in enumerate(L.CONTRASTS.items()):
        lab = np.array([fn(*attrs[r]) for r in rid_list])
        cnt = L.pool_item_counts(item, lab[rater], y, N, K)
        tvd_obs, elig = L.mean_tvd(cnt)
        Ne = int(elig.sum())
        rng = np.random.default_rng(np.random.SeedSequence([L.SEED0, 44, 1, ds_idx, ci]))
        er = np.where(lab >= 0)[0]
        null = np.empty(nperm)
        for t in range(nperm):
            lp = lab.copy()
            lp[er] = rng.permutation(lab[er])
            null[t] = L.mean_tvd(L.pool_item_counts(item, lp[rater], y, N, K))[0]
        p1 = float((1 + (null >= tvd_obs).sum()) / (nperm + 1))
        tvd_rows.append(dict(dataset=dname, contrast=cname, pool_A=la, pool_B=lb, n_items=N, n_eligible_items=Ne,
                             raters_A=int((lab == 0).sum()), raters_B=int((lab == 1).sum()),
                             tvd_observed=tvd_obs, tvd_null_mean=float(np.nanmean(null)),
                             tvd_null_p95=float(np.nanpercentile(null, 95)), n_null_nan=int(np.isnan(null).sum()),
                             p_raw=p1))
        rows_A = (lab[rater] == 0) & elig[item]
        rows_B = (lab[rater] == 1) & elig[item]
        rows_all = elig[item]

        def pool_risk(rows):
            num = np.stack([np.bincount(item[rows], weights=loss[rows, j], minlength=N) for j in range(J)], 1)
            den = np.bincount(item[rows], minlength=N)
            return (num[elig] / den[elig, None]).mean(0)

        RA, RB, Rall = pool_risk(rows_A), pool_risk(rows_B), pool_risk(rows_all)
        emap = -np.ones(N, int)
        emap[np.where(elig)[0]] = np.arange(Ne)
        rA, rB = np.where(lab == 0)[0], np.where(lab == 1)[0]
        mA, mB = -np.ones(R, int), -np.ones(R, int)
        mA[rA], mB[rB] = np.arange(len(rA)), np.arange(len(rB))

        def mats(rows, rmap, nR):
            ri, rr = emap[item[rows]], rmap[rater[rows]]
            Ls = [sparse.csr_matrix((loss[rows, j], (ri, rr)), shape=(Ne, nR)) for j in range(J)]
            return Ls, sparse.csr_matrix((np.ones(int(rows.sum())), (ri, rr)), shape=(Ne, nR))

        LA, MA = mats(rows_A, mA, len(rA))
        LB, MB = mats(rows_B, mB, len(rB))
        rngb = np.random.default_rng(np.random.SeedSequence([L.SEED0, 44, 1, ds_idx, ci, 99]))
        outA, outB = [], []
        for c0 in range(0, B, chunk):
            c = min(chunk, B - c0)
            vA = rngb.multinomial(len(rA), np.full(len(rA), 1 / len(rA)), size=c).T.astype(float)
            vB = rngb.multinomial(len(rB), np.full(len(rB), 1 / len(rB)), size=c).T.astype(float)
            W = rngb.multinomial(Ne, np.full(Ne, 1 / Ne), size=c).T.astype(float)
            a, b = L.pool_draw_risks(LA, MA, LB, MB, vA, vB, W)
            outA.append(a)
            outB.append(b)
        dA, dB = np.concatenate(outA), np.concatenate(outB)
        dpool = dA - dB
        for j, jn in enumerate(L.JUDGES):
            lo, hi = L.pct_ci(dpool[:, j])
            judge_rows.append(dict(dataset=dname, contrast=cname, judge=jn, risk_pool_A=RA[j], risk_pool_B=RB[j],
                                   risk_all_raters=Rall[j], delta_pool=RA[j] - RB[j], ci_lo=lo, ci_hi=hi,
                                   p_raw=L.boot_p(dpool[:, j])))
        tau = L.tau_b(RA, RB)
        for a in range(J):
            for b in range(a + 1, J):
                dd = dpool[:, a] - dpool[:, b]
                lo, hi = L.pct_ci(dd)
                wa, wb = RA[a] - RA[b], RB[a] - RB[b]
                pair_rows.append(dict(dataset=dname, contrast=cname, judge_a=L.JUDGES[a], judge_b=L.JUDGES[b],
                                      diff_pool_A=wa, diff_pool_B=wb, interaction_D=wa - wb, ci_lo=lo, ci_hi=hi,
                                      p_raw=L.boot_p(dd), point_reversal=bool(wa * wb < 0), tau_b_AB=tau))
    return judge_rows, pair_rows, tvd_rows


# ------------------------------------------------------------------ Arm S
def arm_s(dname, d, P, ds_idx, B):
    K, N, J = int(d["K"]), len(d["ids"]), len(L.JUDGES)
    kind = L.PRIMARY[dname]
    cA, cB = np.zeros((N, K)), np.zeros((N, K))
    for i in range(N):
        rng = np.random.default_rng(np.random.SeedSequence([L.SEED0, 44, 2, ds_idx, i]))
        n = int(d["n_avail"][i])
        yv = d["ratings"][i, :n]
        perm = rng.permutation(n)
        cA[i] = np.bincount(yv[perm[: n // 2]], minlength=K)
        cB[i] = np.bincount(yv[perm[n // 2:]], minlength=K)
    agree = cA.max(1) / cA.sum(1)
    order = np.lexsort((np.arange(N), agree))
    m = N // 3
    strata = {"contested": order[:m], "consensus": order[-m:]}
    risk = np.stack([L.risk_items(L.temper(P, t), cB, kind) for t in TAUS])  # [T, N, J]
    risk_ce = np.stack([L.risk_items(L.temper(P, t), cB, "ce") for t in TAUS])
    argmax_ok = all(np.array_equal(L.temper(P, t).argmax(-1), P.argmax(-1)) for t in TAUS)
    i1, i05 = TAUS.index(1.0), TAUS.index(0.5)
    rng = np.random.default_rng(np.random.SeedSequence([L.SEED0, 44, 2, ds_idx, 10 ** 6]))
    Wc, Wt = L.boot_weights(rng, m, B), L.boot_weights(rng, m, B)
    g = risk[i1] - risk[i05]  # [N, J] sharpening gain per item
    Gc_d = (Wc @ g[strata["consensus"]]) / m
    Gt_d = (Wt @ g[strata["contested"]]) / m
    curve_rows, gain_rows = [], []
    for sname, idx in strata.items():
        for ti, t in enumerate(TAUS):
            for j, jn in enumerate(L.JUDGES):
                curve_rows.append(dict(dataset=dname, stratum=sname, tau=t, judge=jn, risk_primary=risk[ti][idx, j].mean(),
                                       risk_ce=risk_ce[ti][idx, j].mean(), median_agreement=float(np.median(agree[idx]))))
    for j, jn in enumerate(L.JUDGES):
        Gc, Gt = g[strata["consensus"], j].mean(), g[strata["contested"], j].mean()
        dG = Gc_d[:, j] - Gt_d[:, j]
        lo, hi = L.pct_ci(dG)
        best = {s: TAUS[int(np.argmin([risk[ti][strata[s], j].mean() for ti in range(len(TAUS))]))] for s in strata}
        gain_rows.append(dict(dataset=dname, judge=jn, gain_consensus=Gc, gain_contested=Gt, delta_G=Gc - Gt, ci_lo=lo,
                              ci_hi=hi, p_raw=L.boot_p(dG), best_tau_consensus=best["consensus"],
                              best_tau_contested=best["contested"], n_per_stratum=m, argmax_unchanged=argmax_ok))
    return curve_rows, gain_rows


# ------------------------------------------------------------------ Arm J
def arm_j(dname, d, P, ds_idx, B, nperm=200):
    N, J = len(d["ids"]), len(L.JUDGES)
    kind = L.PRIMARY[dname]
    counts = np.asarray(d["counts"], float)
    bins = L.human_bins(d["counts"], dname in L.DICES)
    risk0 = L.risk_items(P, counts, kind).mean(0)
    rng = np.random.default_rng(np.random.SeedSequence([L.SEED0, 44, 3, ds_idx]))
    W = L.boot_weights(rng, N, B)
    rows, perm_rows = [], []
    js0 = np.array([L.jsb(counts, P[:, j], bins) for j in range(J)])
    for j, jn in enumerate(L.JUDGES):
        jd = L.jsb_boot(counts, P[:, j], bins, W)
        lo, hi = L.pct_ci(jd)
        max_dj, dr = 0.0, []
        for _ in range(nperm):
            q2 = P[:, j].copy()
            for b in np.unique(bins):
                idx = np.where(bins == b)[0]
                q2[idx] = P[rng.permutation(idx), j]
            max_dj = max(max_dj, abs(L.jsb(counts, q2, bins) - js0[j]))
            dr.append(L.risk_items(q2, counts, kind).mean() - risk0[j])
        perm_rows.append(dict(dataset=dname, judge=jn, jsb=js0[j], max_abs_jsb_change=max_dj, risk_original=risk0[j],
                              risk_permuted_minus_original_mean=float(np.mean(dr)),
                              risk_permuted_minus_original_se=float(np.std(dr, ddof=1) / np.sqrt(nperm))))
        rows.append(dict(dataset=dname, judge=jn, jsb=js0[j], jsb_ci_lo=lo, jsb_ci_hi=hi, primary_risk=risk0[j],
                         n_bins=int(len(np.unique(bins)))))
    tau = L.tau_b(js0, risk0)
    for r in rows:
        r["tau_b_jsb_vs_primary_risk"] = tau
    return rows, perm_rows


# ------------------------------------------------------------------ Arm M
def arm_m(dname, d, P, ds_idx, nsub=2000):
    K, N, J = int(d["K"]), len(d["ids"]), len(L.JUDGES)
    kind = L.PRIMARY[dname]
    counts = np.asarray(d["counts"], float)
    risk0 = L.risk_items(P, counts, kind).mean(0)
    rows, comp = [], {}
    for a in range(J):
        for b in range(a + 1, J):
            D, s2, n, delta = L.pair_components(counts, P[:, a], P[:, b], kind)
            sb2, sw2 = L.variance_components(D, s2, n)
            obs = float(D.mean())
            comp[(a, b)] = (D, s2, n, delta, sb2, sw2, obs)
            row = dict(dataset=dname, judge_a=L.JUDGES[a], judge_b=L.JUDGES[b], observed_diff=obs, sigma_b2=sb2,
                       sigma_w2=sw2, n80_k1=L.n80(sb2, sw2, obs))
            for N_ in N_GRID:
                for k in K_GRID:
                    row[f"mdd_N{N_}_k{k}"] = L.mdd(sb2, sw2, N_, k)
            row["resolvable_N400_k1"] = bool(L.mdd(sb2, sw2, 400, 1) <= abs(obs))
            rows.append(row)
    # subsampling check on the pair with the median |observed difference|
    order = sorted(comp, key=lambda ab: abs(comp[ab][6]))
    ab = order[(len(order) - 1) // 2]
    D, s2, n, delta, sb2, sw2, obs = comp[ab]
    ratings = np.asarray(d["ratings"])
    Rmax = ratings.shape[1]
    check = []
    for N_ in (50, 100, 200):
        for k in (1, 5):
            rng = np.random.default_rng(np.random.SeedSequence([L.SEED0, 44, 4, ds_idx, N_, k]))
            est = []
            for c0 in range(0, nsub, 200):
                c = min(200, nsub - c0)
                items = np.argsort(rng.random((c, N)), 1)[:, :N_]
                rs = ratings[items]  # [c, N_, Rmax]
                keys = np.where(rs < 0, np.inf, rng.random(rs.shape))
                pos = np.argpartition(keys, k - 1, axis=2)[:, :, :k]
                lab = np.take_along_axis(rs, pos, 2)
                est.append(delta[items[:, :, None], lab].mean((1, 2)))
            est = np.concatenate(est)
            vs, vp = L.v_sub(D, s2, n, N_, k), (sb2 + sw2 / k) / N_
            check.append(dict(dataset=dname, judge_a=L.JUDGES[ab[0]], judge_b=L.JUDGES[ab[1]], N=N_, k=k,
                              empirical_var=float(est.var(ddof=1)), v_sub=vs, ratio_to_v_sub=float(est.var(ddof=1) / vs),
                              v_planning=vp, ratio_to_v_planning=float(est.var(ddof=1) / vp),
                              within_0p9_1p1=bool(0.9 <= est.var(ddof=1) / vs <= 1.1)))
    return rows, check


# ------------------------------------------------------------------ driver
def run(out, B, nperm, drop_invalid):
    bundle, bundle_sha = L.load_bundle()
    out.mkdir(parents=True, exist_ok=True)
    allp = dict(j=[], pr=[], tv=[]); alls = dict(c=[], g=[]); allj = dict(j=[], p=[]); allm = dict(m=[], c=[])
    items_used = {}
    for ds_idx, dname in enumerate(L.DATASETS):
        d = bundle[dname]
        P, keep = L.forecast_array(d, drop_invalid)
        d = L.subset(d, keep)
        P = P[keep]
        items_used[dname] = int(keep.sum())
        print(dname, "items", items_used[dname], flush=True)
        if dname in L.DICES:
            a, b, c = arm_p(dname, d, P, ds_idx, B, nperm)
            allp["j"] += a; allp["pr"] += b; allp["tv"] += c
        a, b = arm_s(dname, d, P, ds_idx, B)
        alls["c"] += a; alls["g"] += b
        a, b = arm_j(dname, d, P, ds_idx, B)
        allj["j"] += a; allj["p"] += b
        a, b = arm_m(dname, d, P, ds_idx)
        allm["m"] += a; allm["c"] += b
    # Holm families
    for rows, key, name in ((allp["tv"], "p_raw", "P1"), (allp["j"], "p_raw", "P2"), (allp["pr"], "p_raw", "P3")):
        adj = L.holm([r[key] for r in rows])
        for r, a in zip(rows, adj):
            r["p_holm"] = float(a)
    write_csv(out / "arm_p_tvd.csv", allp["tv"]); write_csv(out / "arm_p_judges.csv", allp["j"]); write_csv(out / "arm_p_pairs.csv", allp["pr"])
    write_csv(out / "arm_s_curves.csv", alls["c"]); write_csv(out / "arm_s_gain.csv", alls["g"])
    write_csv(out / "arm_j_jsb.csv", allj["j"]); write_csv(out / "arm_j_permutation.csv", allj["p"])
    write_csv(out / "arm_m_pairs.csv", allm["m"]); write_csv(out / "arm_m_check.csv", allm["c"])
    # hypotheses
    S = {"protocol_commit": PROTOCOL_COMMIT, "amendment_commit": AMENDMENT_COMMIT, "bundle_sha256": bundle_sha,
         "drop_invalid": drop_invalid, "bootstrap_draws": B, "permutations": nperm, "items_used": items_used}
    sets = [s for s in L.DATASETS if s in L.DICES]
    S["H_P1"] = {s: bool(any(r["p_holm"] < 0.05 for r in allp["tv"] if r["dataset"] == s)) for s in sets}
    S["H_P2"] = {s: bool(any(r["p_holm"] < 0.05 for r in allp["j"] if r["dataset"] == s)) for s in sets}
    S["P3_holm_significant_interactions"] = int(sum(r["p_holm"] < 0.05 for r in allp["pr"]))
    S["P3_point_reversals"] = int(sum(r["point_reversal"] for r in allp["pr"]))
    S["P3_pool_dependent_orderings"] = int(sum(r["p_holm"] < 0.05 and r["point_reversal"] for r in allp["pr"]))
    S["P3_tau_b_AB"] = {f"{r['dataset']}/{r['contrast']}": r["tau_b_AB"] for r in allp["pr"][::10]}
    g = alls["g"]
    S["H_S1"] = {"cells": len(g), "delta_G_positive": int(sum(r["delta_G"] > 0 for r in g)),
                 "ci_excludes_0_above": int(sum(r["ci_lo"] > 0 for r in g)),
                 "ci_excludes_0_below": int(sum(r["ci_hi"] < 0 for r in g)),
                 "argmax_unchanged_all": bool(all(r["argmax_unchanged"] for r in g))}
    S["H_S1"]["supported"] = bool(S["H_S1"]["delta_G_positive"] >= 20 and S["H_S1"]["ci_excludes_0_above"] >= 15)
    pj = allj["p"]
    S["H_J1"] = {"max_abs_jsb_change": float(max(r["max_abs_jsb_change"] for r in pj)),
                 "supported": bool(max(r["max_abs_jsb_change"] for r in pj) <= 1e-12)}
    S["H_J2"] = {"cells": len(pj), "permuted_worse": int(sum(r["risk_permuted_minus_original_mean"] > 0 for r in pj)),
                 "supported": bool(all(r["risk_permuted_minus_original_mean"] > 0 for r in pj))}
    S["J_tau_b_jsb_vs_primary"] = {r["dataset"]: r["tau_b_jsb_vs_primary_risk"] for r in allj["j"][::5]}
    cm = allm["c"]
    S["H_M1"] = {"cells": len(cm), "within_0p9_1p1": int(sum(r["within_0p9_1p1"] for r in cm)),
                 "supported": bool(all(r["within_0p9_1p1"] for r in cm))}
    # H-M2 (identity, not a test): at a fixed total budget B = N k the variance (sigma_b^2 k + sigma_w^2) / B must not decrease in k
    ident = []
    for r in allm["m"]:
        seq = [(float(r["sigma_b2"]) * k + float(r["sigma_w2"])) / 400.0 for k in K_GRID]
        ident.append(all(a <= b + 1e-18 for a, b in zip(seq, seq[1:])))
    S["H_M2"] = {"pairs": len(ident), "nondecreasing_in_k_at_fixed_budget": int(sum(ident)), "identity_holds": bool(all(ident))}
    S["M_resolvable_N400_k1_by_set"] = {s: f"{sum(r['resolvable_N400_k1'] for r in allm['m'] if r['dataset'] == s)}/10" for s in L.DATASETS}
    (out / "summary.json").write_text(json.dumps(S, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in S.items() if k.startswith("H_") or k.startswith("P3")}, indent=1, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--perm", type=int, default=500)
    ap.add_argument("--drop-invalid", action="store_true")
    a = ap.parse_args()
    run(Path(a.out), a.boot, a.perm, a.drop_invalid)
