#!/usr/bin/env python3
"""Experiment 46 analysis: rank judges by correlation/agreement vs by the encoding-matched proper score (PROTOCOL.md, frozen).

Usage: python analyze_rank_reversal.py LABELS_JSON SUMMEVAL_INPUT_DIR LOCAL_JUDGE_DIR OUT_DIR
  LABELS_JSON         labels.json written by 28_*/runners/prepare.py from the public files (counts and group per id)
  SUMMEVAL_INPUT_DIR  directory with summeval_consistency.npz and summeval_relevance.npz from 23_*/summeval/prepare.py
  LOCAL_JUDGE_DIR     *.jsonl written by run_local_judges.py
Writes OUT_DIR/judge_scores.csv, pairwise.csv, reversals.csv, kendall.csv, summary.json.
"""
import csv, glob, itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata, kendalltau

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
E28 = EXP.parent / "28_public_rating_sets_chaosnli_dices_frontier_judges_ce_vs_pearson_start_23Nov2026"
SETS = ["chaosnli_snli", "chaosnli_mnli", "chaosnli_alphanli", "dices350", "dices990", "summeval_consistency", "summeval_relevance"]
PRIMARY = {"chaosnli_snli": "brier", "chaosnli_mnli": "brier", "chaosnli_alphanli": "brier", "dices350": "nrps", "dices990": "nrps",
           "summeval_consistency": "nrps", "summeval_relevance": "nrps"}
FRONTIER = ["gpt-6-astra", "claude-opus-5-5", "kimi-k3", "glm-5.3", "deepseek-v4.1-flash"]
import os
BOOT, SEED, ALPHA = int(os.environ.get("E46_BOOT", 4000)), 20261127, 0.05
CORR_METRICS = ["pearson", "spearman", "top1", "kappa"]


def floor(p):
    p = np.asarray(p, float); k = p.shape[-1]
    return (1 - k * 1e-6) * p + 1e-6


# ---------------- per-item proper scores (expected over the item's own ratings = unbiased per-sample risk) ----------------
def item_scores(q, h):
    F, G = np.cumsum(q, -1)[:, :-1], np.cumsum(h, -1)[:, :-1]
    nrps = (F * F - 2 * F * G + G).mean(-1)
    ce = -(h * np.log(q)).sum(-1)
    brier = (q * q).sum(-1) - 2 * (q * h).sum(-1) + 1
    return {"nrps": nrps, "ce": ce, "brier": brier}


def mode_w(rows):
    t = rows >= rows.max(1, keepdims=True) - 1e-12
    return t / t.sum(1, keepdims=True)


def corr_stats(q, h, w):
    """Weighted (bootstrap multiplicities w) Pearson, Spearman, top-1 agreement, Cohen's kappa."""
    k = q.shape[1]; v = np.arange(k, dtype=float)
    x, y = q @ v, h @ v
    W = w.sum()
    def wcorr(a, b):
        ma, mb = (w * a).sum() / W, (w * b).sum() / W
        ca, cb = a - ma, b - mb
        den = np.sqrt((w * ca * ca).sum() * (w * cb * cb).sum())
        return (w * ca * cb).sum() / den if den > 1e-15 else np.nan
    pear = wcorr(x, y)
    rep = np.repeat(np.arange(len(w)), w.astype(int))
    rx, ry = rankdata(x[rep]), rankdata(y[rep])
    spear = np.corrcoef(rx, ry)[0, 1] if rx.std() > 0 and ry.std() > 0 else np.nan
    top1 = (w * (mode_w(q) * mode_w(h)).sum(1)).sum() / W
    a, b = q.argmax(1), h.argmax(1)
    po = (w * (a == b)).sum() / W
    pa = np.array([(w * (a == c)).sum() / W for c in range(k)]); pb = np.array([(w * (b == c)).sum() / W for c in range(k)])
    pe = (pa * pb).sum()
    kappa = (po - pe) / (1 - pe) if 1 - pe > 1e-12 else np.nan
    return {"pearson": pear, "spearman": spear, "top1": top1, "kappa": kappa}


# ---------------- calibrator for H3 (multinomial logit on [log q, 1], ridge 1e-3, individual-rating CE) ----------------
def fit_cal(q, cnt, ridge=1e-3):
    x = np.column_stack([np.log(q), np.ones(len(q))]); n, d = x.shape; k = q.shape[1]
    tot = cnt.sum()
    def f(wf):
        Wm = wf.reshape(d, k); z = x @ Wm; z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True)); p = np.exp(lp)
        ll = -(cnt * lp).sum() / tot
        g = x.T @ (p * cnt.sum(1, keepdims=True) - cnt) / tot
        pen = Wm.copy(); pen[-1] = 0; pen[:k, :] -= np.eye(k)  # shrink toward identity map
        return ll + 0.5 * ridge * (pen ** 2).sum(), (g + ridge * pen).ravel()
    w0 = np.zeros((d, k)); w0[:k, :] = np.eye(k)
    return minimize(f, w0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x.reshape(d, k)


def apply_cal(Wm, q):
    z = np.column_stack([np.log(q), np.ones(len(q))]) @ Wm; z -= z.max(1, keepdims=True); e = np.exp(z)
    return floor(e / e.sum(1, keepdims=True))


def folds_for(groups, set_index):
    ug = sorted(set(groups))
    perm = np.random.default_rng(np.random.SeedSequence([SEED, 46, 100 + set_index])).permutation(len(ug))
    fold_of = {ug[j]: pos % 5 for pos, j in enumerate(perm)}
    return np.array([fold_of[g] for g in groups])


def load_all(labels_json, se_dir, local_dir):
    lab = json.loads(Path(labels_json).read_text())
    items = {s: [] for s in SETS}
    counts, groups = {}, {}
    for i, v in lab.items():
        s = i.split("/")[0]; items[s].append(i); counts[i] = np.array(v["counts"], float); groups[i] = str(v["group"])
    forecasts = {}  # (judge) -> {id: (p, status)}
    for r in csv.DictReader(open(E28 / "forecasts" / "forecasts_11-23-2026.csv")):
        k = 2 if r["id"].startswith("chaosnli_alphanli") else 3
        forecasts.setdefault(r["judge"], {})[r["id"]] = (np.array([float(r[f"p{j}"]) for j in range(k)]), r["status"])
    for dim in ("consistency", "relevance"):
        z = np.load(Path(se_dir) / f"summeval_{dim}.npz", allow_pickle=False)
        s = f"summeval_{dim}"
        for idx, ident in enumerate(z["id"]):
            i = f"{s}::{ident}"; items[s].append(i); counts[i] = z["counts"][idx].astype(float); groups[i] = str(z["group"][idx])
            forecasts.setdefault("g-eval-gpt-4|sampled20", {})[i] = (z["native"][idx].astype(float), "valid")
    for f in sorted(x for d_ in str(local_dir).split(":") for x in glob.glob(str(Path(d_) / "*.jsonl"))):  # A2: colon-separated dirs (local + api)
        if "/test_" in f:
            continue
        for line in open(f):
            if line.strip():
                r = json.loads(line); forecasts.setdefault(r["judge"], {})[r["id"]] = (np.array(r["p"], float), r["status"])
    return items, counts, groups, forecasts


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0
    for r, j in enumerate(order):
        run = max(run, min(1, (m - r) * p[j])); adj[j] = run
    return adj


def analyze_panel(name, judges, set_items, counts, groups, forecasts, calibrate, rng_seed):
    rows, pair_rows, ken_rows = [], [], []
    for si, s in enumerate(SETS):
        ids = [i for i in set_items.get(s, []) if all(i in forecasts[j] for j in judges)]
        if len(ids) < 20 or len(judges) < 2:
            continue
        C = np.stack([counts[i] for i in ids]); H = C / C.sum(1, keepdims=True); g = [groups[i] for i in ids]
        Q = {j: floor(np.stack([forecasts[j][i][0] for i in ids])) for j in judges}
        inval = {j: float(np.mean([forecasts[j][i][1] != "valid" for i in ids])) for j in judges}
        fold = folds_for(g, si)
        if calibrate:
            for j in judges:
                out = np.zeros_like(Q[j])
                for f in range(5):
                    tr, te = fold != f, fold == f
                    out[te] = apply_cal(fit_cal(Q[j][tr], C[tr]), Q[j][te])
                Q[j] = out
        base = np.zeros_like(H)  # cross-fitted marginal base rate
        for f in range(5):
            tr, te = fold != f, fold == f
            m = C[tr].sum(0); base[te] = floor(m / m.sum())
        prim = PRIMARY[s]
        Sb = item_scores(base, H)[prim]
        per = {j: item_scores(Q[j], H) for j in judges}
        ug, ginv = np.unique(g, return_inverse=True)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 46, si, rng_seed]))
        wb = np.stack([np.bincount(ginv, weights=None, minlength=len(ug))[ginv] * 0 + np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[ginv] for _ in range(BOOT)]).astype(float)
        full_w = np.ones(len(ids))
        point = {j: {**{m_: float(per[j][m_].mean()) for m_ in ("nrps", "ce", "brier")},
                     "skill": float(1 - per[j][prim].mean() / Sb.mean()), **corr_stats(Q[j], H, full_w)} for j in judges}
        bootm = {j: {"S": (wb @ per[j][prim]) / wb.sum(1)} for j in judges}
        # Amendment A1 (reporting only, decision rules unchanged): skill-score differences with CIs, base resampled with the items.
        Sb_n = item_scores(base, H)["nrps"]
        bb, bbn = (wb @ Sb) / wb.sum(1), (wb @ Sb_n) / wb.sum(1)
        for j in judges:
            point[j]["skill_nrps"] = float(1 - per[j]["nrps"].mean() / Sb_n.mean())
            bootm[j]["skill"] = 1 - bootm[j]["S"] / bb
            bootm[j]["skill_nrps"] = 1 - ((wb @ per[j]["nrps"]) / wb.sum(1)) / bbn
        for j in judges:
            cs = [corr_stats(Q[j], H, wb[b]) for b in range(BOOT)]
            for m_ in CORR_METRICS:
                bootm[j][m_] = np.array([c[m_] for c in cs])
        for j in judges:
            rows.append({"panel": name, "set": s, "judge": j, "n_items": len(ids), "primary": prim, "invalid_frac": inval[j],
                         "S_primary": point[j][prim], **point[j], "base_S_primary": float(Sb.mean()), "base_nrps": float(item_scores(base, H)["nrps"].mean())})
        for a, b in itertools.combinations(judges, 2):
            r = {"panel": name, "set": s, "A": a, "B": b, "n_items": len(ids)}
            for m_ in ["S", "skill", "skill_nrps"] + CORR_METRICS:
                pa = point[a][prim] if m_ == "S" else point[a][m_]; pb = point[b][prim] if m_ == "S" else point[b][m_]
                d = bootm[a][m_] - bootm[b][m_]; d = d[np.isfinite(d)]
                p = min(1.0, 2 * min((1 + (d <= 0).sum()) / (len(d) + 1), (1 + (d >= 0).sum()) / (len(d) + 1))) if len(d) else np.nan
                r.update({f"d_{m_}": pa - pb, f"lo_{m_}": float(np.percentile(d, 2.5)) if len(d) else np.nan,
                          f"hi_{m_}": float(np.percentile(d, 97.5)) if len(d) else np.nan, f"p_{m_}": p})
            pair_rows.append(r)
        for m_ in CORR_METRICS:
            xs = [point[j][m_] for j in judges]; ys = [-point[j][prim] for j in judges]
            ken_rows.append({"panel": name, "set": s, "metric": m_, "n_judges": len(judges), "kendall_tau_b_vs_primary": float(kendalltau(xs, ys).statistic),
                             "best_by_metric": judges[int(np.nanargmax(xs))], "best_by_primary": judges[int(np.argmax(ys))]})
    for m_ in ["S", "skill", "skill_nrps"] + CORR_METRICS:  # Holm within metric family across all (set, pair) cells of the panel
        ps = [r[f"p_{m_}"] for r in pair_rows]
        if ps:
            adj = holm(np.nan_to_num(ps, nan=1.0))
            for r, v in zip(pair_rows, adj):
                r[f"holm_{m_}"] = float(v)
    rev = []
    for r in pair_rows:
        for m_ in CORR_METRICS:
            # metric favours A when d_m > 0; proper score favours B when d_S > 0 (A worse). Reversal = opposite preferences.
            opp = (r[f"d_{m_}"] > 0 and r["d_S"] > 0) or (r[f"d_{m_}"] < 0 and r["d_S"] < 0)
            sig = opp and r.get(f"holm_{m_}", 1) < ALPHA and r.get("holm_S", 1) < ALPHA
            if opp:
                rev.append({"panel": r["panel"], "set": r["set"], "A": r["A"], "B": r["B"], "metric": m_, "d_metric": r[f"d_{m_}"],
                            "ci_metric": [r[f"lo_{m_}"], r[f"hi_{m_}"]], "holm_metric": r.get(f"holm_{m_}"), "d_S": r["d_S"],
                            "ci_S": [r["lo_S"], r["hi_S"]], "holm_S": r.get("holm_S"), "significant": bool(sig), "versus": "primary_S"})
    for r in pair_rows:  # A1: correlation vs nRPS skill (skill favours A when d_skill_nrps > 0)
        for m_ in CORR_METRICS:
            opp = (r[f"d_{m_}"] > 0 and r["d_skill_nrps"] < 0) or (r[f"d_{m_}"] < 0 and r["d_skill_nrps"] > 0)
            if opp:
                rev.append({"panel": r["panel"], "set": r["set"], "A": r["A"], "B": r["B"], "metric": m_, "versus": "skill_nrps", "d_metric": r[f"d_{m_}"],
                            "ci_metric": [r[f"lo_{m_}"], r[f"hi_{m_}"]], "holm_metric": r.get(f"holm_{m_}"), "d_S": r["d_skill_nrps"],
                            "ci_S": [r["lo_skill_nrps"], r["hi_skill_nrps"]], "holm_S": r.get("holm_skill_nrps"),
                            "significant": bool(opp and r.get(f"holm_{m_}", 1) < ALPHA and r.get("holm_skill_nrps", 1) < ALPHA)})
    return rows, pair_rows, ken_rows, rev


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]


def main(labels_json, se_dir, local_dir, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    items, counts, groups, fc = load_all(labels_json, se_dir, local_dir)
    subset = json.loads((EXP / "subset_ids.json").read_text())
    sub_items = {s: [(f"{s}::{i}" if s.startswith("summeval") else i) for i in v] for s, v in subset.items()}
    local = sorted(j for j in fc if "|" in j and not j.startswith("g-eval"))
    # local judges must have finished a set to enter it; a judge is kept per panel only where it covers every subset item
    panels = [("frontier_full", FRONTIER, {s: items[s] for s in SETS[:5]})]
    lj_nli = [j for j in local if all(all(i in fc[j] for i in sub_items[s]) for s in SETS[:5])]
    panels.append(("local_plus_frontier_subset", FRONTIER + lj_nli, {s: sub_items[s] for s in SETS[:5]}))
    lj_se = [j for j in local if j.endswith(("|native", "|api-sampled", "|api-logprob")) and all(all(i in fc[j] for i in sub_items[s]) for s in SETS[5:])]
    panels.append(("summeval_geval_plus_local_subset", ["g-eval-gpt-4|sampled20"] + lj_se, {s: sub_items[s] for s in SETS[5:]}))
    # sensitivity: drop items where any frontier judge was invalid
    valid_items = {s: [i for i in items[s] if all(fc[j][i][1] == "valid" for j in FRONTIER)] for s in SETS[:5]}
    panels.append(("frontier_full_valid_only", FRONTIER, valid_items))
    allrows, allpairs, allken, allrev = [], [], [], []
    summary = {"panels": {}}
    for pi, (name, judges, set_items) in enumerate(panels):
        for cal in (False, True):
            tag = name + ("|calibrated" if cal else "|raw")
            rows, pairs, ken, rev = analyze_panel(tag, judges, set_items, counts, groups, fc, cal, pi * 2 + int(cal))
            allrows += rows; allpairs += pairs; allken += ken; allrev += rev
            summary["panels"][tag] = {"judges": judges, "n_pairs_x_sets": len(pairs),
                                      "significant_reversals": {m_: sum(1 for r in rev if r["metric"] == m_ and r["significant"] and r["versus"] == "primary_S") for m_ in CORR_METRICS},
                                      "point_reversals": {m_: sum(1 for r in rev if r["metric"] == m_ and r["versus"] == "primary_S") for m_ in CORR_METRICS},
                                      "A1_significant_reversals_vs_nrps_skill": {m_: sum(1 for r in rev if r["metric"] == m_ and r["significant"] and r["versus"] == "skill_nrps") for m_ in CORR_METRICS},
                                      "A1_point_reversals_vs_nrps_skill": {m_: sum(1 for r in rev if r["metric"] == m_ and r["versus"] == "skill_nrps") for m_ in CORR_METRICS},
                                      "significant_S_differences": sum(1 for r in pairs if r.get("holm_S", 1) < ALPHA)}
            print(tag, json.dumps(summary["panels"][tag]["significant_reversals"]), "point", json.dumps(summary["panels"][tag]["point_reversals"]), flush=True)
    write_csv(out / "judge_scores.csv", allrows); write_csv(out / "pairwise.csv", allpairs)
    write_csv(out / "kendall.csv", allken); write_csv(out / "reversals.csv", [{**r, "ci_metric": json.dumps(r["ci_metric"]), "ci_S": json.dumps(r["ci_S"])} for r in allrev])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
