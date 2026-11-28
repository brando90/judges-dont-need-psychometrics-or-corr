#!/usr/bin/env python3
"""Experiment 48: calibration under dataset, rubric and rater-pool shift (PROTOCOL.md, frozen before this script existed).
Usage: python calibration_shift.py LABELS_JSON SUMMEVAL_INPUT_DIR OUT_DIR
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXPS = HERE.parents[1]
sys.path.insert(0, str(EXPS / "46_real_llm_judge_rank_reversal_corr_vs_proper_scores_start_27Nov2026" / "scripts"))
sys.path.insert(0, str(EXPS / "47_calibration_label_budget_sweep_exp41_heads_start_27Nov2026" / "scripts"))
import analyze_rank_reversal as A  # noqa: E402  (Exp 46 scorers, calibrator, folds, loaders)

BOOT, SEED = 4000, 20261127
K3 = ["chaosnli_snli", "chaosnli_mnli", "dices350", "dices990"]
NEAR = {("chaosnli_snli", "chaosnli_mnli"), ("chaosnli_mnli", "chaosnli_snli"), ("dices350", "dices990"), ("dices990", "dices350")}


def arm_a(labels_json, se_dir):
    items, counts, groups, fc = A.load_all(labels_json, se_dir, "/nonexistent")
    def mats(s, judge):
        ids = items[s]
        C = np.stack([counts[i] for i in ids]); Q = A.floor(np.stack([fc[judge][i][0] for i in ids]))
        return ids, C, Q, [groups[i] for i in ids]
    rows = []
    plan = [(j, src, tgt) for j in A.FRONTIER for src in K3 for tgt in K3 if src != tgt]
    plan += [(j, "pooled_other_three", tgt) for j in A.FRONTIER for tgt in K3]
    plan += [("g-eval-gpt-4|sampled20", "summeval_relevance", "summeval_consistency"), ("g-eval-gpt-4|sampled20", "summeval_consistency", "summeval_relevance")]
    cache = {}
    for ti, (judge, src, tgt) in enumerate(plan):
        ids, C, Q, g = mats(tgt, judge)
        H = C / C.sum(1, keepdims=True); prim = A.PRIMARY[tgt]
        if (judge, tgt) not in cache:
            fold = A.folds_for(g, A.SETS.index(tgt)); qin = np.zeros_like(Q)
            for f in range(5):
                tr, te = fold != f, fold == f
                qin[te] = A.apply_cal(A.fit_cal(Q[tr], C[tr]), Q[te])
            cache[(judge, tgt)] = qin
        qin = cache[(judge, tgt)]
        if src == "pooled_other_three":
            srcs = [s for s in K3 if s != tgt]
            Qs = np.concatenate([mats(s, judge)[2] for s in srcs]); Cs = np.concatenate([mats(s, judge)[1] for s in srcs])
        else:
            _, Cs, Qs, _ = mats(src, judge)
        qsh = A.apply_cal(A.fit_cal(Qs, Cs), Q)
        sc = {k: A.item_scores(v, H) for k, v in (("raw", Q), ("in", qin), ("shift", qsh))}
        ug, ginv = np.unique(g, return_inverse=True)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 48, ti]))
        W = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[ginv] for _ in range(BOOT)]).astype(float)
        def bm(x): return (W @ x) / W.sum(1)
        r = {"judge": judge, "source": src, "target": tgt, "shift": "rubric" if tgt.startswith("summeval") else ("near" if (src, tgt) in NEAR else ("leave_one_set_out" if src.startswith("pooled") else "far")),
             "primary": prim, "n_target_items": len(ids)}
        for m in (prim, "ce"):
            raw, inn, sh = sc["raw"][m], sc["in"][m], sc["shift"][m]
            D = bm(sh) - bm(inn); gain_in = bm(raw) - bm(inn); F = (bm(raw) - bm(sh)) / np.where(np.abs(gain_in) > 1e-12, gain_in, np.nan)
            Dr = bm(sh) - bm(raw)
            r[m] = {"S_raw": float(raw.mean()), "S_in_domain": float(inn.mean()), "S_shifted": float(sh.mean()),
                    "D_shift_minus_in": float(sh.mean() - inn.mean()), "D_ci": [float(np.percentile(D, 2.5)), float(np.percentile(D, 97.5))],
                    "F_gain_retained": float((raw.mean() - sh.mean()) / (raw.mean() - inn.mean())) if abs(raw.mean() - inn.mean()) > 1e-12 else None,
                    "F_ci": [float(np.nanpercentile(F, 2.5)), float(np.nanpercentile(F, 97.5))],
                    "shift_minus_raw": float(sh.mean() - raw.mean()), "shift_minus_raw_ci": [float(np.percentile(Dr, 2.5)), float(np.percentile(Dr, 97.5))]}
        p = r[prim]
        r["reopens"] = bool(p["D_ci"][0] > 0 and p["F_gain_retained"] is not None and p["F_gain_retained"] < 0.5)
        r["calibration_fails_outright"] = bool(p["shift_minus_raw_ci"][0] > 0)
        rows.append(r)
        print(judge, src, "->", tgt, f"raw {p['S_raw']:.4f} in {p['S_in_domain']:.4f} shift {p['S_shifted']:.4f} F {p['F_gain_retained']}", "reopens" if r["reopens"] else "", flush=True)
    return rows


def arm_b():
    from heads41 import build_views, heads, item_nrps_ce, map_from, paired_ci, fit_logit_map, softmax, histograms
    views = build_views(); byname = {v[0]: v for v in views}
    rows = []
    def zstd(x, ref):
        m, s = float(ref.mean()), float(ref.std()); return (x - m) / (s if s > 1e-12 else 1.0)
    # (1) rater-pool shift, SummEval expert <-> crowd
    for K in (2, 3, 5):
        for tgt, src in ((f"summeval_crowd|K{K}", f"summeval_expert|K{K}"), (f"summeval_expert|K{K}", f"summeval_crowd|K{K}")):
            dt, ds = byname[tgt][3], byname[src][3]
            assert dt.texts == ds.texts and list(dt.groups) == list(ds.groups)
            for seed in range(5):
                H = heads(tgt, dt, seed); val, test = H["val"], H["test"]
                ce_raw, _ = item_nrps_ce(H["ce_pt"], dt.ratings, test, K)
                q_in = map_from(H["pearson_mv"], histograms(dt.ratings, val, K), H["pearson_mt"])
                q_sh = map_from(H["pearson_mv"], histograms(ds.ratings, val, K), H["pearson_mt"])  # map fitted on the SOURCE pool's ratings
                s_in, _ = item_nrps_ce(q_in, dt.ratings, test, K); s_sh, _ = item_nrps_ce(q_sh, dt.ratings, test, K)
                g_in, g_sh = paired_ci(s_in, ce_raw), paired_ci(s_sh, ce_raw); dd = paired_ci(s_sh, s_in)
                rows.append({"arm": "rater_pool", "target": tgt, "source": src, "K": K, "seed": seed, "G_in": g_in, "G_shift": g_sh, "shift_minus_in": dd})
    # (2) leave-one-dataset-out within K (Pearson-trained heads)
    for K in (2, 3, 5):
        vs = [v for v in views if v[2] == K]
        for tname, tbase, _, dt in vs:
            for seed in range(5):
                H = heads(tname, dt, seed); val, test = H["val"], H["test"]
                ce_raw, _ = item_nrps_ce(H["ce_pt"], dt.ratings, test, K)
                q_in = map_from(H["pearson_mv"], histograms(dt.ratings, val, K), H["pearson_mt"])
                zs, ts = [], []
                for sname, sbase, _, dsrc in vs:
                    if sbase == tbase:
                        continue
                    Hs = heads(sname, dsrc, seed)
                    zs.append(zstd(Hs["pearson_mv"], Hs["pearson_mv"])); ts.append(histograms(dsrc.ratings, Hs["val"], K))
                if not zs:
                    continue
                w = fit_logit_map(np.concatenate(zs), np.concatenate(ts))
                zt = zstd(H["pearson_mt"], H["pearson_mv"])  # target standardized with its own unlabeled validation scores
                q_sh = softmax(np.column_stack([zt, np.ones_like(zt)]) @ w)
                s_in, _ = item_nrps_ce(q_in, dt.ratings, test, K); s_sh, _ = item_nrps_ce(q_sh, dt.ratings, test, K)
                rows.append({"arm": "leave_one_dataset_out", "target": tname, "source": "other same-K views", "K": K, "seed": seed,
                             "G_in": paired_ci(s_in, ce_raw), "G_shift": paired_ci(s_sh, ce_raw), "shift_minus_in": paired_ci(s_sh, s_in)})
    summ = {}
    for r in rows:
        summ.setdefault((r["arm"], r["target"]), []).append(r)
    table = []
    for (arm, tgt), rs in summ.items():
        gs = np.mean([r["G_shift"][0] for r in rs]); gi = np.mean([r["G_in"][0] for r in rs])
        above = sum(r["G_shift"][1] > 0 for r in rs)
        table.append({"arm": arm, "target": tgt, "source": rs[0]["source"], "G_in_seed_mean": float(gi), "G_shift_seed_mean": float(gs),
                      "G_shift_seed_range": [float(min(r["G_shift"][0] for r in rs)), float(max(r["G_shift"][0] for r in rs))],
                      "seeds_G_shift_ci_above_0": int(above), "reopens": bool(gs > 0.0024 and above >= 3)})
        print(arm, tgt, f"G_in {gi:+.4f} G_shift {gs:+.4f} seeds>0 {above}", "REOPENS" if table[-1]["reopens"] else "", flush=True)
    return rows, table


def main(labels_json, se_dir, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    a = arm_a(labels_json, se_dir)
    (out / "arm_a_llm_judges.json").write_text(json.dumps(a, indent=1))
    rows, table = arm_b()
    (out / "arm_b_exp41_heads.json").write_text(json.dumps({"rows": rows, "table": table}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
