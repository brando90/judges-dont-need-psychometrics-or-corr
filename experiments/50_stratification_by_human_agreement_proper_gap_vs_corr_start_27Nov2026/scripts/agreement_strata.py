#!/usr/bin/env python3
"""Experiment 50: proper-score gaps, skill, correlation and agreement by human-agreement quintile (PROTOCOL.md, frozen first).
Usage: python agreement_strata.py LABELS_JSON SUMMEVAL_INPUT_DIR OUT_DIR
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent; EXPS = HERE.parents[1]
sys.path.insert(0, str(EXPS / "46_real_llm_judge_rank_reversal_corr_vs_proper_scores_start_27Nov2026" / "scripts"))
sys.path.insert(0, str(EXPS / "47_calibration_label_budget_sweep_exp41_heads_start_27Nov2026" / "scripts"))
import analyze_rank_reversal as A  # noqa: E402
SEED, BOOT, NB = 20261127, 2000, 5


def halves(ratings_list, pos0=0):
    """For each item: (half A labels, half B labels, same_ratings_flag). Items with >= 4 ratings are split; others reuse all ratings."""
    out = []
    for p, ys in enumerate(ratings_list):
        ys = np.asarray(ys, int)
        if ys.size >= 4:
            perm = np.random.default_rng(SEED + pos0 + p).permutation(ys.size); h = ys.size // 2
            out.append((ys[perm[:h]], ys[perm[h:]], False))
        else:
            out.append((ys, ys, True))
    return out


def entropy(ys, k):
    c = np.bincount(ys, minlength=k) / ys.size; c = c[c > 0]; return float(-(c * np.log(c)).sum())


def quintiles(ent):
    """Five equal-count bins by entropy (ties broken by a fixed random jitter so bins have equal counts)."""
    jitter = np.random.default_rng(SEED).random(len(ent)) * 1e-9
    order = np.argsort(ent + jitter); b = np.empty(len(ent), int)
    for q, chunk in enumerate(np.array_split(order, NB)):
        b[chunk] = q
    return b


def boot_mean(x, rng):
    idx = rng.integers(0, len(x), (BOOT, len(x))); return x[idx].mean(1)


def wpear(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def arm_a():
    from heads41 import build_views, heads, map_from, histograms
    rows = []
    for vname, base, k, data in build_views():
        for seed in range(5):
            H = heads(vname, data, seed); val, test = H["val"], H["test"]
            hv = halves([data.ratings[i] for i in test], pos0=1000 * seed)
            HB = np.stack([np.bincount(b, minlength=k) / b.size for _, b, _ in hv])
            ent = np.array([entropy(a, k) for a, _, _ in hv]); qb = quintiles(ent)
            same = any(s for _, _, s in hv)
            def nrps(q):
                F, G = np.cumsum(q, 1)[:, :-1], np.cumsum(HB, 1)[:, :-1]; return (F * F - 2 * F * G + G).sum(1) / (k - 1)
            vals = np.asarray(data.class_values, float); vals = (vals - vals.min())
            sc = {}
            for obj in ("ce", "pearson"):
                pt = H[f"{obj}_pt"]; qc = map_from(H[f"{obj}_mv"], histograms(data.ratings, val, k), H[f"{obj}_mt"])
                sc[obj] = {"raw": nrps(pt), "cal": nrps(qc), "mean": pt @ np.arange(k)}
            ymean = HB @ np.arange(k)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 50, seed]))
            per_bin = []
            for q in range(NB):
                m = qb == q
                rg, mg = sc["pearson"]["raw"][m] - sc["ce"]["raw"][m], sc["pearson"]["cal"][m] - sc["ce"]["cal"][m]
                br, bm = boot_mean(rg, rng), boot_mean(mg, rng)
                per_bin.append({"bin": q + 1, "n": int(m.sum()), "entropy_range": [float(ent[m].min()), float(ent[m].max())],
                                "raw_gap": float(rg.mean()), "raw_gap_ci": [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
                                "matched_gap": float(mg.mean()), "matched_gap_ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                                "pearson_ce_head": wpear(sc["ce"]["mean"][m], ymean[m]), "pearson_pearson_head": wpear(sc["pearson"]["mean"][m], ymean[m]),
                                "_br": br})
            d = per_bin[-1]["_br"] - per_bin[0]["_br"]
            for pb in per_bin:
                pb.pop("_br")
            rows.append({"view": vname, "K": k, "seed": seed, "same_ratings_binning": bool(same), "n_test": int(len(test)),
                         "pooled_pearson_ce_head": wpear(sc["ce"]["mean"], ymean), "pooled_pearson_pearson_head": wpear(sc["pearson"]["mean"], ymean),
                         "bins": per_bin, "raw_gap_Q5_minus_Q1": float(per_bin[-1]["raw_gap"] - per_bin[0]["raw_gap"]),
                         "raw_gap_Q5_minus_Q1_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]})
            print(vname, seed, f"Q1 raw {per_bin[0]['raw_gap']:+.4f} Q5 raw {per_bin[-1]['raw_gap']:+.4f}", flush=True)
    return rows


def arm_b(labels_json, se_dir):
    items, counts, groups, fc = A.load_all(labels_json, se_dir, "/nonexistent")
    rows = []
    for si, s in enumerate(A.SETS):
        judges = A.FRONTIER if si < 5 else ["g-eval-gpt-4|sampled20"]
        ids = items[s]; k = len(counts[ids[0]])
        rl = [np.repeat(np.arange(k), counts[i].astype(int)) for i in ids]
        hv = halves(rl, pos0=100000 * si)
        HB = np.stack([np.bincount(b, minlength=k) / b.size for _, b, _ in hv]); ent = np.array([entropy(a, k) for a, _, _ in hv]); qb = quintiles(ent)
        g = [groups[i] for i in ids]; fold = A.folds_for(g, si); C = np.stack([counts[i] for i in ids])
        base = np.zeros((len(ids), k))
        for f in range(5):
            m_ = C[fold != f].sum(0); base[fold == f] = A.floor(m_ / m_.sum())
        prim = A.PRIMARY[s]
        Sb = A.item_scores(base, HB)[prim]
        Q = {j: A.floor(np.stack([fc[j][i][0] for i in ids])) for j in judges}
        S = {j: A.item_scores(Q[j], HB)[prim] for j in judges}
        ym = HB @ np.arange(k)
        for q in range(NB):
            m = qb == q; rec = {"set": s, "bin": q + 1, "n": int(m.sum()), "entropy_range": [float(ent[m].min()), float(ent[m].max())],
                                 "same_ratings_binning": bool(hv[0][2]), "primary": prim, "judges": {}}
            for j in judges:
                top1 = float((A.mode_w(Q[j][m]) * A.mode_w(HB[m])).sum(1).mean())
                rec["judges"][j] = {"S": float(S[j][m].mean()), "skill": float(1 - S[j][m].mean() / Sb[m].mean()),
                                    "pearson": wpear(Q[j][m] @ np.arange(k), ym[m]), "top1": top1}
            if len(judges) > 1:
                xs = [rec["judges"][j]["pearson"] for j in judges]; ys = [-rec["judges"][j]["S"] for j in judges]
                rec["kendall_pearson_vs_S"] = float(kendalltau(xs, ys).statistic)
                rec["best_by_pearson"] = judges[int(np.nanargmax(xs))]; rec["best_by_S"] = judges[int(np.argmax(ys))]
            rows.append(rec)
        pooled = {j: {"pearson": wpear(Q[j] @ np.arange(k), ym), "S": float(S[j].mean()), "top1": float((A.mode_w(Q[j]) * A.mode_w(HB)).sum(1).mean())} for j in judges}
        rows.append({"set": s, "bin": "pooled", "n": len(ids), "judges": pooled})
        print(s, "done", flush=True)
    return rows


def main(labels_json, se_dir, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    b = arm_b(labels_json, se_dir); (out / "arm_b_llm_judges.json").write_text(json.dumps(b, indent=1))
    a = arm_a(); (out / "arm_a_exp41_heads.json").write_text(json.dumps(a, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
