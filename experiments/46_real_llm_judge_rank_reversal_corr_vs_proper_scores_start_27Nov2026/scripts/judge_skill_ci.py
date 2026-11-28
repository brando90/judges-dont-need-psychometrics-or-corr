#!/usr/bin/env python3
"""Experiment 46 A1 (reporting only): per-judge skill score vs the cross-fitted base, with item-cluster bootstrap CIs
(4,000 draws, seed SeedSequence([20261127, 46, 99, set_index])), raw and 5-fold cross-fitted calibrated, frontier panel on all items.
Usage: python judge_skill_ci.py LABELS_JSON SUMMEVAL_INPUT_DIR LOCAL_DIRS OUT_JSON"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_rank_reversal as A  # noqa: E402

def main(labels, se_dir, local, out):
    items, counts, groups, fc = A.load_all(labels, se_dir, local)
    judges = A.FRONTIER + ["g-eval-gpt-4|sampled20"]
    rows = []
    for si, s in enumerate(A.SETS):
        for j in judges:
            ids = [i for i in items[s] if i in fc.get(j, {})]
            if len(ids) < 20: continue
            C = np.stack([counts[i] for i in ids]); H = C / C.sum(1, keepdims=True); g = [groups[i] for i in ids]
            Q = A.floor(np.stack([fc[j][i][0] for i in ids])); fold = A.folds_for(g, si)
            base = np.zeros_like(H); Qc = np.zeros_like(Q)
            for f in range(5):
                tr, te = fold != f, fold == f; m = C[tr].sum(0); base[te] = A.floor(m / m.sum())
                Qc[te] = A.apply_cal(A.fit_cal(Q[tr], C[tr]), Q[te])
            prim = A.PRIMARY[s]; sb = A.item_scores(base, H)[prim]
            ug, ginv = np.unique(g, return_inverse=True)
            rng = np.random.default_rng(np.random.SeedSequence([A.SEED, 46, 99, si]))
            W = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[ginv] for _ in range(4000)]).astype(float)
            bb = (W @ sb) / W.sum(1)
            r = {"set": s, "judge": j, "n_items": len(ids), "primary": prim, "S_base": float(sb.mean())}
            for tag, q in (("raw", Q), ("calibrated", Qc)):
                sj = A.item_scores(q, H)[prim]; ss = 1 - ((W @ sj) / W.sum(1)) / bb
                r[f"skill_{tag}"] = float(1 - sj.mean() / sb.mean()); r[f"skill_{tag}_ci"] = [float(np.percentile(ss, 2.5)), float(np.percentile(ss, 97.5))]
            r.update({m: v for m, v in A.corr_stats(Q, H, np.ones(len(ids))).items()})
            rows.append(r)
            print(f"{s:22s} {j:24s} skill raw {r['skill_raw']:+.3f} [{r['skill_raw_ci'][0]:+.3f},{r['skill_raw_ci'][1]:+.3f}] cal {r['skill_calibrated']:+.3f} [{r['skill_calibrated_ci'][0]:+.3f},{r['skill_calibrated_ci'][1]:+.3f}] pearson {r['pearson']:.3f} spearman {r['spearman']:.3f} top1 {r['top1']:.3f}", flush=True)
    Path(out).write_text(json.dumps(rows, indent=1))

if __name__ == "__main__":
    main(*sys.argv[1:5])
