#!/usr/bin/env python3
"""Experiment 56: label budget of per-judge calibration for the five frozen frontier judges (PROTOCOL.md, frozen first).
Usage: python label_budget_frontier.py LABELS_JSON SUMMEVAL_INPUT_DIR OUT_DIR
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "46_real_llm_judge_rank_reversal_corr_vs_proper_scores_start_27Nov2026" / "scripts"))
import analyze_rank_reversal as A  # noqa: E402

SETS5 = ["chaosnli_snli", "chaosnli_mnli", "chaosnli_alphanli", "dices350", "dices990"]
import os
BUDGETS = [int(x) for x in os.environ.get("E56_BUDGETS", "10,25,50,100,200").split(",")]  # 9999 = every pool item (deviation run)
DRAWS, SEED = 50, 20261127


def main(labels_json, se_dir, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    items, counts, groups, fc = A.load_all(labels_json, se_dir, "/nonexistent")
    rows, pair = [], {}
    for si, s in enumerate(SETS5):
        ids = items[s]; C = np.stack([counts[i] for i in ids]).astype(float); H = C / C.sum(1, keepdims=True)
        g = np.array([groups[i] for i in ids]); ug = np.unique(g); prim = A.PRIMARY[s]; K = C.shape[1]
        for ji, j in enumerate(A.FRONTIER):
            Q = A.floor(np.stack([fc[j][i][0] for i in ids]))
            rec = {n: [] for n in BUDGETS}; raw, full, hurt = [], [], {n: 0 for n in BUDGETS}; neg = {n: 0 for n in BUDGETS}
            for d in range(DRAWS):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, 56, si, ji, d]))
                inP = np.isin(g, rng.permutation(ug)[: len(ug) // 2]); P, T = np.where(inP)[0], np.where(~inP)[0]
                base = np.tile(C[P].sum(0) / C[P].sum(), (len(T), 1))
                sb = A.item_scores(base, H[T])[prim].mean()
                ss = lambda q: 1 - A.item_scores(q, H[T])[prim].mean() / sb
                r0 = ss(Q[T]); f0 = ss(A.apply_cal(A.fit_cal(Q[P], C[P]), Q[T])); raw.append(r0); full.append(f0)
                for n in BUDGETS:
                    S = rng.choice(P, size=min(n, len(P)), replace=False)
                    one = np.zeros((len(S), K))
                    for a, i in enumerate(S):
                        one[a, rng.choice(K, p=H[i])] = 1.0
                    v = ss(A.apply_cal(A.fit_cal(Q[S], one), Q[T])); rec[n].append(v)
                    hurt[n] += v < r0; neg[n] += v < 0
                    if s in ("chaosnli_snli", "chaosnli_mnli") and j in ("gpt-6-astra", "glm-5.3"):
                        pair.setdefault((s, n, d), {})[j] = v
            m = {n: float(np.mean(rec[n])) for n in BUDGETS}; mf = float(np.mean(full)); mr = float(np.mean(raw))
            gain = mf - mr
            nstar = next((n for n in BUDGETS if mf - m[n] <= 0.01), None)
            row = {"set": s, "judge": j, "primary": prim, "n_items": len(ids), "SS_raw": mr, "SS_full": mf, "full_gain": gain,
                   "SS_n": m, "SS_n_p05_p95": {n: [float(np.percentile(rec[n], 5)), float(np.percentile(rec[n], 95))] for n in BUDGETS},
                   "share_hurt": {n: hurt[n] / DRAWS for n in BUDGETS}, "share_negative": {n: neg[n] / DRAWS for n in BUDGETS},
                   "n_star": nstar, "in_nstar_summary": gain >= 0.01}
            rows.append(row)
            print(s, j, f"raw {mr:+.3f} full {mf:+.3f}", " ".join(f"{n}:{m[n]:+.3f}" for n in BUDGETS), "n*", nstar, flush=True)
    rev = {}
    for (s, n, d), v in pair.items():
        rev.setdefault(f"{s}|{n}", []).append(v["gpt-6-astra"] < v["glm-5.3"])
    rev = {k: float(np.mean(v)) for k, v in rev.items()}
    (out / "label_budget_frontier.json").write_text(json.dumps({"rows": rows, "astra_below_glm_share": rev}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
