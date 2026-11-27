"""Can the quoted single-pass logit rows be recomputed from the bridged files?

The two bridged rating CSVs hold 175 items (100 new + 75 old) with their sorted 0-5 scores, the
same 175 item ids as the logit JSONs. Recompute, per item, the mean over its ratings of the
ranked probability score (raw and divided by K-1 = 5) and of -log q(y), then the item average;
also the item-equal mean human score. Compare with the quoted rows (first two columns).
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

P = Path("/tmp/e40_cursor/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/provenance/ttj_efdc8c3")
rows = list(csv.DictReader(open(P / "human_ratings_new_11-26-2026.csv"))) + list(csv.DictReader(open(P / "human_ratings_old_11-26-2026.csv")))
ratings = {r["item"]: [int(float(t)) for t in r["ratings_sorted_0to5_rater_identity_removed"].split()] for r in rows}
split = {r["item"]: r["exp27_split"] for r in rows}


def rps(q, y):
    c = np.cumsum(q)[:-1]
    ind = (np.arange(len(q) - 1) >= y).astype(float)
    return float(np.sum((c - ind) ** 2))


for fn in ("qwen3_4b_logit_6144.json", "qwen3_8b_logit_6144.json"):
    data = json.loads((P / fn).read_text())
    for scope in ("all 175 items", "exp27 test only"):
        nrps, ce, means, n = [], [], [], 0
        for r in data:
            it = r["item"]
            if scope != "all 175 items" and split[it] != "test":
                continue
            q = np.asarray(r["probs"], float)
            ys = ratings[it]
            nrps.append(np.mean([rps(q, y) / 5 for y in ys]))
            ce.append(np.mean([-math.log(max(q[y], 1e-300)) for y in ys]))
            means.append(np.mean(ys))
            n += 1
        print(f"{fn} [{scope}, n={n}]: normalized RPS {np.mean(nrps):.4f}; CE {np.mean(ce):.3f}; "
              f"item-equal mean human score {np.mean(means):.3f}")
print("quoted: Qwen3-4B single-pass logits 0.1482 / 3.589 (mean human 3.742); Qwen3-8B 0.2145 / 5.375")
