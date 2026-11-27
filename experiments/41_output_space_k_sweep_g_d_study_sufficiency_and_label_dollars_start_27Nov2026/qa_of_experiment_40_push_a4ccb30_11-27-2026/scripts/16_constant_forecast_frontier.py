"""Would a forecast that ignores the text meet experiment 40's dollar tolerance?

For each dataset and each grid budget B, estimate a constant (train-marginal) forecast from B
train ratings (one rating from each of B seed-0-permuted train prompts, add-one smoothing; all
train prompts if B exceeds them) and score it on the same held-out test prompts with the same
unbiased per-prompt cross-entropy. Compare with the committed frontier (min over k of the
CE-trained head's test CE) and with the committed target (full-train head + 0.05 nats).
Usage: python3 16_constant_forecast_frontier.py <exp40_dir>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

EXP40 = Path(sys.argv[1])
sys.path.insert(0, str(EXP40))
import run_all as ra  # noqa: E402

res = json.loads(Path("/tmp/e40_cursor/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/results.json").read_text())
specs = [
    ("summeval_expert", lambda: ra.load_summeval("expert")),
    ("summeval_crowd", lambda: ra.load_summeval("crowd")),
    ("popquorn_offensiveness", lambda: ra.load_popquorn("offensiveness")),
    ("popquorn_politeness", lambda: ra.load_popquorn("politeness")),
    ("dices990", lambda: ra._load_dices("990")),
    ("chaosnli_snli_mnli", lambda: ra.load_chaos_3way()),
    ("veribench_fsc_native_main", lambda: ra.load_veribench_fsc_native_main()),
]
for name, loader in specs:
    data = loader()
    if name.startswith("veribench"):
        sp = ra.exp27_split(data)
        train, test = sp["train"], sp["test"]
    else:
        train, _val, test = ra.split_by_group(data.groups, seed=0)
    K = len(data.class_values)
    order = np.random.default_rng(0).permutation(np.asarray(train))
    rng = np.random.default_rng(1)
    block = res["datasets"][name]
    target = block["dollars"]["ce_target"]
    out = []
    first_hit = None
    for row in block["allocation"]["budgets"]:
        B = row["B"]
        cnt = np.ones(K)  # add-one smoothing
        for i in order[:B]:
            ys = data.ratings[int(i)]
            cnt[int(ys[rng.integers(0, ys.size)])] += 1
        q = cnt / cnt.sum()
        ce = float(np.mean([np.mean([-math.log(q[int(y)]) for y in data.ratings[int(i)]]) for i in test]))
        if first_hit is None and ce <= target:
            first_hit = B
        out.append(f"B={B}: const {ce:.4f} / head frontier {row['frontier_ce']:.4f}")
    print(f"{name}: target (full-train head + 0.05) = {target:.4f}; committed head hit B={block['dollars']['B_ce']}; "
          f"constant-forecast hit B={first_hit}")
    print("    " + "; ".join(out))
