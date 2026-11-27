#!/usr/bin/env python3
"""Independent implementation of QA finding 1b on experiment 36: give BOTH committed heads the same post-hoc calibration.

For each real multi-annotator dataset, with experiment 36's own splits (seed 0, 60/20/20 by group) and heads
(cross-entropy head and the Pearson sampled-rating surrogate head, both exactly as committed: L2 = 0.01, at most 35 iterations):
  1. raw head -> logits Z on validation and test items,
  2. calibration fitted on the VALIDATION split only: softmax(Z / T + b) with one temperature T and K class offsets b,
     minimising validation cross-entropy against the individual ratings (identical procedure for both heads),
  3. held-out test metrics on the TEST split: unbiased per-sample cross-entropy and normalized RPS (RPS / (K - 1)),
  4. paired item-clustered percentile bootstrap (B = 1000, seed 1) of
     (Pearson head) minus (CE head) on the SAME metric and the SAME items; positive = the CE head is better.

Usage:  python3 -I qa_exp36_matched_calibration.py <exp36_dir> <data_dir>
Data: real ratings (SummEval experts, ChaosNLI SNLI+MNLI, DICES-990, and the binary views used by experiment 36).
Nothing is chosen on the test split.  This is a diagnostic of experiment 36, not a new headline result.
"""
from __future__ import annotations

import os
import sys

exp36_dir, data_dir = sys.argv[1], sys.argv[2]
os.environ["SANMI36_DATA"] = data_dir
sys.path.insert(0, exp36_dir)

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import fit_eval  # noqa: E402
from data_io import SUMMEVAL_BINARY_CUT, _load_dices, binarize, load_chaos_3way, load_summeval  # noqa: E402
from run_all import prepare  # noqa: E402

BOOT, BOOT_SEED = 1000, 1


def load(name):
    if name.startswith("summeval_expert"):
        d = load_summeval("expert")
        return binarize(d, SUMMEVAL_BINARY_CUT, name) if name.endswith("binary") else d
    if name == "chaosnli_snli_mnli":
        return load_chaos_3way()
    if name.startswith("dices990"):
        d = _load_dices("990")
        return binarize(d, 2, name) if name.endswith("binary_yes") else d
    raise ValueError(name)


def val_ce(P, ratings, idx):
    """Unbiased per-sample cross-entropy on the given items: mean over items of the mean over that item's ratings."""
    tot = 0.0
    for row, i in enumerate(idx):
        y = np.asarray(ratings[i], dtype=int)
        tot += float(-np.log(np.clip(P[row, y], 1e-300, 1.0)).mean())
    return tot / len(idx)


def calibrate(Zva, ratings, val_idx, K):
    def f(th):
        return val_ce(fit_eval.softmax(Zva * np.exp(-th[0]) + th[1:]), ratings, val_idx)
    r = minimize(f, np.zeros(K + 1), method="L-BFGS-B")
    return r.x


def item_metrics(P, d, idx, K):
    v = np.asarray(d.class_values, dtype=float)
    rec = fit_eval._item_records(P, P @ v, d.ratings, idx, np.asarray(d.class_values))
    return rec["ce"], rec["rps"] / (K - 1)


def paired(a, b):
    """mean(b) - mean(a) with a paired item-clustered percentile interval; a = CE head, b = Pearson head."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(a)
    diffs = []
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        diffs.append(float(b[i].mean() - a[i].mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(b.mean() - a.mean()), float(lo), float(hi)


print("dataset | metric (held-out test, per-sample, item-clustered paired bootstrap B=1000) | raw heads: Pearson head minus CE head | after the same validation calibration: Pearson head minus CE head")
for name in ("summeval_expert", "chaosnli_snli_mnli", "dices990", "summeval_expert_binary", "dices990_binary_yes"):
    d = load(name)
    prep = prepare(d)
    K = d.n_classes
    raw, cal = {}, {}
    for obj in ("ce", "pearson"):
        W = fit_eval.fit_objective(prep["x_train"], d.ratings, prep["train"], d.class_values, obj, None, 0)["weight"]
        Zva, Zte = prep["x_val"] @ W, prep["x_test"] @ W
        raw[obj] = item_metrics(fit_eval.softmax(Zte), d, prep["test"], K)
        th = calibrate(Zva, d.ratings, prep["val"], K)
        cal[obj] = item_metrics(fit_eval.softmax(Zte * np.exp(-th[0]) + th[1:]), d, prep["test"], K)
    for j, metric in ((1, "normalized RPS"), (0, "cross-entropy (nats)")):
        r = paired(raw["ce"][j], raw["pearson"][j])
        c = paired(cal["ce"][j], cal["pearson"][j])
        print(f"{name} | {metric} | {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}] | {c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]")
