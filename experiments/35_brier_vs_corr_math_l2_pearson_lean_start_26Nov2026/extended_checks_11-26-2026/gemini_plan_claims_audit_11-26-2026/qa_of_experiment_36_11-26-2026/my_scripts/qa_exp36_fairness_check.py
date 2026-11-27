#!/usr/bin/env python3
"""Independent check of QA finding 1 on experiment 36 (is the Pearson-trained head a fair competitor?).

Uses experiment 36's OWN code and data splits (commit 036cafa) and changes only two things, one at a time:
the L2 penalty (default 1e-2, as committed) and the L-BFGS-B iteration budget (default 35, as committed).
Reports, for the committed Pearson surrogate (correlation of a SAMPLED rating):
  - the training value of that surrogate at the solution (no penalty term),
  - held-out cross-entropy and held-out normalized ranked probability score (RPS) of the head,
and the same two held-out numbers for the committed cross-entropy head and for the constant forecast.

Usage:  python3 -I qa_exp36_fairness_check.py <exp36_dir> <data_dir>
  <exp36_dir>  a copy of experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026
  <data_dir>   directory with the public datasets as downloaded by experiment 36's data_io.py (SANMI36_DATA)
Data: real multi-annotator ratings (SummEval experts, ChaosNLI SNLI+MNLI, DICES-990); test split of experiment 36 (seed 0, 60/20/20 by group).
Nothing is selected on the test split here: this is a diagnostic, not a new result.
"""
from __future__ import annotations

import os
import sys

exp36_dir, data_dir = sys.argv[1], sys.argv[2]
os.environ["SANMI36_DATA"] = data_dir
sys.path.insert(0, exp36_dir)

import numpy as np  # noqa: E402

import fit_eval  # noqa: E402
from data_io import SUMMEVAL_BINARY_CUT, _load_dices, binarize, load_chaos_3way, load_summeval  # noqa: E402,F401
from run_all import prepare  # noqa: E402


def load(name):
    if name == "summeval_expert":
        return load_summeval("expert")
    if name == "chaosnli_snli_mnli":
        return load_chaos_3way()
    if name == "dices990":
        return _load_dices("990")
    raise ValueError(name)


def surrogate_corr(W, x, targets, ybar_value, values):
    """The committed Pearson surrogate WITHOUT the penalty: Corr(sampled rating, human) as coded in fit_eval._fit_softmax."""
    prob = fit_eval.softmax(x @ W)
    mu = prob @ values
    var_c = prob @ (values ** 2) - mu ** 2
    mu_c = mu - mu.mean()
    y_c = ybar_value - ybar_value.mean()
    cov = float(np.mean(mu_c * y_c))
    var_x = float(np.mean(var_c) + np.mean(mu_c ** 2))
    second = targets @ (values ** 2)
    var_y = float(np.mean(second - (targets @ values) ** 2) + np.var(targets @ values))
    return cov / np.sqrt(var_x * var_y)


def held_out(P, d, prep, K):
    v = np.asarray(d.class_values, dtype=float)
    mu = P @ v
    rec = fit_eval._item_records(P, mu, d.ratings, prep["test"], np.asarray(d.class_values))
    return float(rec["ce"].mean()), float(rec["rps"].mean()) / (K - 1)


print("dataset | head | L2 | maxiter | train surrogate corr (no penalty) | held-out CE (nats) | held-out normalized RPS")
for name in ("summeval_expert", "chaosnli_snli_mnli", "dices990"):
    d = load(name)
    prep = prepare(d)
    K = d.n_classes
    v = np.asarray(d.class_values, dtype=float)
    rng = np.random.default_rng(0)
    targets, _, kept = fit_eval._soft_targets(d.ratings, prep["train"], K, None, rng)
    ybar_value = np.array([float(v[ys].mean()) for ys in kept])
    x = prep["x_train"]
    # constant forecast and the committed cross-entropy head
    prob, _ = fit_eval.constant_marginal(d.ratings, prep["train"], K, d.class_values)
    Pm = np.repeat(prob[None, :], len(prep["test"]), 0)
    ce_m, rp_m = held_out(Pm, d, prep, K)
    print(f"{name} | constant train-marginal forecast | - | - | - | {ce_m:.4f} | {rp_m:.4f}")
    fit_eval.MAXITER = 35
    fit_eval._penalty.__defaults__ = (1e-2,)
    pk = fit_eval.fit_objective(x, d.ratings, prep["train"], d.class_values, "ce", None, 0)
    Pc = fit_eval.softmax(prep["x_test"] @ pk["weight"])
    ce_c, rp_c = held_out(Pc, d, prep, K)
    print(f"{name} | cross-entropy head (committed) | 0.01 | 35 | - | {ce_c:.4f} | {rp_c:.4f}")
    for l2, maxit in ((1e-2, 35), (1e-2, 300), (1e-3, 300), (1e-4, 300), (0.0, 300)):
        fit_eval.MAXITER = maxit
        fit_eval._penalty.__defaults__ = (l2,)
        pk = fit_eval.fit_objective(x, d.ratings, prep["train"], d.class_values, "pearson", None, 0)
        W = pk["weight"]
        tr = surrogate_corr(W, x, targets, ybar_value, v)
        P = fit_eval.softmax(prep["x_test"] @ W)
        ce, rp = held_out(P, d, prep, K)
        print(f"{name} | Pearson head | {l2:g} | {maxit} | {tr:.4f} | {ce:.4f} | {rp:.4f}")
