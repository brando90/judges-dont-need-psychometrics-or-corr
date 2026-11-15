"""Experiment 08 (11-14-2026): how many ratings does a loss-trained judge need to beat a correlation-trained one,
as a function of where the target's information lies, and with validation-tuned regularization?

Reuses experiment 13's judge family (softmax of 13 cosine/sine features), objective code, optimizer (L-BFGS-B,
400 iterations / 600 evaluations), starting point and test metrics, imported from its frozen scripts/toy.py.

Scenarios (human ratings R | x ~ BetaBin(5, mu(x) kappa(x), (1 - mu(x)) kappa(x)), x ~ Unif[-1, 1]):
  A  mean nearly constant (mu = 0.5 + 0.08 x), disagreement varies (kappa(x) = exp(1.5 + 2.2 cos(1.5 pi x)))  [exp 13]
  B  mean monotone nonlinear (mu = 0.1 + 0.8 ((x + 1)/2)^2), disagreement varies as in A                       [exp 13]
  C  mean monotone nonlinear as in B, disagreement constant (kappa = exp(1.5))   -- information only in the mean
  D  mean linear (mu = 0.5 + 0.35 x), disagreement constant (kappa = exp(1.5))   -- correlation's most favorable case
Scenarios A and B regenerate experiment 13's exact data.

Judges per (scenario, n, seed):
  constant          the training-task rating histogram, predicted for every test task (no fitting)
  <kind>_fixed      ridge c = 1 as in experiment 13 (kind = ce, rps, pearson; pearson_fixed is formulation 2, ce_fixed 4)
  <kind>_valrps     ridge c chosen from {0.01, 0.1, 1, 10, 100} by 4-fold cross-validated RPS over the training tasks
                    (ce_valrps is formulation 3 with the project's common validation criterion)
  pearson_valcorr   ridge c chosen by 4-fold cross-validated Pearson correlation (formulation 1)
Ties in a validation criterion go to the larger c (more regularization). Test data never enter any choice.
"""
import os
import sys

import numpy as np
from scipy.special import softmax

EXP13 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "archive",
                     "13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026", "scripts")
sys.path.insert(0, os.path.abspath(EXP13))
import toy  # noqa: E402

CODE = {"A": 0, "B": 1, "C": 2, "D": 3}
KAPPA_CONST = float(np.exp(1.5))
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
FOLDS = 4
KINDS = ["ce", "rps", "pearson"]


def truth(x, sc):
    if sc in "AB":
        return toy.truth(x, sc)
    mu = 0.1 + 0.8 * ((x + 1) / 2) ** 2 if sc == "C" else 0.5 + 0.35 * x
    return toy.betabinom(mu, np.full_like(x, KAPPA_CONST))


def test_metrics(Q, Pt, xt, Qconst=None):
    m = toy.metrics(Q, Pt, xt)
    out = {k: float(m[k]) for k in ("rps_excess", "ce_excess", "pearson", "spearman")}
    if Qconst is not None:
        out["tvd_from_constant"] = float(0.5 * np.abs(Q - Qconst).sum(1).mean())
    return out


def val_scores(Phi, P, kind, c, fold, seed):
    """Pooled out-of-fold validation RPS (lower better) and Pearson (higher better) for ridge c."""
    n = len(P)
    Q = np.zeros_like(P)
    for f in range(FOLDS):
        tr, va = fold != f, fold == f
        W = toy.fit(Phi[tr], P[tr], kind, int(tr.sum()), toy.marginal_W0(Phi[tr], P[tr], seed), c=c)
        Q[va] = softmax(Phi[va] @ W, 1)
    pc, qc = np.cumsum(P, 1)[:, :5], np.cumsum(Q, 1)[:, :5]
    rps = float(((qc - pc) ** 2 + pc * (1 - pc)).mean())
    m, h = Q @ toy.GRID, P @ toy.GRID
    corr = float(np.corrcoef(m, h)[0, 1]) if m.std() > 1e-12 and h.std() > 1e-12 else -np.inf
    return rps, corr


def pick(cs, scores, lower_better):
    """Best c among finite scores; ties (within 1e-12) go to the larger c. None if no score is finite.

    Amendment (11-14-2026, found when the first launch crashed, before any analysis): with very few training tasks
    every candidate can have an undefined validation correlation; the caller then falls back to c = 1 and flags it.
    """
    finite = [(c, s) for c, s in zip(cs, scores) if np.isfinite(s)]
    if not finite:
        return None
    best = min(s for _, s in finite) if lower_better else max(s for _, s in finite)
    return max(c for c, s in finite if abs(s - best) <= 1e-12)


def run_one(args):
    sc, n, seed, k, ntest = args
    rng = np.random.default_rng(1000003 * seed + 7919 * n + CODE[sc])
    x = rng.uniform(-1, 1, n)
    P = np.array([rng.multinomial(k, p) for p in truth(x, sc)]) / k
    Phi = toy.phi(x)
    W0 = toy.marginal_W0(Phi, P, seed)
    xt = rng.uniform(-1, 1, ntest)
    Pt = truth(xt, sc)
    Phit = toy.phi(xt)
    fold = np.empty(n, int)
    fold[np.random.default_rng([seed, n, CODE[sc], 99]).permutation(n)] = np.arange(n) % FOLDS
    Qconst = np.tile(P.mean(0), (ntest, 1))
    out = {"constant": test_metrics(Qconst, Pt, xt, Qconst)}
    for kind in KINDS:
        W = toy.fit(Phi, P, kind, n, W0, c=1.0)
        out[f"{kind}_fixed"] = test_metrics(softmax(Phit @ W, 1), Pt, xt, Qconst)
        vs = [val_scores(Phi, P, kind, c, fold, seed) for c in C_GRID]
        choices = {"valrps": pick(C_GRID, [v[0] for v in vs], True)}
        if kind == "pearson":
            choices["valcorr"] = pick(C_GRID, [v[1] for v in vs], False)
        for name, c in choices.items():
            fallback = c is None
            c = 1.0 if fallback else c
            Wc = W if c == 1.0 else toy.fit(Phi, P, kind, n, W0, c=c)
            out[f"{kind}_{name}"] = dict(test_metrics(softmax(Phit @ Wc, 1), Pt, xt, Qconst), c=c, fallback=fallback)
    return sc, n, seed, out
