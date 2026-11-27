"""L2-penalty control for the Experiment 41 sensitivity analyses (NOT part of the preregistered K-sweep).

Why this file exists.  Experiment 36's heads use a FIXED penalty, `L2 = 0.01`, written 0.5 * L2 * sum(W^2) on the MEAN training loss for the
softmax heads (cross-entropy, Brier, RPS, Pearson).  A fixed value does not shrink when the training set grows, and the 24 principal-component
features have standard deviations of only 0.077 to 0.167, so that penalty flattens learning curves (an independent review of Experiment 41 found
this, 11-27-2026).  The MSE head is a different object: `_fit_ridge` adds `l2` to X'X, i.e. a penalty on the SUM of squared errors, which is
about 0.5 * N times weaker than the softmax heads' penalty at the same nominal `l2`.  So the preregistered MSE head is almost unregularized.

What this module does.  `l2_settings(lam)` temporarily sets, inside this process only (Experiment 36's files are never edited):
    softmax heads : penalty 0.5 * lam * sum(W_without_intercept^2)       on the mean loss         (Experiment 36's form, value lam)
    MSE ridge     : (X'X + 0.5 * N * lam * P) beta = X'y                 i.e. mean squared error + 0.5 * lam * sum(beta_without_intercept^2)
so one value `lam` means the same penalty FORM for every head.  The loss scales differ (nats, squared rating units), so `lam` is chosen on
validation for each head separately; it is not claimed to be "the same amount of shrinkage".
"""
from __future__ import annotations

import contextlib

import numpy as np

from e40_common import fe, histograms, fit_logit_map, softmax

LAMBDA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


@contextlib.contextmanager
def l2_settings(lam: float):
    orig_penalty, orig_ridge = fe._penalty, fe._fit_ridge

    def penalty(w, d, k, l2=None):
        weight = w.reshape(d, k)
        return 0.5 * lam * float(np.sum(weight[:-1] ** 2))  # last row = intercept, unpenalized (as in Experiment 36)

    def ridge(x, y, l2=None):
        n, d = x.shape
        pen = np.eye(d)
        pen[-1, -1] = 0.0  # intercept unpenalized
        return np.linalg.solve(x.T @ x + 0.5 * n * lam * pen, x.T @ y)

    fe._penalty, fe._fit_ridge = penalty, ridge
    try:
        yield
    finally:
        fe._penalty, fe._fit_ridge = orig_penalty, orig_ridge


def calibrate_both(mu_val, mu_test, ratings, val_idx, k):
    """Same map as e40_common.calibrate (validation moments, validation histograms) but returns the forecasts on BOTH validation and test.
    The map is fitted on the validation items, so the validation forecast is in-sample for the map's 2K parameters; the test forecast is not."""
    m, s = float(mu_val.mean()), float(mu_val.std())
    s = s if s > 1e-12 else 1.0
    w = fit_logit_map((mu_val - m) / s, histograms(ratings, val_idx, k))
    xv = np.column_stack([(mu_val - m) / s, np.ones_like(mu_val)])
    xt = np.column_stack([(mu_test - m) / s, np.ones_like(mu_test)])
    return softmax(xv @ w), softmax(xt @ w)
