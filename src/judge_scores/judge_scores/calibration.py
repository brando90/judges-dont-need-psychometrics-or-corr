"""Post-hoc calibration of a frozen judge's forecasts and the calibration-error / resolution split of its proper risk.

The map family and its fitting are reproduced exactly from experiments/46_*/scripts/analyze_rank_reversal.py (``fit_cal`` and
``apply_cal``, Experiment 46 hypothesis H3): a multinomial logit on [log q, 1] fitted by the individual-rating cross-entropy with
a ridge penalty of 1e-3 that shrinks the map toward the identity. Fitting is by L-BFGS-B (maxiter 500), so the fitted map is the
result of that optimizer from the identity start, not a certified global optimum (the objective is convex in the map, so the
optimizer's answer is the optimum up to its tolerance).

Split (all risks are per-sample proper risks of the stated rule, lower is better, cross-fitted over item folds):
    UNC = risk of the cross-fitted base-rate forecast        (uncertainty)
    R_raw, R_cal = risk of the raw and the cross-fitted calibrated forecast
    calibration error  CAL = R_raw - R_cal                   (what this map class removes; a lower bound on the population
                                                              miscalibration term E d(q, Pr(Y|q)) within the class)
    resolution         RES = UNC - R_cal                     (discrimination left after recalibration)
    identity           R_raw = UNC - RES + CAL               (holds by construction)
Written in units of UNC: raw skill = (RES - CAL)/UNC and calibrated skill = RES/UNC.
The word "reliability" is avoided on purpose (it means something else in psychometrics): this is the calibration error.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .core import FLOOR, crossfit_base_rate, floor_probs, item_scores


def fit_calibrator(q, counts, ridge: float = 1e-3):
    """Multinomial-logit calibration map on [log q, 1]; q [N, K] floored forecasts, counts [N, K] ratings. Returns [K+1, K]."""
    q = np.asarray(q, dtype=float)
    cnt = np.asarray(counts, dtype=float)
    x = np.column_stack([np.log(q), np.ones(len(q))])
    d = x.shape[1]
    k = q.shape[1]
    tot = cnt.sum()

    def f(wf):
        Wm = wf.reshape(d, k)
        z = x @ Wm
        z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        p = np.exp(lp)
        ll = -(cnt * lp).sum() / tot
        g = x.T @ (p * cnt.sum(1, keepdims=True) - cnt) / tot
        pen = Wm.copy()
        pen[-1] = 0
        pen[:k, :] -= np.eye(k)  # shrink toward the identity map
        return ll + 0.5 * ridge * (pen ** 2).sum(), (g + ridge * pen).ravel()

    w0 = np.zeros((d, k))
    w0[:k, :] = np.eye(k)
    return minimize(f, w0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x.reshape(d, k)


def apply_calibrator(Wm, q, floor: float = FLOOR):
    z = np.column_stack([np.log(q), np.ones(len(q))]) @ Wm
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return floor_probs(e / e.sum(1, keepdims=True), floor)


def crossfit_calibrate(q, counts, folds, ridge: float = 1e-3):
    """Calibrated forecasts [N, K]; the map applied to an item was fitted only on the ratings of the items in the other folds."""
    q = np.asarray(q, dtype=float)
    counts = np.asarray(counts, dtype=float)
    out = np.zeros_like(q)
    for f in np.unique(folds):
        te = folds == f
        out[te] = apply_calibrator(fit_calibrator(q[~te], counts[~te], ridge), q[te])
    return out


def calibration_split(q, counts, rule: str, folds, ridge: float = 1e-3):
    """Per-item risks and the scalar split described in the module docstring. ``q`` must already be floored."""
    q = np.asarray(q, dtype=float)
    base = crossfit_base_rate(counts, folds)
    qc = crossfit_calibrate(q, counts, folds, ridge)
    s_raw, s_cal, s_base = (item_scores(a, counts, rule) for a in (q, qc, base))
    unc, r_raw, r_cal = float(s_base.mean()), float(s_raw.mean()), float(s_cal.mean())
    cal_err, res = r_raw - r_cal, unc - r_cal
    return {
        "unc": unc, "risk_raw": r_raw, "risk_cal": r_cal,
        "calibration_error": cal_err, "resolution": res,
        "skill_raw": 1.0 - r_raw / unc, "skill_cal": 1.0 - r_cal / unc,
        "calibration_share": cal_err / unc, "resolution_share": res / unc,
        "identity_residual": r_raw - (unc - res + cal_err),
        "items_raw": s_raw, "items_cal": s_cal, "items_base": s_base, "forecasts_cal": qc,
    }
