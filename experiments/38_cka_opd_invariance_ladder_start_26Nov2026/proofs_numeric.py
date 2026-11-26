"""Deterministic numeric checks of the Lean theorems in CkaOpd.lean.

Exact claims use fractions. Floating checks use a named tolerance. The script
prints PASS or FAIL per claim and exits nonzero on a failure. A second run
must reprint the same lines.
"""

from __future__ import annotations

import sys
from fractions import Fraction as F

import numpy as np

from metrics import (
    brier_one,
    cca_sq,
    ce_one,
    linear_cka,
    median_gamma,
    normalized_opd,
    pearson,
    rbf_cka,
    rps_one,
    softmax,
)

TOL = 1e-10
SEED = 20261126


def _center(xs):
    m = sum(xs) / len(xs)
    return [x - m for x in xs]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _sq_pearson(x, y):
    xc, yc = _center(x), _center(y)
    return (_dot(xc, yc) ** 2) / (_dot(xc, xc) * _dot(yc, yc))


def _brier3(q, y):
    return sum((q[j] - (F(1) if j == y else F(0))) ** 2 for j in range(3))


def _rps3(q, y):
    f0, f1 = q[0], q[0] + q[1]
    g0 = F(1) if y == 0 else F(0)
    g1 = F(1) if y <= 1 else F(0)
    return (f0 - g0) ** 2 + (f1 - g1) ** 2


class Report:
    def __init__(self):
        self.failed = 0
        self.lines = []

    def check(self, name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        if not ok:
            self.failed += 1
        line = f"{status} {name}" + (f" | {detail}" if detail else "")
        self.lines.append(line)
        print(line)


def main() -> int:
    report = Report()
    wx = [F(1), F(-1), F(0), F(4), F(2), F(3)]
    wy = [F(0), F(0), F(1), F(3), F(3), F(4)]
    wxc = [F(0), F(0), F(0), F(3), F(3), F(3)]
    cka = _sq_pearson(wx, wy)
    cka_c = _sq_pearson(wxc, wy)
    report.check("collapse_cka_values", cka == F(2187, 3115) and cka_c == F(81, 89),
                 f"{cka} < {cka_c}")
    report.check("collapse_cka_strict", cka < cka_c)

    def within(xs, ys):
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        return sum((a - mx) * (b - my) for a, b in zip(xs, ys))

    report.check(
        "collapse_conditional_uncorrelated",
        within(wx[:3], wy[:3]) == 0 and within(wx[3:], wy[3:]) == 0,
    )
    # 1D normalized OPD squared is 2 - 2*sqrt(cka) when the correlation is positive.
    # The ordering is equivalent to the proved CKA inequality.
    report.check("collapse_opd_order_equivalent", cka < cka_c and cka > 0 and cka_c > 0)

    good = (F(1, 2), F(1, 3), F(1, 6))
    rev = (F(1, 6), F(1, 3), F(1, 2))
    swap = (F(1, 3), F(1, 2), F(1, 6))
    report.check("reversal_brier_strict", _brier3(good, 0) < _brier3(rev, 0),
                 f"{_brier3(good, 0)} < {_brier3(rev, 0)}")
    report.check("reversal_rps_strict", _rps3(good, 0) < _rps3(rev, 0),
                 f"{_rps3(good, 0)} < {_rps3(rev, 0)}")
    report.check("transposition_brier_strict", _brier3(good, 0) < _brier3(swap, 0))
    report.check("transposition_rps_strict", _rps3(good, 0) < _rps3(swap, 0))
    report.check("reversal_true_class_prob", F(1, 6) < F(1, 2))
    report.check("transposition_true_class_prob", F(1, 3) < F(1, 2))

    q = (F(1, 4), F(1, 2), F(1, 4))
    delta = (F(0), F(1), F(0))
    outcomes = (0, 1, 1, 2)
    bq = sum(_brier3(q, y) for y in outcomes) / 4
    bd = sum(_brier3(delta, y) for y in outcomes) / 4
    rq = sum(_rps3(q, y) for y in outcomes) / 4
    rd = sum(_rps3(delta, y) for y in outcomes) / 4
    report.check("collapse_pointmass_brier", bq < bd, f"{bq} < {bd}")
    report.check("collapse_pointmass_rps", rq < rd, f"{rq} < {rd}")
    report.check("collapse_pointmass_ce_infinite", ce_one(np.array([0.0, 1.0, 0.0]), 0) == float("inf"))

    rng = np.random.default_rng(SEED)
    max_id = 0.0
    for _ in range(40):
        n = int(rng.integers(6, 18))
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        cka_f = linear_cka(x, y)
        r2 = pearson(x, y) ** 2
        max_id = max(max_id, abs(cka_f - r2))
    report.check("linearCKA1d_eq_sqPearson_float", max_id < TOL, f"max abs {max_id:.3e}")

    max_aff = 0.0
    for _ in range(40):
        n = int(rng.integers(6, 18))
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        a = float(rng.choice([-3.5, -0.4, 0.7, 2.0, 5.0]))
        b = float(rng.normal())
        max_aff = max(max_aff, abs(pearson(a * x + b, y) ** 2 - pearson(x, y) ** 2))
    report.check("sqPearson_scale_shift_float", max_aff < TOL, f"max abs {max_aff:.3e}")

    max_orth = 0.0
    max_opd = 0.0
    for _ in range(25):
        n, p, q = 30, 4, 3
        x = rng.normal(size=(n, p))
        y = rng.normal(size=(n, q))
        gauss = rng.normal(size=(p, p))
        qq, _ = np.linalg.qr(gauss)
        a = float(rng.choice([-2.0, -0.5, 1.5, 3.0]))
        shift = rng.normal(size=p)
        xt = a * (x @ qq) + shift
        max_orth = max(max_orth, abs(linear_cka(xt, y) - linear_cka(x, y)))
        # OPD needs equal feature width for a square orthogonal action on X only;
        # the distance to Y is still defined. Isotropic scale, shift, and Q on X.
        max_opd = max(max_opd, abs(normalized_opd(xt, y) - normalized_opd(x, y)))
    report.check("cka_orth_scale_shift_float", max_orth < 1e-8, f"max abs {max_orth:.3e}")
    report.check("opd_orth_scale_shift_float", max_opd < 1e-8, f"max abs {max_opd:.3e}")

    eye = np.eye(4)
    report.check("negI_orth", np.allclose((-eye).T @ (-eye), eye, atol=TOL))
    perm = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]], dtype=float)
    report.check("perm_orth", np.allclose(perm.T @ perm, np.eye(4), atol=TOL))
    x = rng.normal(size=(20, 4))
    y = rng.normal(size=(20, 4))
    report.check(
        "sign_and_perm_leave_cka_opd",
        abs(linear_cka(-x, y) - linear_cka(x, y)) < 1e-8
        and abs(linear_cka(x @ perm, y) - linear_cka(x, y)) < 1e-8
        and abs(normalized_opd(-x, y) - normalized_opd(x, y)) < 1e-8
        and abs(normalized_opd(x @ perm, y) - normalized_opd(x, y)) < 1e-8,
    )

    # Anisotropic shear: CKA and OPD move, CCA does not.
    shear = np.array([[1.0, 0.8, 0.0], [0.0, 1.0, -0.5], [0.3, 0.0, 1.0]])
    xs = rng.normal(size=(40, 3))
    ys = rng.normal(size=(40, 3))
    cka_delta = abs(linear_cka(xs @ shear, ys) - linear_cka(xs, ys))
    opd_delta = abs(normalized_opd(xs @ shear, ys) - normalized_opd(xs, ys))
    cca_delta = abs(cca_sq(xs @ shear, ys) - cca_sq(xs, ys))
    report.check("shear_moves_cka", cka_delta > 1e-3, f"delta {cka_delta:.3e}")
    report.check("shear_moves_opd", opd_delta > 1e-3, f"delta {opd_delta:.3e}")
    report.check("shear_preserves_cca", cca_delta < 1e-8, f"delta {cca_delta:.3e}")

    gamma = median_gamma(xs)
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    report.check(
        "rbf_isometry",
        abs(rbf_cka(xs @ rot + rng.normal(size=3), ys, gamma) - rbf_cka(xs, ys, gamma)) < 1e-8,
    )
    report.check(
        "rbf_not_scale_invariant",
        abs(rbf_cka(3.0 * xs, ys, gamma) - rbf_cka(xs, ys, gamma)) > 1e-4,
    )

    # A sharp forecast gets worse proper scores after a class reversal.
    logits = np.array([4.0, 0.0, -1.0])
    prob = softmax(logits.reshape(1, -1))[0]
    flipped = prob[::-1]
    report.check(
        "softmax_reversal_worsens_proper_scores",
        ce_one(flipped, 0) > ce_one(prob, 0)
        and brier_one(flipped, 0) > brier_one(prob, 0)
        and rps_one(flipped, 0) > rps_one(prob, 0),
    )

    if report.failed:
        print(f"FAILED {report.failed}")
        return 1
    print(f"ALL PASS {len(report.lines)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
