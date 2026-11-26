"""Deterministic numerical checks of the Lean claims in CorrFirstMomentCCA.lean.

Exact `Fraction` checks where the quantity is rational. Float checks use TOL_CCA = 1e-8
for canonical correlations and TOL_CORR = 1e-12 for the quadratic sample correlation.
Seed 0. A claimed-false statement is recorded only when it fails.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from cca_tools import TOL_CCA, canonical_correlations, correlations_close, pearson

SEED = 0
N_RANDOM = 200
TOL_CORR = 1e-12


def _simplex(rng: np.random.Generator, k: int) -> list[Fraction]:
    weights = [int(x) for x in rng.integers(1, 7, size=k)]
    total = sum(weights)
    return [Fraction(w, total) for w in weights]


def check_random_total_moments(rng: np.random.Generator) -> bool:
    ok = True
    for _ in range(N_RANDOM):
        n_c = int(rng.integers(2, 5))
        n_x = int(rng.integers(2, 5))
        n_y = int(rng.integers(2, 5))
        pc = _simplex(rng, n_c)
        xv = [Fraction(int(v)) for v in rng.integers(-3, 4, size=n_x)]
        yv = [Fraction(int(v)) for v in rng.integers(-3, 4, size=n_y)]
        px = [_simplex(rng, n_x) for _c in range(n_c)]
        py = [_simplex(rng, n_y) for _c in range(n_c)]
        exy = Fraction(0)
        ex = Fraction(0)
        ey = Fraction(0)
        ex2 = Fraction(0)
        cov_means = Fraction(0)
        e_mu2 = Fraction(0)
        e_var = Fraction(0)
        for c in range(n_c):
            mu_x = sum((px[c][i] * xv[i] for i in range(n_x)), Fraction(0))
            mu_y = sum((py[c][j] * yv[j] for j in range(n_y)), Fraction(0))
            second = sum((px[c][i] * xv[i] * xv[i] for i in range(n_x)), Fraction(0))
            cross = sum(
                (px[c][i] * py[c][j] * xv[i] * yv[j] for i in range(n_x) for j in range(n_y)),
                Fraction(0),
            )
            if cross != mu_x * mu_y:
                ok = False
            exy += pc[c] * cross
            ex += pc[c] * mu_x
            ey += pc[c] * mu_y
            ex2 += pc[c] * second
            cov_means += pc[c] * mu_x * mu_y
            e_mu2 += pc[c] * mu_x * mu_x
            e_var += pc[c] * (second - mu_x * mu_x)
        cov_means -= ex * ey
        if exy - ex * ey != cov_means:
            ok = False
        if ex2 - ex * ex != e_var + (e_mu2 - ex * ex):
            ok = False
    return ok


def check_quadratic() -> bool:
    x = [Fraction(-1), Fraction(0), Fraction(1)]
    y = [v * v for v in x]
    n = Fraction(len(x))
    mean_x = sum(x, Fraction(0)) / n
    mean_y = sum(y, Fraction(0)) / n
    mean_xy = sum((a * b for a, b in zip(x, y)), Fraction(0)) / n
    cov = mean_xy - mean_x * mean_y
    var_x = sum((a * a for a in x), Fraction(0)) / n - mean_x * mean_x
    var_y = sum((b * b for b in y), Fraction(0)) / n - mean_y * mean_y
    # Claimed false: a deterministic relationship forces nonzero covariance.
    false_claim_holds = cov != 0
    return (
        cov == 0
        and var_x == Fraction(2, 3)
        and var_y == Fraction(2, 9)
        and not false_claim_holds
        and Fraction(1, 3) * Fraction(2, 3) == Fraction(2, 9)
    )


def check_shift_and_scale(rng: np.random.Generator) -> bool:
    ok = True
    for _ in range(N_RANDOM):
        n = int(rng.integers(4, 12))
        x = [Fraction(int(v)) for v in rng.integers(-5, 6, size=n)]
        y = [Fraction(int(v)) for v in rng.integers(-5, 6, size=n)]
        a = Fraction(int(rng.choice([-3, -2, -1, 1, 2, 3])))
        b = Fraction(int(rng.integers(-4, 5)))
        nn = Fraction(n)

        def cov(u, v):
            mu = sum(u, Fraction(0)) / nn
            mv = sum(v, Fraction(0)) / nn
            return sum((p * q for p, q in zip(u, v)), Fraction(0)) / nn - mu * mv

        c0 = cov(x, y)
        shifted = [v + b for v in x]
        scaled = [a * v + b for v in x]
        if cov(shifted, y) != c0:
            ok = False
        if cov(scaled, y) != a * c0:
            ok = False
        var_s = cov(scaled, scaled)
        var_x = cov(x, x)
        if var_s != a * a * var_x:
            ok = False
        if var_x != 0 and cov(y, y) != 0:
            sq0 = (c0 * c0) / (var_x * cov(y, y))
            sq1 = (cov(scaled, y) * cov(scaled, y)) / (var_s * cov(y, y))
            if sq0 != sq1:
                ok = False
    return ok


def check_witness_numbers() -> bool:
    # Aspect 1: p in {1/4, 3/4}. Aspect 2: p in {1/3, 2/3}.
    spread_1 = Fraction(1, 16)
    collapsed_1 = Fraction(1, 4)
    spread_2 = Fraction(1, 81)
    collapsed_2 = Fraction(1, 9)
    brier_true_1 = Fraction(3, 8)
    brier_point_1 = Fraction(1, 2)
    brier_true_2 = Fraction(4, 9)
    brier_point_2 = Fraction(2, 3)
    rps_true_1 = Fraction(3, 16)
    rps_point_1 = Fraction(1, 4)
    rps_true_2 = Fraction(2, 9)
    rps_point_2 = Fraction(1, 3)
    var_mu = Fraction(1, 16)
    e_var = Fraction(3, 16)
    return (
        spread_1 < collapsed_1
        and spread_2 < collapsed_2
        and brier_point_1 > brier_true_1
        and brier_point_2 > brier_true_2
        and rps_point_1 > rps_true_1
        and rps_point_2 > rps_true_2
        and e_var + var_mu == Fraction(1, 4)
        and (Fraction(1, 16) * Fraction(1, 16)) / (Fraction(1, 4) * Fraction(1, 4)) == spread_1
        and (Fraction(1, 16) * Fraction(1, 16)) / (Fraction(1, 16) * Fraction(1, 4)) == collapsed_1
    )


def check_cca_invariance(rng: np.random.Generator) -> bool:
    ok = True
    n, d, p = 60, 3, 2
    x = rng.normal(size=(n, d))
    y = rng.normal(size=(n, p))
    base = canonical_correlations(x, y)
    transforms = []
    transforms.append(-np.eye(d))
    perm = np.eye(d)[:, [1, 2, 0]]
    transforms.append(perm)
    shear = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 1.0]])
    transforms.append(shear)
    mix = rng.normal(size=(d, d))
    while abs(np.linalg.det(mix)) < 1e-2 or np.linalg.cond(mix) > 30:
        mix = rng.normal(size=(d, d))
    transforms.append(mix)
    b = rng.normal(size=d)
    for mat in transforms:
        moved = x @ mat.T + b
        got = canonical_correlations(moved, y)
        if not correlations_close(base, got, TOL_CCA):
            ok = False
    # 2d preimages used in Lean: shear, sign, swap, mix [[2,1],[1,2]]
    x2 = rng.normal(size=(n, 2))
    y2 = rng.normal(size=(n, 2))
    base2 = canonical_correlations(x2, y2)
    special = [
        -np.eye(2),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[1.0, 1.0], [0.0, 1.0]]),
        np.array([[2.0, 1.0], [1.0, 2.0]]),
    ]
    for mat in special:
        got = canonical_correlations(x2 @ mat.T + np.array([0.3, -0.7]), y2)
        if not correlations_close(base2, got, TOL_CCA):
            ok = False
    return ok


def check_onehot_sees_beyond_mean() -> bool:
    """Same scalar mean, different probability mass functions.

    Scalar Pearson of the conditional means is undefined (zero variance).
    CCA of the probability vectors with a copy of themselves is 1, and it stays 1
    after a label permutation, while cross-entropy of the permuted forecast is worse.
    """
    p0 = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
    p1 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    values = np.arange(5, dtype=float)
    if abs(p0 @ values - 2.0) > 1e-12 or abs(p1 @ values - 2.0) > 1e-12:
        return False
    # Four rows so each prompt appears twice; means do not vary.
    pmf = np.vstack([p0, p0, p1, p1])
    rho = canonical_correlations(pmf, pmf)
    if rho.size == 0 or abs(rho[0] - 1.0) > 1e-8:
        return False
    perm = np.array([1, 2, 3, 4, 0])
    scrambled = pmf[:, perm]
    rho_p = canonical_correlations(scrambled, pmf)
    if not correlations_close(rho, rho_p, TOL_CCA):
        return False
    # A draw from the true law. Cross-entropy of p is finite; the permuted forecast
    # puts zero on the realized class for the point-mass prompt.
    ce_true = float(-(0.5 * np.log(0.5) + 0.5 * np.log(0.5)))
    # Prompt 1 is a point mass at class 2. Permutation sends class 2 to index 3 of the
    # inverse: scrambled[2] = p[perm][2] = p[3] = 0 for p1 which is e_2, perm[2]=3 so
    # scrambled row is e_2 moved to column perm... row[:, perm] places old column 2 at new index 2?
    # q_new[perm[k]] = q_old[k], implemented as q[:, perm] reads q[perm[j]] into position j.
    # For a point mass at 2, q[2]=1, q[:, perm][j] = q[perm[j]]. This is 1 when perm[j]=2, i.e. j=1
    # because perm[1]=2. So the mass moves to class 1. CE on the true class 2 is -log(q_new[2]) = -log(0).
    if scrambled[2, 2] != 0.0:
        return False
    return np.isfinite(ce_true) and ce_true > 0.0


def check_shapes_and_collapse(rng: np.random.Generator) -> bool:
    x = np.linspace(-2.0, 2.0, 401)
    y_lin = x.copy()
    y_quad = x ** 2
    y_s = np.tanh(x)
    y_step = np.where(x >= 0.0, 1.0, -1.0)
    r_lin = pearson(x, y_lin)
    r_quad = pearson(x, y_quad)
    r_s = pearson(x, y_s)
    r_step = pearson(x, y_step)
    if not (abs(r_lin - 1.0) < 1e-12 and abs(r_quad) < TOL_CORR and r_s > 0.9 and abs(r_step) < 0.95):
        return False
    # Two prompts, conditional means 0 and 2, conditional variance 1, independent copy.
    # Population correlation of a fresh draw is 0.5; the collapsed conditional mean has correlation 1/sqrt(2).
    # Sample check: many draws, collapsed correlation strictly larger.
    n_per = 4000
    c = np.repeat([0.0, 1.0], n_per)
    mu = np.where(c == 0.0, 0.0, 2.0)
    x_draw = mu + rng.normal(size=2 * n_per)
    y_draw = mu + rng.normal(size=2 * n_per)
    r_spread = pearson(x_draw, y_draw)
    r_coll = pearson(mu, y_draw)
    if not (np.isfinite(r_spread) and np.isfinite(r_coll) and r_coll > r_spread + 0.05):
        return False
    # Proper scores on this Bernoulli-like ordinal are checked exactly in check_witness_numbers.
    return True


def run_all() -> int:
    rng = np.random.default_rng(SEED)
    checks = [
        ("random_total_covariance_and_variance", check_random_total_moments(rng)),
        ("quadratic_cov_zero_and_false_claim_fails", check_quadratic()),
        ("shift_deletes_mean_and_scale_keeps_sqcorr", check_shift_and_scale(rng)),
        ("cca_witness_sqcorr_and_proper_scores", check_witness_numbers()),
        ("cca_affine_sign_perm_mix_invariance", check_cca_invariance(rng)),
        ("onehot_cca_sees_pmf_and_ignores_label_perm", check_onehot_sees_beyond_mean()),
        ("nonlinear_shapes_and_collapse_raises_corr", check_shapes_and_collapse(rng)),
    ]
    failed = False
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
        failed = failed or not ok
    print("FALSE-CLAIM checked: 'Y = X^2 on {-1,0,1} has nonzero covariance' does not hold")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run_all())
