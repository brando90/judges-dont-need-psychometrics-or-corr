#!/usr/bin/env python3
"""TLDR: Independent checks of every formula in gtheory_lib.py (exact where possible).

Run: python experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/test_metrics.py
Writes results/test_metrics.json with one pass/fail record per check.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gtheory_lib as L  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CHECKS: list[dict] = []


def check(name, ok, detail=''):
    CHECKS.append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
    if not ok:
        print('FAIL', name, detail)


def pmfs_for_tests():
    out = [L.mix(L.binom_pmf(Fr(j, 7)), L.UNIFORM, Fr(1, 5)) for j in range(1, 7)]
    out.append([Fr(1, 2), Fr(0), Fr(1, 4), Fr(0), Fr(0), Fr(1, 4)])   # zeros included
    out.append(L.UNIFORM)
    return out


def rps_bruteforce(q, y):
    """Independent path: cumulative sum of (forecast minus one-hot)."""
    onehot = [Fr(1) if r == y else Fr(0) for r in L.Y]
    run, tot = Fr(0), Fr(0)
    for k in range(5):
        run += Fr(q[k]) - onehot[k]
        tot += run * run
    return tot / 5


def test_losses():
    ps = pmfs_for_tests()
    for i, p in enumerate(ps):
        for j, q in enumerate(ps):
            for y in L.Y:
                check(f'rps_individual[{i},{j},{y}]', L.rps_individual(q, y) == rps_bruteforce(q, y))
            e_rps = sum(p[y] * rps_bruteforce(q, y) for y in L.Y)
            check(f'expected_rps[{i},{j}]', e_rps == L.expected_rps(q, p))
            check(f'rps_excess_identity[{i},{j}]', L.expected_rps(q, p) - L.expected_rps(p, p) == L.rps_excess(q, p))
            brier = lambda f: sum(p[y] * sum((f[r] - (1 if r == y else 0)) ** 2 for r in L.Y) for y in L.Y)
            check(f'brier_excess[{i},{j}]', brier(q) - brier(p) == L.brier_excess(q, p))
            best = max(abs(sum(q[r] - p[r] for r in S)) for n in range(7) for S in itertools.combinations(L.Y, n))
            check(f'tvd_subset_max[{i},{j}]', best == L.tvd(q, p))
            ce = L.expected_ce(q, p)
            ce_ind = sum(float(p[y]) * L.ce_individual(q, y) for y in L.Y if p[y] > 0)
            check(f'expected_ce[{i},{j}]', (math.isinf(ce) and math.isinf(ce_ind)) or abs(ce - ce_ind) < 1e-12)
            if not math.isinf(ce):
                check(f'kl_identity[{i},{j}]', abs(ce - L.expected_ce(p, p) - L.kl(p, q)) < 1e-12)


def test_moments_and_components():
    for s in range(3):
        v = [0] * 6
        for k, c in enumerate((-1, 3, -3, 1)):
            v[s + k] = c
        check(f'third_difference_orthogonal[s={s}]',
              sum(v) == 0 and sum(y * v[y] for y in L.Y) == 0 and sum(y * y * v[y] for y in L.Y) == 0
              and sum(y ** 3 * v[y] for y in L.Y) == 6)
    ps = pmfs_for_tests()
    comp = L.iid_panel_components(ps)
    pbar = [sum(p[y] for p in ps) / len(ps) for y in L.Y]
    check('law_of_total_variance', L.var(pbar) == comp['u'] + comp['e'])
    base = [[Fr(0), Fr(a, 10), Fr(10 - a - c, 10), Fr(c, 10), Fr(0), Fr(0)] for a, c in [(7, 1), (5, 2), (3, 3), (2, 5), (1, 7)]]
    shifted = [[Fr(0)] + list(p[:5]) for p in base]
    c0, c1 = L.iid_panel_components(base), L.iid_panel_components(shifted)
    check('shift_invariant_components', c0 == c1 and all(L.mean(s) - L.mean(p) == 1 for p, s in zip(base, shifted)), (c0, c1))
    zero = L.one_facet_coeffs({'u': Fr(0), 'a': Fr(0), 'e': Fr(0)}, 3)
    check('zero_denominator_undefined', zero['Erho2'] is None and zero['Phi'] is None)
    z2 = L.one_facet_coeffs({'u': Fr(0), 'a': Fr(0), 'e': Fr(1)}, 3)
    check('no_object_variance_gives_zero', z2['Erho2'] == 0 and z2['Phi'] == 0)
    rng = np.random.default_rng(1)
    bad = []
    for t in range(200):
        v = dict(zip(L.COMPS, rng.gamma(0.7, 1.0, 7)))
        for na, nb in [(1, 1), (2, 4), (8, 2)]:
            r = L.d_study(v, na, nb)
            if not r['Phi'] <= r['Erho2'] + 1e-15:
                bad.append((v, na, nb))
    check('phi_le_erho2_600_random_cases', not bad, bad[:2])


def tiny_design(n_u, n_a, n_b):
    return np.array([(i, j, k) for i in range(n_u) for j in range(n_a) for k in range(n_b)])


def covariance(idx, v):
    def same(*cols):
        m = np.ones((len(idx), len(idx)), dtype=bool)
        for c in cols:
            m &= idx[:, None, c] == idx[None, :, c]
        return m.astype(float)
    return (v['u'] * same(0) + v['a'] * same(1) + v['b'] * same(2) + v['ua'] * same(0, 1)
            + v['ub'] * same(0, 2) + v['ab'] * same(1, 2) + v['e'] * np.eye(len(idx)))


def test_d_study_by_covariance():
    """v_Delta = Var(object mean - universe score); v_delta via a contrast between two objects."""
    rng = np.random.default_rng(2)
    for t in range(20):
        v = dict(zip(L.COMPS, rng.gamma(0.8, 1.0, 7)))
        for na, nb in [(1, 1), (2, 3), (4, 2)]:
            idx = tiny_design(2, na, nb)
            V = covariance(idx, v)
            V_noU = V - v['u'] * (idx[:, None, 0] == idx[None, :, 0])
            w0 = (idx[:, 0] == 0) / (na * nb)
            w1 = (idx[:, 0] == 1) / (na * nb)
            v_Delta = w0 @ V_noU @ w0
            v_delta = 0.5 * ((w0 - w1) @ V_noU @ (w0 - w1))   # shared facet effects cancel
            r = L.d_study(v, na, nb)
            check(f'd_study_cov[{t},{na},{nb}]', abs(r['v_Delta'] - v_Delta) < 1e-12 and abs(r['v_delta'] - v_delta) < 1e-12,
                  (r['v_Delta'], v_Delta, r['v_delta'], v_delta))


def projector_strata(n_u, n_a, n_b):
    """Stratum projectors built from Kronecker products, independent of anova3's loops."""
    I = lambda n: np.eye(n)
    J = lambda n: np.ones((n, n)) / n
    C = lambda n: I(n) - J(n)
    kron = lambda a, b, c: np.kron(np.kron(a, b), c)
    return {'u': kron(C(n_u), J(n_a), J(n_b)), 'a': kron(J(n_u), C(n_a), J(n_b)), 'b': kron(J(n_u), J(n_a), C(n_b)),
            'ua': kron(C(n_u), C(n_a), J(n_b)), 'ub': kron(C(n_u), J(n_a), C(n_b)), 'ab': kron(J(n_u), C(n_a), C(n_b)),
            'e': kron(C(n_u), C(n_a), C(n_b))}


def test_ems_and_anova():
    rng = np.random.default_rng(3)
    n_u, n_a, n_b = 4, 3, 3
    idx = tiny_design(n_u, n_a, n_b)
    P = projector_strata(n_u, n_a, n_b)
    Cm = L.ems_matrix(n_u, n_a, n_b)
    df = L.strata_df(n_u, n_a, n_b)
    for t in range(10):
        v = dict(zip(L.COMPS, rng.gamma(0.8, 1.0, 7)))
        V = covariance(idx, v)
        lam = Cm @ np.array([v[c] for c in L.COMPS])
        for s_i, s in enumerate(L.COMPS):
            check(f'ems_trace[{t},{s}]', abs(np.trace(P[s] @ V) - df[s_i] * lam[s_i]) < 1e-9,
                  (np.trace(P[s] @ V), df[s_i] * lam[s_i]))
        X = rng.normal(size=(n_u, n_a, n_b))
        a = L.anova3(X)
        y = X.reshape(-1)
        ss_proj = np.array([y @ P[s] @ y for s in L.COMPS])
        check(f'anova3_ss_projectors[{t}]', np.allclose(a['ss'], ss_proj, atol=1e-10))
        check(f'anova3_ss_total[{t}]', abs(a['ss'].sum() - ((y - y.mean()) ** 2).sum()) < 1e-9)


def test_reml():
    rng = np.random.default_rng(4)
    n_u, n_a, n_b = 5, 3, 3
    X = rng.normal(size=(n_u, n_a, n_b)) + rng.normal(size=(n_u, 1, 1)) + rng.normal(size=(1, n_a, 1))
    a = L.anova3(X)
    Cm = L.ems_matrix(n_u, n_a, n_b)
    diffs = []
    for t in range(6):
        v = dict(zip(L.COMPS, rng.gamma(1.0, 1.0, 7)))
        f, _ = L.reml3_negloglik(np.array([v[c] for c in L.COMPS]), a['ss'], a['df'], Cm)
        diffs.append(L.dense_reml_loglik(X, v) - (-f))
    check('reml_stratum_equals_dense_up_to_constant', max(diffs) - min(diffs) < 1e-8 and
          abs(diffs[0] + 0.5 * math.log(n_u * n_a * n_b)) < 1e-8, diffs)
    # gradient check by finite differences
    v0 = np.array(rng.gamma(1.0, 1.0, 7))
    f0, g0 = L.reml3_negloglik(v0, a['ss'], a['df'], Cm)
    num = np.array([(L.reml3_negloglik(v0 + 1e-6 * e, a['ss'], a['df'], Cm)[0]
                     - L.reml3_negloglik(v0 - 1e-6 * e, a['ss'], a['df'], Cm)[0]) / 2e-6 for e in np.eye(7)])
    check('reml_gradient_finite_difference', np.allclose(num, g0, rtol=1e-5, atol=1e-6), (num, g0))
    # with all ANOVA estimates positive, constrained REML = ANOVA for balanced data
    truth = {'u': 2.0, 'a': 1.5, 'b': 1.0, 'ua': 1.0, 'ub': 0.8, 'ab': 0.6, 'e': 0.5}
    for t in range(40):
        n_u, n_a, n_b = 30, 8, 8
        X = sim(rng, truth, n_u, n_a, n_b)
        a = L.anova3(X)
        if min(a['est'].values()) > 0:
            r = L.reml3(a['ss'], a['df'], n_u, n_a, n_b)
            ok = all(abs(r['est'][c] - a['est'][c]) <= 1e-5 * (1 + abs(a['est'][c])) for c in L.COMPS)
            check('reml_equals_anova_when_positive', ok and r['success'], (r, a['est']))
            break


def sim(rng, v, n_u, n_a, n_b):
    s = {c: math.sqrt(v[c]) for c in L.COMPS}
    return (rng.normal(0, s['u'], (n_u, 1, 1)) + rng.normal(0, s['a'], (1, n_a, 1)) + rng.normal(0, s['b'], (1, 1, n_b))
            + rng.normal(0, s['ua'], (n_u, n_a, 1)) + rng.normal(0, s['ub'], (n_u, 1, n_b))
            + rng.normal(0, s['ab'], (1, n_a, n_b)) + rng.normal(0, s['e'], (n_u, n_a, n_b)))


def test_icc_matches_g_coefficients():
    rng = np.random.default_rng(5)
    for t in range(10):
        X = rng.normal(size=(20, 3)) + rng.normal(size=(20, 1)) + rng.normal(0, 0.5, size=(1, 3))
        icc = L.icc_two_way(X)
        est = L.anova2(X)['est']
        g = L.one_facet_coeffs(est, 1)
        check(f'icc_equals_g_n1[{t}]', abs(icc['ICC_C1'] - g['Erho2']) < 1e-12 and abs(icc['ICC_A1'] - g['Phi']) < 1e-12)
        a2 = L.anova2(X)
        n_u, n_a = X.shape
        tot = ((X - X.mean()) ** 2).sum()
        check(f'anova2_ss_total[{t}]', abs(a2['ms_u'] * a2['df_u'] + a2['ms_a'] * a2['df_a'] + a2['ms_e'] * a2['df_e'] - tot) < 1e-9)


def test_serialization():
    ps = pmfs_for_tests()
    enc = json.dumps([[L.fr_to_str(x) for x in p] for p in ps])
    dec = [[L.str_to_fr(s) for s in row] for row in json.loads(enc)]
    check('fraction_json_roundtrip_exact', dec == ps)
    flo = json.dumps([[float(x) for x in p] for p in ps])
    check('float_json_roundtrip_exact', [[float(x) for x in p] for p in ps] == json.loads(flo))
    check('smooth_normalized', all(abs(sum(L.smooth(p, 1e-6)) - 1) < 1e-12 for p in ps))


def main():
    for f in (test_losses, test_moments_and_components, test_d_study_by_covariance, test_ems_and_anova,
              test_reml, test_icc_matches_g_coefficients, test_serialization):
        f()
    n_fail = sum(not c['pass'] for c in CHECKS)
    out = HERE / 'results' / 'test_metrics.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({'n_checks': len(CHECKS), 'n_fail': n_fail, 'checks': CHECKS}, indent=1) + '\n')
    print(json.dumps({'n_checks': len(CHECKS), 'n_fail': n_fail}))
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
