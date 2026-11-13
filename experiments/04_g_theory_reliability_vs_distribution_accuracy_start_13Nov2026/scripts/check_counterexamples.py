#!/usr/bin/env python3
"""TLDR: Phase A of experiment 04 — exact counterexamples and controls (no model calls, no human data).

A1  same mean and variance, different distribution (PLAN.md epsilon) and A1_max (largest RPS gap
    under a 1/100 floor); exact Fractions; plus a finite-sample ANOVA Monte Carlo.
A2  repeatable but wrong; constant judge; shared +1 offset (separate and pooled panels).
A3  oracle forecast versus sampled-score reliability; tempered forecasts; population dependence.
A4  equal total variance, different decision-study (D-study) recommendations.

Run: python experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/check_counterexamples.py
Writes results/phase_a.json (every number the report shows, plus contract pass/fail records).
"""
from __future__ import annotations

import itertools
import json
import math
import platform
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gtheory_lib as L  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
CFG = PROTOCOL['phase_a']
CONTRACT: list[dict] = []


def contract(cid, ok, observed):
    CONTRACT.append({'id': cid, 'pass': bool(ok), 'observed': observed})


def S(x):
    """Serialize exact values as 'num/den' strings plus a float."""
    if x is None:
        return None
    if isinstance(x, Fr):
        return {'exact': L.fr_to_str(x), 'float': float(x)}
    return float(x)


def vecS(v):
    return [L.fr_to_str(x) for x in v]


def corr3(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {'pearson': None, 'spearman': None, 'kendall': None}
    return {'pearson': float(stats.pearsonr(a, b)[0]), 'spearman': float(stats.spearmanr(a, b)[0]),
            'kendall': float(stats.kendalltau(a, b)[0])}


def coeff_table(v, ns):
    return {str(n): {k: S(x) for k, x in L.one_facet_coeffs(v, n).items()} for n in ns}


def human_population():
    thetas = [Fr(t) for t in CFG['human_population']['theta']]
    return thetas, [L.mix(L.binom_pmf(t), L.UNIFORM, Fr(1, 10)) for t in thetas]


# ------------------------------------------------------------------------ A1
def a1(P):
    v = CFG['A1']['v']
    rows, Q = [], []
    for j, p in enumerate(P, start=1):
        eps = min(p) / 6
        q = [p[y] + eps * v[y] for y in L.Y]
        Q.append(q)
        rows.append({
            'j': j, 'epsilon': S(eps), 'p': vecS(p), 'q': vecS(q),
            'sum_q_is_1': sum(q) == 1, 'q_positive': all(x > 0 for x in q), 'q_differs': q != p,
            'mean_diff': S(L.mean(q) - L.mean(p)), 'var_diff': S(L.var(q) - L.var(p)),
            'third_raw_moment_diff': S(L.raw_moment(q, 3) - L.raw_moment(p, 3)),
            'mean_p': S(L.mean(p)), 'var_p': S(L.var(p)),
            'rps_excess': S(L.rps_excess(q, p)), 'rps_excess_formula': S(Fr(6, 5) * eps ** 2),
            'brier_excess': S(L.brier_excess(q, p)), 'tvd': S(L.tvd(q, p)),
            'kl_nats': L.kl(p, q), 'oracle_expected_rps': S(L.expected_rps(p, p)),
            'oracle_expected_ce': L.expected_ce(p, p),
        })
    exact_ok = all(r['sum_q_is_1'] and r['q_positive'] and r['q_differs'] for r in rows)
    contract('A1.validity', exact_ok, f'{sum(r["sum_q_is_1"] and r["q_positive"] and r["q_differs"] for r in rows)}/11 artifacts valid')
    mom_ok = all(L.mean(q) == L.mean(p) and L.var(q) == L.var(p) and
                 L.raw_moment(q, 3) - L.raw_moment(p, 3) == 6 * min(p) / 6 for p, q in zip(P, Q))
    contract('A1.moments', mom_ok, 'exact Fraction equality of mean and variance; third raw moment differs by 6 epsilon')
    rps_ok = all(L.rps_excess(q, p) == Fr(6, 5) * (min(p) / 6) ** 2 > 0 for p, q in zip(P, Q))
    contract('A1.rps_excess', rps_ok, 'RPS excess == (6/5) epsilon^2 > 0 for all 11')
    bt_ok = all(L.brier_excess(q, p) == 20 * (min(p) / 6) ** 2 and L.tvd(q, p) == 4 * min(p) / 6 for p, q in zip(P, Q))
    contract('A1.brier_tvd', bt_ok, 'Brier excess == 20 epsilon^2 and TVD == 4 epsilon for all 11')
    contract('A1.ce_excess', all(0 < r['kl_nats'] < math.inf for r in rows),
             f'KL range {min(r["kl_nats"] for r in rows):.3e}..{max(r["kl_nats"] for r in rows):.3e} nats')
    mp, mq = [L.mean(p) for p in P], [L.mean(q) for q in Q]
    cp, cq = corr3(mp, mp), corr3(mq, mp)
    contract('A1.correlations', cp == cq, {'oracle': cp, 'perturbed': cq})
    comp_p, comp_q = L.iid_panel_components(P), L.iid_panel_components(Q)
    ns = CFG['A1']['rater_counts_for_coefficients']
    same = comp_p == comp_q and all(L.one_facet_coeffs(comp_p, n) == L.one_facet_coeffs(comp_q, n) for n in ns)
    contract('A1.components', same, {'v_u': S(comp_p['u']), 'v_e': S(comp_p['e'])})
    summary = {
        'n_artifacts': len(P),
        'mean_rps_excess': S(sum(L.rps_excess(q, p) for p, q in zip(P, Q)) / len(P)),
        'mean_oracle_expected_rps': S(sum(L.expected_rps(p, p) for p in P) / len(P)),
        'mean_tvd': S(sum(L.tvd(q, p) for p, q in zip(P, Q)) / len(P)),
        'mean_brier_excess': S(sum(L.brier_excess(q, p) for p, q in zip(P, Q)) / len(P)),
        'mean_kl_nats': float(np.mean([r['kl_nats'] for r in rows])),
        'mean_oracle_expected_ce_nats': float(np.mean([r['oracle_expected_ce'] for r in rows])),
        'correlations_of_forecast_means_with_true_means': {'oracle': cp, 'perturbed': cq},
        'components_p': {k: S(x) for k, x in comp_p.items()}, 'components_q': {k: S(x) for k, x in comp_q.items()},
        'coefficients_p': coeff_table(comp_p, ns), 'coefficients_q': coeff_table(comp_q, ns),
    }
    return rows, Q, summary


# ------------------------------------------------------------------------ A1_max
def solve3(A, b):
    """Exact Gaussian elimination for a 3x3 Fraction system."""
    M = [list(map(Fr, row)) + [Fr(bb)] for row, bb in zip(A, b)]
    for c in range(3):
        piv = next(r for r in range(c, 3) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        for r in range(3):
            if r != c and M[r][c] != 0:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[r][3] / M[r][r] for r in range(3)]


def max_gap_same_moments(p, floor):
    m1, m2 = L.raw_moment(p, 1), L.raw_moment(p, 2)
    best, best_val, best_tri, feasible = None, None, None, 0
    for tri in itertools.combinations(L.Y, 3):
        others = [y for y in L.Y if y not in tri]
        A = [[1, 1, 1], list(tri), [y * y for y in tri]]
        b = [1 - 3 * floor, m1 - floor * sum(others), m2 - floor * sum(y * y for y in others)]
        sol = solve3(A, b)
        if all(s >= floor for s in sol):
            feasible += 1
            q = [floor] * 6
            for y, s in zip(tri, sol):
                q[y] = s
            val = L.rps_excess(q, p)
            if best_val is None or val > best_val:
                best, best_val, best_tri = q, val, tri
    return best, best_tri, feasible


def a1_max(P, Q_a1):
    floor = Fr(CFG['A1_max']['floor'])
    rows, Q = [], []
    for j, (p, qa) in enumerate(zip(P, Q_a1), start=1):
        q, tri, nfeas = max_gap_same_moments(p, floor)
        Q.append(q)
        rows.append({'j': j, 'q': vecS(q), 'free_triple': list(tri), 'feasible_vertices': nfeas,
                     'mean_diff': S(L.mean(q) - L.mean(p)), 'var_diff': S(L.var(q) - L.var(p)),
                     'rps_excess': S(L.rps_excess(q, p)), 'rps_excess_A1': S(L.rps_excess(qa, p)),
                     'oracle_expected_rps': S(L.expected_rps(p, p)), 'tvd': S(L.tvd(q, p)),
                     'brier_excess': S(L.brier_excess(q, p)), 'kl_nats': L.kl(p, q), 'min_q': S(min(q))})
    ok = all(L.mean(q) == L.mean(p) and L.var(q) == L.var(p) and sum(q) == 1 and min(q) >= floor for p, q in zip(P, Q))
    larger = sum(L.rps_excess(q, p) > L.rps_excess(qa, p) for p, q, qa in zip(P, Q, Q_a1))
    contract('A1_max.moments_and_gap', ok and larger == len(P), f'exact moments {ok}; larger RPS gap than A1 in {larger}/{len(P)}')
    comp_p, comp_q = L.iid_panel_components(P), L.iid_panel_components(Q)
    summary = {'floor': L.fr_to_str(floor),
               'mean_rps_excess': S(sum(L.rps_excess(q, p) for p, q in zip(P, Q)) / len(P)),
               'mean_tvd': S(sum(L.tvd(q, p) for p, q in zip(P, Q)) / len(P)),
               'mean_brier_excess': S(sum(L.brier_excess(q, p) for p, q in zip(P, Q)) / len(P)),
               'mean_kl_nats': float(np.mean([r['kl_nats'] for r in rows])),
               'components_equal': comp_p == comp_q,
               'correlations_of_forecast_means_with_true_means': corr3([L.mean(q) for q in Q], [L.mean(p) for p in P])}
    return rows, Q, summary


# ------------------------------------------------------------------------ A1 finite-sample
def sample_panel(rng, pmfs_float, idx, n_raters):
    cdfs = np.cumsum(pmfs_float[idx], axis=1)
    u = rng.random((len(idx), n_raters))
    return (u[:, :, None] > cdfs[:, None, :]).sum(axis=2).astype(float)


def a1_finite(P, Q_a1, Q_max):
    c = CFG['A1_finite_sample']
    rng = np.random.default_rng(c['seed'])
    Pf = np.array([[float(x) for x in p] for p in P])
    panels = {'p': Pf, 'q_A1': np.array([[float(x) for x in q] for q in Q_a1]),
              'q_A1_max': np.array([[float(x) for x in q] for q in Q_max])}
    keys = ['u', 'a', 'e', 'Erho2_n3', 'Phi_n3']
    est = {k: np.empty((c['replications'], len(keys))) for k in panels}
    for r in range(c['replications']):
        idx = rng.integers(0, len(P), c['n_objects_per_panel'])
        for name, pm in panels.items():
            X = sample_panel(rng, pm, idx, c['n_raters'])
            e = L.anova2(X)['est']
            co = L.one_facet_coeffs(e, c['n_raters'])
            est[name][r] = [e['u'], e['a'], e['e'],
                            np.nan if co['Erho2'] is None else co['Erho2'], np.nan if co['Phi'] is None else co['Phi']]
    pop = L.iid_panel_components(P)
    pop_co = L.one_facet_coeffs(pop, c['n_raters'])
    pop_vals = {'u': float(pop['u']), 'a': 0.0, 'e': float(pop['e']),
                'Erho2_n3': float(pop_co['Erho2']), 'Phi_n3': float(pop_co['Phi'])}
    brng = np.random.default_rng(c['bootstrap_for_sd_ratio']['seed'])
    B = c['bootstrap_for_sd_ratio']['draws']
    boot_idx = brng.integers(0, c['replications'], (B, c['replications']))
    out = {'population_values': pop_vals, 'replications': c['replications'], 'comparisons': {}}
    covers = []
    for name in ('q_A1', 'q_A1_max'):
        comp = {}
        for i, k in enumerate(keys):
            a, b = est['p'][:, i], est[name][:, i]
            d = b - a
            m, se = float(np.nanmean(d)), float(np.nanstd(d, ddof=1) / math.sqrt(np.sum(~np.isnan(d))))
            tcrit = stats.t.ppf(0.975, np.sum(~np.isnan(d)) - 1)
            sd_ratio = float(np.nanstd(b, ddof=1) / np.nanstd(a, ddof=1))
            bs = np.array([np.nanstd(b[ix], ddof=1) / np.nanstd(a[ix], ddof=1) for ix in boot_idx])
            comp[k] = {'mean_p_panel': float(np.nanmean(a)), 'mean_q_panel': float(np.nanmean(b)),
                       'sd_p_panel': float(np.nanstd(a, ddof=1)), 'sd_q_panel': float(np.nanstd(b, ddof=1)),
                       'paired_mean_diff_q_minus_p': m, 'ci95': [m - tcrit * se, m + tcrit * se],
                       'sd_ratio_q_over_p': sd_ratio, 'sd_ratio_ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                       'n_undefined': int(np.isnan(b).sum() + np.isnan(a).sum())}
            if k in ('u', 'a', 'e'):
                covers.append(comp[k]['ci95'][0] <= 0 <= comp[k]['ci95'][1])
        out['comparisons'][name] = comp
    contract('A1.finite_sample', all(covers), f'{sum(covers)}/{len(covers)} component-difference intervals cover 0')
    return out


# ------------------------------------------------------------------------ A2
def a2(P):
    c = CFG['A2']
    S_wrong = set(c['wrong_subset_j'])
    W = [P[(12 - j) - 1] if j in S_wrong else P[j - 1] for j in range(1, 12)]
    pbar = [sum(p[y] for p in P) / len(P) for y in L.Y]
    Cst = [pbar] * len(P)
    def det(F):
        """Deterministic vector output: every call returns the same vector, so its scalar summary (the
        forecast mean) has no rater/call variance and no residual; v_u is the variance of forecast means."""
        ms = [L.mean(f) for f in F]
        mbar = sum(ms) / len(ms)
        return {'u': sum((m - mbar) ** 2 for m in ms) / len(ms), 'a': Fr(0), 'e': Fr(0)}
    judges = {'oracle': P, 'repeatable_wrong': W, 'constant_marginal': Cst}
    out = {'wrong_subset_j': sorted(S_wrong), 'judges': {}}
    for name, F in judges.items():
        per = []
        for j, (p, f) in enumerate(zip(P, F), start=1):
            per.append({'j': j, 'rps_excess': S(L.rps_excess(f, p)), 'tvd': S(L.tvd(f, p)), 'kl_nats': L.kl(p, f),
                        'abs_mean_error': S(abs(L.mean(f) - L.mean(p)))})
        v_det = det(F)
        v_samp = L.iid_panel_components(F)
        ns = [1, c['calls']]
        out['judges'][name] = {
            'per_artifact': per,
            'mean_expected_rps': S(sum(L.expected_rps(f, p) for p, f in zip(P, F)) / len(P)),
            'mean_rps_excess': S(sum(L.rps_excess(f, p) for p, f in zip(P, F)) / len(P)),
            'mean_tvd': S(sum(L.tvd(f, p) for p, f in zip(P, F)) / len(P)),
            'mean_kl_nats': float(np.mean([L.kl(p, f) for p, f in zip(P, F)])),
            'mae_of_means': S(sum(abs(L.mean(f) - L.mean(p)) for p, f in zip(P, F)) / len(P)),
            'bias_of_means': S(sum(L.mean(f) - L.mean(p) for p, f in zip(P, F)) / len(P)),
            'judge_mean': S(sum(L.mean(f) for f in F) / len(P)), 'human_mean': S(sum(L.mean(p) for p in P) / len(P)),
            'correlations_of_means': corr3([L.mean(f) for f in F], [L.mean(p) for p in P]),
            'deterministic_vector_output': {'components': {k: S(x) for k, x in v_det.items()}, 'coefficients': coeff_table(v_det, ns)},
            'sampled_rating_output': {'components': {k: S(x) for k, x in v_samp.items()}, 'coefficients': coeff_table(v_samp, ns),
                                      f'mean_expected_rps_{c["calls"]}call_histogram':
                                          S(sum(L.expected_rps(f, p) + L.rps_irreducible(f) / c['calls'] for p, f in zip(P, F)) / len(P))},
        }
    jw = out['judges']['repeatable_wrong']
    ok_rel = (jw['deterministic_vector_output']['components']['e']['exact'] == '0/1'
              and all(jw['deterministic_vector_output']['coefficients'][str(n)][k]['exact'] == '1/1'
                      for n in (1, c['calls']) for k in ('Erho2', 'Phi'))
              and out['judges']['oracle']['deterministic_vector_output']['coefficients'] == jw['deterministic_vector_output']['coefficients'])
    on = all((Fr(r['rps_excess']['exact']) > 0 and Fr(r['tvd']['exact']) > 0) == (r['j'] in S_wrong) for r in jw['per_artifact'])
    contract('A2.repeatable_wrong', ok_rel and on and Fr(jw['mae_of_means']['exact']) > 0,
             {'coefficients_all_1': ok_rel, 'errors_exactly_on_subset': on, 'mae': jw['mae_of_means']})
    jc = out['judges']['constant_marginal']
    undefined = all(jc['deterministic_vector_output']['coefficients'][str(n)][k] is None for n in (1, 3) for k in ('Erho2', 'Phi'))
    zero = all(jc['sampled_rating_output']['coefficients'][str(n)][k]['exact'] == '0/1' for n in (1, 3) for k in ('Erho2', 'Phi'))
    contract('A2.constant', undefined and zero, {'deterministic_undefined': undefined, 'sampled_zero': zero})

    # shared offset
    H = [[Fr(0), Fr(a, 10), Fr(10 - a - cc, 10), Fr(cc, 10), Fr(0), Fr(0)] for a, cc in c['offset_population']]
    G = [[Fr(0)] + h[:5] for h in H]
    eta = Fr(6, 100)
    Gs = [L.mix(g, L.UNIFORM, eta) for g in G]
    ch, cg = L.iid_panel_components(H), L.iid_panel_components(G)
    ns = c['pooled_rater_counts']
    pi = Fr(c['pooled_judge_rater_fraction'])
    pooled = {'u': ch['u'], 'a': pi * (1 - pi) * c['offset'] ** 2, 'e': ch['e']}
    off = {
        'human_components': {k: S(x) for k, x in ch.items()}, 'judge_components': {k: S(x) for k, x in cg.items()},
        'human_coefficients': coeff_table(ch, ns), 'judge_coefficients': coeff_table(cg, ns),
        'pooled_components': {k: S(x) for k, x in pooled.items()}, 'pooled_coefficients': coeff_table(pooled, ns),
        'bias_per_artifact': [S(L.mean(g) - L.mean(h)) for h, g in zip(H, G)],
        'human_mean': S(sum(L.mean(h) for h in H) / len(H)), 'judge_mean': S(sum(L.mean(g) for g in G) / len(G)),
        'mean_expected_rps_oracle': S(sum(L.expected_rps(h, h) for h in H) / len(H)),
        'mean_expected_rps_shifted': S(sum(L.expected_rps(g, h) for h, g in zip(H, G)) / len(H)),
        'mean_expected_rps_shifted_smoothed': S(sum(L.expected_rps(g, h) for h, g in zip(H, Gs)) / len(H)),
        'mean_expected_ce_oracle_nats': float(np.mean([L.expected_ce(h, h) for h in H])),
        'mean_expected_ce_shifted_nats': float(np.mean([L.expected_ce(g, h) for h, g in zip(H, G)])),
        'mean_expected_ce_shifted_smoothed_nats': float(np.mean([L.expected_ce(g, h) for h, g in zip(H, Gs)])),
        'correlations_of_means': corr3([L.mean(g) for g in G], [L.mean(h) for h in H]),
        'smoothing_eta': L.fr_to_str(eta),
    }
    ok = (ch == cg and all(L.mean(g) - L.mean(h) == 1 for h, g in zip(H, G))
          and all(L.rps_excess(g, h) > 0 for h, g in zip(H, G)) and math.isinf(off['mean_expected_ce_shifted_nats'])
          and off['mean_expected_ce_oracle_nats'] < off['mean_expected_ce_shifted_smoothed_nats'] < math.inf
          and all(L.one_facet_coeffs(pooled, n)['Erho2'] == L.one_facet_coeffs(ch, n)['Erho2']
                  and L.one_facet_coeffs(pooled, n)['Phi'] < L.one_facet_coeffs(ch, n)['Phi'] for n in ns))
    contract('A2.offset', ok, {'bias': '+1 each', 'ce_shifted': off['mean_expected_ce_shifted_nats'],
                               'ce_smoothed': off['mean_expected_ce_shifted_smoothed_nats']})
    out['offset'] = off
    return out


# ------------------------------------------------------------------------ A3
def a3(P):
    c = CFG['A3']
    Pf = [[float(x) for x in p] for p in P]

    def tempered(p, T):
        if T == 'mode':
            # Amendment 1: the T -> 0 limit of p^(1/T) splits mass equally over tied modes
            # (theta = 1/2 has p(2) = p(3)); the frozen prediction is left unchanged.
            m = max(p)
            k = sum(x == m for x in p)
            return [1.0 / k if x == m else 0.0 for x in p]
        w = [x ** (1.0 / float(Fr(T))) for x in p]
        s = sum(w)
        return [x / s for x in w]

    fam = {}
    for T in c['temperatures']:
        F = [tempered(p, T) for p in Pf]
        comp = L.iid_panel_components(F)
        fam[T] = {
            'mean_expected_rps': float(np.mean([L.expected_rps(f, p) for p, f in zip(Pf, F)])),
            'mean_expected_ce_nats': float(np.mean([L.expected_ce(f, p) for p, f in zip(Pf, F)])),
            'mean_tvd': float(np.mean([L.tvd(f, p) for p, f in zip(Pf, F)])),
            'mae_of_means': float(np.mean([abs(L.mean(f) - L.mean(p)) for p, f in zip(Pf, F)])),
            'mean_conditional_variance': float(np.mean([L.var(f) for f in F])),
            'human_mean_conditional_variance': float(np.mean([L.var(p) for p in Pf])),
            'sampled_components': {k: float(x) for k, x in comp.items()},
            'sampled_Erho2': {str(n): L.as_float(L.one_facet_coeffs(comp, n)['Erho2']) for n in c['sampled_rating_counts']},
            'mean_expected_rps_3call_histogram': float(np.mean([L.expected_rps(f, p) + L.rps_irreducible(f) / 3 for p, f in zip(Pf, F)])),
        }
    order = c['temperatures']
    rps_min = min(order, key=lambda T: fam[T]['mean_expected_rps'])
    ce_min = min(order, key=lambda T: fam[T]['mean_expected_ce_nats'])
    contract('A3.oracle', rps_min == '1' and ce_min == '1' and fam['1']['mean_expected_rps'] > 0
             and math.isinf(fam['mode']['mean_expected_ce_nats']),
             {'rps_argmin': rps_min, 'ce_argmin': ce_min, 'oracle_rps': fam['1']['mean_expected_rps']})
    e1 = [fam[T]['sampled_Erho2']['1'] for T in order]
    mono = all(a < b for a, b in zip(e1, e1[1:]))
    contract('A3.sampled_reliability', mono and abs(fam['mode']['sampled_Erho2']['1'] - 1) < 1e-15,
             dict(zip(order, e1)))
    # population dependence with identical per-artifact distributions up to shift
    base = [Fr(x) for x in c['shift_family_base']]

    def shift_pmf(cen):
        q = [Fr(0)] * 6
        for k, b in enumerate(base):
            q[cen - 1 + k] = b
        return q

    pops = {}
    for name in ('narrow', 'wide'):
        F = [shift_pmf(cc) for cc in c[f'{name}_centers']]
        comp = L.iid_panel_components(F)
        pops[name] = {'centers': c[f'{name}_centers'],
                      'oracle_mean_expected_rps': S(sum(L.expected_rps(f, f) for f in F) / len(F)),
                      'oracle_mae': S(Fr(0)),
                      'components': {k: S(x) for k, x in comp.items()},
                      'sampled_coefficients': coeff_table(comp, c['sampled_rating_counts'])}
    ok = (pops['narrow']['oracle_mean_expected_rps']['exact'] == '3/40' == pops['wide']['oracle_mean_expected_rps']['exact']
          and pops['narrow']['sampled_coefficients']['1']['Erho2']['exact'] == '1/3'
          and pops['wide']['sampled_coefficients']['1']['Erho2']['exact'] == '5/7')
    contract('A3.population_dependence', ok, {n: pops[n]['sampled_coefficients']['1']['Erho2'] for n in pops})
    return {'tempered_family': fam, 'population_dependence': pops}


# ------------------------------------------------------------------------ A4
def a4():
    c = CFG['A4']
    cfgs = {k: {kk: Fr(vv) for kk, vv in v.items()} for k, v in c['configs'].items()}
    lv = c['facet_levels']
    out = {'configs': {}, 'optimal': {}}
    for name, v in cfgs.items():
        out['configs'][name] = {
            'marginal_single_rating_variance': S(sum(v.values())),
            'components': {k: L.fr_to_str(x) for k, x in v.items()},
            'designs': {f'{na}x{nb}': {k: S(x) for k, x in L.d_study(v, na, nb).items()} for na in lv for nb in lv},
        }
        out['optimal'][name] = {}
        for budget, designs in c['budgets'].items():
            for target, key in (('absolute', 'v_Delta'), ('relative', 'v_delta')):
                vals = {f'{a}x{b}': L.d_study(v, a, b)[key] for a, b in designs}
                out['optimal'][name][f'budget{budget}_{target}'] = min(vals, key=vals.get)
    ok = (out['configs']['C1']['marginal_single_rating_variance'] == out['configs']['C2']['marginal_single_rating_variance']
          and all(out['optimal']['C1'][f'budget16_{t}'] == '8x2' and out['optimal']['C2'][f'budget16_{t}'] == '2x8'
                  for t in ('absolute', 'relative')))
    contract('A4.design', ok, out['optimal'])
    return out


def main():
    t0 = time.time()
    thetas, P = human_population()
    assert all(L.is_pmf(p, positive=True) for p in P)
    rows_a1, Q_a1, sum_a1 = a1(P)
    rows_max, Q_max, sum_max = a1_max(P, Q_a1)
    fin = a1_finite(P, Q_a1, Q_max)
    r2 = a2(P)
    r3 = a3(P)
    r4 = a4()
    tj = CFG['human_population']['trace_artifact_j']
    trace = {'j': tj, 'theta': CFG['human_population']['theta'][tj - 1], 'p': rows_a1[tj - 1]['p'],
             'q_A1': rows_a1[tj - 1]['q'], 'q_A1_max': rows_max[tj - 1]['q'],
             'cdf_p': vecS(L.cdf(P[tj - 1])), 'cdf_q_A1': vecS(L.cdf(Q_a1[tj - 1])),
             'rps_individual_p_by_rating': vecS([L.rps_individual(P[tj - 1], y) for y in L.Y]),
             'rps_individual_qA1_by_rating': vecS([L.rps_individual(Q_a1[tj - 1], y) for y in L.Y]),
             'expected_rps_p': S(L.expected_rps(P[tj - 1], P[tj - 1])), 'expected_rps_qA1': S(L.expected_rps(Q_a1[tj - 1], P[tj - 1])),
             'serialization_roundtrip_exact': [L.str_to_fr(s) for s in rows_a1[tj - 1]['q']] == Q_a1[tj - 1]}
    out = {
        'experiment': PROTOCOL['experiment'], 'phase': 'A', 'protocol_version': PROTOCOL['protocol_version'],
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'command': 'python experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/check_counterexamples.py',
        'runtime_seconds': None,
        'A1': {'per_artifact': rows_a1, 'summary': sum_a1},
        'A1_max': {'per_artifact': rows_max, 'summary': sum_max},
        'A1_finite_sample': fin, 'A2': r2, 'A3': r3, 'A4': r4, 'trace': trace,
        'contract': CONTRACT,
    }
    out['runtime_seconds'] = round(time.time() - t0, 2)
    path = HERE / 'results' / 'phase_a.json'
    path.write_text(json.dumps(out, indent=1, default=str) + '\n')
    n_fail = sum(not c['pass'] for c in CONTRACT)
    print(json.dumps({'contract_checks': len(CONTRACT), 'failed': [c['id'] for c in CONTRACT if not c['pass']],
                      'runtime_seconds': out['runtime_seconds']}))
    return n_fail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
