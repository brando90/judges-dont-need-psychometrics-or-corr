#!/usr/bin/env python3
"""TLDR: Independent recomputation of experiment 04 headline numbers from saved outputs, plus two
post-hoc diagnostics (labeled; they never replace a frozen result).

Checks (different code paths from the measuring scripts):
  A  per-artifact A1/A1_max moments and RPS/TVD via numpy matrix algebra; A4 optima by brute force.
  B  every summary in phase_b.json recomputed from the per-replication CSVs; design choices recomputed
     from the saved REML estimates.
  C  point metrics recomputed from the private dataset with separate RPS/ICC/ANOVA code.
Post-hoc diagnostics: (1) A1 finite-sample v_u difference with 20,000 new replications (seed 4099);
(2) Phase B C1 4x8 absolute-error formula check with 8,192 new replications (condition id 901).

Run: python experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/verify_results.py
Writes results/verification.json.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent.parent
RES = HERE / 'results'
CHECKS: list[dict] = []


def check(name, ok, detail=''):
    CHECKS.append({'check': name, 'pass': bool(ok), 'detail': detail})
    if not ok:
        print('FAIL', name, detail)


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol * (1 + abs(b))


# ------------------------------------------------------------------ Phase A
def verify_a():
    a = json.loads((RES / 'phase_a.json').read_text())
    y = np.arange(6.0)
    T = np.tril(np.ones((5, 6)))          # row k sums categories 0..k
    for name in ('A1', 'A1_max'):
        for r_a1, r in zip(a['A1']['per_artifact'], a[name]['per_artifact']):
            p = np.array([float(Fr(s)) for s in r_a1['p']])
            q = np.array([float(Fr(s)) for s in r['q']])
            check(f'{name}.moments[{r["j"]}]', abs(q @ y - p @ y) < 1e-12 and abs(q @ y ** 2 - p @ y ** 2) < 1e-12)
            rps_ex = float(((T @ q - T @ p) ** 2).sum() / 5)
            check(f'{name}.rps_excess[{r["j"]}]', close(rps_ex, r['rps_excess']['float'], 1e-9))
            check(f'{name}.tvd[{r["j"]}]', close(0.5 * np.abs(q - p).sum(), r['tvd']['float'], 1e-9))
    # A4 optima by brute force over floats
    cfgs = a['A4']['configs']
    for cname, c in cfgs.items():
        v = {k: float(Fr(x)) for k, x in c['components'].items()}
        for target, key in (('absolute', 'v_Delta'), ('relative', 'v_delta')):
            best = None
            for na, nb in [(2, 8), (4, 4), (8, 2)]:
                e = v['ua'] / na + v['ub'] / nb + v['e'] / (na * nb)
                if key == 'v_Delta':
                    e += v['a'] / na + v['b'] / nb + v['ab'] / (na * nb)
                if best is None or e < best[0]:
                    best = (e, f'{na}x{nb}')
            check(f'A4.optimum[{cname},{target}]', best[1] == a['A4']['optimal'][cname][f'budget16_{target}'])
    # A3 identity: 3-call histogram RPS = RPS + irreducible(q)/3, recomputed for the oracle row
    fam = a['A3']['tempered_family']['1']
    check('A3.oracle_3call_identity', close(fam['mean_expected_rps_3call_histogram'],
                                            fam['mean_expected_rps'] * 4 / 3, 1e-9))
    return a


# ------------------------------------------------------------------ Phase B
def tmean(x):
    x = np.asarray(x, float)
    return float(x.mean())


def verify_b():
    b = json.loads((RES / 'phase_b.json').read_text())
    est = list(csv.DictReader(open(RES / 'phase_b_estimates.csv')))
    obs = list(csv.DictReader(open(RES / 'phase_b_design_errors.csv')))
    sel = list(csv.DictReader(open(RES / 'phase_b_selection.csv')))
    comps = ('u', 'a', 'b', 'ua', 'ub', 'ab', 'e')
    for cname, fc in b['formula_checks'].items():
        for key, rec in fc.items():
            design, et = key.split('_')
            vals = [float(r[f'{et}_err']) for r in obs if r['config'] == cname and r['design'] == design]
            check(f'B.formula_mean[{cname},{key}]', len(vals) == 256 and close(tmean(vals), rec['mean'], 1e-12))
    # recompute REML-based choices at budget 16 (absolute) from saved estimates
    for cname in ('C1', 'C2'):
        mism = 0
        for r in (x for x in est if x['config'] == cname):
            v = {c: float(r[f'reml_{c}']) for c in comps}
            errs = {}
            for na, nb in [(2, 8), (4, 4), (8, 2)]:
                errs[f'{na}x{nb}'] = (v['ua'] / na + v['ub'] / nb + v['e'] / (na * nb) + v['a'] / na + v['b'] / nb + v['ab'] / (na * nb))
            choice = min(errs, key=errs.get)
            saved = next(s['chosen'] for s in sel if s['config'] == cname and s['rep'] == r['rep'] and s['budget'] == '16'
                         and s['target'] == 'absolute' and s['selector'] == 'reml')
            mism += choice != saved
        check(f'B.choices_from_saved_estimates[{cname}]', mism == 0, f'{mism} mismatches')
        recs = [s for s in sel if s['config'] == cname and s['budget'] == '16' and s['target'] == 'absolute' and s['selector'] == 'reml']
        prop = sum(s['chosen'] == s['oracle_design'] for s in recs) / len(recs)
        check(f'B.oracle_proportion[{cname}]', close(prop, b['selection'][cname]['budget16_absolute_reml']['chose_oracle_design']['proportion']))
        eq = {r['rep']: float(r['abs_err']) for r in obs if r['config'] == cname and r['design'] == '4x4'}
        d = [float(s['obs_err_chosen']) - eq[s['rep']] for s in recs]
        check(f'B.paired_vs_equal[{cname}]', close(tmean(d), b['selection'][cname]['budget16_absolute_reml']['paired_obs_err_chosen_minus_equal']['mean'], 1e-12))
    return b


# ------------------------------------------------------------------ Phase C
def verify_c():
    c = json.loads((RES / 'phase_c.json').read_text())
    src = HERE.parent / '02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/private/historical_exp06/cached_v1/dataset.json'
    data = json.loads(src.read_text())
    rows = data['rows']
    ix = [i for i, r in enumerate(rows) if r['split'] == 'test']
    eps = 1e-6

    def rps_cum(q, y):
        Q = np.cumsum(q)[:5]
        H = (np.arange(5) >= y).astype(float)
        return float(((Q - H) ** 2).mean())   # mean over 5 thresholds == (1/5) sum

    def icc_a1(x, yv):
        """ICC(A,1) via variance components of a 2-column matrix, computed without anova2."""
        M = np.column_stack([x, yv])
        n, k = M.shape
        gm = M.mean()
        ssr = k * ((M.mean(1) - gm) ** 2).sum()
        ssc = n * ((M.mean(0) - gm) ** 2).sum()
        sse = ((M - gm) ** 2).sum() - ssr - ssc
        msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
        return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

    pt = c['sets']['test']['metrics']
    for j in ('H0_sol', 'H1_sol', 'H2_sol', 'H0_opus', 'H2_opus'):
        rps, mae, jm_all, hm_all = [], [], [], []
        for i in ix:
            calls = data['outputs'][j][i]
            q = np.bincount(np.array(calls, int), minlength=6) / 3.0
            q = (1 - 6 * eps) * q + eps
            ys = rows[i]['ratings']
            rps.append(np.mean([rps_cum(q, yy) for yy in ys]))
            jm_all.append(np.mean(calls))
            hm_all.append(np.mean(ys))
        check(f'C.rps[{j}]', close(float(np.mean(rps)), pt[j]['RPS']['point'], 1e-9))
        check(f'C.mae[{j}]', close(float(np.mean(np.abs(np.array(jm_all) - np.array(hm_all)))), pt[j]['MAE']['point'], 1e-12))
        check(f'C.icc_a1[{j}]', close(float(icc_a1(np.array(jm_all), np.array(hm_all))), pt[j]['ICC_A1']['point'], 1e-9))
        X = np.array([data['outputs'][j][i] for i in ix], float)
        # test-retest single-call Erho2 = (MS_u - MS_e)/(MS_u + 2 MS_e) via an explicit residual matrix
        n, k = X.shape
        R = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
        mse = (R ** 2).sum() / ((n - 1) * (k - 1))
        msu = k * ((X.mean(1) - X.mean()) ** 2).sum() / (n - 1)
        check(f'C.erho2_1[{j}]', close((msu - mse) / (msu + (k - 1) * mse), pt[j]['Erho2_1']['point'], 1e-9))
    o_rel = c['sets']['test']['order_by_reliability']
    o_rps = c['sets']['test']['order_by_rps']
    tau = stats.kendalltau([-o_rel.index(j) for j in o_rel], [-o_rps.index(j) for j in o_rel])[0]
    check('C.kendall_tau_from_orders', close(float(tau), c['sets']['test']['kendall_tau_reliability_vs_accuracy'], 1e-12))
    return c


# ------------------------------------------------------------------ post-hoc diagnostics
def diagnostics():
    import check_counterexamples as A
    import gtheory_lib as L
    import run_d_study as B
    out = {}
    _, P = A.human_population()
    _, Q, _ = A.a1(P)
    Pf = np.array([[float(x) for x in p] for p in P])
    Qf = np.array([[float(x) for x in q] for q in Q])
    rng = np.random.default_rng(4099)
    d = []
    for _ in range(20000):
        idx = rng.integers(0, len(P), 50)
        d.append(L.anova2(A.sample_panel(rng, Qf, idx, 3))['est']['u'] - L.anova2(A.sample_panel(rng, Pf, idx, 3))['est']['u'])
    d = np.array(d)
    se = d.std(ddof=1) / math.sqrt(len(d))
    out['A1_finite_sample_v_u_difference_rerun'] = {
        'label': 'post hoc diagnostic; frozen result unchanged', 'replications': 20000, 'seed': 4099,
        'mean_q_minus_p': float(d.mean()), 'ci95': [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]}
    v = B.CFG['configs']['C1']
    x = np.array([B.eval_errors(B.rng_for(r, 901, 1), v, (4, 8))[0] for r in range(8192)])
    pop = L.d_study(v, 4, 8)['v_Delta']
    se = x.std(ddof=1) / math.sqrt(len(x))
    frozen = [float(r['abs_err']) for r in csv.DictReader(open(RES / 'phase_b_design_errors.csv'))
              if r['config'] == 'C1' and r['design'] == '4x8']
    out['B_C1_4x8_absolute_formula_rerun'] = {
        'label': 'post hoc diagnostic; frozen result unchanged', 'replications': 8192, 'condition_id': 901,
        'population': pop, 'mean': float(x.mean()), 'z': float((x.mean() - pop) / se),
        'skewness': float(stats.skew(x)), 'frozen_mean': float(np.mean(frozen)),
        'frozen_sample_sd': float(np.std(frozen, ddof=1)), 'large_sample_sd': float(x.std(ddof=1)),
        'frozen_z_using_large_sample_sd': float((np.mean(frozen) - pop) / (x.std(ddof=1) / math.sqrt(len(frozen))))}
    return out


def main():
    verify_a()
    verify_b()
    verify_c()
    diag = diagnostics()
    n_fail = sum(not c['pass'] for c in CHECKS)
    (RES / 'verification.json').write_text(json.dumps({'n_checks': len(CHECKS), 'n_fail': n_fail,
                                                       'post_hoc_diagnostics': diag, 'checks': CHECKS}, indent=1) + '\n')
    print(json.dumps({'n_checks': len(CHECKS), 'n_fail': n_fail, 'post_hoc_diagnostics': diag}, indent=1))
    return n_fail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
