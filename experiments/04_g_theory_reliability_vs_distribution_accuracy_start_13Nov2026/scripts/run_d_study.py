#!/usr/bin/env python3
"""TLDR: Phase B of experiment 04 — a fair test of decision-study (D-study) allocation.

Crossed objects x a x b Gaussian random-effects model with known components. Each of 256
replications runs the complete procedure: a 64 x 8 x 8 calibration generalizability study (G study),
constrained REML (and truncated ANOVA) component estimates, design choice at fixed rating budgets,
and evaluation of every candidate design on fresh objects, facet levels and residuals.
Then four assumption violations (correlated raters, group heterogeneity, population shift and
dependent calibration occasions) are tested separately.

Run: python experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/run_d_study.py
Writes results/phase_b.json (summaries + contract) and results/phase_b_*.csv (per replication).
"""
from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gtheory_lib as L  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
CFG = PROTOCOL['phase_b']
R = CFG['replications']
CAL = CFG['calibration']
N_EVAL = CFG['evaluation']['n_objects']
DESIGNS = [tuple(d) for d in CFG['evaluation']['designs']]
BUDGETS = {b: [tuple(d) for d in ds] for b, ds in CFG['budgets'].items()}
CONTRACT: list[dict] = []


def contract(cid, ok, observed):
    CONTRACT.append({'id': cid, 'pass': bool(ok), 'observed': observed})


def rng_for(rep, cond, stage):
    return np.random.default_rng(np.random.SeedSequence([rep, cond, stage]))


def gen(rng, v, n_u, n_a, n_b, ua_sd_by_object=None, rater_pair_rho=None, ub_dep_rho=None):
    """One crossed data set X[u, a, b] and the universe scores U (grand mean 0)."""
    sd = {c: math.sqrt(v[c]) for c in L.COMPS}
    U = rng.normal(0, sd['u'], n_u)
    B = rng.normal(0, sd['b'], n_b)
    if rater_pair_rho is None:
        A = rng.normal(0, 1, n_a)
        UA = rng.normal(0, 1, (n_u, n_a))
        AB = rng.normal(0, 1, (n_a, n_b))
    else:
        rho, h = rater_pair_rho, n_a // 2
        k = math.sqrt(1 - rho ** 2)

        def paired(first, axis):
            second = rho * first + k * rng.normal(0, 1, first.shape)
            return np.stack([first, second], axis=axis + 1).reshape(
                first.shape[:axis] + (2 * first.shape[axis],) + first.shape[axis + 1:])
        A = paired(rng.normal(0, 1, h), 0)
        UA = paired(rng.normal(0, 1, (n_u, h)), 1)
        AB = paired(rng.normal(0, 1, (h, n_b)), 0)
    A = A * sd['a']
    UA = UA * (sd['ua'] if ua_sd_by_object is None else np.asarray(ua_sd_by_object)[:, None])
    AB = AB * sd['ab']
    if ub_dep_rho is None:
        UB = rng.normal(0, sd['ub'], (n_u, n_b))
    else:
        W = rng.normal(0, 1, (n_u, 1))
        UB = sd['ub'] * (math.sqrt(ub_dep_rho) * W + math.sqrt(1 - ub_dep_rho) * rng.normal(0, 1, (n_u, n_b)))
    E = rng.normal(0, sd['e'], (n_u, n_a, n_b))
    X = U[:, None, None] + A[None, :, None] + B[None, None, :] + UA[:, :, None] + UB[:, None, :] + AB[None, :, :] + E
    return X, U


def eval_errors(rng, v, design, n_obj=N_EVAL, **kw):
    X, U = gen(rng, v, n_obj, design[0], design[1], **kw)
    d = X.mean((1, 2)) - U
    return float(np.mean(d ** 2)), float(np.var(d, ddof=1)), d


def tci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = float(x.mean())
    if len(x) < 2:
        return {'mean': m, 'ci95': None, 'n': int(len(x))}
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    t = stats.t.ppf(0.975, len(x) - 1)
    return {'mean': m, 'ci95': [m - t * se, m + t * se], 'se': se, 'n': int(len(x))}


def wilson(k, n):
    if n == 0:
        return None
    z, p = 1.959963984540054, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {'proportion': p, 'k': k, 'n': n, 'ci95': [c - h, c + h]}


def dkey(d):
    return f'{d[0]}x{d[1]}'


def choose(v, cands, key):
    vals = [L.d_study(v, *d)[key] for d in cands]
    return cands[int(np.argmin(vals))]       # ties keep the first listed design


def main():
    t0 = time.time()
    rows_est, rows_obs, rows_sel = [], [], []
    cal_cache = {}
    summary = {'formula_checks': {}, 'estimates': {}, 'selection': {}, 'fixed_designs_budget16': {}}
    keymap = {'absolute': 'v_Delta', 'relative': 'v_delta'}
    for cond, cname in ((1, 'C1'), (2, 'C2')):
        v = CFG['configs'][cname]
        obs_all = {d: {'abs': [], 'rel': []} for d in DESIGNS}
        est_rec = {s: {c: [] for c in L.COMPS} for s in ('reml', 'anova_raw')}
        n_success = 0
        sel = {}
        for r in range(R):
            Xc, _ = gen(rng_for(r, cond, 0), v, CAL['n_u'], CAL['n_a'], CAL['n_b'])
            an = L.anova3(Xc)
            rm = L.reml3(an['ss'], an['df'], CAL['n_u'], CAL['n_a'], CAL['n_b'])
            n_success += rm['success']
            est = {'reml': rm['est'], 'anova_trunc': {c: max(0.0, an['est'][c]) for c in L.COMPS}, 'oracle': v}
            cal_cache[(cname, r)] = est
            for c in L.COMPS:
                est_rec['reml'][c].append(rm['est'][c])
                est_rec['anova_raw'][c].append(an['est'][c])
            rows_est.append({'config': cname, 'rep': r, 'reml_success': rm['success'],
                             **{f'reml_{c}': rm['est'][c] for c in L.COMPS}, **{f'anova_{c}': an['est'][c] for c in L.COMPS}})
            rng_ev = rng_for(r, cond, 1)
            obs = {}
            for d in DESIGNS:
                a_err, r_err, _ = eval_errors(rng_ev, v, d)
                obs[d] = {'abs': a_err, 'rel': r_err}
                obs_all[d]['abs'].append(a_err)
                obs_all[d]['rel'].append(r_err)
                rows_obs.append({'config': cname, 'rep': r, 'design': dkey(d), 'abs_err': a_err, 'rel_err': r_err})
            for budget, cands in BUDGETS.items():
                for target, key in keymap.items():
                    best_true = min(L.d_study(v, *d)[key] for d in cands)
                    for s, ev in est.items():
                        ch = choose(ev, cands, key)
                        true_err = L.d_study(v, *ch)[key]
                        pred = L.d_study(ev, *ch)
                        coef = 'Phi' if target == 'absolute' else 'Erho2'
                        rec = {'config': cname, 'rep': r, 'budget': budget, 'target': target, 'selector': s,
                               'chosen': dkey(ch), 'oracle_design': dkey(choose(v, cands, key)),
                               'true_err_chosen': true_err, 'regret': true_err - best_true,
                               'obs_err_chosen': obs[ch]['abs' if target == 'absolute' else 'rel'],
                               'pred_err_chosen': pred[key], 'pred_coef_chosen': pred[coef],
                               'true_coef_chosen': L.d_study(v, *ch)[coef]}
                        rows_sel.append(rec)
                        sel.setdefault((budget, target, s), []).append(rec)
        # formula checks
        fc = {}
        for d in DESIGNS:
            pop = L.d_study(v, *d)
            for et, key in (('abs', 'v_Delta'), ('rel', 'v_delta')):
                ci = tci(obs_all[d][et])
                z = (ci['mean'] - pop[key]) / ci['se']
                fc[f'{dkey(d)}_{et}'] = {'population': pop[key], **ci, 'z': z, 'within_tolerance': abs(z) <= 3.5}
        summary['formula_checks'][cname] = fc
        summary['estimates'][cname] = {
            'true': v, 'reml_converged': n_success, 'replications': R,
            'reml': {c: {'mean': float(np.mean(est_rec['reml'][c])), 'sd': float(np.std(est_rec['reml'][c], ddof=1)),
                         'n_at_zero': int(np.sum(np.asarray(est_rec['reml'][c]) <= 1e-10))} for c in L.COMPS},
            'anova_raw': {c: {**tci(est_rec['anova_raw'][c]), 'sd': float(np.std(est_rec['anova_raw'][c], ddof=1)),
                              'n_negative': int(np.sum(np.asarray(est_rec['anova_raw'][c]) < 0))} for c in L.COMPS},
        }
        selsum = {}
        for (budget, target, s), recs in sel.items():
            k = sum(x['chosen'] == x['oracle_design'] for x in recs)
            entry = {'oracle_design': recs[0]['oracle_design'],
                     'choice_counts': {c: sum(x['chosen'] == c for x in recs) for c in sorted({x['chosen'] for x in recs})},
                     'chose_oracle_design': wilson(k, len(recs)),
                     'regret_true_err': tci([x['regret'] for x in recs]),
                     'obs_err_chosen': tci([x['obs_err_chosen'] for x in recs]),
                     'pred_minus_true_err': tci([x['pred_err_chosen'] - x['true_err_chosen'] for x in recs]),
                     'pred_minus_true_coef': tci([x['pred_coef_chosen'] - x['true_coef_chosen'] for x in recs])}
            if budget == '16':
                et = 'abs' if target == 'absolute' else 'rel'
                for bname, bd in CFG['fixed_baselines_budget_16'].items():
                    diffs = [x['obs_err_chosen'] - obs_all[tuple(bd)][et][x['rep']] for x in recs]
                    entry[f'paired_obs_err_chosen_minus_{bname}'] = tci(diffs)
                    entry[f'true_err_chosen_minus_{bname}'] = tci(
                        [x['true_err_chosen'] - L.d_study(v, *bd)[keymap[target]] for x in recs])
            selsum[f'budget{budget}_{target}_{s}'] = entry
        summary['selection'][cname] = selsum
        summary['fixed_designs_budget16'][cname] = {
            bname: {'abs_obs': tci(obs_all[tuple(bd)]['abs']), 'abs_pop': L.d_study(v, *bd)['v_Delta'],
                    'rel_obs': tci(obs_all[tuple(bd)]['rel']), 'rel_pop': L.d_study(v, *bd)['v_delta'],
                    'Phi_pop': L.d_study(v, *bd)['Phi'], 'Erho2_pop': L.d_study(v, *bd)['Erho2']}
            for bname, bd in CFG['fixed_baselines_budget_16'].items()}

    n_fc = sum(len(x) for x in summary['formula_checks'].values())
    n_ok = sum(y['within_tolerance'] for x in summary['formula_checks'].values() for y in x.values())
    contract('B.formula', n_ok == n_fc, f'{n_ok}/{n_fc} within 3.5 Monte Carlo SE')
    sel_ok, sel_obs = True, {}
    for cname in ('C1', 'C2'):
        e = summary['selection'][cname]['budget16_absolute_reml']
        p = e['chose_oracle_design']['proportion']
        d = e['paired_obs_err_chosen_minus_equal']
        sel_obs[cname] = {'chose_oracle': p, 'paired_diff_vs_equal': d['mean'], 'ci95': d['ci95']}
        sel_ok &= p >= 0.9 and d['ci95'][1] < 0
    contract('B.selection', sel_ok, sel_obs)

    vul = vulnerabilities(cal_cache)
    out = {'experiment': PROTOCOL['experiment'], 'phase': 'B', 'protocol_version': PROTOCOL['protocol_version'],
           'software': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
           'command': 'python experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/run_d_study.py',
           **summary, 'vulnerabilities': vul, 'contract': CONTRACT, 'runtime_seconds': round(time.time() - t0, 2)}
    (HERE / 'results' / 'phase_b.json').write_text(json.dumps(out, indent=1) + '\n')
    for name, rows in (('estimates', rows_est), ('design_errors', rows_obs), ('selection', rows_sel)):
        with open(HERE / 'results' / f'phase_b_{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({'contract_checks': len(CONTRACT), 'failed': [c['id'] for c in CONTRACT if not c['pass']],
                      'runtime_seconds': out['runtime_seconds']}))
    return sum(not c['pass'] for c in CONTRACT)


def vulnerabilities(cal_cache):
    V = CFG['vulnerabilities']
    out = {}
    # V1 correlated raters
    c1 = CFG['configs']['C1']
    v1 = V['V1_correlated_raters']
    d = tuple(v1['design'])
    naive = L.d_study(c1, *d)['v_Delta']
    res = {}
    for i, rho in enumerate(v1['rho']):
        vmod = dict(c1)
        for c in ('a', 'ua', 'ab'):
            vmod[c] = c1[c] * (1 + rho)
        analytic = L.d_study(vmod, *d)['v_Delta']
        obs = [eval_errors(rng_for(r, 11 + i, 1), c1, d, rater_pair_rho=rho)[0] for r in range(R)]
        pred = [L.d_study(cal_cache[('C1', r)]['reml'], *d)['v_Delta'] for r in range(R)]
        o = tci(obs)
        res[str(rho)] = {'naive_population_prediction': naive, 'analytic_actual': analytic, 'observed_abs_err': o,
                         'reml_prediction': tci(pred), 'naive_Phi': L.d_study(c1, *d)['Phi'],
                         'actual_Phi': L.d_study(vmod, *d)['Phi'],
                         'z_vs_analytic': (o['mean'] - analytic) / o['se'], 'z_vs_naive': (o['mean'] - naive) / o['se']}
    out['V1_correlated_raters'] = res
    contract('B.V1', all(x['observed_abs_err']['ci95'][0] > x['naive_population_prediction'] and abs(x['z_vs_analytic']) <= 3.5
                         for x in res.values()), {k: (x['observed_abs_err']['mean'], x['analytic_actual']) for k, x in res.items()})

    # V2 heterogeneity across predeclared groups
    v2 = V['V2_group_heterogeneity']
    d = tuple(v2['design'])
    sdL, sdH = math.sqrt(v2['groups']['L']['ua']), math.sqrt(v2['groups']['H']['ua'])
    pred, obsL, obsH, diff = [], [], [], []
    for r in range(R):
        nc = CAL['n_u']
        Xc, _ = gen(rng_for(r, 21, 0), c1, nc, CAL['n_a'], CAL['n_b'], ua_sd_by_object=[sdL] * (nc // 2) + [sdH] * (nc // 2))
        an = L.anova3(Xc)
        rm = L.reml3(an['ss'], an['df'], nc, CAL['n_a'], CAL['n_b'])
        pred.append(L.d_study(rm['est'], *d)['v_Delta'])
        _, _, dv = eval_errors(rng_for(r, 21, 1), c1, d, ua_sd_by_object=[sdL] * (N_EVAL // 2) + [sdH] * (N_EVAL // 2))
        eL, eH = float(np.mean(dv[:N_EVAL // 2] ** 2)), float(np.mean(dv[N_EVAL // 2:] ** 2))
        obsL.append(eL)
        obsH.append(eH)
        diff.append(eH - eL)
    gL, gH = dict(c1, ua=v2['groups']['L']['ua']), dict(c1, ua=v2['groups']['H']['ua'])
    out['V2_group_heterogeneity'] = {
        'pooled_reml_prediction': tci(pred), 'pooled_population': L.d_study(c1, *d)['v_Delta'],
        'group_L': {'analytic': L.d_study(gL, *d)['v_Delta'], 'observed': tci(obsL), 'Phi': L.d_study(gL, *d)['Phi']},
        'group_H': {'analytic': L.d_study(gH, *d)['v_Delta'], 'observed': tci(obsH), 'Phi': L.d_study(gH, *d)['Phi']},
        'pooled_Phi': L.d_study(c1, *d)['Phi'], 'paired_obs_H_minus_L': tci(diff)}
    v2o = out['V2_group_heterogeneity']
    contract('B.V2', abs(v2o['pooled_reml_prediction']['mean'] - v2o['pooled_population']) < 0.03
             and v2o['paired_obs_H_minus_L']['ci95'][0] <= 0.1 <= v2o['paired_obs_H_minus_L']['ci95'][1],
             {'pooled_pred': v2o['pooled_reml_prediction']['mean'], 'H_minus_L': v2o['paired_obs_H_minus_L']})

    # V3 population shift; design chosen by REML on the unshifted C1 calibration (budget 16, absolute)
    v3 = V['V3_population_shift']
    cands = BUDGETS['16']
    res = {}
    for i, (sname, change) in enumerate(v3['shifts'].items()):
        vs = dict(c1, **change)
        best_shift = choose(vs, cands, 'v_Delta')
        chosen_counts, pred_err, pred_phi, real_phi, obs_ch, obs_best_minus_ch = {}, [], [], [], [], []
        for r in range(R):
            ch = choose(cal_cache[('C1', r)]['reml'], cands, 'v_Delta')
            chosen_counts[dkey(ch)] = chosen_counts.get(dkey(ch), 0) + 1
            p = L.d_study(cal_cache[('C1', r)]['reml'], *ch)
            pred_err.append(p['v_Delta'])
            pred_phi.append(p['Phi'])
            real_phi.append(L.d_study(vs, *ch)['Phi'])
            rng = rng_for(r, 31 + i, 1)
            e_ch = eval_errors(rng, vs, ch)[0]
            e_best = eval_errors(rng, vs, best_shift)[0]
            obs_ch.append(e_ch)
            obs_best_minus_ch.append(e_best - e_ch)
        res[sname] = {'shifted_components': vs, 'chosen_design_counts': chosen_counts,
                      'best_design_under_shift': dkey(best_shift),
                      'analytic_err_by_design_under_shift': {dkey(dd): L.d_study(vs, *dd)['v_Delta'] for dd in cands},
                      'analytic_Phi_by_design_under_shift': {dkey(dd): L.d_study(vs, *dd)['Phi'] for dd in cands},
                      'calibration_predicted_err': tci(pred_err), 'calibration_predicted_Phi': tci(pred_phi),
                      'realized_Phi_of_chosen': tci(real_phi), 'observed_err_chosen': tci(obs_ch),
                      'paired_obs_err_best_minus_chosen': tci(obs_best_minus_ch)}
    out['V3_population_shift'] = res
    s1, s2 = res['S1_restricted_object_range'], res['S2_occasion_instability']
    s1_err_ok = abs(s1['observed_err_chosen']['mean'] - s1['calibration_predicted_err']['mean']) < 0.05
    s1_phi_drop = s1['realized_Phi_of_chosen']['mean'] < s1['calibration_predicted_Phi']['mean'] - 0.2
    s2_mostly_wrong = s2['chosen_design_counts'].get(s2['best_design_under_shift'], 0) < R / 2
    contract('B.V3', s1_err_ok and s1_phi_drop and s2_mostly_wrong,
             {'S1_pred_Phi': s1['calibration_predicted_Phi']['mean'], 'S1_real_Phi': s1['realized_Phi_of_chosen']['mean'],
              'S2_best': s2['best_design_under_shift'], 'S2_chosen': s2['chosen_design_counts']})

    # V4 dependent occasions in calibration
    v4 = V['V4_dependent_occasions_in_calibration']
    c2 = CFG['configs']['C2']
    d = tuple(v4['design'])
    pred_phi, pred_phi_indep, obs = [], [], []
    for r in range(R):
        Xc, _ = gen(rng_for(r, 41, 0), c2, CAL['n_u'], CAL['n_a'], CAL['n_b'], ub_dep_rho=v4['rho'])
        an = L.anova3(Xc)
        rm = L.reml3(an['ss'], an['df'], CAL['n_u'], CAL['n_a'], CAL['n_b'])
        pred_phi.append(L.d_study(rm['est'], *d)['Phi'])
        pred_phi_indep.append(L.d_study(cal_cache[('C2', r)]['reml'], *d)['Phi'])
        obs.append(eval_errors(rng_for(r, 41, 1), c2, d)[0])
    real = L.d_study(c2, *d)
    out['V4_dependent_occasions_in_calibration'] = {
        'rho': v4['rho'], 'design': dkey(d),
        'predicted_Phi_dependent_calibration': tci(pred_phi), 'predicted_Phi_independent_calibration': tci(pred_phi_indep),
        'realized_Phi': real['Phi'], 'realized_err_population': real['v_Delta'], 'observed_deployment_err': tci(obs),
        'paired_pred_Phi_dependent_minus_realized': tci([p - real['Phi'] for p in pred_phi])}
    v4o = out['V4_dependent_occasions_in_calibration']
    contract('B.V4', v4o['paired_pred_Phi_dependent_minus_realized']['ci95'][0] > 0,
             {'pred_Phi': v4o['predicted_Phi_dependent_calibration']['mean'], 'realized_Phi': real['Phi']})
    return out


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
