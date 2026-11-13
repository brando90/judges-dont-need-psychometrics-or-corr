#!/usr/bin/env python3
"""TLDR: Phase C of experiment 04 — retrospective reliability versus accuracy on existing ratings.

Reads the private, git-ignored historical dataset already imported by experiment 02 (75 artifacts,
25 tasks, 2-3 expert ratings each; five cached scalar judges x 3 repeat calls). No new model calls,
no new ratings. Publishes aggregates only.

For the human experts: inter-rater G study (artifacts x raters). For each judge: test-retest G study
(artifacts x repeat calls) and external agreement with the expert ratings (RPS primary, CE, TVD, MAE,
bias, correlations, ICC). Question: does the judges' internal reliability order match their
accuracy order?

Run: python experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/run_historical.py
Writes results/phase_c.json.
"""
from __future__ import annotations

import hashlib
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
REPO = HERE.parent.parent
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
CFG = PROTOCOL['phase_c']
SRC = REPO / 'experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/private/historical_exp06/cached_v1/dataset.json'
EPS = 1e-6
JUDGES = ['H0_sol', 'H1_sol', 'H2_sol', 'H0_opus', 'H2_opus']
CONTRACT: list[dict] = []


def contract(cid, ok, observed):
    CONTRACT.append({'id': cid, 'pass': bool(ok), 'observed': observed})


def hist(scores):
    q = np.zeros(6)
    for s in scores:
        q[int(s)] += 1
    return q / q.sum()


def per_artifact_metrics(q, ratings):
    """RPS and CE averaged over the artifact's expert ratings; empirical TVD against their histogram."""
    rps = float(np.mean([L.rps_individual(list(q), y) for y in ratings]))
    ce = float(np.mean([L.ce_individual(list(q), y) for y in ratings]))
    tvd = 0.5 * float(np.abs(np.asarray(q) - hist(ratings)).sum())
    return rps, ce, tvd


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None, None, None
    return float(stats.pearsonr(a, b)[0]), float(stats.spearmanr(a, b)[0]), float(stats.kendalltau(a, b)[0])


def build(rows, outputs, train_marginal):
    """Precompute everything per artifact so a bootstrap only reindexes."""
    art = {'task': [r['task'] for r in rows],
           'human_p12': np.array([[r['ratings_by_expert'][0], r['ratings_by_expert'][1]] for r in rows], float),
           'expert_mean': np.array([np.mean(r['ratings']) for r in rows]),
           'n_ratings': np.array([len(r['ratings']) for r in rows]),
           'flagged': np.array([bool(r['ambiguous']) or not bool(r['ordinary']) for r in rows])}
    q0 = L.smooth(list(train_marginal), EPS)
    art['baseline'] = np.array([per_artifact_metrics(q0, r['ratings']) for r in rows])
    for j in JUDGES:
        sc = np.array(outputs[j], float)
        art[f'{j}_calls'] = sc
        art[f'{j}_mean'] = sc.mean(1)
        art[f'{j}_ext'] = np.array([per_artifact_metrics(L.smooth(list(hist(s)), EPS), r['ratings'])
                                    for s, r in zip(outputs[j], rows)])
    return art


def coefs(X, n):
    est = L.anova2(X)['est']
    c = L.one_facet_coeffs(est, n)
    return est, c


def metrics(art, ix):
    """All headline statistics on artifacts ix (indices may repeat in a bootstrap)."""
    out = {}
    est, c1 = coefs(art['human_p12'][ix], 1)
    _, c2 = coefs(art['human_p12'][ix], 2)
    out['human'] = {'v_u': est['u'], 'v_rater': est['a'], 'v_e': est['e'], 'Erho2_1': c1['Erho2'], 'Phi_1': c1['Phi'],
                    'Erho2_2': c2['Erho2'], 'Phi_2': c2['Phi'], 'mean': float(art['human_p12'][ix].mean())}
    b = art['baseline'][ix].mean(0)
    out['baseline_train_marginal'] = {'RPS': float(b[0]), 'CE': float(b[1]), 'TVD': float(b[2])}
    hm = art['expert_mean'][ix]
    for j in JUDGES:
        X = art[f'{j}_calls'][ix]
        est, c1 = coefs(X, 1)
        _, c3 = coefs(X, 3)
        jm = art[f'{j}_mean'][ix]
        ext = art[f'{j}_ext'][ix].mean(0)
        pr, sp, kt = corr(jm, hm)
        icc = L.icc_two_way(np.column_stack([jm, hm]))
        out[j] = {'v_u': est['u'], 'v_call': est['a'], 'v_e': est['e'],
                  'Erho2_1': c1['Erho2'], 'Phi_1': c1['Phi'], 'Erho2_3': c3['Erho2'], 'Phi_3': c3['Phi'],
                  'repeat_sd': float(np.sqrt(np.mean(X.var(1, ddof=1)))),
                  'RPS': float(ext[0]), 'CE': float(ext[1]), 'TVD': float(ext[2]),
                  'MAE': float(np.mean(np.abs(jm - hm))), 'bias': float(np.mean(jm - hm)),
                  'judge_mean': float(jm.mean()), 'expert_mean': float(hm.mean()),
                  'pearson': pr, 'spearman': sp, 'kendall': kt, 'ICC_A1': icc['ICC_A1'], 'ICC_C1': icc['ICC_C1']}
    return out


def bootstrap(art, ix_set, draws, seed):
    tasks = sorted({art['task'][i] for i in ix_set})
    by_task = {t: [i for i in ix_set if art['task'][i] == t] for t in tasks}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        pick = rng.integers(0, len(tasks), len(tasks))
        ix = [i for k in pick for i in by_task[tasks[k]]]
        samples.append(metrics(art, ix))
    return samples


def pct(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return {'ci95': None, 'n_defined': 0}
    return {'ci95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], 'n_defined': int(len(v))}


def attach_ci(point, samples):
    res = {}
    for grp, d in point.items():
        res[grp] = {}
        for k, x in d.items():
            res[grp][k] = {'point': x, **pct([s[grp][k] for s in samples])}
    return res


def main():
    t0 = time.time()
    raw = SRC.read_bytes()
    data = json.loads(raw)
    assert data['source_commit'] == CFG['upstream_source_commit'].split()[0]   # amendment 2: hash token only
    rows = data['rows']
    assert len(rows) == 75 and all(len(data['outputs'][j]) == 75 for j in JUDGES)
    assert all(r['ratings_by_expert'][0] is not None and r['ratings_by_expert'][1] is not None for r in rows)
    train = [y for r in rows if r['split'] == 'train' for y in r['ratings']]
    train_marginal = hist(train)
    art = build(rows, data['outputs'], train_marginal)
    sets = {'test': [i for i, r in enumerate(rows) if r['split'] == 'test'], 'all75': list(range(75))}
    bc = CFG['bootstrap']
    out = {'experiment': PROTOCOL['experiment'], 'phase': 'C', 'protocol_version': PROTOCOL['protocol_version'],
           'software': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
           'command': 'python experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/run_historical.py',
           'source_sha256': hashlib.sha256(raw).hexdigest(), 'upstream_source_commit': data['source_commit'],
           'judges': CFG['judges'], 'probability_floor_epsilon': EPS, 'new_model_calls': 0, 'sets': {}}
    for name, ix in sets.items():
        point = metrics(art, ix)
        samples = bootstrap(art, ix, bc['draws'] if name == 'test' else 2000, bc['seed'])
        res = attach_ci(point, samples)
        rel = {j: point[j]['Erho2_1'] for j in JUDGES}
        rps = {j: point[j]['RPS'] for j in JUDGES}
        most_rel = max(JUDGES, key=lambda j: rel[j])
        best_rps = min(JUDGES, key=lambda j: rps[j])
        tau = float(stats.kendalltau([rel[j] for j in JUDGES], [-rps[j] for j in JUDGES])[0])
        paired = {}
        if most_rel != best_rps:
            for k in ('RPS', 'Erho2_1', 'MAE', 'CE'):
                d = point[most_rel][k] - point[best_rps][k]
                paired[k] = {'point': d, **pct([s[most_rel][k] - s[best_rps][k] for s in samples])}
        judge_minus_human = {j: {'point': point[j]['Erho2_1'] - point['human']['Erho2_1'],
                                 **pct([s[j]['Erho2_1'] - s['human']['Erho2_1'] for s in samples])} for j in JUDGES}
        tasks = sorted({art['task'][i] for i in ix})
        out['sets'][name] = {
            'n_artifacts': len(ix), 'n_tasks': len(tasks), 'n_expert_ratings': int(art['n_ratings'][ix].sum()),
            'n_flagged_artifacts': int(art['flagged'][ix].sum()),
            'bootstrap_draws': len(samples), 'bootstrap_unit': 'task', 'metrics': res,
            'order_by_reliability': sorted(JUDGES, key=lambda j: -rel[j]), 'order_by_rps': sorted(JUDGES, key=lambda j: rps[j]),
            'kendall_tau_reliability_vs_accuracy': tau, 'most_repeatable_judge': most_rel, 'lowest_rps_judge': best_rps,
            'paired_most_repeatable_minus_lowest_rps': paired, 'judge_minus_human_Erho2_1': judge_minus_human,
        }
        if name == 'test':
            keep = [i for i in ix if not art['flagged'][i]]
            sens = metrics(art, keep)
            out['sets'][name]['sensitivity_excluding_flagged'] = {
                'n_artifacts': len(keep),
                'order_by_reliability': sorted(JUDGES, key=lambda j: -sens[j]['Erho2_1']),
                'order_by_rps': sorted(JUDGES, key=lambda j: sens[j]['RPS']),
                'point': sens}
    t = out['sets']['test']
    above = sum(t['metrics'][j]['Erho2_1']['point'] > t['metrics']['human']['Erho2_1']['point'] for j in JUDGES)
    contract('C.reliability_vs_accuracy', above == len(JUDGES),
             {'judges_more_repeatable_than_expert_agreement': f'{above}/5', 'tau': t['kendall_tau_reliability_vs_accuracy']})
    out['contract'] = CONTRACT
    out['runtime_seconds'] = round(time.time() - t0, 2)
    (HERE / 'results' / 'phase_c.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'contract_checks': len(CONTRACT), 'failed': [c['id'] for c in CONTRACT if not c['pass']],
                      'runtime_seconds': out['runtime_seconds']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
