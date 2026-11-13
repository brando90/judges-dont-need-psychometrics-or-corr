#!/usr/bin/env python3
"""Frozen, finite-population conditional-distribution identification study."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

# Set before numpy/scipy import; an analysis uses at most eight native threads.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '8'
import numpy as np
import scipy
from scipy.stats import kendalltau, rankdata, t as student_t

ROOT = Path(__file__).resolve().parent
SCORES = np.arange(6, dtype=float)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def population(t):
    t = np.asarray(t)
    return np.stack([math.comb(5, r) * t**r * (1-t)**(5-r) for r in range(6)], axis=-1)


def items(config):
    rows = []
    for split, families in config['family_splits'].items():
        for f in families:
            for j in range(config['pairs_per_family']):
                d = (f + (j+0.5)/config['pairs_per_family']) / 30
                for sign in (-1, 1):
                    rows.append({'id': f'f{f:02d}_j{j:02d}_s{sign:+d}', 'split': split,
                                 'family': f, 't': 0.5+sign*d})
    return rows


def candidates(t, train_ratings, config):
    """Only constant_train accesses train ratings; none accesses val/test ratings."""
    t = np.asarray(t)
    p = population(t)
    endpoint = np.zeros_like(p)
    endpoint[:, 0], endpoint[:, 5] = 1-t, t
    q = [(1-lam)*p + lam*endpoint for lam in config['lambdas']]
    # Analytic shared means prevent floating-point residue from creating rank ties.
    means = [5*t for _ in config['lambdas']]
    q += [population(1-t), np.tile(population(np.array([0.5])), (len(t), 1)),
          np.full_like(p, 1/6)]
    means += [5*(1-t), np.full(len(t), 2.5), np.full(len(t), 2.5)]
    counts = np.bincount(train_ratings.ravel(), minlength=6).astype(float)
    counts += config['train_constant_laplace_alpha']
    constant = counts/counts.sum()
    q.append(np.tile(constant, (len(t), 1)))
    means.append(np.full(len(t), constant @ SCORES))
    return np.stack(q), np.stack(means)


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x-x.mean(), y-y.mean()
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/denominator) if denominator > 1e-14 else None


def associations(mu, target):
    if np.ptp(mu) < 1e-14 or np.ptp(target) < 1e-14:
        return {'pearson': None, 'spearman': None, 'kendall': None}
    return {'pearson': corr(mu, target), 'spearman': corr(rankdata(mu), rankdata(target)),
            'kendall': float(kendalltau(mu, target).statistic)}


def empirical_metrics(q, mu, ratings):
    """Average individual-rating proper losses; no plug-in population histogram."""
    n, m = ratings.shape
    counts = np.eye(6)[ratings].mean(axis=1)
    cdf_q, cdf_y = np.cumsum(q, axis=1)[:, :-1], np.cumsum(counts, axis=1)[:, :-1]
    # E[(Q - 1{R<=k})²], not (Q - empirical_CDF)².
    out = {'ce': float(-np.sum(counts*np.log(q))/n),
           'brier': float(np.mean(np.sum(q*q - 2*q*counts, axis=1)+1)),
           'rps': float(np.mean(cdf_q*cdf_q - 2*cdf_q*cdf_y+cdf_y)),
           'nmse': float(np.mean(((mu[:, None]-ratings)/5)**2)),
           'mae': float(np.mean(np.abs(mu[:, None]-ratings)/5)),
           'bias': float((mu.mean()-ratings.mean())/5),
           'judge_mean': float(mu.mean()/5), 'human_mean': float(ratings.mean()/5)}
    out.update(associations(mu, ratings.mean(axis=1)))
    return out


def expected_metrics(p, q, mu):
    """Exact equal-item expectation over future individual ratings."""
    p_cdf, q_cdf = np.cumsum(p, axis=1)[:, :-1], np.cumsum(q, axis=1)[:, :-1]
    target = p @ SCORES
    ce = float(-np.mean(np.sum(p*np.log(q), axis=1)))
    out = {'ce': ce, 'kl': float(np.mean(np.sum(p*np.log(p/q), axis=1))),
           'brier': float(np.mean(np.sum(q*q-2*q*p, axis=1)+1)),
           'rps': float(np.mean(q_cdf*q_cdf-2*q_cdf*p_cdf+p_cdf)),
           'nmse': float(np.mean(np.sum(p*((mu[:, None]-SCORES)/5)**2, axis=1))),
           'mae': float(np.mean(np.sum(p*np.abs(mu[:, None]-SCORES)/5, axis=1))),
           'mean_mae': float(np.mean(np.abs(mu-target)/5)),
           'mean_mse': float(np.mean(((mu-target)/5)**2)),
           'bias': float(np.mean(mu-target)/5),
           'judge_mean': float(mu.mean()/5), 'human_mean': float(target.mean()/5)}
    # Diagnostics of conditional means, not rank correlations of random ratings.
    out.update({k+'_conditional_mean': v for k, v in associations(mu, np.round(target, 12)).items()})
    variance_r = float(np.mean(p @ (SCORES**2)) - target.mean()**2)
    out['pearson_individual_rating'] = (float(np.mean((mu-mu.mean())*(target-target.mean())) /
                                       math.sqrt(np.var(mu)*variance_r)) if np.ptp(mu)>1e-14 else None)
    return out


def select(metric_rows, objective, tolerance=1e-12):
    values = np.array([row[objective] if row[objective] is not None else np.nan for row in metric_rows])
    if objective in ('pearson', 'spearman', 'kendall'):
        values = -values
    valid = np.isfinite(values)
    if not valid.any():
        return []
    best = values[valid].min()
    return np.flatnonzero(valid & (np.abs(values-best) <= tolerance)).tolist()


def summary(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return {'n_defined': 0, 'mean': None, 'sd': None, 'mc_ci95': None, 'repeat_q95': None}
    sd = float(values.std(ddof=1)) if len(values)>1 else 0.0
    delta = float(student_t.ppf(.975, len(values)-1)*sd/math.sqrt(len(values))) if len(values)>1 else 0.0
    mean = float(values.mean())
    return {'n_defined': len(values), 'mean': mean, 'sd': sd,
            'mc_ci95': [mean-delta, mean+delta], 'repeat_q95': np.quantile(values,[.025,.975]).tolist()}


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze():
    config = json.loads((ROOT/'protocol.json').read_text())
    write_json(ROOT/'input_manifest.json', {'items': items(config), 'configuration': config})
    names = ['protocol.json', 'PROTOCOL.md', 'input_manifest.json', 'run_study.py', 'test_study.py']
    output = ROOT/'FREEZE.json'
    if output.exists():
        raise RuntimeError('Refusing to replace an existing freeze receipt')
    write_json(output, {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                        'hashes': {name: sha(ROOT/name) for name in names},
                        'status': 'fixed before synthetic ratings or metrics are generated'})
    print('Freeze SHA-256', sha(output))


def check_freeze():
    receipt = json.loads((ROOT/'FREEZE.json').read_text())
    for name, digest in receipt['hashes'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('Frozen input changed: '+name)


def run(out):
    check_freeze()
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists() or (out/'selection_rows.csv').exists():
        raise RuntimeError('Refusing to overwrite completed evidence; use --out for deterministic verification')
    config = json.loads((ROOT/'protocol.json').read_text())
    manifest = json.loads((ROOT/'input_manifest.json').read_text())
    grids = {s: np.array([r['t'] for r in manifest['items'] if r['split']==s]) for s in config['family_splits']}
    names = config['candidates']
    all_selected, all_candidates, trace = [], [], []
    summaries = {}
    dummy_train = np.array([[0,1,2,3,4,5]])
    q, means = candidates(grids['test'], dummy_train, config)
    population_rows = []
    for i,name in enumerate(names[:-1]):
        population_rows.append({'candidate': name, **expected_metrics(population(grids['test']),q[i],means[i])})
    csv_write(out/'population_metrics.csv', population_rows)
    reverse = population(1-grids['test'])
    p = population(grids['test'])
    write_json(out/'analytic_controls.json', {
        'permutation_marginal_max_absolute_difference': float(np.max(abs(p.mean(axis=0)-reverse.mean(axis=0)))),
        'shared_mean_max_absolute_difference': float(np.max(abs(q[:5]@SCORES - 5*grids['test']))),
        'oracle_kl': population_rows[0]['kl'],
        'permutation_conditional_kl': population_rows[5]['kl'],
        'oracle_mean_mse': population_rows[0]['mean_mse'],
        'population_expected_identity': 'CE(q)-CE(p)=KL(p||q); Brier excess=mean sum(q-p)^2; RPS excess=mean(CDF(q)-CDF(p))^2'
    })
    for rep in range(config['replications']):
        if time.monotonic()-started > config['analysis_wall_seconds_max']:
            raise TimeoutError('Fixed analysis budget exhausted')
        ratings = {}
        for split_no, (split, grid) in enumerate(grids.items()):
            rng = np.random.default_rng(np.random.SeedSequence([config['seed'],rep,split_no]))
            ratings[split] = rng.binomial(5, grid[:,None], size=(len(grid),3))
        for m in config['ratings_per_item']:
            val_q, val_mu = candidates(grids['validation'],ratings['train'][:,:m],config)
            val_rows = [empirical_metrics(qq,mm,ratings['validation'][:,:m]) for qq,mm in zip(val_q,val_mu)]
            selections = {objective: select(val_rows,objective) for objective in config['selection_objectives']}
            # Test ratings are not passed to any fitting or selection function.
            test_q, test_mu = candidates(grids['test'],ratings['train'][:,:m],config)
            test_expected = [expected_metrics(p,qq,mm) for qq,mm in zip(test_q,test_mu)]
            test_observed = [empirical_metrics(qq,mm,ratings['test'][:,:m]) for qq,mm in zip(test_q,test_mu)]
            for i,name in enumerate(names):
                all_candidates.append({'replication': rep,'ratings_per_item':m,'candidate':name,
                    **{'validation_'+k:v for k,v in val_rows[i].items()},
                    **{'test_expected_'+k:v for k,v in test_expected[i].items()},
                    **{'test_sample_'+k:v for k,v in test_observed[i].items()}})
            for objective, chosen in selections.items():
                row = {'replication':rep, 'ratings_per_item':m, 'objective':objective,
                       'status':'complete' if chosen else 'undefined_selection',
                       'tie_size':len(chosen), 'selected_candidates':'|'.join(names[i] for i in chosen),
                       'oracle_in_tie':int(0 in chosen), 'oracle_uniform_tie_probability':1/len(chosen) if 0 in chosen else 0.0}
                for label, metric_rows in [('expected',test_expected),('sample',test_observed)]:
                    for metric in metric_rows[0]:
                        vals = [metric_rows[i][metric] for i in chosen]
                        defined = [v for v in vals if v is not None]
                        for stat, fun in [('mean',np.mean),('min',np.min),('max',np.max)]:
                            row[f'test_{label}_{metric}_tie_{stat}'] = float(fun(defined)) if len(defined)==len(vals) and defined else None
                all_selected.append(row)
            if rep==0:
                # Synthetic trace is public and contains no human data.
                for split, grid in grids.items():
                    for j in range(min(3,len(grid))):
                        trace.append({'split':split,'m':m,'t':float(grid[j]),'ratings':ratings[split][j,:m].tolist(),
                                      'p':population(grid[j:j+1])[0].tolist()})
        if rep%64==0:
            print(f'Completed {rep+1}/{config["replications"]} independent rating replications',flush=True)
    for m in config['ratings_per_item']:
        summaries[str(m)] = {}
        for objective in config['selection_objectives']:
            rows=[r for r in all_selected if r['ratings_per_item']==m and r['objective']==objective]
            keys=[k for k in rows[0] if k.startswith('test_')]+['oracle_uniform_tie_probability','tie_size']
            summaries[str(m)][objective] = {'replications':len(rows),'failed':sum(r['status']!='complete' for r in rows),
                'oracle_in_tie_count':sum(r['oracle_in_tie'] for r in rows),
                'metrics':{k:summary([r[k] for r in rows]) for k in keys},
                'selected_set_counts':{key:sum(r['selected_candidates']==key for r in rows) for key in sorted({r['selected_candidates'] for r in rows})}}
    csv_write(out/'candidate_rows.csv',all_candidates)
    csv_write(out/'selection_rows.csv',all_selected)
    write_json(out/'summary.json',summaries)
    write_json(out/'synthetic_trace.json',trace)
    receipt = {'status':'complete','replications':config['replications'], 'simulation_conditions':config['replications']*3,
               'selection_rows':len(all_selected),'candidate_rows':len(all_candidates),
               'failures':sum(r['status']!='complete' for r in all_selected),'provider_calls':0,'provider_cost_usd':0,
               'freeze_sha256':sha(ROOT/'FREEZE.json'),
               'wall_seconds':time.monotonic()-started,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
               'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               'artifact_hashes':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out/'COMPLETE.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    if args.freeze:
        freeze()
    else:
        run(args.out)
