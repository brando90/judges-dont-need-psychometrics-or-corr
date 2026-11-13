"""Reproduce one historical failing fit without reading any held-out outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'historical'))
from fit_heads import objective, GRID
from metrics import histogram, pearson
from run_cached import partitions, dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--historical-results', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    assert not out.exists(), 'Never overwrite a diagnostic attempt.'
    cfg = json.loads((HERE/'historical/protocol.json').read_text())
    raw = Path(args.data).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == cfg['dataset_sha256']
    data = json.loads(raw)
    saved = json.loads(Path(args.historical_results).read_text())
    failures = [f for f in saved['fits'] if f['loss'] == 'pearson' and not f['success']]
    rec = failures[0]
    rows = data['rows']
    x = np.array([np.mean(data['outputs'][name], axis=1)/5 for name in cfg['candidate_order']]).T
    p = np.array([histogram(r['ratings']) for r in rows])
    si = cfg['seeds'].index(rec['seed'])
    outer, inners = partitions(rows, cfg['seeds'])[si]
    fold, inner = rec['outer_fold'], rec['inner_fold']
    train = np.flatnonzero(outer != fold)
    if inner is not None:
        train = train[[inners[fold][rows[i]['family']] != inner for i in train]]
    x, p = x[train], p[train]
    z = np.column_stack([np.ones(len(x)), (x-x.mean(0))/np.maximum(x.std(0), 1e-8)])
    seed = rec['seed'] + 100*fold + (99 if inner is None else inner)
    initial = np.random.default_rng(seed).normal(0, .02, size=(z.shape[1], 6)).ravel()

    def inspect(w, strength):
        value, grad = objective(w, z, p, 'pearson', strength)
        mu = softmax(z@w.reshape(z.shape[1], 6), axis=1)@GRID
        return {'objective': value, 'penalty': float(strength*(w@w)/2),
                'one_minus_stabilized_rho': objective(w,z,p,'pearson',0)[0],
                'true_rho': pearson(mu,p@GRID), 'prediction_sd': float(mu.std()),
                'prediction_mean': float(mu.mean()), 'coefficient_norm': float(np.linalg.norm(w)),
                'gradient_inf': float(np.max(np.abs(grad)))}

    attempts = []
    final = None
    # An iteration increase and penalty removal are diagnostic ablations only.
    for label, strength, maxiter, maxfun in [('exact_original', rec['l2'], 300, 600),
            ('larger_budget_diagnostic', rec['l2'], 3000, 6000),
            ('no_l2_diagnostic', 0., 300, 600)]:
        trace = [inspect(initial, strength)]
        start = time.monotonic()
        opt = minimize(objective, initial.copy(), args=(z,p,'pearson',strength),
                       method='L-BFGS-B', jac=True, callback=lambda w: trace.append(inspect(w,strength)),
                       options={'maxiter':maxiter,'maxfun':maxfun,'ftol':1e-10,'gtol':1e-6})
        attempts.append({'label':label,'strength':strength,'maxiter':maxiter,'maxfun':maxfun,
                         'success':bool(opt.success),'status':int(opt.status),'message':str(opt.message),
                         'nit':int(opt.nit),'nfev':int(opt.nfev),'elapsed_seconds':time.monotonic()-start,
                         'final':inspect(opt.x,strength),'trace':trace})
        if label == 'exact_original': final = opt.x.copy()
    scales = [{'scale':s, **inspect(final*s,rec['l2'])} for s in [10.,3.,1.,.3,.1,.03,.01,.001]]
    # Central directional differences at several step sizes avoid interpreting the
    # default absolute check_grad step as meaningful near constant predictions.
    direction = np.random.default_rng(19).normal(size=len(initial))
    direction /= np.linalg.norm(direction)
    gradient_checks = []
    for scale in [1.,1e-2,1e-4,1e-6,0.]:
        w = initial*scale
        _, g = objective(w,z,p,'pearson',rec['l2'])
        exact = float(g@direction)
        for h in [1e-6,1e-8,1e-10]:
            numeric = (objective(w+h*direction,z,p,'pearson',rec['l2'])[0]-
                       objective(w-h*direction,z,p,'pearson',rec['l2'])[0])/(2*h)
            gradient_checks.append({'scale':scale,'step':h,'analytic':exact,'central':numeric,
                                    'relative_error':abs(exact-numeric)/max(1.,abs(exact),abs(numeric))})
    shifted = final.reshape(z.shape[1],6).copy()
    shifted += np.arange(z.shape[1])[:,None]*.1
    q0 = softmax(z@final.reshape(z.shape[1],6),axis=1)
    q1 = softmax(z@shifted,axis=1)
    counts = {loss: {'fits':sum(f['loss']==loss for f in saved['fits']),
                    'failed':sum(f['loss']==loss and not f['success'] for f in saved['fits'])}
              for loss in ['ce','brier','rps','pearson']}
    dump(out, {'historical_counts':counts,'chosen_historical_receipt':rec,'initialization_seed':seed,
               'training_rows':len(train),'attempts':attempts,'scaling_final_coefficients':scales,
               'directional_gradient_checks':gradient_checks,
               'softmax_row_shift_max_probability_error':float(np.max(np.abs(q0-q1))),
               'dataset_sha256':cfg['dataset_sha256'],
               'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'held_out_outcomes_used':False})
    print(json.dumps({'out':str(out),'exact_original':attempts[0]['final'],
                      'termination':attempts[0]['message']}))


if __name__ == '__main__':
    main()
