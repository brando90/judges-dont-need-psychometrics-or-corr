"""Experiment 16: RPS-selected arm and probability-floor ablation on the historical scalar-vote pools (experiments 12 and 14 protocol).
TLDR: same 1,000 task-grouped 20/5 splits, same cached judges, same human ratings as experiments 12 and 14; adds (i) judges picked or mixed by
train RPS, (ii) a Spearman-picked single judge, and (iii) a sweep of the probability floor epsilon that smooths each judge's 3-sample histogram.
CPU only, zero model calls, $0.
Usage: python3 run_rps_floor.py --data dataset_pool.json --pool all|open|old5 --eps 1e-6 --out results/all_eps1e-06 [--splits 1000 --boot 10000 --workers 24]"""
import argparse, json, time, platform, sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, scipy
from scipy.optimize import minimize
from scipy.stats import rankdata
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE/'exp10_scripts')); sys.path.insert(0, str(HERE/'vendor_exp03')); sys.path.insert(0, str(HERE))
from models import histogram, smooth            # noqa
from common import GRID, split_tasks, REPEAT_SEED0, FIXED_SEED, ce, rps, tvd, pearson, spearman, score   # noqa
from ce_mixture import fit                      # noqa
from nnls_pearson import fit_pearson            # noqa

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--pool', choices=['all', 'open', 'old5'], required=True)
ap.add_argument('--eps', type=float, default=1e-6); ap.add_argument('--out', required=True)
ap.add_argument('--splits', type=int, default=1000); ap.add_argument('--boot', type=int, default=10000)
ap.add_argument('--spearman_draws', type=int, default=4000); ap.add_argument('--workers', type=int, default=8)
a = ap.parse_args(); t0 = time.time()

d = json.load(open(a.data)); rows = d['rows']; order_all = d['candidate_order']
OLD5 = ['H0_sol', 'H0_opus', 'H1_sol', 'H2_sol', 'H2_opus']
order = {'all': order_all, 'old5': OLD5, 'open': [n for n in order_all if n not in OLD5]}[a.pool]
p = np.array([histogram(r['ratings']) for r in rows])
q = np.stack([smooth(np.array([histogram(x) for x in d['outputs'][n]]), a.eps) for n in order], axis=1)
tasks = np.array([r['task'] for r in rows]); ut = sorted(set(tasks)); T = len(ut); tidx = np.array([ut.index(t) for t in tasks])
N, J = len(rows), len(order); mu_q = q @ GRID
METH = ['constant', 'equal_mixture', 'corr_single', 'corr_single_spearman', 'corr_mixture_pearson', 'corr_mixture_spearman',
        'ce_single', 'ce_mixture', 'rps_single', 'rps_mixture']
LABEL = {'constant': 'constant (train marginal)', 'equal_mixture': 'equal-weight mixture', 'corr_single': '② best single judge by train Pearson',
         'corr_single_spearman': '② best single judge by train Spearman', 'corr_mixture_pearson': '② mixture maximizing train Pearson',
         'corr_mixture_spearman': '② mixture maximizing train Spearman', 'ce_single': '④ best single judge by train CE',
         'ce_mixture': '④ mixture minimizing train CE', 'rps_single': '④ best single judge by train RPS (new)',
         'rps_mixture': '④ mixture minimizing train RPS (new)'}
FW_TOL = 1e-8

def fit_rps(qt, pt):
    """Convex QP over the simplex: minimize mean_{i,k} (sum_j w_j Qc_ijk - Pc_ik)^2; train RPS differs from it by a constant. SLSQP, certificate = Frank-Wolfe gap."""
    n, m, _ = qt.shape
    A = np.cumsum(qt, -1)[..., :5].transpose(0, 2, 1).reshape(n*5, m); b = np.cumsum(pt, -1)[:, :5].reshape(n*5); sc = 5.0*n
    f = lambda w: float(((A@w-b)**2).sum()/sc); g = lambda w: 2*A.T@(A@w-b)/sc
    w0 = np.full(m, 1/m)
    opt = minimize(f, w0, jac=g, method='SLSQP', bounds=[(0., 1.)]*m, constraints={'type': 'eq', 'fun': lambda w: w.sum()-1, 'jac': lambda w: np.ones(m)},
                   options={'maxiter': 1000, 'ftol': 1e-15})
    w = np.maximum(opt.x, 0); w /= w.sum(); gr = g(w); gap = float(gr@w-gr.min())
    return w, {'status': 'optimal' if gap <= FW_TOL else 'certificate_failed', 'fw_gap': gap}

def fit_all(tr, rng):
    pt, qt = p[tr], q[tr]; hm = pt@GRID; W = {}; info = {}
    W['constant'] = None; W['equal_mixture'] = np.full(J, 1/J)
    pe = np.nan_to_num(np.array([pearson(mu_q[tr, j], hm) for j in range(J)]), nan=-2)
    W['corr_single'] = np.eye(J)[int(np.argmax(pe))]; info['corr_single'] = int(np.argmax(pe))
    se = np.nan_to_num(np.array([spearman(mu_q[tr, j], hm) for j in range(J)]), nan=-2)
    W['corr_single_spearman'] = np.eye(J)[int(np.argmax(se))]; info['corr_single_spearman'] = int(np.argmax(se))
    w, rec = fit_pearson(qt, pt); W['corr_mixture_pearson'] = w; info['pm_ok'] = rec['status'] == 'optimal_cone_projection'
    cand = np.vstack([np.eye(J), np.full((1, J), 1/J), rng.dirichlet(np.ones(J), size=a.spearman_draws)])
    mu = mu_q[tr]@cand.T; rk = rankdata(mu, axis=0); rt = rankdata(hm); rk -= rk.mean(0); rtc = rt-rt.mean()
    den = np.linalg.norm(rk, axis=0)*np.linalg.norm(rtc)
    sp = np.where(den > 1e-12, (rk*rtc[:, None]).sum(0)/np.where(den > 1e-12, den, 1), -2); W['corr_mixture_spearman'] = cand[int(np.argmax(sp))]
    cs = [ce(pt, qt[:, j]).mean() for j in range(J)]; W['ce_single'] = np.eye(J)[int(np.argmin(cs))]; info['ce_single'] = int(np.argmin(cs))
    w, rec = fit(qt, pt, 'ce'); W['ce_mixture'] = w; info['cm_ok'] = rec['status'] == 'optimal'
    rs = [rps(pt, qt[:, j]).mean() for j in range(J)]; W['rps_single'] = np.eye(J)[int(np.argmin(rs))]; info['rps_single'] = int(np.argmin(rs))
    w, rec = fit_rps(qt, pt); W['rps_mixture'] = w; info['rm_ok'] = rec['status'] == 'optimal'
    return W, smooth(pt.mean(0), a.eps), info

def predict(W, marg, te):
    return {m: (np.tile(marg, (len(te), 1)) if W[m] is None else np.einsum('njr,j->nr', q[te], W[m])) for m in METH}

def one_split(s):
    tr, te = split_tasks(tasks, REPEAT_SEED0+s); rng = np.random.default_rng(REPEAT_SEED0+s)
    W, marg, info = fit_all(tr, rng); Q = predict(W, marg, te); out = {'s': s, 'te': te, 'info': info, 'sc': {}, 'w': {m: (None if W[m] is None else W[m]) for m in METH}}
    for m in METH:
        sc = score(p[te], Q[m]); out['sc'][m] = {k: (np.asarray(v) if k in ('ce', 'rps', 'tvd') else float(v)) for k, v in sc.items()}
    return out

if __name__ == '__main__':
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True); log = open(out_dir/'run.log', 'a')
    def L(*x):
        print(*x, file=log, flush=True); print(*x, flush=True)
    L(f'START {time.strftime("%Y-%m-%d %H:%M:%S")} pool={a.pool} J={J} eps={a.eps} splits={a.splits} boot={a.boot} workers={a.workers} N={N} T={T}')
    SUM = {mt: {m: np.zeros(N) for m in METH} for mt in ['ce', 'rps', 'tvd']}; CNT = np.zeros(N)
    CSUM = {mt: {m: np.zeros(T) for m in METH} for mt in ['pearson', 'spearman']}; CCNT = {mt: {m: np.zeros(T) for m in METH} for mt in ['pearson', 'spearman']}
    CSPLIT = {mt: {m: [] for m in METH} for mt in ['pearson', 'spearman']}; wts = {m: [] for m in METH}
    fails = {'pm': 0, 'cm': 0, 'rm': 0}; picks = {k: np.zeros(J, int) for k in ['corr_single', 'corr_single_spearman', 'ce_single', 'rps_single']}
    done = 0
    with Pool(a.workers) as pool:
        for r in pool.imap(one_split, range(a.splits), chunksize=4):
            te = r['te']; CNT[te] += 1; tt = np.unique(tidx[te]); done += 1
            fails['pm'] += int(not r['info']['pm_ok']); fails['cm'] += int(not r['info']['cm_ok']); fails['rm'] += int(not r['info']['rm_ok'])
            for k in picks: picks[k][r['info'][k]] += 1
            for m in METH:
                sc = r['sc'][m]; wts[m].append(r['w'][m])
                for mt in SUM: SUM[mt][m][te] += sc[mt]
                for mt in CSUM:
                    CSPLIT[mt][m].append(sc[mt])
                    if np.isfinite(sc[mt]): CSUM[mt][m][tt] += sc[mt]; CCNT[mt][m][tt] += 1
            if done % 100 == 0: L(f'  {done}/{a.splits} splits, {time.time()-t0:.0f}s')
    assert (CNT > 0).all(), 'every artifact must be tested at least once'
    per = {mt: {m: SUM[mt][m]/CNT for m in METH} for mt in SUM}
    rng = np.random.default_rng(1210); draws = rng.integers(0, T, size=(a.boot, T))
    tsum = lambda x: (np.bincount(tidx, weights=x, minlength=T), np.bincount(tidx, minlength=T))
    def boot_paired(dt, nt):
        point = float(dt.sum()/nt.sum()); bs = dt[draws].sum(1)/nt[draws].sum(1); lo, hi = np.percentile(bs, [2.5, 97.5])
        return {'point': point, 'ci95': [float(lo), float(hi)], 'excludes_zero': bool(lo > 0 or hi < 0)}
    PAIRS = [('corr_single', 'rps_single'), ('corr_single_spearman', 'rps_single'), ('corr_mixture_pearson', 'rps_mixture'), ('corr_mixture_spearman', 'rps_mixture'),
             ('corr_single', 'ce_single'), ('corr_mixture_pearson', 'ce_mixture'), ('corr_mixture_spearman', 'ce_mixture'),
             ('ce_single', 'rps_single'), ('ce_mixture', 'rps_mixture'), ('rps_single', 'constant'), ('rps_mixture', 'constant'),
             ('corr_single', 'constant'), ('ce_single', 'constant'), ('ce_mixture', 'constant'), ('rps_mixture', 'equal_mixture')]
    res = {'experiment': 16, 'pool': a.pool, 'judges': order, 'n_judges': J, 'epsilon': a.eps, 'n_artifacts': N, 'n_tasks': T, 'splits': a.splits, 'train_tasks': 20, 'test_tasks': 5,
           'bootstrap_draws': a.boot, 'spearman_search_draws': a.spearman_draws, 'optimizer_certificate_failures': fails, 'inference_calls': 0, 'provider_cost_usd': 0,
           'test_count_per_artifact_min_max': [int(CNT.min()), int(CNT.max())], 'metrics': {}, 'corr_metrics': {}, 'paired': {}, 'mean_weights': {},
           'single_judge_choice_counts': {k: {order[i]: int(v[i]) for i in range(J) if v[i]} for k, v in picks.items()}, 'labels': LABEL}
    for mt in SUM:
        res['metrics'][mt] = {m: float(per[mt][m].mean()) for m in METH}
        for x, y in PAIRS:
            st, nt = tsum(per[mt][x]-per[mt][y]); r_ = boot_paired(st, nt); r_['tasks_first_better'] = int((st < 0).sum()); res['paired'][f'{mt}: {x} minus {y}'] = r_
    for mt in CSUM:
        tm = {m: np.where(CCNT[mt][m] > 0, CSUM[mt][m]/np.maximum(CCNT[mt][m], 1), np.nan) for m in METH}
        res['corr_metrics'][mt] = {m: {'mean_over_splits': float(np.nanmean(CSPLIT[mt][m])) if np.isfinite(CSPLIT[mt][m]).any() else None,
                                       'undefined_splits': int(np.isnan(CSPLIT[mt][m]).sum())} for m in METH}
        for x, y in PAIRS:
            ok = np.isfinite(tm[x]) & np.isfinite(tm[y]); dt = np.where(ok, tm[x]-tm[y], 0.0); n = ok.astype(float)
            bs = dt[draws].sum(1)/np.maximum(n[draws].sum(1), 1); lo, hi = np.percentile(bs, [2.5, 97.5])
            res['paired'][f'{mt} (test-set correlation, task-attributed): {x} minus {y}'] = {'point': float(dt.sum()/n.sum()), 'ci95': [float(lo), float(hi)], 'excludes_zero': bool(lo > 0 or hi < 0)}
    for m in METH:
        if wts[m][0] is not None: res['mean_weights'][m] = {order[i]: float(v) for i, v in enumerate(np.mean(wts[m], 0)) if v > 0.005}
    res['seconds'] = time.time()-t0; res['environment'] = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}
    json.dump(res, open(out_dir/'results.json', 'w'), indent=1)
    json.dump({'per_artifact': {mt: {m: per[mt][m].tolist() for m in METH} for mt in per}}, open(out_dir/'per_artifact.json', 'w'))
    L(f'DONE {time.strftime("%Y-%m-%d %H:%M:%S")} {res["seconds"]:.0f}s failures={fails}')
    open(out_dir/'DONE', 'w').write('done\n')
