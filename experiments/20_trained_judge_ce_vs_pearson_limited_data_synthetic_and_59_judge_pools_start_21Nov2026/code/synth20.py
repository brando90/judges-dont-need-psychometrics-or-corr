"""Experiment 20, part A (11-21-2026): controlled synthetic test, known per-item human rating distribution, CE-trained vs Pearson-trained judge, training-size sweep.
TLDR: items have d=6 features x ~ N(0,I). The true human rating distribution of an item is p(r|x) ∝ exp(alpha*(a.x) u + gam(x) u^2 + dlt*(c.x) u^3), u=(r-2.5)/2.5, r in 0..5.
World 'mean_only': spread and shape are the same for every item (only the mean moves). World 'mean_spread_shape': mean, spread and skew each move with x, along different directions, so items with the
same mean have different spread and shape. Judges: softmax-linear head, see lib20.py (the true p is inside the judge's model family, so differences are estimation, not misspecification).
Train: m items x n_r individual ratings drawn from p. Test: 2000 fresh items scored against the TRUE p. Usage: python3 synth20.py --seeds 200 --out out_synth.json [--procs 16]"""
import argparse, json, time
import numpy as np
from multiprocessing import Pool
from lib20 import *

D = 6; SIZES = [4, 8, 16, 32, 64, 128, 256, 512]; NRS = [1, 3]; LAMS = [0.001, 0.01, 0.1]; ARMS = ['ce', 'corr', 'corr_cal', 'constant']

def world(kind, rng):
    u = lambda: (lambda v: v / np.linalg.norm(v))(rng.normal(size=D))
    a, b, c = u(), u(), u()
    def logits(X):
        mean_t = 1.5 * (X @ a)[:, None] * U[None, :]
        if kind == 'mean_only': return mean_t - 0.8 * U[None, :] ** 2
        gam = (0.8 - 1.4 * (X @ b))[:, None]          # u^2 coefficient: -gam; gam<0 -> bimodal, gam large -> narrow
        return mean_t - gam * U[None, :] ** 2 + 0.8 * (X @ c)[:, None] * U[None, :] ** 3
    return logits

def one(args):
    kind, seed = args; rng = np.random.default_rng(np.random.SeedSequence([20261121, 20, seed, 0 if kind == 'mean_only' else 1]))
    lg = world(kind, rng); Xte = rng.normal(size=(2000, D)); Pte = softmax(lg(Xte)); mp = Pte @ R
    bins = np.digitize(mp, np.quantile(mp, [.2, .4, .6, .8]))
    out = []
    Xall = rng.normal(size=(max(SIZES), D)); Pall = softmax(lg(Xall)); u_draw = rng.random((max(SIZES), 3))
    cum = Pall.cumsum(1)
    for nr in NRS:
        for m in SIZES:
            Xtr = Xall[:m]; C = np.zeros((m, 6))
            for j in range(nr): C[np.arange(m), (u_draw[:m, j][:, None] > cum[:m]).sum(1)] += 1
            for lam in LAMS:
                q = fit_arms(Xtr, C, lam, init_seed=seed)
                for arm in ARMS:
                    r = metrics_known(q[arm](Xte), Pte, bins); r.update(world=kind, seed=seed, n_items=m, ratings_per_item=nr, n_ratings=m * nr, lam=lam, arm=arm); out.append(r)
        # oracle row (true distribution) for reference, once per seed
    r = metrics_known(Pte, Pte, bins); r.update(world=kind, seed=seed, n_items=0, ratings_per_item=0, n_ratings=0, lam=0, arm='oracle'); out.append(r)
    mq, vq, sq = moments(Pte); inb = [float(np.std(vq[bins == k])) for k in range(5)]
    out.append({'world': kind, 'seed': seed, 'arm': 'world_diag', 'mean_sd': float(mq.std()), 'var_sd': float(vq.std()), 'skew_sd': float(sq.std()), 'corr_mean_var': float(np.corrcoef(mq, vq)[0, 1]),
                'within_mean_bin_var_sd': float(np.mean(inb)), 'mean_range': [float(mq.min()), float(mq.max())], 'var_range': [float(vq.min()), float(vq.max())], 'frac_bimodal': float(np.mean([(np.diff(p) > 0).any() and (np.diff(p)[np.argmax(np.diff(p) > 0):] < 0).any() for p in Pte]))})
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=200); ap.add_argument('--out', default='out_synth.json'); ap.add_argument('--procs', type=int, default=8); ap.add_argument('--worlds', default='mean_only,mean_spread_shape')
    a = ap.parse_args(); t0 = time.time(); jobs = [(k, s) for k in a.worlds.split(',') for s in range(a.seeds)]
    with Pool(a.procs) as p: res = [r for rs in p.imap_unordered(one, jobs) for r in rs]
    json.dump(res, open(a.out, 'w')); print('rows', len(res), 'seconds', round(time.time() - t0))
