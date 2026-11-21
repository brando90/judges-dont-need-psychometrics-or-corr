"""Experiment 20, part B (11-21-2026): the same trained-judge comparison on the REAL 75-artifact / 195-rating data, with saved judge outputs only (CPU, $0, no model calls).
TLDR: for each judge pool (closed-weight native probabilities N15; closed-weight 3-vote histograms H5; open-weight 3-vote histograms H39; all 59 = N15+H44) and each split / training size / ratings per artifact
we score seven arms on the held-out artifacts: sel_ce (pick the pool judge with the best training CE), sel_pearson (pick the pool judge with the best training Pearson), sel_pearson_cal (that judge + 3-parameter CE calibration),
head_ce / head_corr (softmax-linear head on the pool judges' expected scores, fit by CE vs by maximising Pearson, same init, same ridge, see lib20.py), head_corr_cal, and constant (training marginal).
Splits: the 250 family splits of experiments 18/19 (3, 7, 15 training families; 4 held-out families), ratings per training artifact 1, 2 or all. Test: every available rating of the held-out artifacts.
Also: --codex_check reproduces Codex's experiment-19 'closed pair' grid-calibration numbers (gpt-5.4-mini P0 + gpt-6.1-sol P0 mixture, 27-point calibration grid, CE vs Pearson) with an independent implementation.
Usage: python3 real20.py --pool_json judge_forecasts.json --data dataset_pool.json --out out_real.json [--procs 16] [--lam 0.1]"""
import argparse, json, time, itertools
import numpy as np
from multiprocessing import Pool
from scipy.stats import rankdata
from lib20 import *

ap = argparse.ArgumentParser(); ap.add_argument('--pool_json', default='judge_forecasts.json'); ap.add_argument('--data', default='dataset_pool.json'); ap.add_argument('--out', default='out_real.json')
ap.add_argument('--procs', type=int, default=8); ap.add_argument('--lam', type=float, default=0.1); ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--splits', type=int, default=250)
ap.add_argument('--codex_check', action='store_true')
a = ap.parse_args()
pj = json.load(open(a.pool_json)); ds = json.load(open(a.data)); items = pj['artifacts']; keys = [r['item'] for r in items]; EPS = pj['probability_floor']
by_item = {r['item']: r for r in ds['rows']}; assert set(keys) == set(by_item) and len(keys) == 75
ratings = [np.asarray(by_item[k]['ratings'], int) for k in keys]
fam_names, fid = np.unique([r['family'] for r in items], return_inverse=True); fam_col = fam_names[fid]
names15 = [s['judge_id'] for s in pj['arm_a_specs']]
raw = np.array([[{r['item']: r for r in pj['native_prob_forecasts'][n]}[k]['probabilities'] for k in keys] for n in names15]); raw = np.moveaxis(raw, 0, 1); raw = raw / raw.sum(-1, keepdims=True)
q15 = (1 - 6 * EPS) * raw + EPS
row_pos = {r['item']: i for i, r in enumerate(ds['rows'])}; order44 = ds['candidate_order']; CLOSED3 = ['H0_sol', 'H0_opus', 'H1_sol', 'H2_sol', 'H2_opus']; OPEN39 = [n for n in order44 if n not in CLOSED3]
def hist(v): return np.bincount(np.asarray(v, int), minlength=6) / len(v)
def smooth(p, e): p = np.maximum(p, e); return p / p.sum(-1, keepdims=True)
def hist_pool(names): return np.stack([smooth(np.array([hist(ds['outputs'][n][row_pos[k]]) for k in keys]), EPS) for n in names], axis=1)
POOLS = {'closed_native_15': q15, 'closed_3vote_5': hist_pool(CLOSED3), 'open_3vote_39': hist_pool(OPEN39), 'all_59': np.concatenate([q15, hist_pool(order44)], axis=1), 'hist_3vote_44': hist_pool(order44)}
NAMES = {'closed_native_15': names15, 'closed_3vote_5': CLOSED3, 'open_3vote_39': OPEN39, 'all_59': names15 + order44, 'hist_3vote_44': order44}
assert POOLS['all_59'].shape[1] == 59
full_hist = np.array([np.bincount(r, minlength=6) / len(r) for r in ratings]); nrat = np.array([len(r) for r in ratings])
svar = np.array([r.var(ddof=1) for r in ratings]); rmean = full_hist @ R
ARMS = ['sel_ce', 'sel_pearson', 'sel_pearson_cal', 'head_ce', 'head_corr', 'head_corr_cal', 'constant', 'uniform']
MET = ['rps', 'ce', 'tvd', 'mean_ae', 'mean_bias', 'var_sqerr', 'var_bias']
def art_losses(Q, idx):
    """Per held-out artifact losses of forecast Q (n_te x 6), averaged over that artifact's individual ratings; individual-rating RPS = mean over 5 thresholds of (Q-P)^2+P(1-P)."""
    P = full_hist[idx]; cq, cp = Q.cumsum(1)[:, :-1], P.cumsum(1)[:, :-1]; mq = Q @ R; vq = Q @ R ** 2 - mq ** 2
    return np.column_stack([(((cq - cp) ** 2 + cp * (1 - cp)).mean(1)), -(P * np.log(Q)).sum(1), 0.5 * np.abs(Q - P).sum(1), np.abs(mq - rmean[idx]), mq - rmean[idx], (vq - svar[idx]) ** 2, vq - svar[idx]])
def corr_metrics(Q, idx):
    mq = Q @ R; mo = rmean[idx]
    if np.ptp(mq) < 1e-12 or np.ptp(mo) < 1e-12: return np.nan, np.nan
    return np.corrcoef(mq, mo)[0, 1], np.corrcoef(rankdata(mq), rankdata(mo))[0, 1]
def counts(perms, k, idx):
    C = np.zeros((len(idx), 6))
    for j, i in enumerate(idx):
        r = ratings[i] if k == 'all' else perms[i][:k]; C[j] = np.bincount(r, minlength=6)
    return C
def pearson_per_judge(mu, y):  # mu: n x J
    mc = mu - mu.mean(0); yc = y - y.mean(); den = np.sqrt((mc ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(all='ignore'): c = (mc * yc[:, None]).sum(0) / den
    c[~np.isfinite(c)] = -np.inf; c[np.ptp(mu, 0) < 1e-12] = -np.inf; return c
def forecasts(q, tr, te, C, lam):
    """q: n x J x 6 pool. Returns dict arm -> Q_test (n_te x 6)."""
    qt = q[tr]; J = q.shape[1]; n_tr = C.sum()
    ce_j = -(C[:, None, :] * np.log(qt)).sum(2).sum(0) / n_tr; mu_j = qt @ R; y = (C @ R) / C.sum(1); pe = pearson_per_judge(mu_j, y)
    j_ce = int(np.argmin(ce_j)); j_pe = int(np.argmax(pe)) if np.isfinite(pe).any() else 0
    out = {'sel_ce': q[te, j_ce], 'sel_pearson': q[te, j_pe]}
    cal = calibrate(np.log(qt[:, j_pe]), C); L = cal[0] * np.log(q[te, j_pe]) + cal[1] * U + cal[2] * U ** 2; out['sel_pearson_cal'] = softmax(L)
    X = mu_j; sd = X.std(0); sd[sd < 1e-9] = 1.0; Z = (X - X.mean(0)) / sd; Zte = ((q[te] @ R) - X.mean(0)) / sd
    Xd = design(Z); Tc = fit(Xd, C, 'ce', lam, init_seed=1); To = fit(Xd, C, 'corr', lam, init_seed=1)
    out['head_ce'] = softmax(design(Zte) @ Tc); out['head_corr'] = softmax(design(Zte) @ To)
    cal2 = calibrate(Xd @ To, C); out['head_corr_cal'] = softmax(cal2[0] * (design(Zte) @ To) + cal2[1] * U + cal2[2] * U ** 2)
    m = (C.sum(0) + 0.5) / (C.sum() + 3.0); out['constant'] = np.tile(m, (len(te), 1)); out['uniform'] = np.full((len(te), 6), 1 / 6)
    return out, (j_ce, j_pe)

famsorted = sorted(set(r['family'] for r in items)); ordf = sorted({r['family'] for r in items if r['ordinary']}); phf = sorted({r['family'] for r in items if not r['ordinary']})
def split_list():
    rng = np.random.default_rng(20261116); out = []
    for s in range(250):
        test = set(rng.choice(ordf, 3, replace=False)) | set(rng.choice(phf, 1, replace=False)); train = rng.permutation([f for f in famsorted if f not in test])
        out.append((s, train, np.flatnonzero(np.isin(fam_col, list(test))), test))
    return out
SPL = split_list()[:a.splits]; SIZES = (3, 7, 15); KS = (1, 2, 'all')
wboot = np.vstack([np.ones((1, len(fam_names))), np.random.default_rng(20261123).exponential(size=(a.boot, len(fam_names)))])

def run_split(s):
    s_, train, te, test = SPL[s]; rng = np.random.default_rng(np.random.SeedSequence([20261121, 190, s_])); perms = [rng.permutation(r) for r in ratings]
    wte = wboot[:, fid[te]]; wte = wte / wte.sum(1, keepdims=True); res = {}
    for pname, q in POOLS.items():
        for n in SIZES:
            tr = np.flatnonzero(np.isin(fam_col, train[:n]))
            for k in KS:
                C = counts(perms, k, tr); F, picks = forecasts(q, tr, te, C, a.lam); cell = {}
                for arm in ARMS:
                    L = art_losses(F[arm], te); cell[arm] = (wte @ L, corr_metrics(F[arm], te))
                res[(pname, n, k)] = (cell, picks, len(tr), int(C.sum()))
    return s, res

if __name__ == '__main__' and not a.codex_check:
    t0 = time.time()
    with Pool(a.procs) as p: allres = dict(p.imap_unordered(run_split, range(len(SPL))))
    rows = []
    for pname in POOLS:
        for n in SIZES:
            for k in KS:
                agg = {arm: np.zeros((len(wboot), len(MET))) for arm in ARMS}; cs = {arm: [[], []] for arm in ARMS}; picks = []; ntr = []; nr = []
                for s in range(len(SPL)):
                    cell, pk, nt, nrt = allres[s][(pname, n, k)]; ntr.append(nt); nr.append(nrt); picks.append(pk)
                    for arm in ARMS: agg[arm] += cell[arm][0]; cs[arm][0].append(cell[arm][1][0]); cs[arm][1].append(cell[arm][1][1])
                for arm in ARMS: agg[arm] /= len(SPL)
                for arm in ARMS:
                    row = {'pool': pname, 'n_train_families': n, 'ratings_per_artifact': k, 'arm': arm, 'train_artifacts_mean': float(np.mean(ntr)), 'train_ratings_mean': float(np.mean(nr)), 'pearson': float(np.nanmean(cs[arm][0])), 'spearman': float(np.nanmean(cs[arm][1])), 'pearson_valid_splits': int(np.sum(np.isfinite(cs[arm][0])))}
                    for mi, mname in enumerate(MET):
                        row[mname] = float(agg[arm][0, mi]); d = agg[arm][1:, mi] - agg['sel_ce' if arm.startswith('sel') else 'head_ce'][1:, mi]
                        row[mname + '_vs_ce_lo'], row[mname + '_vs_ce_hi'] = [float(x) for x in np.quantile(d, [.025, .975])]
                    rows.append(row)
                # head_corr minus head_ce and sel_pearson minus sel_ce (point + interval) as explicit rows
                for (A, B, name) in (('sel_pearson', 'sel_ce', 'diff_sel_pearson_minus_ce'), ('head_corr', 'head_ce', 'diff_head_corr_minus_ce')):
                    row = {'pool': pname, 'n_train_families': n, 'ratings_per_artifact': k, 'arm': name}
                    for mi, mname in enumerate(MET):
                        d = agg[A][:, mi] - agg[B][:, mi]; row[mname] = float(d[0]); row[mname + '_lo'], row[mname + '_hi'] = [float(x) for x in np.quantile(d[1:], [.025, .975])]
                    rows.append(row)
                rows.append({'pool': pname, 'n_train_families': n, 'ratings_per_artifact': k, 'arm': 'picks', 'sel_ce_judges_chosen': sorted({NAMES[pname][p[0]] for p in picks}, key=str)[:8], 'sel_ce_top': NAMES[pname][int(np.bincount([p[0] for p in picks]).argmax())], 'sel_pearson_top': NAMES[pname][int(np.bincount([p[1] for p in picks]).argmax())]})
    json.dump({'lam': a.lam, 'splits': len(SPL), 'rows': rows}, open(a.out, 'w')); print('rows', len(rows), 'seconds', round(time.time() - t0))
