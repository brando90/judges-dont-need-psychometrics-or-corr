"""Experiment 18, reconciliation (11-21-2026): single-judge selection by Pearson vs CE, same pools and splits under both protocols.
TLDR: runs (pool x split protocol x training ratings per artifact) cells of 'select one judge by train CE / Pearson / RPS, score held-out RPS'.
Pools: N15 (exp 10 native-probability judges, as in the local Codex run), H44 and H5 (historical 3-vote scalar judges, as in the real_n_sweep run),
N15s3 (SIMULATED: the 15 native judges re-expressed as 3-vote histograms), N15+H44 (union).
Protocols: C = Codex (250 family splits, 4 test families, train 3/7/15 families, ratings 1/2/all), M = real_n_sweep (task splits, 5 test tasks, train 4/8/12/16 tasks, ratings 1/2/all).
Selection, scoring and Bayesian-bootstrap code follow the Codex run's analyze.py (re-implemented; asserts reproduce its numbers). CPU only, no model calls.
Usage: python3 reconcile.py --pool_json judge_forecasts.json --data dataset_pool.json --out out/ [--m_splits 400 --draws 10]"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

R = np.arange(6)
ap = argparse.ArgumentParser()
ap.add_argument('--pool_json', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--m_splits', type=int, default=400); ap.add_argument('--draws', type=int, default=10); ap.add_argument('--boot', type=int, default=1000)
a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); T0 = time.time()

# ---------------- data
pj = json.load(open(a.pool_json)); ds = json.load(open(a.data))
items = pj['artifacts']; keys = [r['item'] for r in items]; EPS = pj['probability_floor']
by_item = {r['item']: r for r in ds['rows']}
assert set(keys) == set(by_item) and len(keys) == 75
ratings = [np.asarray(by_item[k]['ratings'], int) for k in keys]
assert sorted(map(len, ratings)) == [2]*30 + [3]*45
fam_names, fid = np.unique([r['family'] for r in items], return_inverse=True)
tasks = np.array([by_item[k]['task'] for k in keys]); assert all(by_item[k]['task'] == r['task'] for k, r in zip(keys, items))
fam_col = fam_names[fid]
# pool N15 (Codex): native probabilities, floor applied as (1-6e)q+e once
names15 = [s['judge_id'] for s in pj['arm_a_specs']]
raw = np.array([[ {r['item']: r for r in pj['native_prob_forecasts'][n]}[k]['probabilities'] for k in keys] for n in names15])  # J,N,6
raw = np.moveaxis(raw, 0, 1); raw = raw / raw.sum(-1, keepdims=True)
q15 = (1 - 6*EPS)*raw + EPS
# historical pools H44 / H5: 3-vote histograms, floor by max(p, e) and renormalize (exp 10 'smooth')
row_pos = {r['item']: i for i, r in enumerate(ds['rows'])}
order44 = ds['candidate_order']; OLD5 = ['H0_sol', 'H0_opus', 'H1_sol', 'H2_sol', 'H2_opus']
def hist(v): return np.bincount(np.asarray(v, int), minlength=6)/len(v)
def smooth(p, e): p = np.maximum(p, e); return p/p.sum(-1, keepdims=True)
def hist_pool(names): return np.stack([smooth(np.array([hist(ds['outputs'][n][row_pos[k]]) for k in keys]), EPS) for n in names], axis=1)
q44 = hist_pool(order44); q5 = hist_pool(OLD5)
def n15_3vote(seed):
    rng = np.random.default_rng(seed); out = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            out[i, j] = np.bincount(rng.choice(6, 3, p=raw[i, j]), minlength=6)/3
    return smooth(out, EPS)
POOLS = {'N15': q15, 'H44': q44, 'H5': q5, 'N15+H44': np.concatenate([q15, q44], axis=1)}
NAMES = {'N15': names15, 'H44': order44, 'H5': OLD5, 'N15+H44': names15 + order44}

# ---------------- scoring
def hists(rs): return np.array([np.bincount(r, minlength=6)/len(r) for r in rs])
def losses(q, p):
    t = p[:, None, :]; Q, P = q.cumsum(-1)[..., :-1], t.cumsum(-1)[..., :-1]
    return (((Q-P)**2 + P*(1-P)).mean(-1), -(t*np.log(q)).sum(-1))
def best_min(v): return np.argmax(v <= np.min(v, -1, keepdims=True) + 1e-12, -1)
def select(w, mu, h, ce, rps):
    w = w/w.sum(1, keepdims=True); mb, hb = w@mu, w@h
    cov = w@(mu*h[:, None]) - mb*hb[:, None]; vm = np.maximum(w@(mu**2)-mb**2, 0); vh = np.maximum(w@(h**2)-hb**2, 0)
    valid = (np.ptp(mu, 0) > 1e-12) & (np.ptp(h) > 1e-12); den = np.sqrt(vm*vh[:, None])
    nc = np.full_like(cov, np.inf)
    if valid.any(): nc[:, valid] = -cov[:, valid]/den[:, valid]
    return np.column_stack([best_min(w@ce), best_min(nc), best_min(w@rps)])   # CE, Pearson, RPS choice per weight row

# ---------------- split generators: yield (train_units_ordered_or_full_train_idx function, test_idx)
def splits_C():
    famsorted = sorted(set(r['family'] for r in items))
    ordf = sorted({r['family'] for r in items if r['ordinary']}); phf = sorted({r['family'] for r in items if not r['ordinary']})
    rng = np.random.default_rng(20261116)
    for s in range(250):
        test = set(rng.choice(ordf, 3, replace=False)) | set(rng.choice(phf, 1, replace=False))
        train = rng.permutation([f for f in famsorted if f not in test])
        te = np.flatnonzero(np.isin(fam_col, list(test)))
        yield s, {n: np.flatnonzero(np.isin(fam_col, train[:n])) for n in (3, 7, 15)}, te
REPEAT_SEED0 = 20261210
def splits_M(n_splits):
    ut = sorted(set(tasks))
    for s in range(n_splits):
        rng = np.random.default_rng(REPEAT_SEED0+s); test_t = set(rng.permutation(ut)[:5].tolist())
        te_mask = np.array([t in test_t for t in tasks]); tr = np.flatnonzero(~te_mask); te = np.flatnonzero(te_mask)
        trt = np.array(sorted(set(tasks[tr]))); perm = np.random.default_rng(REPEAT_SEED0+s+777).permutation(trt)
        yield s, {n: np.array([i for i in tr if tasks[i] in set(perm[:n].tolist())]) for n in (4, 8, 12, 16)}, te
PROTOS = {'C': (splits_C, (3, 7, 15)), 'M': (lambda: splits_M(a.m_splits), (4, 8, 12, 16))}
KS = (1, 2, 'all')

full = hists(ratings)
wboot = np.vstack([np.ones((1, len(fam_names))), np.random.default_rng(20261123).exponential(size=(a.boot, len(fam_names)))])
SEL = ['ce', 'pearson', 'rps']

def run_cell_set(q, proto):
    gen, sizes = PROTOS[proto]; means = q@R; full_rps, full_ce = losses(q, full)
    est = {(n, k): np.zeros((len(wboot), 3)) for n in sizes for k in KS}; const = {(n, k): [] for n in sizes for k in KS}
    choice = {(n, k): np.zeros((3, q.shape[1]), int) for n in sizes for k in KS}; ntr = {(n, k): [] for n in sizes for k in KS}
    nsp = 0
    for s, trs, te in gen():
        nsp += 1; wte = wboot[:, fid[te]]; test_rps = (wte/wte.sum(1, keepdims=True))@full_rps[te]
        rng = np.random.default_rng(np.random.SeedSequence([20261121, 190, s])); perms = [rng.permutation(r) for r in ratings]
        for k in KS:
            obs = full if k == 'all' else hists([r[:k] for r in perms])
            tr_rps, tr_ce = losses(q, obs); h = obs@R
            for n, tr in trs.items():
                c = select(wboot[:, fid[tr]], means[tr], h[tr], tr_ce[tr], tr_rps[tr])
                est[(n, k)] += np.take_along_axis(test_rps, c, 1); choice[(n, k)][np.arange(3), c[0]] += 1; ntr[(n, k)].append(len(tr))
                marg = (obs[tr]*np.array([len(ratings[i]) if k == 'all' else k for i in tr])[:, None]).sum(0) + 0.5; marg /= marg.sum()
                const[(n, k)].append(float(losses(np.broadcast_to(marg, (len(te), 1, 6)), full[te])[0].mean()))
    rows = []
    for (n, k), e in est.items():
        sc = e/nsp; d_ce = sc[:, 1]-sc[:, 0]; d_rps = sc[:, 1]-sc[:, 2]
        lo, hi = np.quantile(d_ce[1:], [.025, .975]); lo2, hi2 = np.quantile(d_rps[1:], [.025, .975])
        rows.append({'proto': proto, 'n_train_units': n, 'k': k, 'splits': nsp, 'train_artifacts_mean': float(np.mean(ntr[(n, k)])),
                     'rps_ce_selected': sc[0, 0], 'rps_pearson_selected': sc[0, 1], 'rps_rps_selected': sc[0, 2], 'rps_constant': float(np.mean(const[(n, k)])),
                     'pearson_minus_ce': d_ce[0], 'ci_lo': lo, 'ci_hi': hi, 'pearson_minus_rpsel': d_rps[0], 'ci2_lo': lo2, 'ci2_hi': hi2,
                     'top_choice': {s_: int(np.argmax(choice[(n, k)][i])) for i, s_ in enumerate(SEL)},
                     'top_choice_share': {s_: float(choice[(n, k)][i].max()/nsp) for i, s_ in enumerate(SEL)}})
    return rows

# ---------------- in-sample diagnostics (all 75 artifacts, all ratings)
def diagnostics(q, names):
    mu = q@R; h = full@R; rps_, ce_ = losses(q, full); pe = np.array([np.corrcoef(mu[:, j], h)[0, 1] if np.ptp(mu[:, j]) > 1e-12 else np.nan for j in range(q.shape[1])])
    jp, jc, jr = int(np.nanargmax(pe)), int(np.argmin(ce_.mean(0))), int(np.argmin(rps_.mean(0)))
    mp = full.mean(0); const_rps = float(losses(np.broadcast_to(mp, (75, 1, 6)), full)[0].mean())
    floor_hit = float(((q <= EPS*1.0001) & (full[:, None, :] > 0)).any(-1).mean())
    return {'J': q.shape[1], 'pearson_top': names[jp], 'pearson_top_r': float(pe[jp]), 'pearson_top_rps': float(rps_.mean(0)[jp]), 'pearson_top_ce': float(ce_.mean(0)[jp]),
            'ce_top': names[jc], 'ce_top_r': float(pe[jc]), 'ce_top_rps': float(rps_.mean(0)[jc]), 'ce_top_ce': float(ce_.mean(0)[jc]),
            'rps_top': names[jr], 'rps_top_rps': float(rps_.mean(0)[jr]), 'constant_rps_insample': const_rps,
            'share_judge_item_pairs_with_floor_mass_on_an_observed_rating': floor_hit,
            'median_judge_rps': float(np.median(rps_.mean(0))), 'median_judge_ce': float(np.median(ce_.mean(0)))}

result = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'm_splits': a.m_splits, 'draws_N15s3': a.draws, 'boot': a.boot, 'diagnostics': {}, 'cells': []}
for pn, q in POOLS.items():
    result['diagnostics'][pn] = diagnostics(q, NAMES[pn])
    for proto in PROTOS:
        for r in run_cell_set(q, proto): r['pool'] = pn; result['cells'].append(r)
    print(pn, 'done', f'{time.time()-T0:.0f}s', flush=True)
# N15s3: average of cells over simulated 3-vote draws
acc = {}
diag_s3 = []
for d in range(a.draws):
    qd = n15_3vote(900+d); diag_s3.append(diagnostics(qd, names15))
    for proto in PROTOS:
        for r in run_cell_set(qd, proto):
            key = (r['proto'], r['n_train_units'], r['k']); acc.setdefault(key, []).append(r)
    print('N15s3 draw', d, f'{time.time()-T0:.0f}s', flush=True)
for (proto, n, k), rs in acc.items():
    m = lambda f: float(np.mean([x[f] for x in rs]))
    result['cells'].append({'pool': 'N15s3', 'proto': proto, 'n_train_units': n, 'k': k, 'splits': rs[0]['splits'], 'draws': len(rs),
        **{f: m(f) for f in ['rps_ce_selected', 'rps_pearson_selected', 'rps_rps_selected', 'rps_constant', 'pearson_minus_ce', 'ci_lo', 'ci_hi', 'pearson_minus_rpsel']},
        'draws_with_ci_above_zero': int(sum(x['ci_lo'] > 0 for x in rs)), 'draws_with_ci_below_zero': int(sum(x['ci_hi'] < 0 for x in rs))})
result['diagnostics']['N15s3_mean_over_draws'] = {f: float(np.mean([x[f] for x in diag_s3])) for f in ['pearson_top_rps', 'ce_top_rps', 'rps_top_rps', 'constant_rps_insample', 'share_judge_item_pairs_with_floor_mass_on_an_observed_rating', 'median_judge_rps', 'median_judge_ce', 'pearson_top_r', 'ce_top_r']}
json.dump(result, open(OUT/'reconcile.json', 'w'), indent=1, default=float)
print('DONE', f'{time.time()-T0:.0f}s')
