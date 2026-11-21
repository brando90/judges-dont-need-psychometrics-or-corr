"""Experiment 15: item signal versus rater severity on the 75 expert-rated artifacts (Sanmi Koyejo's 11-16 point 4).
TLDR: the three expert raters (participants 1-3) are the same people on every artifact (1 and 2 rate all 75, 3 rates 45), so rater severity (a constant offset per rater)
is estimable even with 2-3 ratings per artifact; what is not separable is a rater-by-item interaction from noise. Estimates the rater offsets, the share of
within-artifact rating variance they explain, ICC consistency vs agreement and Lin concordance, variance components,
and whether correcting for rater severity changes which cached judge looks best. CPU only, no model calls, $0.
Usage: python3 analyze_raters.py --items items.jsonl --pool dataset_pool.json --out results [--boot 10000]"""
import warnings; warnings.filterwarnings('ignore')
import argparse, json, time, platform, sys, itertools
from pathlib import Path
import numpy as np, scipy
from scipy.stats import rankdata, kendalltau
ap = argparse.ArgumentParser(); ap.add_argument('--items', required=True); ap.add_argument('--pool', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--boot', type=int, default=10000); a = ap.parse_args(); t0 = time.time()
items = [json.loads(l) for l in open(a.items)]; pool = json.load(open(a.pool))
assert [r['item'] for r in pool['rows']] == [i['item'] for i in items]
RK = ['p1', 'p2', 'p3']; Rr = len(RK)
Y = np.array([[i[k] if i.get(k) is not None else np.nan for k in RK] for i in items], float)       # 75 x 3, NaN = not rated
assert [int(v) for v in Y[~np.isnan(Y)]] == [x for r in pool['rows'] for x in r['ratings']], 'ratings must match the pool dataset'
tasks = np.array([i['task'] for i in items]); ut = sorted(set(tasks)); T = len(ut); tidx = np.array([ut.index(t) for t in tasks]); N = len(items)
GRID = np.arange(6)/5

def additive_fit(Y):
    """Least squares rating = item effect + rater offset (offsets sum to zero over raters weighted equally); returns offsets, SSE of item-only and additive, dfs."""
    obs = ~np.isnan(Y); n = obs.sum(); im = np.nanmean(Y, 1, keepdims=True)
    sse_item = float(np.nansum((Y-im)**2)); df_item = int(n-obs.any(1).sum())
    # alternating projections for the two-way additive model on an incomplete table
    beta = np.zeros(Y.shape[1]); alpha = np.nanmean(Y, 1)
    for _ in range(500):
        alpha_new = np.nanmean(Y-beta[None, :], 1); beta_new = np.nanmean(Y-alpha_new[:, None], 0); beta_new -= beta_new.mean()
        if np.abs(alpha_new-alpha).max() < 1e-13 and np.abs(beta_new-beta).max() < 1e-13: alpha, beta = alpha_new, beta_new; break
        alpha, beta = alpha_new, beta_new
    resid = Y-alpha[:, None]-beta[None, :]; sse_add = float(np.nansum(resid**2)); df_add = int(df_item-(Y.shape[1]-1))
    return beta, sse_item, df_item, sse_add, df_add, resid

def icc_pair(x, y):
    """Two-way ICC for k=2 raters on n targets (McGraw and Wong): consistency ICC(C,1) and agreement ICC(A,1); Lin concordance."""
    n = len(x); d = np.stack([x, y], 1); gm = d.mean(); rm = d.mean(1); cm = d.mean(0)
    msr = 2*((rm-gm)**2).sum()/(n-1); msc = n*((cm-gm)**2).sum()/1; sse = ((d-rm[:, None]-cm[None, :]+gm)**2).sum(); mse = sse/((n-1)*1)
    icc_c = (msr-mse)/(msr+mse); icc_a = (msr-mse)/(msr+mse+2*(msc-mse)/n)
    sxy = ((x-x.mean())*(y-y.mean())).mean(); ccc = 2*sxy/(x.var()+y.var()+(x.mean()-y.mean())**2)
    return float(icc_c), float(icc_a), float(ccc), float(np.corrcoef(x, y)[0, 1])

def icc_k3(M):
    """ICC(C,1), ICC(A,1) for the complete 3-rater subset (n x 3)."""
    n, k = M.shape; gm = M.mean(); rm = M.mean(1); cm = M.mean(0)
    msr = k*((rm-gm)**2).sum()/(n-1); msc = n*((cm-gm)**2).sum()/(k-1); sse = ((M-rm[:, None]-cm[None, :]+gm)**2).sum(); mse = sse/((n-1)*(k-1))
    return float((msr-mse)/(msr+(k-1)*mse)), float((msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n))

def stats_for(Yb, tb_idx=None):
    beta, sse_i, df_i, sse_a, df_a, _ = additive_fit(Yb); s2w = sse_i/df_i; s2r = sse_a/df_a
    o = {'offsets': beta, 's2_within_item': s2w, 's2_after_offsets': s2r, 'share_explained_by_offsets': 1-s2r/s2w}
    pairs = {}
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        m = ~np.isnan(Yb[:, i]) & ~np.isnan(Yb[:, j]); x, y = Yb[m, i], Yb[m, j]
        pairs[f'{RK[i]}-{RK[j]}'] = dict(zip(['icc_consistency', 'icc_agreement', 'lin_ccc', 'pearson'], icc_pair(x, y))) | {'n_items': int(m.sum()), 'mean_diff': float((x-y).mean())}
    full = ~np.isnan(Yb).any(1); c, ag = icc_k3(Yb[full]); pairs['p1-p2-p3 (complete 45)'] = {'icc_consistency': c, 'icc_agreement': ag, 'n_items': int(full.sum())}
    o['pairs'] = pairs; return o

obs = stats_for(Y)
# task-cluster bootstrap: resample the 25 tasks with replacement, refit
rng = np.random.default_rng(1210); members = [np.flatnonzero(tidx == t) for t in range(T)]
B = {'off': [], 'share': [], 'pairs': {k: {m: [] for m in ['icc_consistency', 'icc_agreement', 'lin_ccc', 'mean_diff']} for k in obs['pairs'] if 'complete' not in k}, 'full_c': [], 'full_a': []}
for b in range(a.boot):
    if b % 1000 == 0: print('boot', b, round(time.time()-t0, 1), flush=True)
    draw = rng.integers(0, T, size=T); idx = np.concatenate([members[t] for t in draw]); s = stats_for(Y[idx])
    B['off'].append(s['offsets']); B['share'].append(s['share_explained_by_offsets'])
    for k in B['pairs']:
        for m in B['pairs'][k]: B['pairs'][k][m].append(s['pairs'][k][m])
    B['full_c'].append(s['pairs']['p1-p2-p3 (complete 45)']['icc_consistency']); B['full_a'].append(s['pairs']['p1-p2-p3 (complete 45)']['icc_agreement'])
ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
res = {'experiment': 15, 'n_items': N, 'n_tasks': T, 'n_ratings': int((~np.isnan(Y)).sum()), 'raters': RK, 'bootstrap': 'task-cluster, 10,000 draws', 'bootstrap_draws': a.boot,
       'ratings_per_item': {'2': int((np.isnan(Y).sum(1) == 1).sum()), '3': int((np.isnan(Y).sum(1) == 0).sum())},
       'rater_means_over_items_rated': {RK[j]: float(np.nanmean(Y[:, j])) for j in range(Rr)}, 'rater_items': {RK[j]: int((~np.isnan(Y[:, j])).sum()) for j in range(Rr)},
       'offsets': {RK[j]: {'estimate': float(obs['offsets'][j]), 'ci95': ci(np.array(B['off'])[:, j])} for j in range(Rr)},
       's2_within_item': obs['s2_within_item'], 's2_after_offsets': obs['s2_after_offsets'],
       'share_of_within_item_variance_explained_by_rater_offsets': {'estimate': obs['share_explained_by_offsets'], 'ci95': ci(B['share'])},
       'pairs': {}}
for k, v in obs['pairs'].items():
    if 'complete' in k: res['pairs'][k] = v | {'icc_consistency_ci95': ci(B['full_c']), 'icc_agreement_ci95': ci(B['full_a'])}
    else: res['pairs'][k] = v | {f'{m}_ci95': ci(B['pairs'][k][m]) for m in B['pairs'][k]}

beta, _, _, _, _, resid = additive_fit(Y)
# variance components (method of moments on the additive fit): item signal vs rater offsets vs residual
mean_item = np.nanmean(Y-beta[None, :], 1); res['variance_components'] = {
    'residual_sd_after_item_and_rater_effects': float(np.sqrt(obs['s2_after_offsets'])), 'sd_of_artifact_means': float(np.std(mean_item, ddof=1)),
    'sd_of_rater_offsets': float(np.std(obs['offsets'], ddof=1)),
    'note': 'the artifact means carry noise of sd about residual_sd/sqrt(ratings per artifact), so their sd overstates item signal slightly'}

# does correcting for rater severity change how the cached judges look?
names = pool['candidate_order']; mu = np.array([np.array([np.mean(r) for r in pool['outputs'][n]], float) for n in names])       # judges x 75 expected rating (mean of 3 cached scalar ratings)
raw_t = np.nanmean(Y, 1); adj_t = np.nanmean(Y-beta[None, :], 1)
def sp(x, y): return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
def pe(x, y): return float(np.corrcoef(x, y)[0, 1])
rows = []
for n, m in zip(names, mu): rows.append({'judge': n, 'spearman_raw': sp(m, raw_t), 'spearman_adj': sp(m, adj_t), 'pearson_raw': pe(m, raw_t), 'pearson_adj': pe(m, adj_t),
                                          'mean_bias_raw': float(m.mean()-raw_t.mean()), 'mean_bias_adj': float(m.mean()-adj_t.mean())})
adj_hist = np.zeros((N, 6)); raw_hist = np.zeros((N, 6)); 
for i in range(N):
    for j in range(Rr):
        if not np.isnan(Y[i, j]):
            raw_hist[i, int(Y[i, j])] += 1; adj_hist[i, int(np.clip(np.rint(Y[i, j]-beta[j]), 0, 5))] += 1
raw_hist /= raw_hist.sum(1, keepdims=True); adj_hist /= adj_hist.sum(1, keepdims=True)
def rps_h(P, Q): pc = np.cumsum(P, -1)[:, :5]; qc = np.cumsum(Q, -1)[:, :5]; return float(np.mean((qc-pc)**2+pc*(1-pc)))
def hist_of(sc):
    h = np.zeros((N, 6)); 
    for i in range(N):
        for v in sc[i]: h[i, int(v)] += 1
    return h/h.sum(1, keepdims=True)
for r_, n in zip(rows, names):
    Q = np.maximum(hist_of(pool['outputs'][n]), 0.05); Q /= Q.sum(1, keepdims=True)
    r_['rps_vs_raw_hist_floor0.05'] = rps_h(raw_hist, Q); r_['rps_vs_adj_hist_floor0.05'] = rps_h(adj_hist, Q)
res['judges_vs_target'] = rows
def rank_change(k1, k2, desc=True):
    v1 = np.array([r[k1] for r in rows]); v2 = np.array([r[k2] for r in rows]); s = -1 if desc else 1
    return {'kendall_tau_between_rankings': float(kendalltau(v1, v2)[0]), 'best_judge_raw': rows[int(np.argmax(s*-1*v1) if False else (np.argmax(v1) if desc else np.argmin(v1)))]['judge'],
            'best_judge_adj': rows[int(np.argmax(v2) if desc else np.argmin(v2))]['judge']}
res['ranking_stability'] = {'spearman_raw_vs_adj': rank_change('spearman_raw', 'spearman_adj'), 'pearson_raw_vs_adj': rank_change('pearson_raw', 'pearson_adj'),
                            'rps_raw_vs_adj_floor0.05': rank_change('rps_vs_raw_hist_floor0.05', 'rps_vs_adj_hist_floor0.05', desc=False)}
res['judge_means'] = {'median_spearman_raw': float(np.median([r['spearman_raw'] for r in rows])), 'median_spearman_adj': float(np.median([r['spearman_adj'] for r in rows])),
                      'max_abs_change_in_spearman': float(max(abs(r['spearman_raw']-r['spearman_adj']) for r in rows))}
res['notes'] = ['Raters are identified only by their fixed participant slot (p1, p2, p3); the same slot is the same person on every artifact per the experiment 100 protocol.',
                'Rater offset = constant shift per rater in the additive model rating = artifact + rater; it cannot be separated from a rater-by-artifact interaction, which needs repeated ratings of the same artifact by the same rater.',
                'Severity-adjusted ratings subtract each rater offset and round to the nearest integer in 0-5 (only used for the RPS check against adjusted histograms).']
res['seconds'] = time.time()-t0; res['environment'] = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); json.dump(res, open(out/'results.json', 'w'), indent=1)
(out/'DONE').write_text('done\n'); print(json.dumps({k: res[k] for k in ['offsets', 'share_of_within_item_variance_explained_by_rater_offsets', 'pairs', 'variance_components', 'ranking_stability', 'judge_means']}, indent=1))
