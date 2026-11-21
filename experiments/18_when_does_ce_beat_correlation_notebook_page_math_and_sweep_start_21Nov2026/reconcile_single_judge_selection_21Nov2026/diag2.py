"""Experiment 18 reconciliation, mechanism check (11-21-2026): judge-level table and who each selector picks at 3 training families x 1 rating.
TLDR: reuses the loading and scoring code of reconcile.py (exec of its top part); prints judge-level r / RPS / CE and the pick frequencies of Pearson and CE.
Usage: python3 diag2.py --pool_json judge_forecasts.json --data dataset_pool.json --out out/full"""
import sys
src = open('reconcile.py').read()
head = src[:src.index('# ---------------- in-sample diagnostics')]
sys.argv = sys.argv[:1] + sys.argv[1:] + ['--draws', '1']
exec(head)
from scipy.stats import spearmanr
res = {}
mu_all = {pn: q@R for pn, q in POOLS.items()}; h = full@R
for pn in ['N15', 'H44', 'H5']:
    q = POOLS[pn]; rps_, ce_ = losses(q, full); r_ = np.array([np.corrcoef(mu_all[pn][:, j], h)[0, 1] if np.ptp(mu_all[pn][:, j]) > 1e-12 else np.nan for j in range(q.shape[1])])
    ok = np.isfinite(r_); rr = rps_.mean(0); cc = ce_.mean(0); fh = ((q <= EPS*1.0001) & (full[:, None, :] > 0)).any(-1).mean(0)
    top5 = np.argsort(-np.where(ok, r_, -9))[:5]
    res[pn] = {'spearman_r_vs_minus_rps_across_judges': float(spearmanr(r_[ok], -rr[ok])[0]), 'spearman_r_vs_minus_ce_across_judges': float(spearmanr(r_[ok], -cc[ok])[0]),
               'best_rps': float(rr.min()), 'rps_of_top5_by_r': [float(rr[j]) for j in top5], 'r_of_top5_by_r': [float(r_[j]) for j in top5], 'top5_by_r': [NAMES[pn][j] for j in top5],
               'n_judges_better_than_constant_rps': int((rr < 0.17369876543209878).sum()), 'undefined_r_judges': int((~ok).sum()),
               'judges': [{'judge': NAMES[pn][j], 'r': float(r_[j]), 'rps': float(rr[j]), 'ce': float(cc[j]), 'floor_hit_share': float(fh[j])} for j in np.argsort(rr)][:20 if pn != 'N15' else 15]}
# picks at C, n=3, k=1 for N15 and H44
for pn in ['N15', 'H44']:
    q = POOLS[pn]; means = q@R; full_rps, _ = losses(q, full); J = q.shape[1]
    picks = {s_: np.zeros(J, int) for s_ in SEL}; reg = {s_: [] for s_ in SEL}; best_test = []; n_r_ties = 0
    for s, trs, te in splits_C():
        rng = np.random.default_rng(np.random.SeedSequence([20261121, 190, s])); perms = [rng.permutation(r) for r in ratings]
        obs = hists([r[:1] for r in perms]); tr_rps, tr_ce = losses(q, obs); hh = obs@R; tr = trs[3]
        c = select(np.ones((1, len(fam_names)))[:, fid[tr]], means[tr], hh[tr], tr_ce[tr], tr_rps[tr])[0]
        tr_test = full_rps[te].mean(0)
        for i, s_ in enumerate(SEL): picks[s_][c[i]] += 1; reg[s_].append(tr_test[c[i]] - tr_test.min())
    res['picks_C_n3_k1_' + pn] = {s_: {'top3': [(NAMES[pn][j], int(picks[s_][j])) for j in np.argsort(-picks[s_])[:3]], 'distinct_judges_picked': int((picks[s_] > 0).sum()),
                                      'mean_regret_vs_best_judge_on_same_test': float(np.mean(reg[s_]))} for s_ in SEL}
json.dump(res, open(OUT/'diag2.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)[:6000])
