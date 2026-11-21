"""Experiment 20, part A aggregation (11-21-2026): turn out_synth.json (one row per world x seed x training size x lambda x arm) into tables.
TLDR: mean over seeds with 95% t-interval; paired difference (corr minus ce, same seed and same training sample) with its interval and the share of seeds where CE is better.
Usage: python3 agg_synth.py out_synth.json tables_synth.md summary_synth.json"""
import json, sys, collections
import numpy as np
from scipy import stats
rows = json.load(open(sys.argv[1])); MET = ['ce', 'tvd', 'rps_excess', 'mean_rmse', 'var_rmse', 'skew_rmse', 'pearson', 'spearman', 'var_spearman_in_mean_bin']
LOWER = {'ce': 1, 'tvd': 1, 'rps_excess': 1, 'mean_rmse': 1, 'var_rmse': 1, 'skew_rmse': 1, 'pearson': 0, 'spearman': 0, 'var_spearman_in_mean_bin': 0}
HDR = {'ce': 'test CE (nats)', 'tvd': 'TVD', 'rps_excess': 'RPS excess', 'mean_rmse': 'mean RMSE', 'var_rmse': 'variance RMSE', 'skew_rmse': 'skew RMSE', 'pearson': 'Pearson', 'spearman': 'Spearman', 'var_spearman_in_mean_bin': 'var Spearman within mean bin'}
D = collections.defaultdict(dict)
for r in rows:
    if r['arm'] in ('world_diag', 'oracle'): continue
    D[(r['world'], r['ratings_per_item'], r['lam'], r['n_items'], r['arm'])][r['seed']] = r
def ci(x):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) < 3: return (float('nan'), float('nan'), len(x))
    return (x.mean(), stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)), len(x))
def cell(x, d=3): m, h, n = ci(x); return 'n/a' if m != m else f'{m:.{d}f} ± {h:.{d}f}'
def vals(key, met): return [v[met] for v in D[key].values()]
out = []; summ = {}
oracle = collections.defaultdict(list)
for r in rows:
    if r['arm'] == 'oracle': oracle[r['world']].append(r['ce'])
diag = collections.defaultdict(list)
for r in rows:
    if r['arm'] == 'world_diag': diag[r['world']].append(r)
out.append('## World diagnostics (mean over seeds; 2000 test items per seed)\n\n| World | SD of true mean | SD of true variance | SD of true skew | corr(mean, variance) | SD of true variance inside a mean bin (same-mean items) | oracle test CE (nats) |\n|---|---|---|---|---|---|---|')
for w, rs in diag.items():
    out.append(f"| {w} | {np.mean([r['mean_sd'] for r in rs]):.2f} | {np.mean([r['var_sd'] for r in rs]):.2f} | {np.mean([r['skew_sd'] for r in rs]):.2f} | {np.mean([r['corr_mean_var'] for r in rs]):+.2f} | {np.mean([r['within_mean_bin_var_sd'] for r in rs]):.2f} | {np.mean(oracle[w]):.3f} |")
SIZES = sorted({k[3] for k in D}); worlds = sorted({k[0] for k in D})
for w in worlds:
    for nr in (3, 1):
        out.append(f'\n## {w}: {nr} rating(s) per training item, ridge lambda 0.01 (primary). Mean over 200 seeds ± 95% interval, scored against the TRUE distribution of 2000 fresh items\n')
        out.append('| training items | arm | ' + ' | '.join(HDR[m] for m in MET) + ' |\n|' + '---|' * (len(MET) + 2))
        for m_ in SIZES:
            for arm in ('ce', 'corr', 'corr_cal', 'constant'):
                k = (w, nr, 0.01, m_, arm)
                out.append(f'| {m_} | {arm} | ' + ' | '.join(cell(vals(k, m)) for m in MET) + ' |')
    # paired differences
    out.append(f'\n## {w}: paired difference (Pearson-trained minus CE-trained), same training sample. Positive = CE better for error columns, negative = CE better for correlation columns. Share = seeds where CE is better\n')
    out.append('| ratings/item | lambda | training items | ' + ' | '.join(f'{HDR[m]}: diff [95%] (CE better in %)' for m in ('ce', 'tvd', 'rps_excess', 'mean_rmse', 'var_rmse', 'skew_rmse', 'spearman')) + ' |\n|' + '---|' * 10)
    for nr in (3, 1):
        for lam in (0.01, 0.1):
            for m_ in SIZES:
                parts = []
                for met in ('ce', 'tvd', 'rps_excess', 'mean_rmse', 'var_rmse', 'skew_rmse', 'spearman'):
                    a, b = D[(w, nr, lam, m_, 'corr')], D[(w, nr, lam, m_, 'ce')]
                    d = [a[s][met] - b[s][met] for s in a if s in b and a[s][met] == a[s][met] and b[s][met] == b[s][met]]
                    mn, h, n = ci(d); better = np.mean([(x > 0) if LOWER[met] else (x < 0) for x in d]) * 100 if d else float('nan')
                    parts.append(f'{mn:+.3f} [{mn-h:+.3f}, {mn+h:+.3f}] ({better:.0f}%)' if mn == mn else 'n/a')
                    summ[f'{w}|nr{nr}|lam{lam}|m{m_}|{met}'] = {'diff_corr_minus_ce': mn, 'lo': mn - h, 'hi': mn + h, 'ce_better_pct': better}
                out.append(f'| {nr} | {lam} | {m_} | ' + ' | '.join(parts) + ' |')
    out.append(f'\n## {w}: sensitivity to ridge lambda, RPS excess (CE / Pearson / constant), 3 ratings per item, mean over seeds\n\n| training items | ' + ' | '.join(f'lambda {l}: CE / Pearson' for l in (0.001, 0.01, 0.1, 1.0)) + ' | constant |\n|' + '---|' * 6)
    for m_ in SIZES:
        out.append(f'| {m_} | ' + ' | '.join(f"{ci(vals((w,3,l,m_,'ce'),'rps_excess'))[0]:.4f} / {ci(vals((w,3,l,m_,'corr'),'rps_excess'))[0]:.4f}" for l in (0.001, 0.01, 0.1, 1.0)) + f" | {ci(vals((w,3,0.01,m_,'constant'),'rps_excess'))[0]:.4f} |")
open(sys.argv[2], 'w').write('\n'.join(out) + '\n'); json.dump(summ, open(sys.argv[3], 'w'))
print('\n'.join(out[:6]))
