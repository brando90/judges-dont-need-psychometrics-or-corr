"""Experiment 20, part B aggregation (11-21-2026): out_real_lam<L>.json -> markdown tables (full panel per pool, difference tables with Bayesian family-bootstrap intervals).
Usage: python3 agg_real.py out_real_lam0.1.json tables_real_lam0.1.md [more.json ...]"""
import json, sys
rows = json.load(open(sys.argv[1]))['rows']; lam = json.load(open(sys.argv[1]))['lam']
POOLS = ['closed_native_15', 'closed_3vote_5', 'open_3vote_39', 'all_59', 'hist_3vote_44']; DESC = {'closed_native_15': 'closed-weight, 15 judges with native probabilities (OpenAI, Mistral)', 'closed_3vote_5': 'closed-weight, 5 judges as 3-vote histograms (Sol, Opus)', 'open_3vote_39': 'open-weight, 39 judges as 3-vote histograms', 'all_59': 'all 59 judges (15 native + 44 3-vote)', 'hist_3vote_44': 'the 44 historical 3-vote judges (5 closed + 39 open), experiment 18 pool H44'}
ARMS = ['sel_ce', 'sel_pearson', 'sel_pearson_cal', 'head_ce', 'head_corr', 'head_corr_cal', 'constant', 'uniform']
cols = [('rps', 'test RPS'), ('ce', 'test CE (nats)'), ('tvd', 'TVD'), ('mean_ae', 'mean abs error (pts)'), ('mean_bias', 'mean bias (pts)'), ('var_sqerr', 'variance sq. error'), ('pearson', 'Pearson'), ('spearman', 'Spearman')]
idx = {(r['pool'], r['n_train_families'], r['ratings_per_artifact'], r['arm']): r for r in rows}
out = [f'# Real-data tables, ridge lambda {lam} (head arms). 250 family splits, 4 held-out families, all available ratings of held-out artifacts scored. Skew is not computed: each artifact has only 2-3 ratings.\n']
for p in POOLS:
    out.append(f'\n## Pool: {p}: {DESC[p]}\n')
    out.append('### Difference rows: Pearson-based minus CE-based (positive = CE better for every column except Pearson/Spearman). Bracket = 95% Bayesian bootstrap over the 19 held-out/test families (test weighting only; training is not refit)\n')
    out.append('| training families | ratings/artifact | contrast | RPS | CE | TVD | mean abs error | variance sq. error |\n|---|---|---|---|---|---|---|---|')
    for n in (3, 7, 15):
        for k in (1, 2, 'all'):
            for arm in ('diff_sel_pearson_minus_ce', 'diff_head_corr_minus_ce'):
                r = idx[(p, n, k, arm)]
                out.append(f'| {n} | {k} | {arm.replace("diff_", "").replace("_minus_ce", "")} minus CE | ' + ' | '.join(f"{r[m]:+.4f} [{r[m+'_lo']:+.4f}, {r[m+'_hi']:+.4f}]" for m in ('rps', 'ce', 'tvd', 'mean_ae', 'var_sqerr')) + ' |')
    out.append('\n### Full panel\n')
    out.append('| training families | ratings/artifact | arm | ' + ' | '.join(h for _, h in cols) + ' |\n|' + '---|' * (len(cols) + 3))
    for n in (3, 7, 15):
        for k in (1, 2, 'all'):
            for arm in ARMS:
                r = idx[(p, n, k, arm)]; out.append(f'| {n} | {k} | {arm} | ' + ' | '.join(f'{r[m]:.4f}' if r[m] == r[m] else 'n/a' for m, _ in cols) + ' |')
    pk = idx[(p, 3, 1, 'picks')]; out.append(f"\nMost-chosen judge at 3 families / 1 rating: by CE {pk['sel_ce_top']}; by Pearson {pk['sel_pearson_top']}. At 15 families / all ratings: by CE {idx[(p,15,'all','picks')]['sel_ce_top']}; by Pearson {idx[(p,15,'all','picks')]['sel_pearson_top']}.")
open(sys.argv[2], 'w').write('\n'.join(out) + '\n')
