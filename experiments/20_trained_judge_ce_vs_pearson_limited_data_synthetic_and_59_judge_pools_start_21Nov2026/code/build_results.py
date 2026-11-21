"""Experiment 20 (11-21-2026): assemble RESULTS_11-21-2026.md tables from the aggregate outputs. Usage: python3 build_results.py <folder>"""
import json, sys, re, collections
F = sys.argv[1]; real = json.load(open(f'{F}/results/out_real_lam0.1.json'))['rows']; idx = {(r['pool'], r['n_train_families'], r['ratings_per_artifact'], r['arm']): r for r in real}
syn = open(f'{F}/results/tables_synth.md').read().split('\n')
def synth_table(title_start):
    i = next(j for j, l in enumerate(syn) if l.startswith(title_start)); out = [syn[i + 2], syn[i + 3]]
    for l in syn[i + 4:]:
        if not l.startswith('|'): break
        if l.split('|')[2].strip() in ('ce', 'corr', 'constant'): out.append(l)
    return '\n'.join(out)
A = synth_table('## mean_spread_shape: 3 rating'); A1 = synth_table('## mean_spread_shape: 1 rating'); A0 = synth_table('## mean_only: 3 rating')
cols = [('ce', 'test CE (nats)'), ('tvd', 'TVD'), ('rps', 'test RPS'), ('mean_ae', 'mean abs error (pts)'), ('var_sqerr', 'variance sq. error'), ('pearson', 'Pearson'), ('spearman', 'Spearman')]
def real_table(ratings):
    o = ['| pool | training families | arm | ' + ' | '.join(h for _, h in cols) + ' | skew |', '|' + '---|' * (len(cols) + 4)]
    for p in ['closed_native_15', 'closed_3vote_5', 'open_3vote_39', 'all_59', 'hist_3vote_44']:
        for n in (3, 7, 15):
            for arm in ('sel_ce', 'sel_pearson', 'head_ce', 'head_corr', 'constant'):
                r = idx[(p, n, ratings, arm)]; o.append(f'| {p} | {n} | {arm} | ' + ' | '.join(f'{r[m]:.4f}' if r[m] == r[m] else 'n/a' for m, _ in cols) + ' | n/a (2-3 ratings per artifact) |')
    return '\n'.join(o)
def diff_table(ratings):
    o = ['| pool | training families | selection: Pearson minus CE (RPS) | trained head: Corr minus CE (RPS) | trained head: Corr minus CE (variance sq. error) |', '|---|---|---|---|---|']
    for p in ['closed_native_15', 'closed_3vote_5', 'open_3vote_39', 'all_59', 'hist_3vote_44']:
        for n in (3, 7, 15):
            a, b = idx[(p, n, ratings, 'diff_sel_pearson_minus_ce')], idx[(p, n, ratings, 'diff_head_corr_minus_ce')]
            f = lambda r, m: f"{r[m]:+.4f} [{r[m+'_lo']:+.4f}, {r[m+'_hi']:+.4f}]"
            o.append(f'| {p} | {n} | {f(a,"rps")} | {f(b,"rps")} | {f(b,"var_sqerr")} |')
    return '\n'.join(o)
chk = json.load(open(f'{F}/results/check_codex.json'))
t = open(f'{F}/RESULTS_TEMPLATE.md').read()
for k, v in {'{{A}}': A, '{{A1}}': A1, '{{A0}}': A0, '{{B_ALL}}': real_table('all'), '{{B_1}}': real_table(1), '{{D_ALL}}': diff_table('all'), '{{D_1}}': diff_table(1), '{{NCHECK}}': str(chk['n_compared']), '{{MAXDIFF}}': f"{chk['max_abs_diff']:.1e}"}.items(): t = t.replace(k, v)
open(f'{F}/RESULTS_11-21-2026.md', 'w').write(t)
