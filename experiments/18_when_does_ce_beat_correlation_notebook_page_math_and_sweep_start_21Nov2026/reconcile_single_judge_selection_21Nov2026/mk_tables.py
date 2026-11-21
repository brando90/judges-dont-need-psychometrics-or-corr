"""Make tables.md and the figure for the exp 18 reconciliation from reconcile.json (aggregates only)."""
import json, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r = json.load(open('reconcile.json')); cells = r['cells']
POOLN = {'N15': 'N15 (15 native-probability judges, local Codex pool)', 'H44': 'H44 (44 historical 3-vote judges)', 'H5': 'H5 (5 historical 3-vote judges)',
         'N15s3': 'N15s3 (N15 re-expressed as simulated 3-vote histograms, mean of 20 draws)', 'N15+H44': 'N15+H44 (union, 59 judges)'}
star = lambda c: '*' if (c['ci_lo'] > 0 or c['ci_hi'] < 0) else ''
L = []
for proto, unit in [('C', 'training families'), ('M', 'training tasks')]:
    L.append(f"\n### Protocol {proto}: {'Codex splits (250 family splits, 4 test families)' if proto=='C' else 'real_n_sweep splits (400 task splits, 5 test tasks)'}\nCells: held-out RPS of CE-selected / Pearson-selected / constant, and Pearson-selected minus CE-selected [95% interval]; positive = CE better; * = interval excludes 0. Interval: Bayesian bootstrap over the 19 task families (1,000 draws, reselecting on each draw).\n")
    L.append(f"| pool | {unit} | ratings per artifact | constant | CE-selected | Pearson-selected | Pearson minus CE [95%] |\n|---|---|---|---|---|---|---|")
    for pn in ['N15', 'N15s3', 'N15+H44', 'H44', 'H5']:
        for c in [c for c in cells if c['pool'] == pn and c['proto'] == proto]:
            L.append(f"| {pn} | {c['n_train_units']} | {c['k']} | {c['rps_constant']:.4f} | {c['rps_ce_selected']:.4f} | {c['rps_pearson_selected']:.4f} | {c['pearson_minus_ce']:+.4f} [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]{star(c)} |")
open('tables.md', 'w').write('\n'.join(L) + '\n')
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
col = {'N15': '#1f77b4', 'N15s3': '#9ecae1', 'N15+H44': '#2ca02c', 'H44': '#d62728', 'H5': '#ff9896'}
for a_, (proto, unit) in zip(ax, [('C', 'training families'), ('M', 'training tasks')]):
    for pn in ['N15', 'N15s3', 'N15+H44', 'H44', 'H5']:
        cs = sorted([c for c in cells if c['pool'] == pn and c['proto'] == proto and c['k'] == 1], key=lambda c: c['n_train_units'])
        a_.plot([c['n_train_units'] for c in cs], [c['pearson_minus_ce'] for c in cs], marker='o', color=col[pn], label=pn)
    a_.axhline(0, color='k', lw=.8); a_.set_xlabel(unit + (' (Codex splits)' if proto == 'C' else ' (real_n_sweep splits)')); a_.grid(True, alpha=.3)
ax[0].set_ylabel('Pearson-selected minus CE-selected test RPS\n(positive = CE better), 1 rating per artifact')
ax[0].legend(fontsize=8); fig.suptitle('Single-judge selection, same data: the sign follows the judge pool, not the split protocol (11-21-2026)', fontsize=10)
fig.tight_layout(); fig.savefig('fig_reconcile.png', dpi=140)
