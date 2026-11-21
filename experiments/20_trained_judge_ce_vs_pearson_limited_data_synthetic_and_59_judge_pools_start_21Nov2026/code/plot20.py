"""Experiment 20 figures (11-21-2026): Pearson-trained minus CE-trained gaps. Usage: python3 plot20.py summary_synth.json out_real_lam0.1.json fig_synth.png fig_real.png"""
import json, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
s = json.load(open(sys.argv[1])); real = json.load(open(sys.argv[2]))['rows']; M = [4, 8, 16, 32, 64, 128, 256, 512]
fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for i, w in enumerate(['mean_only', 'mean_spread_shape']):
    for j, (met, tt) in enumerate([('rps_excess', 'RPS excess: Pearson-trained minus CE-trained'), ('var_rmse', 'variance RMSE: Pearson minus CE'), ('spearman', 'Spearman of predicted vs true mean: Pearson minus CE')]):
        for nr, c in ((3, 'tab:blue'), (1, 'tab:orange')):
            d = [s[f'{w}|nr{nr}|lam0.01|m{m}|{met}'] for m in M]
            ax[i, j].plot(M, [x['diff_corr_minus_ce'] for x in d], '-o', color=c, label=f'{nr} rating(s)/item'); ax[i, j].fill_between(M, [x['lo'] for x in d], [x['hi'] for x in d], color=c, alpha=.2)
        ax[i, j].axhline(0, color='k', lw=.8); ax[i, j].grid(True); ax[i, j].set_xscale('log', base=2); ax[i, j].set_title(f'{w}\n{tt}', fontsize=8)
        if i == 1: ax[i, j].set_xlabel('training items')
ax[0, 0].legend(fontsize=7); fig.suptitle('Experiment 20 part A (11-21-2026): above 0 = CE better for error columns, below 0 = CE better for Spearman. 200 seeds, 95% interval, ridge 0.01', fontsize=9); fig.tight_layout(); fig.savefig(sys.argv[3], dpi=130)
idx = {(r['pool'], r['n_train_families'], r['ratings_per_artifact'], r['arm']): r for r in real}
P = ['closed_native_15', 'closed_3vote_5', 'open_3vote_39', 'all_59', 'hist_3vote_44']; fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for j, arm in enumerate(['diff_sel_pearson_minus_ce', 'diff_head_corr_minus_ce']):
    for p in P:
        d = [idx[(p, n, 'all', arm)] for n in (3, 7, 15)]; ax[j].errorbar([3, 7, 15], [x['rps'] for x in d], yerr=[[x['rps'] - x['rps_lo'] for x in d], [x['rps_hi'] - x['rps'] for x in d]], marker='o', capsize=3, label=p)
    ax[j].axhline(0, color='k', lw=.8); ax[j].grid(True); ax[j].set_title(arm.replace('diff_', '').replace('_', ' ') + ' (test RPS, all ratings/artifact; >0 = CE better)', fontsize=9); ax[j].set_xlabel('training families')
ax[0].legend(fontsize=7); fig.suptitle('Experiment 20 part B (11-21-2026): real 75 artifacts, 250 family splits, 95% family-bootstrap interval (test weighting only)', fontsize=9); fig.tight_layout(); fig.savefig(sys.argv[4], dpi=130)
