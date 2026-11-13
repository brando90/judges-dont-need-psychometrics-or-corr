#!/usr/bin/env python3
"""TLDR: Draw results/fig_reliability_vs_rps.png from saved Phase A and Phase C outputs.

Panel A (synthetic, exact): tempered forecasts q_T ~ p^(1/T); x = relative coefficient Erho2 of one
sampled rating, y = expected RPS against the true distribution. Panel B (historical, retrospective):
five cached judges; x = one-call test-retest Erho2, y = held-out test RPS against expert ratings,
95% task-bootstrap intervals on both axes.

Run: python experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/scripts/make_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RES = HERE / 'results'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SOL, OPUS = '#2a78d6', '#eb6834'   # validated categorical slots 1-2 (all-pairs, light)


def main():
    a = json.loads((RES / 'phase_a.json').read_text())
    c = json.loads((RES / 'phase_c.json').read_text())
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.titlesize': 10, 'axes.titleweight': 'bold'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    fam = a['A3']['tempered_family']
    names = {'2': 'T = 2 (flatter)', '1': 'T = 1: true distribution', '1/2': 'T = 1/2', '1/4': 'T = 1/4',
             'mode': 'T → 0 (mode)'}
    xs = [fam[t]['sampled_Erho2']['1'] for t in names]
    ys = [fam[t]['mean_expected_rps'] for t in names]
    ax1.plot(xs, ys, color=INK2, linewidth=1.5, zorder=1)
    for t, x, y in zip(names, xs, ys):
        oracle = t == '1'
        ax1.scatter([x], [y], s=70 if oracle else 40, color=INK if oracle else INK2, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax1.annotate(names[t], (x, y), xytext={'1': (8, -12), '1/2': (8, -12), 'mode': (-78, -4)}.get(t, (6, 8)), textcoords='offset points',
                     color=INK if oracle else INK2, fontsize=8.5, fontweight='bold' if oracle else 'normal')
    ax1.set_xlim(0.15, 1.02)
    ax1.set_ylim(0.11, 0.152)
    ax1.set_title('A. Synthetic: sharper forecasts are more repeatable\nbut predict the human distribution worse', loc='left')
    ax1.set_xlabel('Reliability of one sampled rating, relative coefficient Eρ²\n(synthetic population, exact; higher = more repeatable)')
    ax1.set_ylabel('Expected RPS against the true rating distribution\n(synthetic, exact; lower = better)')

    m = c['sets']['test']['metrics']
    judges = [('H0_sol', SOL, 'o'), ('H1_sol', SOL, 'o'), ('H2_sol', SOL, 'o'), ('H0_opus', OPUS, 's'), ('H2_opus', OPUS, 's')]
    offsets = {'H0_sol': (-50, -4), 'H1_sol': (-44, 6), 'H2_sol': (-44, 7), 'H0_opus': (-52, 4), 'H2_opus': (-30, -17)}
    for j, col, mk in judges:
        x, y = m[j]['Erho2_1'], m[j]['RPS']
        ax2.errorbar(x['point'], y['point'], xerr=[[x['point'] - x['ci95'][0]], [x['ci95'][1] - x['point']]],
                     yerr=[[y['point'] - y['ci95'][0]], [y['ci95'][1] - y['point']]], fmt='none', ecolor=col, alpha=0.45,
                     elinewidth=1, capsize=0, zorder=1)
        ax2.scatter([x['point']], [y['point']], s=55, marker=mk, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax2.annotate(j, (x['point'], y['point']), xytext=offsets[j], textcoords='offset points', color=INK, fontsize=8.5,
                     bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.6))
    base = m['baseline_train_marginal']['RPS']['point']
    hum = m['human']['Erho2_1']['point']
    ax2.axhline(base, color=INK2, linestyle='--', linewidth=1)
    ax2.text(0.505, base - 0.011, f'constant train-marginal forecast: {base:.3f}', color=INK2, fontsize=8)
    ax2.axvline(hum, color=INK2, linestyle=':', linewidth=1)
    ax2.text(hum + 0.005, 0.385, f'two experts agree\nwith each other: {hum:.2f}', color=INK2, fontsize=8, va='top')
    ax2.scatter([], [], marker='o', color=SOL, label='GPT-6 Sol (high reasoning)')
    ax2.scatter([], [], marker='s', color=OPUS, label='Claude Opus 5.5 (max effort)')
    ax2.legend(loc='upper left', frameon=False, fontsize=8)
    ax2.set_xlim(0.5, 1.01)
    ax2.set_ylim(0.11, 0.39)
    ax2.set_title('B. Historical judges: the most repeatable judge\nhad the worst held-out RPS', loc='left')
    ax2.set_xlabel('Judge test–retest reliability Eρ² for one call\n(historical test split: 30 artifacts, 10 tasks; previously exposed)')
    ax2.set_ylabel('Held-out test RPS of the 3-call histogram vs expert ratings\n(lower = better; bars: 95% task bootstrap)')
    fig.tight_layout()
    out = RES / 'fig_reliability_vs_rps.png'
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)


if __name__ == '__main__':
    main()
