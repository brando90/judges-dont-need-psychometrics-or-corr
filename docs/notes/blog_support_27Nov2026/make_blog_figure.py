#!/usr/bin/env python3
"""Figure for the evidence-led blog post: five frozen frontier judges on DICES-350, top-1 agreement vs skill over the human base rate.

Reads only committed results of experiment 46 (results/judge_skill_ci.json): per judge the top-1 agreement with the human majority
rating and the normalized-RPS skill over the cross-fitted human base rate with its 95% item-clustered bootstrap interval (B = 4,000).
The dashed line is the top-1 agreement of a constant forecast that always answers No, 269/350 (experiment 28 method hist; recomputed from the
public DICES-350 file by check_always_no_dices350.py). No new computation. Every axis has ax.grid(True). Colors: series 1 and text tokens of the dataviz reference palette (light mode,
surface #ffffff; the single series color passed the palette validator).

Run: python3 -I make_blog_figure.py [repo_root] [out_png]
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "dices350_agreement_vs_skill.png"
SRC = sorted(glob.glob(str(ROOT / "experiments" / "46_*" / "results" / "judge_skill_ci.json")))[0]
ALWAYS_NO = 100 * 269 / 350  # always answering No: unique human modal label No on 269 of 350 conversations, ties count as wrong
NAMES = {"gpt-6-astra": "GPT-6 Astra", "claude-opus-5-5": "Claude Opus 5.5", "kimi-k3": "Kimi K3", "glm-5.3": "GLM 5.3", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash"}

BLUE, INK, INK2, GRID, SURFACE = "#257de0", "#191919", "#666666", "#e6e6e6", "#ffffff"
rows = [e for e in json.load(open(SRC)) if e["set"] == "dices350"]
rows.sort(key=lambda e: -e["skill_raw"])  # best skill on top
y = list(range(len(rows) - 1, -1, -1))

plt.rcParams.update({"font.family": "Inter", "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 4.1), sharey=True, gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.06}, facecolor=SURFACE)

# left: top-1 agreement with the human majority rating
t1 = [100 * e["top1"] for e in rows]
a1.scatter(t1, y, s=46, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.2)
for xv, yv in zip(t1, y):
    a1.annotate(f"{xv:.1f}%", (xv, yv), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
a1.axvline(ALWAYS_NO, color=INK2, linestyle="--", linewidth=1.3, zorder=2)
a1.annotate(f"always answering No: {ALWAYS_NO:.1f}%", (ALWAYS_NO, -0.52), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=INK2)
a1.set_xlim(74, 86)
a1.set_xticks(range(74, 87, 2))
a1.set_yticks(y)
a1.set_yticklabels([NAMES[e["judge"]] for e in rows])
a1.set_ylim(-0.7, len(rows) - 0.3)
a1.set_xlabel("Top-1 agreement with the human majority rating\n(DICES-350, 350 conversations; higher is better)", fontsize=9)
a1.set_title("Agreement: no pair of judges differs significantly", fontsize=9.5, color=INK, loc="left", pad=8)

# right: skill over the human base rate, 95% interval
sk = [e["skill_raw"] for e in rows]
lo = [e["skill_raw_ci"][0] for e in rows]
hi = [e["skill_raw_ci"][1] for e in rows]
a2.axvline(0, color=INK2, linewidth=1.2, zorder=2)
a2.hlines(y, lo, hi, color=BLUE, linewidth=2.0, zorder=3)
a2.scatter(sk, y, s=46, color=BLUE, zorder=4, edgecolor=SURFACE, linewidth=1.2)
for xv, yv in zip(sk, y):
    a2.annotate(f"{xv:+.3f}".replace("-", "−"), (xv, yv), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
a2.set_xlim(-0.27, 0.12)
a2.set_xlabel("Normalized-RPS skill over the human base rate, with 95% interval\n(DICES-350; higher is better; 0 = no better than the base rate)", fontsize=9)
a2.set_title("Skill: eight of ten pairs differ significantly", fontsize=9.5, color=INK, loc="left", pad=8)
a2.annotate("\u2190 worse than the human base rate", (0, -0.52), xytext=(-5, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK2)
a2.annotate("better \u2192", (0, -0.52), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=INK2)

for ax in (a1, a2):
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.tick_params(length=3)

fig.suptitle("Five frozen frontier judges on DICES-350: near-identical agreement, very different skill", x=0.012, ha="left", fontsize=11, color=INK, y=0.995)
fig.text(0.012, 0.012, "Skill = 1 − (judge's normalized RPS ÷ base-rate forecast's normalized RPS); the base rate is the cross-fitted histogram of the other items' human ratings.\n"
         "Judges used as given (verbalized probabilities, temperature 0), no training. Intervals: item-clustered bootstrap, B = 4,000. Pairwise tests: Holm-adjusted, p < 0.05.\n"
         "Dashed line: a constant forecast that always answers No (No is the unique human majority label on 269 of 350 conversations; ties count as wrong).",
         ha="left", va="bottom", fontsize=7.4, color=INK2)
fig.subplots_adjust(left=0.165, right=0.985, top=0.80, bottom=0.30)
fig.savefig(OUT, dpi=200, facecolor=SURFACE)
print("wrote", OUT)
