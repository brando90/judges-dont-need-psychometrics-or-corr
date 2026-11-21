"""Plot primary two-ratings-per-answer test losses and the paired RPS improvements."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
with (ROOT / "results/scores.csv").open() as f:
    scores = list(csv.DictReader(f))
with (ROOT / "results/contrasts.csv").open() as f:
    contrasts = list(csv.DictReader(f))
colors = {"ce_fit":"#17667b","pearson_fit":"#b45322","uniform":"#555b61","train_hist":"#6a568f","identity":"#829768"}
names = {"ce_fit":"CE fit","pearson_fit":"Pearson fit","uniform":"Uniform","train_hist":"Training histogram","identity":"Original judge"}
fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.8), sharex=True, sharey="row")
for j, (judge, title) in enumerate((("qwen","Qwen3.5-9B (open weights)"),("claude","Claude Sonnet 5 (closed weights)"))):
    top, bottom = axes[:,j]
    top.set_title(title, fontsize=13, loc="left")
    for method in names:
        vals = sorted((r for r in scores if (r["judge"],r["ratings_per_answer"],r["method"],r["metric"])
                       == (judge,"2",method,"rps")),key=lambda r:int(r["training_ratings"]))
        x = [int(r["training_ratings"]) for r in vals]
        top.plot(x,[float(r["mean"]) for r in vals],marker="o" if method.endswith("fit") else None,
                 lw=2 if method.endswith("fit") else 1.5,ls="-" if method.endswith("fit") else "--",
                 color=colors[method],label=names[method],markersize=4)
    for offset, method in ((-.045, "pearson_fit"),(.045,"uniform")):
        vals = sorted((r for r in contrasts if (r["judge"],r["ratings_per_answer"],r["comparator"],r["metric"])
                       == (judge,"2",method,"rps")),key=lambda r:int(r["training_ratings"]))
        mid=np.array([float(r["improvement"]) for r in vals])
        low=np.array([float(r["lower_one_sided_975"]) for r in vals])
        xx=np.array([int(r["training_ratings"]) for r in vals]) * 2**offset
        bottom.errorbar(xx, mid, yerr=mid-low, fmt="o-", capsize=2, color=colors[method], lw=1.6,
                        ms=4,label=names[method]+" minus CE fit")
    bottom.axhline(0,color="black",lw=.9)
    bottom.set_xlabel("Individual training ratings (two per answer)")
    bottom.set_xscale("log",base=2);bottom.set_xticks(x, labels=[str(k) for k in x])
    for ax in (top,bottom):
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(LogLocator(base=2, subs=[np.sqrt(2)]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True,which="major",color="#d4d8d9",lw=.65)
        ax.grid(True,which="minor",color="#e6e8e8",lw=.35,alpha=.65)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes[0,0].set_ylabel("Held-out ranked probability score\n(lower is better)")
axes[1,0].set_ylabel("Held-out ranked probability score\ncomparator minus CE fit (higher favors CE)")
fig.suptitle("Grade forecasts for unseen homework assignments",x=.09,ha="left",fontsize=17,fontweight="bold")
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=5,loc="upper center",bbox_to_anchor=(.5,.937),frameon=False)
axes[1,1].legend(loc="best",frameon=False,fontsize=8)
fig.text(.09,.018,"160 answers, two graders, ten held-out homework folds. Each objective exhaustively selects one of 27 transforms per fit.\n"
         "Differences use the same assignments and fits; bars are approximate overlap-adjusted 95% t intervals. CE = cross-entropy.",fontsize=9)
fig.subplots_adjust(top=.83,bottom=.145,left=.1,right=.98,hspace=.22,wspace=.13)
fig.savefig(ROOT/"results/heldout_rps.png",dpi=170)
