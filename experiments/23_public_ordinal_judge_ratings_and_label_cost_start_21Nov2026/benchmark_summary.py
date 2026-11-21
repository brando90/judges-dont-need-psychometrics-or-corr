"""Build the cross-domain scores and figure from the separately registered studies."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TRACKS = [
    ("SummEval · factual consistency", "results/gpt4_consistency", "40 articles / 640 summaries"),
    ("KonIQ-10k · image quality", "results/image_quality", "2,015 photographs"),
    ("USR · Topical-Chat, published scores", "usr/results/topical_usr", "24 contexts / 120 responses"),
    ("USR · Topical-Chat, lexical features", "usr/results/topical_lexical", "24 contexts / 144 responses"),
    ("USR · PersonaChat, lexical features", "usr/results/persona_lexical", "24 contexts / 120 responses"),
    ("MovieLens 100K · submitted stars", "movielens/results/all_users", "400 titles"),
]
METHODS = {
    "ce": ("CE fit", "#176479","o"),
    "pearson": ("Pearson fit", "#a44829","v"),
    "uniform": ("Uniform", "#666e7c","s"),
    "histogram": ("Training histogram", "#a06a13","D"),
    "calibrated_pearson": ("Pearson + calibration", "#534098","^"),
}
ALIASES = {"direct_ce":"ce", "direct_pearson":"pearson", "train_histogram":"histogram"}


def read_max(directory):
    with (ROOT / directory / "scores.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("depth", "1") == "1"]
    budget = max(int(r["budget"]) for r in rows)
    result = {}
    for row in rows:
        method = ALIASES.get(row["method"], row["method"])
        if int(row["budget"]) == budget and method in METHODS:
            result[method] = [float(row["rps"]),float(row.get("ce",row.get("ce_nats"))),float(row["tvd"])]
    assert result.keys() == METHODS.keys()
    return budget, result


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.5), sharey=True)
    fig.subplots_adjust(left=.275, right=.992, bottom=.30, top=.79, wspace=.24)
    metrics = ["Test ranked probability\nscore (RPS)", "Test cross-entropy\n(CE, nats per grade)",
               "Test total variation\ndistance (TVD)"]
    observations = [read_max(path) for _, path, _ in TRACKS]
    lines = ["# Distribution scores across the public human-rating benchmarks | 11-21-2026", "",
             "As of 11-21-2026. All numbers are held-out test losses from 64 matched draws, one human rating per training item. The first two comparisons were prespecified together. The remaining comparisons were added after the first two results, with separate protocols published before fitting each archive. Each row uses the largest tested budget within its task. All three losses are lower-is-better. Uniform assigns probability 0.2 to each of five grades.", "",
             "| Held-out rating population | Test units | Training grades | Fitted procedure | Ranked probability score (RPS) | Cross-entropy (CE), nats | Total variation distance (TVD) |",
             "|---|---|---:|---|---:|---:|---:|"]
    for i, ((name,path,units),(budget,scores)) in enumerate(zip(TRACKS,observations)):
        for j,(method,(label,color,marker)) in enumerate(METHODS.items()):
            values = scores[method]
            lines.append(f"| [{name}]({path}/scores.csv) | {units} | {budget:,} | {label} | " +
                         " | ".join(f"{v:.5f}" for v in values)+" |")
            for ax, value in zip(axes, values):
                ax.scatter(value, -i+(2-j)*.083, s=34, color=color, marker=marker,
                           label=label if i==0 else None, zorder=3)
    for k, ax in enumerate(axes):
        ax.set_xlabel(metrics[k], fontsize=10)
        ax.set_yticks(-np.arange(len(TRACKS)),[f"{n}\n{b:,} training grades" for (n,_,_), (b,_) in zip(TRACKS,observations)],fontsize=10)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", alpha=.22)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(axis="y",length=0,pad=9)
        ax.margins(y=.09)
    fig.suptitle("How well do fitted judges predict the grade a person gives?",y=.963,x=.035,ha="left",fontsize=18,fontweight="bold")
    fig.text(.035,.883,"Held-out distribution losses at the largest tested training budget in each setting. Lower is better on all three axes.",fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower left",bbox_to_anchor=(.026,.105),ncol=5,frameon=False,handletextpad=.3,columnspacing=1.2)
    fig.text(.035,.08,"Every method uses the same human training grades. CE or Pearson selects from the fixed forecast library by exhaustive search. Pearson + calibration then estimates grade frequencies by score quartile.\n"
             "SummEval and KonIQ were prespecified primary benchmarks. USR and MovieLens are exploratory replications. Each point averages 64 matched training draws. RPS and empirical-histogram TVD range from 0 to 1.",fontsize=8.4,va="top")
    for ext in ("png","pdf"):
        metadata={"CreationDate":None,"ModDate":None} if ext=="pdf" else {}
        fig.savefig(ROOT/"figures"/f"cross_domain_test_scores.{ext}",dpi=155,facecolor="white",metadata=metadata)
    plt.close(fig)
    lines += ["", "RPS averages squared cumulative probability error over four grade thresholds (0–1). CE is negative log probability per individual rating in nats. TVD compares the forecast to an item's empirical grade frequencies (0–1). The generative language benchmark uses archived GPT-4 probabilities. For the other applications, fixed feature-based judges use metadata, text features, or published metric scores as specified in each protocol. Paired uncertainty belongs to comparisons on the same held-out observations; comparing absolute scores across different human populations also reflects differences in their grade distributions.", "",
              "[Primary comparisons and intervals](BUDGET_TABLES_11-21-2026.md) · [Dialogue methods and results](usr/RESULTS_11-21-2026.md) · [MovieLens methods and results](movielens/RESULTS_11-21-2026.md) · [Cross-domain figure](figures/cross_domain_test_scores.pdf)"]
    (ROOT/"CROSS_DOMAIN_SCORES_11-21-2026.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
