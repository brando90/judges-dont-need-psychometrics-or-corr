"""Print the metric tally and the Markdown table for the survey in README.md.

Deterministic: reads only papers.py. Run: python3 tally.py
Codes: C = correlation (Pearson, Spearman, Kendall); A = raw agreement, accuracy, precision/recall/F1;
K = chance-corrected agreement (Cohen/Fleiss kappa, Scott's pi, Krippendorff alpha);
P = proper score or distributional divergence (CE/NLL, KL, JSD, Brier/MSE vs. soft label) against item-level human ratings;
Ps = proper score at the system (model-pair) level only.
Groups: D = direct-score judge or automatic metric; S = judge-based benchmark validated against a human leaderboard;
W = pairwise-preference judge or reward model; X = disagreement/distribution-aware judge validation.
"""
from collections import Counter
from papers import R

def codes(r):
    return r[5].split(",")

def link(i):
    if i.startswith("ACL:"):
        return f"https://aclanthology.org/{i[4:]}/"
    return f"https://arxiv.org/abs/{i}"

def main():
    n = len(R)
    print(f"Papers coded: {n}")
    for c, name in [("C", "correlation"), ("A", "raw agreement/accuracy"), ("K", "chance-corrected agreement"),
                    ("P", "item-level proper score / distributional divergence"), ("Ps", "system-level proper score")]:
        k = sum(c in codes(r) for r in R)
        print(f"  {name} ({c}): {k}/{n} = {100*k/n:.0f}%")
    print(f"  correlation is the only metric family: {sum(codes(r)==['C'] for r in R)}/{n}")
    print(f"  no item-level proper score: {sum('P' not in codes(r) for r in R)}/{n}")
    for g, name in [("D", "direct-score judges and metrics"), ("S", "system-level benchmarks"),
                    ("W", "pairwise judges and reward models"), ("X", "distribution-aware")]:
        rs = [r for r in R if r[3].split("+")[0] == g]
        cnt = Counter(x for r in rs for x in codes(r))
        print(f"  group {g} ({name}): {len(rs)} papers; " + ", ".join(f"{k}={cnt[k]}" for k in ["C", "A", "K", "P", "Ps"]))
    ds = [r for r in R if r[3].split("+")[0] in ("D", "S")]
    print(f"  direct-score + system-level papers reporting correlation: {sum('C' in codes(r) for r in ds)}/{len(ds)}")
    w = [r for r in R if r[3].split("+")[0] == "W"]
    print(f"  pairwise papers reporting agreement/accuracy or kappa: {sum(bool({'A','K'} & set(codes(r))) for r in w)}/{len(w)}")
    print()
    print("| # | Paper | Year | Venue | Judge type | Validation metric(s) reported | Codes | Link |")
    print("|---|---|---|---|---|---|---|---|")
    for j, r in enumerate(R, 1):
        print(f"| {j} | {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | [{r[6]}]({link(r[6])}) |")

if __name__ == "__main__":
    main()
