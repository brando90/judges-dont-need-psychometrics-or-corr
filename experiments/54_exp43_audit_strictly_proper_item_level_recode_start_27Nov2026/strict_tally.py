"""Experiment 54: stricter recode of the Experiment 43 audit (49 papers), answering the advisor's note S2/S8.

Reads ../43_llm_judge_validation_practice_survey_start_27Nov2026/papers.py READ-ONLY. Experiment 43's code P lumps
"proper score OR distributional divergence against item-level human ratings". The advisor's narrower claim is about an
item-level, STRICTLY proper score. This script splits P into:
  P_strict : an item-level strictly proper score against human ratings (log score / CE, KL(human || judge), Brier or MSE to a
             soft label, RPS),
  P_div    : a divergence that is not a proper score of the judge's item-level forecast (e.g. binned JSD on pooled bins),
and records, for P_strict papers, whether the score is order-sensitive on an ordinal scale (RPS) or not.
Each recode cites what was read in the paper's PDF on 11-27-2026 (pdftotext, section named in EVIDENCE).
Deterministic: python3 strict_tally.py (run twice; identical output).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "43_llm_judge_validation_practice_survey_start_27Nov2026"))
from papers import R  # noqa: E402

# arXiv id -> (strict code, order-sensitive ordinal score?, evidence)
RECODE = {
    "2410.03775": ("P_div", False, "Elangovan et al. (ICLR 2025), Sec. 3.1 Eq. 1-2: binned JSD between pooled human and machine label PMFs within bins of items grouped by median human rating; a population-level divergence, not an item-level proper score of the judge's forecast."),
    "2503.05965": ("P_strict", False, "Guerdan et al. (NeurIPS 2025): KL-D(h,j) and cross-entropy between human and judge response distributions per item (categorical and ordinal scales); KL(human||judge) is CE minus a judge-free term, hence strictly proper; not order-sensitive."),
    "2505.12301": ("P_strict", False, "Chen et al. (NeurIPS 2025), Eq. 10: KL(p||q_theta) from the human distribution to the model's, per item, on SNLI/MNLI and SummEval 1-5 Likert; strictly proper, not order-sensitive."),
    "2503.03064": ("P_strict", False, "Wang, Zhang & Choi (Findings EMNLP 2025): MSE between the judge's preference probability and the averaged human preference, i.e. a binary Brier score against a soft label; strictly proper, binary."),
    "2510.00263": ("P_strict", False, "Li et al. (Findings EMNLP 2026): evaluated by MSE between the predicted P(B>A) and each item's mean of ten votes (Sec. 5.1; squared error to the vote mean, a Brier-type strictly proper comparison); the votes are Gemini-2.5-Flash persona annotations, not human (Sec. 4); Table 1 Brier/ECE are against the binarized majority (App. C.1); trained with Brier/log rewards (Eqs. 1-2); human PandaLM scored by majority agreement (Sec. 5.3); ordinal left to future work (App. E). Kept as P_strict, the conservative choice for our gap count"),
    "2610.37868": ("P_strict", True, "Lyu, Do & Pergola (arXiv, 27 Oct 2026): annotation-level RPS on ordinal labels (Brier on nominal), plus Jensen-Shannon distance as the primary metric; RPS is strictly proper and order-sensitive."),
}


def main():
    n = len(R)
    p_rows = [r for r in R if "P" in r[5].split(",")]
    assert {r[6] for r in p_rows} == set(RECODE), "Experiment 43's P set changed; update RECODE"
    strict = [r for r in p_rows if RECODE[r[6]][0] == "P_strict"]
    ordinal = [r for r in strict if RECODE[r[6]][1]]
    print(f"Papers coded: {n}")
    print(f"Experiment 43 broad code P (proper score or divergence, item level): {len(p_rows)}/{n}; no P: {n - len(p_rows)}/{n}")
    print(f"Strict: item-level strictly proper score: {len(strict)}/{n}; none: {n - len(strict)}/{n}")
    print(f"Strict and order-sensitive on an ordinal scale (RPS): {len(ordinal)}/{n}")
    print("The six broad-P papers (the 49 - 43):")
    for r in p_rows:
        code, ordn, ev = RECODE[r[6]]
        print(f"  - {r[0]} ({r[2]}; {r[6]}): Exp43 codes {r[5]} -> {code}{' (ordinal RPS)' if ordn else ''}. {ev}")
    groups = {}
    for r in R:
        if "P" not in r[5].split(","):
            groups.setdefault(r[3].split("+")[0], 0)
            groups[r[3].split("+")[0]] += 1
    print("The 43 without any item-level proper score or divergence, by Experiment 43 group:", dict(sorted(groups.items())))


if __name__ == "__main__":
    main()
