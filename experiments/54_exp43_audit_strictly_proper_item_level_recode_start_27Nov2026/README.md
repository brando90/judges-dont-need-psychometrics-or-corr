# Experiment 54: the Experiment 43 audit recoded for item-level strictly proper scores (27 Nov 2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/54_exp43_audit_strictly_proper_item_level_recode_start_27Nov2026/README.md>

> **Note 11-28-2026 (QA; the coding is unchanged).** (1) The binned Jensen–Shannon divergence of Elangovan et al. is defined in their Sec. 3.2 (Eqs. 1 and 2); the table cell below was corrected from Sec. 3.1. (2) The six papers coded P_div or P_strict are Elangovan et al., Guerdan et al., Chen et al., Wang et al., Li et al. and Lyu et al. (43 of 49 report no item-level proper score or divergence: 30 direct or benchmark papers and 13 pairwise or reward-model papers). The audit supports a statement about how many papers report such a score; it does not support "no metric compares judges with rating distributions", which is why the paper credits all six.


**TLDR:** Experiment 43's code P counts any "proper score or distributional divergence against item-level human ratings". By that code, 43 of 49 papers report none. The advisor's narrower claim needs a stricter code: an item-level, *strictly proper* score. Under it, 44 of 49 papers report none. Elangovan et al.'s binned Jensen–Shannon divergence compares pooled distributions within bins of items, so it is neither item-level nor proper. Both papers the advisor named are in the audit. Elangovan et al. is coded P in Experiment 43, and P_div here. Li et al. is coded P, and P_strict here (Brier and log scores on pairwise preference distributions). Of the five strict papers, only Lyu, Do and Pergola (arXiv, Oct 2026) use an order-aware score (RPS) on ordinal ratings. Chen et al. and Guerdan et al. use KL or cross-entropy on ordinal scales, which is strictly proper but ignores category order. So "no item-level, strictly proper score on ordinal scales" is false as an absolute claim. The defensible version is "rare: 5 of 49, and 1 of 49 order-aware".

**As of:** 11-27-2026 (Friday, Pacific time). Literature recode only; no model calls. Experiment 43's files are read, not edited, and its broad counts are unchanged.

## How the 49 were selected (from Experiment 43, Section 2)

The 49 form a purposive convenience sample of papers or tools that propose or validate an LLM judge, an automatic text-quality metric, or a judge-based benchmark. Sources were the canonical set named in the original brief, targeted searches (arXiv, ACL Anthology, OpenReview), the AlpacaEval family and a later literature sweep. Highly cited 2020–2026 work is over-represented. The sample is not a systematic review, and there was a single coder.

## The six papers outside the 43 (Experiment 43 group X, distribution-aware)

| paper | Experiment 43 codes | strict recode | what it reports |
|---|---|---|---|
| Elangovan et al., ICLR 2025 (2410.03775) | P,C,K | P_div | binned JSD between pooled human and judge label distributions within bins of items grouped by median human rating (Sec. 3.2, Eqs. 1–2) |
| Guerdan et al., NeurIPS 2025 (2503.05965) | P,A,K | P_strict | KL(human‖judge) and cross-entropy per item, on categorical and ordinal scales |
| Chen et al., NeurIPS 2025 (2505.12301) | P,A | P_strict | KL(p‖q) per item (Eq. 10) on SNLI/MNLI and SummEval 1–5 Likert |
| Wang, Zhang & Choi, Findings EMNLP 2025 (2503.03064) | P,A,C | P_strict | MSE of the judge's preference probability against the averaged human preference (binary Brier) |
| Li et al., Findings EMNLP 2026 (2510.00263) | P | P_strict | MSE of the stated P(B>A) to each item's ten-vote mean (Sec. 5.1), a Brier-type strictly proper comparison; votes are Gemini-2.5-Flash persona annotations, not human (Sec. 4), and the Table 1 Brier/ECE use the binarized majority (App. C.1); kept as P_strict, the conservative choice for the gap count; ordinal left to future work (App. E) |
| Lyu, Do & Pergola, arXiv Oct 2026 (2610.37868) | P,C | P_strict, ordinal RPS | annotation-level RPS (ordinal) or Brier (nominal), plus JS distance as the primary metric |

The 43 others are 25 direct-score judges and metrics, 5 judge-based benchmarks and 13 pairwise judges or reward models.

Evidence: the Elangovan, Chen and Guerdan rows were re-read from the arXiv PDFs on 11-27-2026 (`pdftotext`, sections as cited). The Wang, Li and Lyu rows use Experiment 43's PDF reading.

## Reproduce

`python3 strict_tally.py`. It reads `../43_*/papers.py` read-only, asserts that Experiment 43's P set has not changed, and is deterministic. Its output, identical on two runs, is in [`strict_tally_output.txt`](strict_tally_output.txt).
