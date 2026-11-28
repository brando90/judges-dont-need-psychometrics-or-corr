# One-slide note for the 10 December 2026 defense: the ongoing evaluation work

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/DEFENSE_ONE_SLIDE_EVAL_NOTE.md>

**TLDR:** A standalone note for the single "ongoing work" slide about this project. It holds a slide title, three bullets, one headline table to show, and a 45-second script. The defense deck itself is not touched.

**As of:** 11-27-2026 (Friday, Pacific time).

## Slide title

Ongoing: scoring an AI judge as a forecaster of the human rating distribution

## Three bullets

1. The target is a distribution of human ratings, not one number (Pavlick and Kwiatkowski 2019; Plank 2022). The open question is the ruler: forecasts with the same average tie on correlation and squared error, so correlation cannot identify that distribution in general (machine-checked in Lean for finite laws); a strictly proper score matched to the rating scale can.
2. Practice has not moved: 43 of 49 judge-validation papers report no item-level proper score, and 29 of 30 score-based papers validate by correlation (a purposive sample, one coder).
3. On public rating sets, frontier judges' skill over the base rate falls from 0.16 to 0.91 on high-agreement items to minus 0.38 to 0.11 on contested ones, while their Pearson correlation on contested NLI items stays at 0.38 to 0.86. Whether real judge rankings reverse under a proper score is still being tested and is not claimed.

## Headline table to show

The "Arm B, frontier judges, Q1 (most agreement) to Q5 (least)" table in [`experiments/50_stratification_by_human_agreement_proper_gap_vs_corr_start_27Nov2026/README.md`](experiments/50_stratification_by_human_agreement_proper_gap_vs_corr_start_27Nov2026/README.md): skill over the base rate, Pearson and top-1 agreement for five frontier judges on five rating sets, by agreement fifth. Data: public multi-annotator rating sets (ChaosNLI SNLI, MNLI and alphaNLI; DICES-350 and DICES-990), five frozen frontier judges, saved forecasts, no fitting. Skill is one minus the judge's held-out proper-score risk over the base rate's; zero means no better than the base rate, one is perfect.

## 45-second script

This is ongoing work on a question under all three of my papers: how do we know an AI judge can be trusted? The field checks judges by correlation with human scores. But the human target is not one number, it is a distribution of ratings, and correlation cannot identify that distribution. Two judges with the same average tie on correlation, which we machine-checked in Lean. A proper score matched to the rating scale does tell them apart. In a forty-nine-paper audit, forty-three papers use no such score. On public rating sets, frontier judges look fine by correlation but add nothing over the base rate on contested items. Whether real judge rankings flip is still running, so I am not claiming that yet.

**TLDR-end:** [judge objectives: defense slide note] One slide: the target is a distribution (not our claim); correlation cannot identify it in general (Lean-checked); a proper score can; 43 of 49 papers do not use one; skill by agreement stratum shows where judges stop being useful; real-judge rank reversal is not yet claimed.
