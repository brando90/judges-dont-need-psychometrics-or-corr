# Experiment 55 protocol: does the judge verdict depend on which strictly proper rule is used? (frozen 11-27-2026, before the analysis script was run)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/55_all_three_proper_rules_skill_ranking_strata_power_start_27Nov2026/PROTOCOL.md>

**TLDR:** The thesis is a strictly proper score matched to the label encoding, not "use RPS". Experiments 44, 46 and 50 score each set with its matched rule (multiclass Brier for the unordered NLI sets and the binary alphaNLI set, normalized ranked probability score for the ordinal DICES sets) and cross-entropy as secondary. This experiment reads the same saved forecasts and reports risk, base-rate skill, judge ranking, human-agreement strata and paired-comparison power under all three rules on every set, with the matched rule labeled, so a reader can see what changes when the rule changes. CPU only, $0, no model calls.

**As of:** 11-27-2026 (Friday, Pacific time). Written by Claude (Sonnet 5.5) in Claude Code for Brando Miranda, to close the "all three proper rules" item of the generality addendum (`docs/prompts/claude_code_paper1_prompt_corrected_27Nov2026.md`). Experiments 44, 46 and 50 are read, not edited.

## Rule-to-encoding table (stated before any score is computed)

| set | labels | encoding | matched rule(s) | other rules |
|---|---|---|---|---|
| ChaosNLI SNLI, MNLI-m (K = 3) | entailment / neutral / contradiction, unordered | one-hot | multiclass Brier and cross-entropy (CE) | normalized RPS is **not matched** (the label order is arbitrary); reported as a sensitivity diagnostic only |
| ChaosNLI alphaNLI (K = 2) | binary | one-hot | Brier and CE (normalized RPS equals Brier divided by 2 here, so it is matched and adds nothing) | none |
| DICES-350, DICES-990 (K = 3) | No < Unsure < Yes, ordinal | cumulative thresholds | normalized RPS (distance-sensitive) | Brier and CE are proper but ignore the order |

Scores for a floored forecast q (sums to 1, floor 1e-6 as in Experiments 28, 44 and 46) and a rating y: Brier = sum_k (q_k - 1{y = k})^2; CE = -log q_y; normalized RPS = (K - 1)^-1 sum_{m < K-1} (Q_m - 1{y <= m})^2 with Q the cumulative forecast. The per-item risk is the mean over the item's ratings (unbiased per-sample risk, valid at one rating per item); the set risk is the mean over items. Lower is better.

## Data

The five frozen frontier judges of Experiment 28 (gpt-6-astra, claude-opus-5-5, kimi-k3, glm-5.3, deepseek-v4.1-flash), temperature 0, one verbalized-distribution forecast per item, invalid forecasts replaced by the uniform forecast (the Experiment 28, 44 and 46 rule), on the five sets above, through Experiment 44's data layer (`experiments/44_*/src/lib44.py`, bundle SHA-256 recorded in the output). Items and ratings are exactly Experiment 44's. No fitting happens here, so there is no train/test split; every item is scored once. The analysis uses ratings that Experiments 44, 46 and 50 already scored.

## What the analyst had seen before freezing

- Experiment 46's `results/judge_scores.csv` lists each judge's raw risk under normalized RPS, CE and Brier on every set (so the three raw risks of the five judges on DICES-350 were seen), and its skill under the matched rule. The base-rate risks under the unmatched rules, the strata under CE and Brier on DICES and CE on NLI, and the power under the unmatched rules were not seen. H1 below is therefore a prediction made with partial sight of the raw risks; its base-rate half was not seen.
- Experiment 46's frontier-panel result under the matched rules (4 of 5 judges below the base rate on DICES-350) and Experiment 50's matched-rule strata ranges were read.

## Quantities and hypotheses

Seeds: numpy `SeedSequence([20261127, 55, ...])`. Resampling unit: the item (items drawn with replacement, 4,000 draws, percentile 95% intervals). A two-sided bootstrap p-value is p = min(1, 2 min(P*(skill <= 0), P*(skill >= 0))) with the (B + 1) correction. Holm-Bonferroni is applied over all 75 declared cells (5 sets x 3 rules x 5 judges). A judge is **significantly below the base rate** under a rule when its point skill is below 0 and its Holm-adjusted p is below 0.05.

- **Base rate.** The marginal rating histogram of the set, cross-fitted: items are split into 5 random folds (SeedSequence([20261127, 55, 0, set index])), and each fold's base forecast is the floored histogram of the other four folds' ratings. (Experiment 46 cross-fits by group, for example by SNLI premise; this experiment has no group labels, so folds are by item, and base-rate risks differ from Experiment 46's at about the third decimal.) Skill = 1 - judge risk / base risk, per rule.
- **Q1. Skill under each rule** with item-bootstrap intervals and Holm-adjusted p (75 cells).
- **Q2. Ranking.** Per set, the judge ordering by skill under each rule, Kendall tau-b between each pair of rules, whether the best judge is the same under all three rules, and the number of the 10 judge pairs whose order differs between two rules (point estimates, descriptive).
- **Q3. Strata.** The Experiment 50 definition: items with at least 4 ratings are split at random into half A (defines agreement through its Shannon entropy in nats) and half B (scores); items fall into five equal-count entropy quintiles within each set (Q1 = most agreement, Q5 = least); skill is computed per quintile with the same cross-fitted base forecast scored on half B. The random split uses SeedSequence([20261127, 55, 1, set index, item index]), so it is not the same split as Experiment 50's and the numbers can differ within noise.
- **Q4. Power.** Minimum detectable difference (MDD) for a paired judge comparison at N = 400 items and k = 1 rating per item, 80% power, 5% two-sided level, from Experiment 44's variance components (`pair_components`, `variance_components`, `mdd`), per rule, as the median and range over the 10 judge pairs of a set.
- **H1 (the base-rate verdict is not an artifact of RPS).** On DICES-350, at least 3 of the 5 judges are significantly below the base rate under each of the three rules (normalized RPS, Brier, CE). Experiment 46 found 4 of 5 under normalized RPS. H1 is false if any rule gives fewer than 3.
- **H2 (strata direction).** Under each of the three rules, the best judge's skill in Q1 exceeds its skill in Q5 on all five sets (15 of 15 set-by-rule cells). H2 is false if any cell fails.
- Descriptive, no hypothesis: Q2 and Q4. Both are reported whether or not they look favorable.

## Not tested, and what would change the reading

No claim is made about which rule a practitioner should prefer beyond the matching table. A set where the rules disagree about the best judge is reported as such. Local judges, SummEval and the API panel of Experiment 46 are out of scope here (no frozen frontier forecasts of all five judges on SummEval). Failures and missing cells are reported.

## Outputs

`results/skill_all_rules.csv`, `results/rank_agreement.csv`, `results/strata_all_rules.csv`, `results/mdd_all_rules.csv`, `results/summary.json`, `results/key_numbers.md`; `tests/test_exp55.py`; `README.md` and `HOW_TO_READ_RESULTS.md` written after the first run. Amendments after this freeze go in `AMENDMENT_A1.md` with a date.
