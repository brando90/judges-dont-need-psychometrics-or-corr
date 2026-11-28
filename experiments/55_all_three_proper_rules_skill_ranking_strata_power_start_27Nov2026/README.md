# Experiment 55: does the judge verdict depend on which strictly proper rule you use? (27 Nov 2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/55_all_three_proper_rules_skill_ranking_strata_power_start_27Nov2026/README.md>

**TLDR:** No, not for the base-rate verdict, and only a little for the ranking. On the five frozen frontier judges and five public rating sets, both pre-registered predictions held. On DICES-350, four of five judges are significantly below the base rate under normalized ranked probability score (RPS), under Brier and under cross-entropy (CE) alike (H1: 4, 4, 4). Skill falls from the most-agreed to the least-agreed fifth of items for the best judge in all 15 set-by-rule cells (H2: 15 of 15). The thesis is a strictly proper score matched to the label encoding, so the table says which rule matches which set; RPS is the matched rule only for the two ordinal DICES sets (and equals Brier divided by two on binary alphaNLI), and on the unordered NLI sets it is shown as a sensitivity diagnostic only. Judge orderings agree across rules on most sets (Kendall tau-b 0.6 to 1.0), the best judge changes with the rule on alphaNLI only, and cross-entropy is the harshest rule on over-confident judges.

**As of:** 11-27-2026 (Friday, Pacific time). CPU only, $0, no model calls. Protocol frozen first (commit `4daeb50`); the script was run after.

## What was done

The five frozen frontier judges of Experiment 28 (verbalized probability forecasts at temperature 0) on ChaosNLI SNLI (600 items), MNLI-m (600), alphaNLI (400) and DICES-350 (350 items) and DICES-990 (400 items), through Experiment 44's data layer. For each set and each of the three rules (multiclass Brier, CE, normalized RPS): the unbiased per-sample risk, the skill over a cross-fitted marginal base-rate forecast with item-bootstrap intervals (4,000 draws) and Holm-adjusted p over 75 cells, the ordering of the judges and its agreement across rules, skill by human-agreement quintile (agreement measured on one random half of each item's ratings, score on the other half), and the minimum detectable difference of a paired judge comparison at N = 400 items and one rating per item. The full tables are in [`results/key_numbers.md`](results/key_numbers.md) and the CSVs next to it.

## Rule-to-encoding table

| set | labels | matched rule(s) | other rules |
|---|---|---|---|
| ChaosNLI SNLI, MNLI-m (K = 3) | unordered | Brier, CE | normalized RPS not matched (label order arbitrary); sensitivity only |
| ChaosNLI alphaNLI (K = 2) | binary | Brier, CE | normalized RPS equals Brier / 2 exactly |
| DICES-350, DICES-990 (K = 3) | ordinal No < Unsure < Yes | normalized RPS | Brier and CE are proper but ignore the order |

## Results

1. **H1 holds: the base-rate verdict is not an artifact of RPS.** On DICES-350, Astra, Kimi K3, GLM 5.3 and DeepSeek V4.1 Flash are Holm-significantly below the base rate under all three rules; Claude Opus 5.5 is not significantly below under any (skill +0.028 Brier, +0.008 CE, +0.032 normalized RPS). Skill ranges from -0.179 to -0.097 in normalized RPS (Experiment 46 reported -0.177 to -0.096 with group folds), from -0.171 to -0.090 in Brier and from -0.466 to -0.209 in CE.
2. **H2 holds: skill falls with disagreement under every rule.** For the best judge of each set and rule, skill in the most-agreed fifth exceeds skill in the least-agreed fifth in 15 of 15 cells. Across the five judges the least-agreed fifth ranges from about -0.60 to +0.16 and the most-agreed fifth from about -0.12 to +0.92, depending on the set and the rule (full table in `results/key_numbers.md`).
3. **The matched rule matters most where the labels are unordered.** Every judge is Holm-significantly above the base rate on SNLI and alphaNLI under all three rules. On MNLI-m, GPT-6 Astra and DeepSeek V4.1 Flash are indistinguishable from the base rate under Brier (skill +0.043 and +0.009) and under CE (+0.001 and +0.007), and Kimi K3 is too under CE (+0.038), but all five are significantly above it under normalized RPS (+0.103 to +0.216). The ordinal rule on unordered labels flatters the judges; this is why the table labels it "not matched".
4. **Ranking.** Kendall tau-b between the orderings of the five judges is 1.0 for Brier versus normalized RPS on the two unordered three-class sets, 0.6 for Brier versus CE and for CE versus normalized RPS on them (two of the ten judge pairs swap), 1.0 for Brier versus CE and 0.8 for either against normalized RPS on the two DICES sets (one pair swaps). The best judge is Claude Opus 5.5 under all three rules on four of five sets; on alphaNLI it is GPT-6 Astra under Brier and normalized RPS and Opus under CE.
5. **Power depends on the rule's scale.** The minimum detectable paired difference at N = 400 and one rating per item is, as a median over the ten judge pairs, 0.036 to 0.056 in Brier, 0.065 to 0.099 in CE (nats) and 0.014 to 0.020 in normalized RPS, across the five sets.

## What this does not show

- It does not show that every proper rule is interchangeable: CE punishes a near-zero forecast on a rating that occurs far more than Brier or RPS does, so it separates over-confident judges more (its skill values are the most negative), and ordering of close judges can change with the rule (alphaNLI).
- It scores five verbalized-forecast judges at temperature 0 on five public sets. The judges' forecasts were fixed in Experiment 28; there is no fitting here.
- The base rate is cross-fitted by random item folds (Experiment 46 cross-fits by group), and the half-split behind the strata differs from Experiment 50's, so numbers match those experiments only within noise.
- Normalized RPS on the three-class NLI sets uses the stored label order, which is arbitrary; it is reported to show that the verdict does not hinge on it, not as a valid score for those sets.
- Intervals resample items and ignore rater sampling variance (as Experiments 44 and 46 do). No Lean claim is made; H1 and H2 are empirical statements about these panels.

## Files and reproduce

`PROTOCOL.md` (frozen first), `scripts/run_all_rules.py`, `scripts/summarize.py`, `tests/test_exp55.py` (9 tests, all pass), `results/` (CSVs, `summary.json`, `key_numbers.md`), `HOW_TO_READ_RESULTS.md`.

```bash
python3 experiments/55_*/scripts/run_all_rules.py   # about one second; identical output on two runs (checked with cmp)
python3 experiments/55_*/scripts/summarize.py
python3 -m pytest -q experiments/55_*/tests
```
