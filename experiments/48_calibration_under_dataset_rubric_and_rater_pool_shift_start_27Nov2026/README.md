# Experiment 48: does a calibration map transfer across datasets, rubrics and rater pools?

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/48_calibration_under_dataset_rubric_and_rater_pool_shift_start_27Nov2026/README.md>

**TLDR:** Mostly it does not. "Calibrate once and the correlation-validated judge is fine" fails under shift. For the five frozen frontier LLM judges, a map fitted on one set and applied to another reopens the proper-score gap in 53 of 82 (judge, source, target) cases. In 33 of those it makes the judge significantly worse than not calibrating at all. The exception is the near shift between the two DICES rater pools, where the map transfers (0 of 10 reopen). For Experiment 41's linear heads, the gap reopens in 5 of 6 expert-to-crowd rater-pool shifts and in all 16 leave-one-dataset-out views. So each new target needs its own held-out human labels scored with a proper rule, which is the paper's recommendation.

**As of:** 11-27-2026 (Friday, Pacific time). CPU, \$0 (no API calls; frozen forecasts only). Protocol frozen first: [PROTOCOL.md](PROTOCOL.md), commit 83b9782.

## Setup (from PROTOCOL.md)

- **Arm A (primary).** Frozen verbalized forecasts of GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3 and DeepSeek V4.1 Flash from Experiment 28, on ChaosNLI SNLI and MNLI and DICES-350 and DICES-990 (all K = 3), plus the G-Eval GPT-4 SummEval forecasts of Experiment 23 (K = 5, 20 samples per item). Calibrator: multinomial logit on $[\log q, 1]$, ridge $10^{-3}$, fitted by individual-rating cross-entropy. In-domain reference: 5-fold group cross-fit on the target. Shifted: fitted on all items of the source. Primary score: Brier for nominal NLI, normalized RPS for ordinal DICES and SummEval.
- $D = S_{\text{shift}} - S_{\text{in}}$ (the reopened gap) and $F = (S_{\text{raw}} - S_{\text{shift}})/(S_{\text{raw}} - S_{\text{in}})$ (the fraction of the in-domain calibration gain that survives). Paired item-cluster bootstrap, 4,000 draws, 95% intervals.
- **Decision rule (frozen).** The gap *reopens* if the interval of $D$ lies above 0 and $F < 0.5$. Calibration *fails outright* if $S_{\text{shift}} > S_{\text{raw}}$ with an interval excluding 0.
- **Arm B (secondary).** Experiment 41's views and linear heads (via Experiment 47's `heads41.py`). $G_{\text{shift}}$ = nRPS(Pearson-trained head plus shifted map) − nRPS(CE-trained head, raw), split seeds 0 to 4, 2,000 draws per seed. Reopens if the seed mean exceeds 0.0024 (Exp 41's in-domain bound) and at least 3 of 5 seed intervals lie above 0.

## Results

Full tables: [results/shift_tables.md](results/shift_tables.md). Raw rows: `results/arm_a_llm_judges.json` and `results/arm_b_exp41_heads.json`.

**Arm A, frontier LLM judges (82 cases):**

| shift | cases | gap reopens | worse than raw |
|---|---|---|---|
| near: SNLI <-> MNLI | 10 | 9 | 6 |
| near: DICES-350 <-> DICES-990 | 10 | 0 | 0 |
| far: NLI <-> DICES | 40 | 33 | 21 |
| leave one set out (fit on other three) | 20 | 10 | 6 |
| rubric: SummEval relevance <-> consistency | 2 | 1 | 0 |

- **DICES pools transfer.** DICES-990 to DICES-350 keeps $F$ = 0.89 to 1.00 of the gain. DICES-350 to DICES-990 keeps 0.54 to 0.98, with $D \le 0.0066$.
- **DICES to SNLI fails for every judge.** Brier rises by +0.032 (GPT-6 Astra, DICES-350 source) to +0.338 (DeepSeek V4.1 Flash, DICES-990 source) over the uncalibrated judge. Example: Claude Opus 5.5 with a DICES-350 map scores Brier 0.665 on SNLI, vs 0.400 raw and 0.396 in-domain. The difference from raw is +0.265, 95% CI [+0.243, +0.288].
- **SNLI <-> MNLI.** The in-domain calibration gain on SNLI is small: Brier falls 0.0041 to 0.043 (e.g. 0.4004 to 0.3963 for Opus). So $F$ is unstable there (down to −5.1), and $D$ is the readable quantity: $D$ = +0.024 to +0.055 in both directions.
- **Rubric shift (same items and experts).** A relevance map applied to consistency keeps $F$ = 0.59: nRPS 0.119 vs 0.215 raw and 0.052 in-domain, so it does not reopen. A consistency map applied to relevance keeps $F$ = −0.06: nRPS 0.204 vs 0.199 raw, with shift minus raw = +0.005, CI [−0.013, +0.023], so the gain is lost.
- **Leave one set out.** The gap reopens on SNLI (5 of 5 judges) and MNLI (4 of 5) but not on DICES-350 (0 of 5) or DICES-990 (1 of 5). The pooled map is dominated by the DICES-like sets.

**Arm B, Experiment 41 linear heads:**

| shift | views reopened | in-domain G | shifted G (seed mean) |
|---|---|---|---|
| rater pool, SummEval expert <-> crowd (K = 2, 3, 5, both directions) | 5 of 6 | −0.0017 to +0.0001 | +0.0019 to +0.0273 |
| leave one dataset out (map pooled from the other same-K views) | 16 of 16 | −0.0057 to +0.0001 | +0.0085 to +0.2576 |

- At K = 5 the rater-pool shift reopens the gap to +0.024 (expert to crowd) and +0.027 (crowd to expert), with all 5 seeds' intervals above 0. The in-domain values are +0.0001 and −0.0017.
- The one non-reopen is expert to crowd at K = 3: $G_{\text{shift}}$ = +0.0019, below the 0.0024 bound, though 4 of 5 seeds' intervals are above 0.

## Skill-score terms (amendment A1, reporting only)

The advisor asked for skill-score differences with CIs. [scripts/skill_terms.py](scripts/skill_terms.py) re-expresses both arms; decision rules are unchanged. Full per-case tables are in [results/skill_terms_tables.md](results/skill_terms_tables.md); raw values are in `results/skill_terms.json`.

**Arm A.**
- Skill: $SS = 1 - S/S_{\text{base}}$, with $S_{\text{base}}$ the target's cross-fitted marginal (the Experiment 46 base).
- Bootstrap: same plan and same 4,000-draw item-cluster bootstrap (same seeds) as the main run, with the base resampled together with the items.
- Quantities: $\Delta SS_{\text{shift-in}} = SS(\text{shifted map}) - SS(\text{in-domain map})$ and $\Delta SS_{\text{shift-raw}} = SS(\text{shifted map}) - SS(\text{raw judge})$.

| shift | cases | in-domain calibration gain in skill | $\Delta SS_{\text{shift-in}}$ range | CI below 0 | $\Delta SS_{\text{shift-raw}}$ range | CI below 0 (worse than raw) |
|---|---|---|---|---|---|---|
| near: SNLI <-> MNLI | 10 | +0.007 to +0.153 | −0.087 to −0.038 | 10 | −0.050 to +0.077 | 6 |
| near: DICES-350 <-> DICES-990 | 10 | +0.030 to +0.230 | −0.031 to −0.000 | 6 | +0.016 to +0.220 | 0 |
| far: NLI <-> DICES | 40 | +0.007 to +0.230 | −0.567 to −0.027 | 40 | −0.537 to +0.143 | 21 |
| leave one set out | 20 | +0.007 to +0.230 | −0.094 to −0.008 | 20 | −0.051 to +0.196 | 6 |
| rubric (G-Eval, SummEval) | 2 | +0.593, +2.135 | −0.883, −0.629 | 2 | +1.252, −0.036 | 0 |

- **A shifted map is almost always worse than an in-domain one.** In skill terms it loses skill relative to an in-domain map in 78 of 82 cases (CI below 0). It makes the judge worse than uncalibrated in 33 of 82, the same 33 as the frozen "fails outright" rule.
- **Worst cases.** DeepSeek V4.1 Flash with a DICES-990 map on SNLI: skill falls from +0.299 raw to −0.238, $\Delta SS_{\text{shift-raw}}$ = −0.537 [−0.593, −0.485]. Claude Opus 5.5 with a DICES-350 map on SNLI: from +0.362 to −0.060, −0.422 [−0.467, −0.381]. In both cases the calibrated judge is worse than the base rate.
- **The DICES pools transfer.** A shifted map keeps the judge well above raw: +0.016 to +0.220. It is at most 0.031 skill points below in-domain.

**Arm B.** $\Delta SS = SS(\text{Pearson-trained head} + \text{map}) - SS(\text{CE-trained head, raw})$, with $SS = 1 - \mathrm{nRPS}/S_{\text{const}}$. Per-seed CIs are the paired item-bootstrap CIs of G divided by $S_{\text{const}}$ (base held fixed).
- **In-domain**, $\Delta SS$ is −0.001 to +0.033 on all 22 cases.
- **Rater-pool shift** moves it by −0.014 to −0.201 skill points. At K = 5: −0.190 (expert to crowd) and −0.201 (crowd to expert), with 5 of 5 seed CIs below 0.
- **Leave one dataset out** moves it by −0.048 (alphaNLI K2, 3 of 5 seeds) to −1.99 (POPQUORN offensiveness K2). 15 of 16 views have all 5 seed CIs below 0.
- **Context: the CE-trained heads' own skill over the base rate is at most 0.071 on any view** (Experiment 47 A1). These shifts cost many times the skill the heads have.

## Does this support or undercut the thesis?

It supports the thesis. Calibration can close the gap only in-domain. Once the dataset, rubric or rater pool changes, the calibrated judge must be revalidated with a proper score on target labels, because how a map transfers is not predictable in advance: the two DICES pools transfer and SNLI/MNLI do not. A correlation check on the target cannot detect any of this, because it never scores the calibrated probabilities against target ratings.

**Caveats.**
- "Far" pairs mix a nominal task (NLI, scored by Brier) with an ordinal one (DICES, scored by normalized RPS), so far-shift failure partly reflects label-semantics mismatch. That is still a realistic way to reuse a calibrator by mistake, but it is the extreme case.
- Arm B's leave-one-dataset-out pools include the same-K view of a sibling rater pool (e.g. SummEval crowd when the target is SummEval expert). The largest values (up to +0.26 on POPQUORN offensiveness) are likely driven by different label marginals across datasets; this was not separately tested.

**Proposed paper text (not inserted):** "Calibration is local. A map fitted on one task and applied to another reopened the proper-score gap in 53 of 82 frontier-judge cases and made the judge worse than uncalibrated in 33 of them. For example, Claude Opus 5.5 with a DICES map reaches Brier 0.665 on SNLI vs 0.400 raw. The map transferred only between two rater pools of the same task (DICES-350 and DICES-990). Every new deployment therefore needs its own held-out human ratings scored with a proper rule."

## Reproduce

```
# inputs: Experiment 28 prepare.py -> labels.json (request bundle SHA 6004ef74...ee64); Experiment 23 SummEval npz inputs
export E40_DATA_CACHE=/path/to/cache
python scripts/calibration_shift.py /path/to/labels.json /path/to/e23_inputs results   # imports Experiment 46 and 47 scripts
python scripts/summarize_shift.py results
python scripts/skill_terms.py /path/to/labels.json /path/to/e23_inputs results   # A1 skill-score terms, about 10 seconds
```

Seeds are fixed in the script: bootstrap 4,000 draws (Arm A) and 2,000 (Arm B), split seeds 0 to 4. The script imports the loaders in Experiment 46's `analyze_rank_reversal.py` and Experiment 47's `heads41.py`.
