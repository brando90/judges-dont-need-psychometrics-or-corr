# Experiment 46: do real LLM judges rank differently under correlation and agreement than under proper scores?

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/46_real_llm_judge_rank_reversal_corr_vs_proper_scores_start_27Nov2026/README.md>

**Status: interim, 11-27-2026 (Friday, Pacific time).**
- The frontier panel is complete (frozen Experiment 28 forecasts on all items).
- The local CPU panel is partial: 3 of 12 model-by-style judges are done, and the rest are running on the box.
- The API panel of [AMENDMENT_A2_API_PANEL.md](AMENDMENT_A2_API_PANEL.md) has not run, because the box holds no gateway key.
- **API spend so far: \$0.00.**

**TLDR:** On real frontier LLM judges and real multi-annotator labels, correlation and agreement metrics hide differences that proper skill scores show clearly.

1. **H1 is supported: two significant rank reversals.** GPT-6 Astra beats GLM 5.3 on Spearman on SNLI (+0.021, 95% CI [+0.010, +0.034]) and MNLI (+0.034 [+0.014, +0.056]). On the same items it has significantly *lower* Brier skill: −0.039 [−0.056, −0.023] and −0.098 [−0.117, −0.080]. Both survive Holm correction (Holm-adjusted p = 0.025 and 0.042).
2. **High agreement, negative skill.** Uncalibrated, four of five frontier judges on DICES-350 are significantly *worse than the base rate*, with skill between −0.096 and −0.177 (every CI below 0). Their top-1 agreement with the human majority is 0.80–0.82 and their Spearman is 0.42–0.63. On DICES-990 three of five are significantly negative. G-Eval GPT-4 on SummEval consistency has Spearman 0.50 and skill −1.82 [−2.20, −1.49].
3. **Agreement is blind to a large skill gap.** On DICES-350, GPT-6 Astra and Claude Opus 5.5 tie on top-1 agreement (difference +0.001 [−0.022, +0.023]). The skill difference is −0.175 [−0.200, −0.152], and Astra is the best judge by Spearman, top-1 and kappa.
4. **Calibration removes the reversals (H3).** After per-judge 5-fold calibration on held-out labels, there are 0 significant reversals and every judge's skill is positive. Experiments 47 and 48 measure what that remedy costs (100–200 labels per task) and how fragile it is under shift (it fails in 33 of 82 transfers).

## Data, judges, tests

These are fixed in [PROTOCOL.md](PROTOCOL.md) (commit 83b9782, before analysis).

- **Sets.**
  - ChaosNLI SNLI (600 items) and MNLI-m (600), 100 ratings per item, nominal: multiclass Brier.
  - ChaosNLI alphaNLI (400), binary: Brier.
  - DICES-350 (350, 123 ratings per item) and DICES-990 (400, 69–76 ratings), ordinal: RPS/(K−1).
  - SummEval consistency and relevance (1,600 each, 3 experts), ordinal: RPS/(K−1).
- **Judges.**
  - Frontier panel: GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3, DeepSeek V4.1 Flash (Experiment 28 verbalized distributions).
  - G-Eval GPT-4: 20 sampled scores per item.
  - Local panel: Qwen2.5 0.5B/1.5B/3B, Llama-3.2 1B/3B, Gemma-2 2B, as Q4_K_M GGUF under llama.cpp. Native label-token probabilities and verbalized, on the frozen subset (150 per ChaosNLI/DICES set, 240 per SummEval dimension).
- **Skill.** $SS = 1 - S/S_{\text{base}}$, where the base is the set's marginal rating histogram, cross-fitted over 5 group folds.
- **Tests.**
  - Paired item-cluster bootstrap, B = 4,000, seed 20261127.
  - Holm correction within each metric family across all (set, pair) cells of a panel.
  - A *significant reversal* means the correlation or agreement metric favours A, the primary score favours B, and both Holm-adjusted p < 0.05.
- **Amendment A1 (reporting only, decision rules unchanged).** Skill differences, both primary-score skill and nRPS skill, with bootstrap CIs in which the base is resampled with the items. Also per-judge skill CIs ([scripts/judge_skill_ci.py](scripts/judge_skill_ci.py)).

## Results (frozen B = 4,000)

**Significant reversals by panel** (Pearson / Spearman / top-1 / kappa vs the primary score; point reversals in parentheses):

| panel | judges | (set, pair) cells | significant S differences | significant reversals | point reversals |
|---|---|---|---|---|---|
| frontier, all items, raw | 5 | 50 | 31 | 0 / **2** / 0 / 0 | 6 / 10 / 12 / 12 |
| frontier, all items, calibrated | 5 | 50 | 22 | 0 / 0 / 0 / 0 | 3 / 2 / 9 / 7 |
| frontier, items where every judge was valid, raw | 5 | 50 | 33 | 0 / 1 / 0 / 0 | 7 / 15 / 18 / 17 |
| local + frontier, subset, raw (interim: 3 local judges) | 8 | 140 | 0 | 0 / 0 / 0 / 0 | 11 / 17 / 18 / 21 |

Against nRPS skill instead of the primary score (A1), the frontier raw panel has the same 2 significant Spearman reversals.

**Correction to the frozen design.** With B = 4,000 and Holm over 140 cells, the smallest attainable Holm-adjusted p is $140 \cdot 2/4001 \approx 0.07$. The local+frontier panel therefore *cannot* reach significance under the frozen B, and its "0 significant" is uninformative. A sensitivity run at B = 20,000 (floor about 0.014) is labelled as a deviation in `results_B20000/`. The frontier panel's two reversals already sit at its floor of $50 \cdot 2/4001 = 0.025$.

**Per-judge skill, frontier panel, all items** (raw → calibrated; 95% CI in brackets; [results/judge_skill_ci.json](results/judge_skill_ci.json)):

| set | judge | raw skill | calibrated skill | Spearman | top-1 |
|---|---|---|---|---|---|
| DICES-350 | GPT-6 Astra | −0.142 [−0.186, −0.099] | +0.071 [+0.051, +0.091] | 0.627 | 0.823 |
| DICES-350 | Claude Opus 5.5 | +0.033 [+0.005, +0.061] | +0.069 [+0.049, +0.090] | 0.587 | 0.822 |
| DICES-350 | Kimi K3 | −0.137 [−0.179, −0.095] | +0.053 [+0.033, +0.072] | 0.496 | 0.805 |
| DICES-350 | GLM 5.3 | −0.096 [−0.136, −0.057] | +0.057 [+0.037, +0.076] | 0.534 | 0.809 |
| DICES-350 | DeepSeek V4.1 Flash | −0.177 [−0.222, −0.131] | +0.053 [+0.033, +0.075] | 0.423 | 0.803 |
| DICES-990 | GPT-6 Astra | −0.088 [−0.123, −0.053] | +0.106 [+0.086, +0.127] | 0.759 | 0.882 |
| DICES-990 | Claude Opus 5.5 | +0.090 [+0.067, +0.114] | +0.120 [+0.100, +0.140] | 0.809 | 0.876 |
| DICES-990 | Kimi K3 | −0.087 [−0.119, −0.051] | +0.090 [+0.071, +0.111] | 0.713 | 0.848 |
| DICES-990 | GLM 5.3 | −0.025 [−0.056, +0.006] | +0.108 [+0.088, +0.128] | 0.780 | 0.862 |
| DICES-990 | DeepSeek V4.1 Flash | −0.113 [−0.148, −0.076] | +0.095 [+0.076, +0.115] | 0.752 | 0.855 |
| MNLI | DeepSeek V4.1 Flash | +0.009 [−0.022, +0.039] | +0.162 [+0.144, +0.179] | 0.797 | 0.642 |
| SummEval consistency | G-Eval GPT-4 | −1.815 [−2.200, −1.485] | +0.321 [+0.265, +0.375] | 0.504 | 0.295 |
| SummEval relevance | G-Eval GPT-4 | −0.460 [−0.574, −0.351] | +0.133 [+0.108, +0.158] | 0.590 | 0.285 |

On SNLI and alphaNLI every frontier judge has raw skill between +0.30 and +0.56 with CIs above 0.

**Interim local judges** (subset): Qwen2.5-0.5B (native and verbalized) and Llama-3.2-1B (native) have negative skill on every set (−0.09 to −1.19), except Qwen2.5-0.5B verbalized on alphaNLI (+0.002, essentially a constant 50/50 forecast). Their Pearson ranges from −0.13 to 0.46. Qwen2.5-0.5B verbalized output is invalid on 47–60% of ChaosNLI-SNLI/MNLI and DICES items.

## Does this support or undercut the thesis?

It supports the thesis on real judges, with one honest qualification.

- **Support:**
  - Correlation and agreement rank frontier judges differently from proper scores in significant, Holm-corrected cases.
  - High top-1 agreement and moderate correlation coexist with judges that are significantly worse than the base rate.
  - That is the "artifact" claim: correlation-validated judges can carry no calibrated information about the human label distribution.
- **Qualification:** Per-judge calibration on held-out labels removes the reversals (H3, frozen reading: "correlation ranks judges correctly once each is calibrated with held-out labels"). The paper should therefore pair this result with Experiments 47 and 48: calibration needs 100–200 held-out labels per task and does not transfer across tasks. Either way, proper-score validation on held-out labels is required.

**Proposed paper text (not inserted):** "On five frontier LLM judges and five public multi-annotator sets, Spearman correlation and the proper score disagree in Holm-significant cases. GPT-6 Astra out-correlates GLM 5.3 on SNLI and MNLI while having 0.04 and 0.10 lower Brier skill. Four of five judges agree with the human majority on 80–82% of DICES-350 items yet are significantly worse than predicting the base rate, with skill −0.10 to −0.18. Held-out calibration makes every judge's skill positive, but that remedy is itself a proper-score validation step."

## Spend and compute

| item | spend |
|---|---|
| API calls (Amendment A2 runner) | \$0.00: no gateway key on the box |
| Frontier forecasts | \$0 now (frozen Experiment 28 receipts) |
| Local judges | CPU on the box, \$0 |
| **Cumulative Exp 46 spend** | **\$0.00** |

**To run the API panel** (on a host holding the Stanford gateway key):

```
python scripts/make_subset.py ...                                                # writes summeval_prompts.json (derivable, not committed)
python scripts/api_judges.py --config api/api_panel.json --list-models           # pick expansion judges, add them to api_panel.json with rates
python scripts/api_judges.py --config api/api_panel.json --stage full --styles verbalized --dry-run   # about $41 conservative for 5 judges
python scripts/api_judges.py --config api/api_panel.json --stage pilot --styles sampled --judges kimi-k3 glm-5.3 deepseek-v4.1-flash
python scripts/analyze_rank_reversal.py LABELS.json E23_INPUTS localjudges:api/forecasts results
```

## Reproduce

```
python scripts/analyze_rank_reversal.py e28prep/labels.json e23inputs localjudges results            # B = 4,000 (E46_BOOT overrides), about 6 minutes
E46_BOOT=20000 python scripts/analyze_rank_reversal.py e28prep/labels.json e23inputs localjudges results_B20000
python scripts/judge_skill_ci.py e28prep/labels.json e23inputs /nonexistent results/judge_skill_ci.json
bash run_all_local.sh   # local judges (see scripts/run_local_judges.py; Q4_K_M GGUFs from Hugging Face)
```

`labels.json` is rebuilt by Experiment 28's `runners/prepare.py` (request bundle SHA-256 6004ef74…ee64, checked). The SummEval inputs are rebuilt by Experiment 23's `summeval/prepare.py` (SHA-256 matches). Seeds: bootstrap SeedSequence([20261127, 46, set, panel]); folds as in the protocol.
