# Experiment 46, amendment A2: API judge panel, staged spend, stopping rule (frozen 11-27-2026, before any API call)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/46_real_llm_judge_rank_reversal_corr_vs_proper_scores_start_27Nov2026/AMENDMENT_A2_API_PANEL.md>

**TLDR:** Brando raised the API ceiling to \$2,000 total, a ceiling and not a target. It is shared with Claude Code's overnight runs and excludes Claude/Codex subscription credits. This amendment fixes, before any call:
- which API judges, elicitations and items are run;
- how spend is staged and logged;
- when scaling stops.

The analysis is the unchanged `scripts/analyze_rank_reversal.py`, with the same tests, Holm families and reversal definition.

**As of:** 11-27-2026 (Friday, Pacific time). Written by Grok Bot for Brando Miranda.

## Route and credentials

- **Route.** OpenAI-compatible chat completions through the Stanford AI API Gateway, the route of Experiments 26 and 28, via `scripts/api_judges.py`.
- **Credentials.** The key is read from a file named by `E46_API_KEY_FILE` and is never printed or stored.
- **No subscription credits.** No Claude CLI or Codex plan route exists in the script.
- **The box that prepared this amendment has no key.** The run happens wherever the key file lives: Brando's lab host, or the box once he places a key there.

## Judges

- **Base panel.** The five Experiment 28 judges at the same gateway ids: GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3, DeepSeek V4.1 Flash.
- **Expansion to 8–10 judges.**
  - Who: 3–5 more models from providers or sizes not yet represented, e.g. a Gemini, a Grok, an open Llama/Qwen and a small closed model.
  - How they are picked: by exact id from `api_judges.py --list-models`.
  - Recording: each is added to `api/api_panel.json` with its list rate. This file records the final list in a dated line before that judge's first call.
- **Local panel.** The six local CPU judges of the original protocol remain as the small open-weight tier.

## Elicitations (per judge)

1. **verbalized.** Experiment 28's exact prompt, temperature 0, 2,048 max tokens, Exp 28 parser (invalid → uniform).
2. **sampled.**
   - Prompt: the one-word-answer prompt of the local `native` style.
   - Sampling: 10 samples at temperature 1, max 1,024 tokens to leave room for reasoning.
   - Forecast: the histogram of parsed labels. Unparseable samples are dropped; all-unparseable → uniform, counted as invalid.
3. **logprob.**
   - Prompt: the same one-word prompt, temperature 0, top-20 log-probabilities of the first answer token.
   - Forecast: label mass summed over case and space variants, renormalized.
   - Scope: only judges whose endpoint returns log-probabilities.

## Items

- **Main sets.** The frozen `subset_ids.json`: 150 items per ChaosNLI/DICES set, 240 per SummEval dimension. Verbalized skips SummEval, as in the local protocol.
- **Pilot.** The first 50 subset items per set.
- **New datasets.** Additional datasets (MT-Bench human judgments, Wikipedia Detox toxicity, …) enter only through a further dated amendment that fixes the loader, item sample and primary score before their forecasts are joined to labels.

## Stages and spend

| stage | what | conservative estimate (configured rates; \$0-listed models priced at the 1.4/4.4 stand-in) | gate to next stage |
|---|---|---|---|
| P1 | verbalized, all base judges, full subset; sampled on the pilot items for the three \$0-listed judges | ≤ \$45 | runner works, invalid rate < 10% per judge |
| P2 | sampled on the pilot items for GPT-6 Astra and Opus; expansion judges, verbalized on full subset | ≤ \$120 cumulative | any pre-registered reversal or a significant S difference with discordant correlation in the pilot |
| S | sampled and logprob on the full subset for judges and sets that showed signal; new datasets (separate amendment) | ≤ \$600 cumulative | stopping rule below |

Spend accounting:
- **Ledger.** Every call is logged in `api/spend_ledger.jsonl` with token counts and estimated USD at the configured rate. The gateway does not expose billing.
- **Hard stop.** The runner stops at `budget_usd`: \$150 in the committed config, raised only by editing the config in a commit that states the new cumulative ceiling.
- **Reporting.** Cumulative spend is reported in the Exp 46 README and in `docs/STATE_OF_EVIDENCE.md`.

## Stopping rule (Brando, 11-27-2026)

Scale only until the evidence would convince Sanmi and conference area chairs, then stop. Concretely, stop once all four hold:

1. **Decisive reversals.** At least one judge pair, on at least two sets, has a significant Pearson- (or Spearman-) vs primary-score reversal, Holm-adjusted p < 0.05 with B = 4,000. The bootstrap interval of the skill difference must exclude 0 by more than 0.01.
2. **Robustness.** The direction holds for at least two elicitations, or for at least two judge families.
3. **Answer to "just recalibrate".** The calibrated panel is reported. If reversals vanish after per-judge calibration on held-out labels (H3), that reading is stated, and Experiments 47–48 quantify the label cost and the shift fragility of that remedy.
4. **Answer to circularity.**
   - Primary scores use held-out human ratings, never the judge's own outputs.
   - Bases are cross-fitted.
   - This amendment and the subset are committed before any API forecast is joined to labels.

If the bar is not met by the \$600 stage ceiling, stop and report a null.
