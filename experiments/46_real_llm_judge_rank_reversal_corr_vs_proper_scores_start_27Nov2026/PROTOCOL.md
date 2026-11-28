# Experiment 46 protocol: do real LLM judges rank differently under correlation and agreement than under proper scores? (frozen 11-27-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/46_real_llm_judge_rank_reversal_corr_vs_proper_scores_start_27Nov2026/PROTOCOL.md>

**TLDR:** Rank a panel of real LLM judges on real multi-annotator rating sets twice: by Pearson, Spearman, top-1 agreement and Cohen's kappa, and by the strictly proper score matched to each set's label encoding plus the base-rate skill score. A rank reversal is a judge pair where correlation (or agreement) and the proper score disagree, each with a Holm-corrected paired bootstrap interval that excludes zero. This file is committed before the analysis script is written or run and before any local-judge forecast is joined to a label.

**As of:** 11-27-2026 (Friday, Pacific time). Written by Grok Bot for Brando Miranda (task: "do it go", 11-27-2026).

## What the analyst had seen before freezing

- Experiment 28 (committed 11-23-2026) already published raw-judge scores and correlations for the five frontier judges below (`28_*/results/raw_judge_scores.csv`, `correlations.csv`). The author of this protocol has **not opened those two files** and will not before the analysis runs. Because they are public, the frontier-judge panel is labelled "exposure possible" in the results.
- The local-judge forecasts do not exist yet.

## Data (public, multi-annotator)

| set | items | classes (score coding) | ratings per item | encoding, primary proper score |
|---|---|---|---|---|
| ChaosNLI SNLI | 600 (Exp 28 subsample) | contradiction, neutral, entailment (0,1,2) | 100 | nominal: multiclass Brier (one-hot) |
| ChaosNLI MNLI-m | 600 | same | 100 | nominal: multiclass Brier |
| ChaosNLI alphaNLI | 400 | hypothesis 1, hypothesis 2 (0,1) | 100 | binary: Brier |
| DICES-350 | 350 | No, Unsure, Yes (0,1,2) | 123 | ordinal harm scale: RPS / (K-1) |
| DICES-990 | 400 | No, Unsure, Yes (0,1,2) | 69-76 | ordinal: RPS / (K-1) |
| SummEval consistency | 1,600 | grades 1-5 | 3 experts | ordinal: RPS / (K-1) |
| SummEval relevance | 1,600 | grades 1-5 | 3 experts | ordinal: RPS / (K-1) |

Item ids and label counts for the first five sets are rebuilt by `28_*/runners/prepare.py` from the hash-checked public files; the rebuilt request bundle must reproduce Experiment 28's SHA-256 `6004ef74...ee64` (checked before freezing: it does). SummEval inputs are rebuilt by `23_*/summeval/prepare.py` (output SHA-256 must match Experiment 23's documented values; checked: they do). The ordinal coding of NLI is an assumption (Exp 28); it is used only for Pearson, Spearman and RPS, never for the nominal primary score.

## Judges

1. **Frontier panel (frozen, $0 now):** Experiment 28's verbalized-distribution forecasts of GPT-6 Astra, Claude Opus 5.5, Kimi K3, GLM 5.3 and DeepSeek V4.1 Flash on the first five sets (temperature 0, one call per item). Invalid responses were replaced by the uniform forecast in Exp 28; primary analysis keeps that rule, a sensitivity analysis drops items where any judge was invalid.
2. **G-Eval GPT-4 (frozen, $0):** the archived 20 sampled completions per item on SummEval (Experiment 23's `native` histogram), i.e. a sampled-score distribution with n = 20.
3. **Local open-weight panel (CPU, $0):** Qwen2.5-0.5B/1.5B/3B-Instruct, Llama-3.2-1B/3B-Instruct, Gemma-2-2B-it, 4-bit GGUF (Q4_K_M) under llama.cpp. Two elicitation styles: (a) **native**: next-token probabilities of the label words (or digits 1-5 on SummEval) after a one-word-answer prompt, summed over casing/space variants of each label's first token and renormalized over the K labels; (b) **verbalized**: Experiment 28's exact system prompt and user JSON, greedy decoding, at most 64 new tokens, Exp 28's parser (invalid -> uniform). Verbalized is run on the ChaosNLI/DICES sets only. Because the box has no GPU, the local panel is run on a fixed subsample: the first 150 items per ChaosNLI/DICES set and the first 240 items per SummEval dimension (15 source documents x 16 systems) in a permutation drawn from `numpy.random.default_rng(SeedSequence([20261127, 46, set_index]))` over item ids (SummEval: over documents). If a model cannot finish in the box's compute window the missing cells are reported, not imputed.
4. **API judges:** none. The box holds no LLM API credentials, and Brando's Claude/Codex subscription credits are excluded. Any further API panel would be a separate, separately preregistered amendment.

All forecasts are floored as in Exp 28: q <- (1 - K * 1e-6) q + 1e-6.

## Metrics (all on the raw judge forecast; no fitting unless stated)

- **Primary proper score S** (lower is better), matched to the encoding in the table, computed as the unbiased per-sample risk: the mean over each item's individual ratings of the score of q against the one-hot rating, then the mean over items.
- **Skill score** SS = 1 - S(judge)/S(base), base = the set's marginal rating histogram estimated from the other four folds' ratings (5-fold group cross-fit; groups = SNLI premise image id, SummEval source document, otherwise item). Higher is better; 0 = no better than the base rate.
- Secondary proper scores: CE (nats), multiclass Brier, normalized RPS (each reported everywhere).
- Correlation and agreement (higher is better): Pearson and Spearman between the judge's expected score E_q[score] and the item's mean human score; top-1 agreement (argmax q equals the modal human label; human modal ties split credit); Cohen's kappa between argmax q and the human majority label (ties broken to the lowest index).

## Tests (fixed)

- For every set and every unordered judge pair (A, B) in a panel, paired item-cluster bootstrap over groups, 4,000 draws, seed 20261127: Delta_S = S(A) - S(B), Delta_r = metric(A) - metric(B) for each correlation/agreement metric. Two-sided percentile-bootstrap p-value p = min(1, 2 min(P*(Delta <= 0), P*(Delta >= 0))) with the (B+1) correction.
- Holm-Bonferroni within each metric family across all (set, pair) cells of a panel.
- **Significant rank reversal** for (set, A, B, metric M): Delta_r favors A (Delta_r > 0), Delta_S favors B (Delta_S > 0, i.e. A has the worse proper score), and both Holm-adjusted p < 0.05. A **point reversal** is the same with point estimates only (descriptive).
- Descriptive: Kendall tau-b between the judge ranking by M and by S, per set.
- "A beats B" (repository definition) = lower mean held-out S and a significant pre-registered paired test.

## Hypotheses

- **H1 (primary).** At least one significant rank reversal between Pearson and the primary proper score exists in at least one set of the frontier panel or the local panel.
- **H2.** The same holds for top-1 agreement or kappa.
- **H3 (secondary, calibration).** After a per-judge post-hoc calibrator (multinomial logit on [log q, 1], ridge 1e-3, fitted by individual-rating CE on four folds, applied to the fifth; 5-fold cross-fit), the number of significant Pearson-vs-S reversals falls. If it falls to zero the result is read as "correlation ranks judges correctly once each is calibrated with held-out labels"; that reading is reported if it occurs.
- Every set, panel and pair is reported, including those where the rankings agree.

## Compute and cost

CPU only (8 cores). API spend: $0 planned. Brando approved up to $200 of API spend for this experiment (11-27-2026); it is not used because no API credentials are available on the box.
