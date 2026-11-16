# Experiment 10 Live Results — Frontier Artificial Intelligence (AI) Judges: Native Probabilities, Prompt Optimization (`iii.a`), and Harness Optimization (`iii.b`)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/results.md>

**Status:** COMPLETE (verified 11-16-2026 14:45 PST)
**Last updated:** 11-16-2026 14:45 PST

**Failure handling:** Saved losses include uniform forecasts substituted after unsuccessful parsing. `ce_iter2_c1`: 11/75 overall, including 7/30 held-out test artifacts. Harness results also include uniform or upstream scaffold fallbacks. These measure the complete pipeline with fallback behavior, not successful model responses alone. Status metadata was recovered from exact local cached requests without new calls; the failure table below retains every scheduled entry.
**Canonical report:** [RESULTS_frontier_ce_vs_corr.md](RESULTS_frontier_ce_vs_corr.md) · [Protocol](PROTOCOL.md) · [Reading guide](HOW_TO_READ_RESULTS.md) · [Checkpoint](CKPT_frontier_ce_vs_corr.md)

## Phase summary

Table notation: ranked probability score (RPS; cumulative-probability squared error, 0–1), cross-entropy (CE; negative log-probability, nats), total variation distance (TVD; half the histogram L1 distance, 0–1), and mean squared error (MSE; rating squared) are lower-is-better. CI denotes a confidence interval, CV cross-validation, ICC intraclass correlation, SD standard deviation, and AST abstract syntax tree. Bias is judge minus human mean on 0–5; correlations range from −1 to 1. Each difference compares the same held-out metric, correlation-trained/selected minus proper-loss-trained/selected, unless explicitly stated otherwise.

| Phase | Description | Status | Spend (USD) | Headline held-out / sanity result |
|---|---|---|---:|---|
| 0 | Pre-registration freeze (`PROTOCOL.md`, `FREEZE.json`) | DONE | $0.00 | Frozen in `FREEZE.json` (pre-run worktree `aa09730`, integrated in `a346d5c`); `Exp 02` parity verified to `< 1e-6` |
| 1 | Karpathy sanity controls & `k=3` repeat SD (`gpt-6.1-sol`) | DONE | $0.50 | Label-in-prompt `r = 1.0000`, `oracle` `1.000` vs `blank` `0.000`, repeat SD `0.0178` |
| 2 | Multi-family native-prob & logprob pool (7 OpenAI/Mistral models) | DONE | $10.87 | Native-15 `5x5` CV held-out `CE = 1.416` nats (vs `5.491` scalar & `1.928` constant); feature-head `② minus ④` RPS `+0.0250 [+0.0238, +0.0262] p-val=n/a` |
| 3 | Axis `iii.a`: Blind 4x2 prompt search (`Track_Corr` vs `Track_CE`) | DONE | $2.97 | Held-out test RPS `① Corr-selected minus ③ Loss-selected = +0.0418 [-0.0200, +0.1813] bootstrap-tail=0.2526; p-val=n/a` |
| 4 | Axis `iii.b`: Frontier harnesses (`H1`–`H4`) & `2x2` interaction | DONE | $3.51 | Joint `2x2` `[Prompt ① + Harness ①] minus [Prompt ③ + Harness ③]` test RPS `-0.0060 [-0.0493, +0.0613] bootstrap-tail=0.8738; p-val=n/a` |
| 5 | Grouped `250`-split learning curves, figures & `verify_results.py` | DONE | $0.00 | Positive sampled mean differences from **5 training task families (~51.7 human ratings)** (no population threshold); total spend **`$17.85` / `$18.00`** |

**TLDR-end:** [judges-corr: experiment 10 live results] Completed all 5 phases across 7 OpenAI and Mistral frontier models (`$17.85` spent): the native-probability pool has lower CE than the historical scalar pool (`5.49` -> `1.416` nats), proper-loss fits have lower measured loss on continuous mixtures (`+0.0775` RPS) and feature heads (`+0.0250` RPS; conditional split variability only), and on matched `gpt-5.4-mini` `H3` two-step obligation harnesses reach `0.1136–0.1174` held-out RPS (`2x2` discrete pipeline contrast `-0.0060`, `bootstrap-tail=0.8738; p-val=n/a`).
