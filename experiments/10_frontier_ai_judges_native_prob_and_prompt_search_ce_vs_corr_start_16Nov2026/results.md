# Experiment 10 Live Results — Frontier AI Judges: Native Probabilities, Prompt Optimization (`iii.a`), and Harness Optimization (`iii.b`)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/results.md>

**Status:** COMPLETE (verified 11-16-2026 12:41 PST)
**Last updated:** 11-16-2026 12:41 PST
**Canonical report:** [RESULTS_frontier_ce_vs_corr.md](RESULTS_frontier_ce_vs_corr.md) · [Protocol](PROTOCOL.md) · [Reading guide](HOW_TO_READ_RESULTS.md) · [Checkpoint](CKPT_frontier_ce_vs_corr.md)

## Phase summary

| Phase | Description | Status | Spend (USD) | Headline held-out / sanity result |
|---|---|---|---:|---|
| 0 | Pre-registration freeze (`PROTOCOL.md`, `FREEZE.json`) | DONE | $0.00 | Committed at `aa09730` prior to paid runs; `Exp 02` parity verified to `< 1e-6` |
| 1 | Karpathy sanity controls & `k=3` repeat SD (`gpt-6.1-sol`) | DONE | $0.50 | Label-in-prompt `r = 1.0000`, `oracle` `1.000` vs `blank` `0.000`, repeat SD `0.0178` |
| 2 | Multi-family native-prob & logprob pool (7 OpenAI/Mistral models) | DONE | $10.87 | Native-15 `5x5` CV held-out `CE = 1.416` nats (vs `5.491` scalar & `1.928` constant); feature-head `② minus ④` RPS `+0.0250 [+0.0238, +0.0262] p-val=0.0000` |
| 3 | Axis `iii.a`: Blind 4x2 prompt search (`Track_Corr` vs `Track_CE`) | DONE | $2.97 | Held-out test RPS `① Corr-selected minus ③ Loss-selected = +0.0418 [-0.0200, +0.1813] p-val=0.2526` |
| 4 | Axis `iii.b`: Frontier harnesses (`H1`–`H4`) & `2x2` interaction | DONE | $3.51 | Joint `2x2` `[Prompt ① + Harness ①] minus [Prompt ③ + Harness ③]` test RPS `-0.0060 [-0.0493, +0.0613] p-val=0.8738` |
| 5 | Grouped `250`-split learning curves, figures & `verify_results.py` | DONE | $0.00 | Crossover at **5 training task families (~51.7 human ratings)**; total spend **`$17.85` / `$18.00`** |

**TLDR-end:** [judges-corr: experiment 10 live results] Completed all 5 phases across 7 OpenAI and Mistral frontier models (`$17.85` spent): native probabilities fix the historical 3-sample scalar CE failure (`5.49` -> `1.416` nats), proper-loss optimization beats correlation optimization on continuous mixtures (`+0.0775` RPS) and feature heads (`+0.0250` RPS from ~52 ratings), and `H3` two-step obligation harnesses reach `0.1136` held-out RPS (`2x2` discrete pipeline contrast `-0.0060`, `p-val=0.8738`).
