# Folder 10 — PROTOCOL AMENDMENT 01 (Provider Wire & Quota Remedies During Phase 2)

**Recorded:** 11-16-2026 11:30 PST (`PROTOCOL.md` and `FREEZE.json` remain unmodified as committed at `aa09730`).

## Observed provider endpoint behavior during Phase 2 (`$0.00` wasted spend)

During Phase 2 execution (`task-128`), all 8 OpenAI JSON-probability judges (`gpt-6-astra`, `gpt-6.1-sol` x 3, `gpt-5.6-sol`, `gpt-5.4-mini` x 3) completed `75/75` cleanly (`600/600` artifacts), while two wire/account limits surfaced at `$0.00` charge on the remaining slots before any held-out analysis ran:

1. **OpenAI `/v1/chat/completions` `top_logprobs` ceiling on `gpt-5.4-mini`:** Sending `top_logprobs=20` returned HTTP 400 (`"Invalid value for 'top_logprobs': must be less than or equal to 5"`). Verified HTTP 200 with `top_logprobs=5`.
2. **Mistral Codestral key routing (`~/keys/mistral_api_key.txt`):** Although `GET https://api.mistral.ai/v1/models` returned HTTP 200 listing `mistral-medium-3.5` and `magistral-medium-latest`, `POST https://api.mistral.ai/v1/chat/completions` returned HTTP 429 with header `x-ratelimit-limit-req-minute: 0` on `mistral-medium-3.5` and `magistral-medium-latest`, and `x-ratelimit-limit-req-minute: 2` on `codestral-2508`. Routing `codestral-2508` to Mistral's dedicated `https://codestral.mistral.ai/v1/chat/completions` endpoint succeeded `8/8` concurrently at HTTP 200 with zero rate-limit throttling.

## Prospective remedies (preserving 15 frontier pool judges, 7 models, 2 model families, and `$18.00` cap)

1. Set `top_logprobs=5` in `scripts/common.py` (`call_logprobs_6cat`) so `gpt54_mini__L0_logprobs` completes `75/75`.
2. Route `codestral-2508` to `https://codestral.mistral.ai/v1/chat/completions` across `codestral_2508__P0`, `codestral_2508__P1`, and `codestral_2508__P2` in Phase 2 and as the Mistral cross-family member in Phase 4 (`H1_single_codestral` and `H2_ast_codestral`).
3. Replace the three remaining zero-quota `api.mistral.ai` slots in the 15-judge pool with the already-authorized `koyejolab-openai-bm` frontier models `gpt-6.1-terra` (`gpt61_terra__P0`, `gpt61_terra__P2`, `effort="low"`) and `gpt-5.6-terra` (`gpt56_terra__P0`, `effort="none"`), preserving **15 zero-exposure native-probability & logprob frontier judges across 7 distinct frontier models in 2 model families (OpenAI & Mistral)**.
4. **Budget-preserving execution backbone for Phases 3 & 4 (`$6.63` remaining under `$18.00` after `$11.37` Phase 1+2 completion):** Keep `gpt-6.1-sol` (`effort="medium"`) as the blind prompt proposer in Phase 3 (`iii.a`, full 4 iterations x 2 candidates = 8 prompt candidates per track) with `gpt-5.4-mini` (`effort="low"`, `$0.0027/call`) as the fast inner-loop frontier evaluator across all 16 candidates (`~$2.40`), and in Phase 4 (`iii.b`) evaluate all selected prompt tracks (`P0_untuned`, `P_corr_val`, `P_ce_val`) across `H1_single` (`gpt-6.1-sol`, `gpt-5.4-mini`, and Mistral `codestral-2508`), `H2_ast_obligation_harness` (`gpt-5.4-mini` and `codestral-2508`), `H3_two_step_decompose_aggregate` (`gpt-5.4-mini` Step 1 + `gpt-5.4-mini` and `gpt-6.1-sol` Step 2), and `H4_multi_family_ensemble_calibration_harness` (`OpenAI gpt-6.1-sol + gpt-5.4-mini + Mistral codestral-2508`), keeping total Experiment 10 spend around `~$16.60` (`<= $18.00` hard cap).

**TLDR-end:** [judges-corr: experiment 10 amendment 01] Documented zero-cost wire/endpoint remedies (`top_logprobs=5` on `gpt-5.4-mini`, `codestral.mistral.ai` endpoint for Mistral `codestral-2508`, `gpt-6.1-terra` / `gpt-5.6-terra` for zero-RPM `api.mistral.ai` tiers, and budget-fitted `gpt-6.1-sol` proposer + `gpt-5.4-mini`/`gpt-6.1-sol`/`codestral-2508` Phase 3–4 execution under `$18.00`) before any held-out analysis.
