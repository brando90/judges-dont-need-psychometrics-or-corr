# Mega QA of Experiments 09 and 10: Builder Notes

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/qa/mega_qa_11-16-2026/BUILDER_NOTES.md>

**TLDR:** Three sequential strongest-model reviews (`OpenAI Codex v0.169.0 gpt-6-astra ultra` -> `Antigravity Gemini 4 Carbon High` -> fresh `OpenAI Codex v0.169.0 gpt-6-astra ultra`) audited Experiment 09 (`ii.a–c` open-weight real judges), Experiment 10 (`iii.a–b` frontier artificial intelligence (AI) judges), and `experiments/README.md` from original base `233d5047d20da8591b4609ab52242ceed8717152` on 11-16-2026. Final verdict: **FIXED / ACCEPT** with `0` critical issues, all `13 + 4 + 6 = 23` major issue groups resolved and deterministically verified (`917` checks in Experiment 09; `14` frozen files, `28` status matrices, and `75/75` certificates per objective in Experiment 10).

## Chain & Model Provenance

| Stage | Reviewer (harness, model, effort, account, session) | Time (PST) | Findings: critical / major | Verdict | Commit |
| --- | --- | --- | --- | --- | --- |
| Builder | Antigravity `Gemini 4 Carbon (High)` (`7fa0b312` + subagent `4b9ddc42`) | 10:35–14:01 | Pre-QA self-audit fixed `verify_results.py` assertion, `make_reports.py` column alignment, and `RESULTS_frontier_ce_vs_corr.md` null-contrast wording | — | `a346d5c` |
| `qa1_codex_vals` | OpenAI Codex CLI `v0.169.0`, `gpt-6-astra`, `ultra`, Vals profile (`CODEX_HOME=~/.codex-vals`), session `01a1e662-1640-7fc1-973a-16e56951f441` (`176,604` tokens) | 14:02–14:19 (17 min) | `0 / 13` (all 13 fixed; detailed receipt in `../full_base_review_11-16-2026/REVIEW.md`) | `FIXED` | `c20517c` |
| `qa2_carbon` | Antigravity `Gemini 4 Carbon (High)`, fresh subagent `24c24eed-ed73-4686-b30b-cd5de5931bed` (middle builder-family review per `qa-correctness.md` after `claude-fable-5-1` failed auth on `~/.claude-su` and hit Fable daily quota until 23:00 PST on `~/.claude`) | 14:23–14:50 (27 min) | `0 / 4` (all 4 fixed; expanded Exp 09 verifier from `394` to `916` checks) | `FIXED` | `9546ab0` |
| `qa3_codex_vals` | OpenAI Codex CLI `v0.169.0`, `gpt-6-astra`, `ultra`, Vals profile (`CODEX_HOME=~/.codex-vals`), fresh session `01a1e68f-08c0-7472-a10a-c76bfaabf1b3` (`158,962` tokens) | 14:51–15:02 (11 min) | `0 / 6` (all 6 fixed; saved numerical forecasts and all 9 figures unchanged; `917` checks in Exp 09) | `FIXED` (accept) | landing commit |

## Reconciliation of Material Findings Across Stages

1. **Stage 1 (`qa1_codex_vals`, 13 fixes):**
   - Fixed the maximum-entropy logit-shift bug in Experiment 09's `compute_human_ceiling_and_info_gap` (`logsumexp(z)` + analytic gradient + mean-matching assertion `< 1e-7`) and regenerated `analysis_summary.json` (updating `pearson_G_vs_rps_diff` to `+0.198` and `spearman_G_vs_rps_diff` to `+0.136`; primary loss metrics and saved forecasts unchanged).
   - Clarified overlapping `250`-split Monte Carlo interval semantics in Experiment 10 (`conditional split variability` vs population task-family sampling), recovered parse/fallback status provenance across `28` Phase 3/4 forecast vectors (`recover_forecast_status.py`), fixed missing-control `-1 -> n/a` display in Experiment 09, aligned sanity-check match labels in Experiment 10, disclosed `P2`'s historical summary statistics and `60/75` reference-hash provenance, hardened unique cost-ledger IDs and UTF-8 token reservation bounds in `common.py`, and enforced sequential `FREEZE.json` SHA-256 verification across all `14` frozen files in Experiment 10.
2. **Stage 2 (`qa2_carbon`, 4 fixes):**
   - Caught and fixed an exact BLAS `dgemv` vs NumPy reduction `1-ulp` rounding discrepancy in Experiment 09's `family_bootstrap_and_signflip` that had produced `p_val = 0.0` on the `2^14 = 16,384`-configuration ordinary-task test (enforcing `pval >= 2 / 2^m_active` in `verify_results.py`).
   - Documented Experiment 10's Phase-4 legacy harness JSON key -> executed model mapping (`H2_ast_gpt61__*` and `H3_twostep_gpt61__*` executed `gpt-5.4-mini`; `*_mistral35__*` executed `codestral-2508`) and updated headline comparisons to matched `gpt-5.4-mini` `H1 -> H3` numbers (`0.181 -> 0.160` at `P0`, `0.183 -> 0.114` at `①`, `0.141 -> 0.117` at `③`).
   - Corrected Stage 1's held-out family count wording in Experiment 09 `HOW_TO_READ_RESULTS.md` (`4` test families held out at every learning-curve point) and polished clipped/unlabeled figure elements and remaining first-use acronyms across all 9 PNGs and active docs.
3. **Stage 3 (`qa3_codex_vals`, 6 fixes):**
   - Verified all Stage 1 and Stage 2 fixes remained intact with zero numerical drift across saved forecasts and figures, and resolved 6 remaining precision items (`qa_amendment_3`): statistical inference phrasing, Experiment 10's exact smoothed ranked probability score (RPS) disagreement-floor formula vs Experiment 09's same-mean Kullback-Leibler (KL) information gap `G_i`, reference-scale and aggregation disclosures, optimizer/initialization scope statements, and `undetermined` placeholder status in Experiment 09's prospective scaffold helper.

## Final Builder Deterministic Verification After Stage 3

- `python3 experiments/09_.../scripts/verify_results.py`: **PASS** (`917` deterministic checks succeeded).
- `python3 experiments/10_.../scripts/verify_results.py`: **PASS** (`freeze_files_verified: 14`, `forecast_status_vectors_verified: 28`, `75/75` simplex certificates per objective, `exp02_historical_parity_max_abs_diff: 0.0`, `settled_actual_usd: 17.851074 <= 18.0`, `secret_hits: 0`, `forbidden_public_json_keys: 0`).
- `git diff --check 233d5047d20da8591b4609ab52242ceed8717152`: **PASS** (zero whitespace/conflict issues; zero files outside `experiments/09_...`, `experiments/10_...`, and `experiments/README.md`).

**TLDR-end:** [judges-corr: mega QA 11-16-2026 builder notes] Final verdict accept after 3 sequential strongest-model stages (`Codex gpt-6-astra ultra` -> `Gemini 4 Carbon High` -> fresh `Codex gpt-6-astra ultra`); all 23 findings fixed and deterministically verified.
