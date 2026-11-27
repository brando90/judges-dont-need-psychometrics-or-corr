# Mega QA 11-26-2026: Three-Stage Sequential Review (Experiment 36 + `experiments/README.md` Row 36)

**Date:** 11-26-2026
**Original Review Base (`QA_BASE`):** `036cafa7ab08c9ad9d6c06516cd691538e644520`
**Owned Paths (`QA_PATHS`):** `experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/` and `experiments/README.md` (Row 36 only)
**External Verified Commits:**
- `brando90/trustworthy-theorem-judges@460466dd10b735c5ff25f9e9cb21f0441fcf960b` (`Exp 09` live sync: `250` non-coordinator numeric main ratings across `6` active raters; `T30` HTTP 401 unverified)
- `brando90/veribench@d1c473559d3a7d5f7e7b4f708a9d7b9469afd779` (`Exp 100` `items.jsonl` + all `225` artifact files SHA-256 verified)
- `brando90/cert-judge@f6047849853e65fd533a6e5df2416d6a9d9447d6` (`Exp 68` Study A `E63` + Study B `E39` + `Exp 67` interim 100 native files)
- `Jiaxin-Pei/Potato-Prolific-Dataset@dbb118c29b3d9bcce5aa172d8249ded9ece4df84` (`POPQUORN` `offensiveness` + `politeness_rating`)

## Stage Chain & Model Metadata

| Stage | Attempted / Executed Model | Verdict | Critical | Major | Fixes Applied | Structural | Report |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Stage 1 (Opposite-company clean eyes) | `claude-opus-5-5-high` (`agy`) | `FIXED` | `0` | `4` | `11` | `IMPROVED` | [qa1_stage1_11-26-2026.md](qa1_stage1_11-26-2026.md) |
| Stage 2 (Carbon Orchestrator — user-requested) | `gemini-pro-orchestrator` (`agy`, no `--effort` flag) | `FIXED` | `0` | `4` | `4` | `IMPROVED` | [qa2_stage2_11-26-2026.md](qa2_stage2_11-26-2026.md) |
| Stage 3 (attempts 1–3) | `gpt-6-astra` (`codex` usage limit until Nov 29); `claude-fable-5-1` (`claude` OAuth expired); `claude-opus-5-5-high` (`agy` third-party HTTP 429, resets in 4h33m) | Failed (quota/auth) | — | — | — | — | — |
| Stage 3 (Strongest-tier fresh-context fallback) | `gemini-4-neon-xhigh` (`agy`) | `FIXED` | `0` | `1` | `3` | `IMPROVED` | [qa3_stage3_11-26-2026.md](qa3_stage3_11-26-2026.md) |

**Fallback & Diversity Disclosure:** Opposite-company review was achieved in Stage 1 (`claude-opus-5-5-high`) alongside Google Carbon Orchestrator in Stage 2 (`gemini-pro-orchestrator`). For Stage 3, OpenAI `codex` (`gpt-6-astra`) had hit its usage limit until Nov 29, `claude` CLI OAuth was expired, and `agy` `claude-opus-5-5-high` hit third-party HTTP 429 (`4h33m1s` reset), so Stage 3 ran in a fresh context on `gemini-4-neon-xhigh` (`agy`).

## Consolidated Findings Fixed & Verified Across Stages 1–3

1. **VeriBench `R_v1` & Sibling Pinning (`data_io.py`, Stages 1 & 2):** Replaced unpinned working-tree reads (the sibling `veribench` working tree was dirty, matching only `15/75` gold Lean hashes) with `git show <rev>:<path>` and full per-file SHA-256 verification across all `225` `R_v1` artifacts (`75/75` match), `TTJ_MANIFEST_SHA`, all `33` `native_main` sheets, `TTJ_PACKET_TREE_SHA`, and all `4` `CERTJUDGE_SHAS` (`results.json` regenerated deterministically).
2. **Claim Scoping & Numeric Parity (`write_reports.py`, `experiments/README.md`, Stages 1 & 2):** Scoped `SANMI_PITCH.md` bottom line and ready-to-send note to `9/9` ordinal & 3-class views (disclosing `popquorn_politeness_binary` Pearson edge, `E39` `40/56` RPS wins, and `Exp 67` `dRPS` CIs crossing `0`); fixed `6 of 7` finite binary MSE count; synchronized `experiments/README.md` Row 36 and added parity + loader hygiene tests.
3. **Matrix Heatmaps, Rollout Bars & Crossover Mathtext (`make_figures.py`, Stages 2 & 3):** Fixed inverted `viridis_r` colormap & invisible `inf` cells; plotted floored 8-rep MC rollout means (`M1`, `M5`, `M20`) in `single_pass_vs_rollout.png`; fixed `$` mathtext mangling (`0.05overexpert5`) and `3 x 10^0` minor ticks on `_dollar_axis`; widened matrix heatmap panels (`7.5 in`/panel, `fontsize=5.8`, `overlaps=0, spine_overflows=0` verified).
4. **Structural Cleanups (`data_io.py`, `run_all.py`, `pitch_addenda.py`, `write_reports.py`, Stages 1–3):** Reused preloaded `R_v1` & `native_main` sets in `load_veribench_fsc_pooled` (`191.0s -> 169.5s`), removed dead `_jsonable` branches, duplicate `dices990` oracle call, unused imports, and import-time side effects.

## Post-Stage-3 Verification Receipts

- `python3 experiments/36_.../run_all.py`: exit `0` in `169.5s`; `results.json` numerically identical (excluding `seconds`).
- `python3 -m py_compile ...`: exit `0`.
- `python3 -m pytest experiments/36_.../tests -v`: **12 passed**.
- `git diff --check`: exit `0`; `SanmiPitch.lean` and `lean_check.txt` unmodified (`0` diff vs `036cafa7`); `data_cache/` gitignored; secret scan clean.
