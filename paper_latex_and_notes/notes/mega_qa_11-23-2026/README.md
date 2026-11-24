# Mega QA 11-23-2026: Three-Stage Sequential Review (`judges-dont-need-psychometrics-or-corr`)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/paper_latex_and_notes/notes/mega_qa_11-23-2026/README.md>

Date: 11-23-2026

**TLDR:** Completed a three-stage sequential Mega quality assurance (QA) review through the Antigravity command-line interface (`agy`) from base commit `78e2e5e` through pre-QA `HEAD` `0c6c9dd` (`09:23 PST`–`10:19 PST`, `11-23-2026`). Planned opposite-company `claude-opus-5-5-high` hit `RESOURCE_EXHAUSTED` (HTTP 429, resetting in `59h36m31s`), so three distinct fresh-context strongest Google `agy` reviewers (`gemini-4-neon-xhigh` → `gemini-pro-orchestrator-neon` → `gemini-4-hydrogen-xhigh`) ran sequentially, applying and verifying `21` cumulative fixes across the AAAI-27 (Association for the Advancement of Artificial Intelligence 2027) and arXiv paper, Version 28 one-pager, `experiments/README.md` index, and Experiments 24–25 scripts/docs while keeping all frozen SHA-256 locks, Stanford Digital Repository deposit (`DOI: 10.25740/bh388th6008`), and Experiment 28 files unmodified (`0` open critical issues; `1` open major hardcoded-path issue in read-only Experiment 26). Before final landing, upstream commit `5b7257f` completed Experiment 28's break-even analysis, and `experiments/README.md` was updated to reflect those published results while `experiments/28_.../` remained completely unmodified by QA.

## Reviewer Metadata & Chain Summary

- **Original Review Base / Pre-QA `HEAD`:** `78e2e5e` (close of `11-20-2026` Mega QA follow-up) → `0c6c9dd` (includes builder `59e97d4` plus upstream `scalable-coworker[bot]` commits `de92402`–`0c6c9dd` in Experiments 26–28; final rebase target `5b7257f` landed Experiment 28 break-even results before push).
- **Builder & Coordinator:** Antigravity (`gemini-4-carbon-high` built Experiment 25, paper integration, and Version 28 one-pager in `59e97d4`).
- **Task Classification & Role:** `critical` (`acceptance` across all three Mega QA stages, per `~/agents-config/workflows/qa-correctness.md`; no Trigger Rule 43 benchmark reference gold edits were in scope).
- **Missing Company-Diversity Disclosure:** Planned Anthropic reviewer `claude-opus-5-5-high` failed at Stage 1 invocation 1 with `RESOURCE_EXHAUSTED` (`code 429`, `"Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 59h36m31s."`, `error_id: f31a16ad-4ecf-4564-8525-9fe752e70a57-1`). Per pre-registered fallback policy (preserving strongest-tier floor rather than downgrading to `claude-sonnet-5-5-high`), Stages 1–3 executed in fresh contexts across distinct Google flagship models in `agy`. Cross-company Anthropic re-verification remains pending quota reset (`11-25-2026` night PST) if desired.

| Stage | Requested Model (`agy`) | Actual Model / Effort Reported | Verdict | Critical | Major | Fixes | Structural | Report |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Stage 1 (attempt 1) | `claude-opus-5-5-high` | — (`RESOURCE_EXHAUSTED` 429, `59h36m31s`) | Failed (quota) | — | — | — | — | [COMMON_11-23-2026.md](COMMON_11-23-2026.md) |
| Stage 1 (fallback) | `gemini-4-neon-xhigh` | `gemini-4-neon-xhigh` (`XHigh`) | `FIXED` | `0` | `1` (open read-only `O01`) | `10` | `IMPROVED` | [qa1_stage1_11-23-2026.md](qa1_stage1_11-23-2026.md) |
| Stage 2 | `gemini-pro-orchestrator-neon` | `Gemini 4 Neon` (`High` via Orchestrator Neon) | `FIXED` | `0` | `1` (open read-only `O01`) | `5` | `IMPROVED` | [qa2_stage2_11-23-2026.md](qa2_stage2_11-23-2026.md) |
| Stage 3 | `gemini-4-hydrogen-xhigh` | `gemini-4-hydrogen-xhigh` (`XHigh`) | `FIXED` | `0` | `1` (open read-only `O01`) | `6` | `PASS` | [qa3_stage3_11-23-2026.md](qa3_stage3_11-23-2026.md) |

## Consolidated Findings & Builder Reconciliation

All `21` fixes across Stages 1–3 were verified after Stage 3's final edit by independent builds, file inspection, and deterministic test scripts.

### Key Substantive Scientific & Factual Fixes

| ID | Stage | Severity | Location | Issue & Verification | Status |
| --- | --- | --- | --- | --- | --- |
| `S3-01` | 3 | **Major** | `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/10-supplementary.tex` L167, L218 | Experiment 10 Joint 2×2 validation numbers `$0.1529$` and `$0.1590$` had inverted labels (`0.1529` is Pearson-selected `[Prompt ① + Harness ①]` and `0.1590` is ranked probability score (RPS)-selected `[Prompt ③ + Harness ③]`, diff `0.1529 - 0.1590 = -0.0060 [-0.0493, +0.0613]` matching `analysis_heldout_tables.json` and `RESULTS_frontier_ce_vs_corr.md`). | **Accepted & Verified** |
| `F01` | 1 | **Major** | `paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/6-case-study.tex` L13 | Claimed calibrated `Qwen3-4B` cross-entropy (CE) low-rank adaptation (LoRA) stayed lower on RPS `"at every budget"`, whereas Experiment 24 `scores.csv`/`contrasts.csv` show calibrated Pearson had lower mean RPS at `B_total = 16` (`0.21117` vs `0.21197`, `-0.00080`) and `B_total = 32` (`0.19389` vs `0.19539`, `-0.00150`, `p = 0.0452`). Qualified to `"for B_total >= 64"` (`6/8` Exp 24, `5/5` Exp 25) while keeping `Conclusion` on Page 4. | **Accepted & Verified** |
| `F02`, `S2-03`, `S3-04` | 1, 2, 3 | **Major / Minor** | `paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/judge-objectives-status.html` L109, L121, L242 | Scoped `20/20` raw draw wins to Experiment 25 (`B = 512..8,192`) vs Experiment 24 (`8/10` at `16`, `9/10` at `32`, `10/10` at `64..2,048`), separated Exp 24 crossed p-values (`0.0187`, `0.0390`, `0.0002`), clarified calibrated threshold `B_total >= 64`, and expanded `low-rank adaptation (LoRA)` / `total variation distance (TVD)` at first use. | **Accepted & Verified** |
| `F03` | 1 | **Major** | `experiments/24_.../HOW_TO_READ_RESULTS.md` L8, `README.md` L13, `scripts/plot_and_report.py` | Corrected uniform baseline RPS from `~0.224` (`base_raw` `0.22354`) to exact saved `0.21800` (wired dynamically from `scores.csv`) and added `sync_gdoc_and_drive.py` to script docs. | **Accepted & Verified** |
| `F05`, `S2-01`, `S2-02`, `S3-05` | 1, 2, 3 | **Major / Minor** | `experiments/README.md` | Updated index date to `11-23-2026`, added full folder links for Exps 05 and 15–17, indexed Exp 27 (`DOI: 10.25740/bh388th6008`, `exact 3-item (K = 3, n = 3) counterexample` after Stage 2 caught Stage 1's `"2-item"` error) and Exp 28 (`ChaosNLI SNLI/MNLI-m/alphaNLI + DICES-350/990`, `2,350` items after Stage 2 caught Stage 1's partial dataset title), tightened Row 25 CE p-value from `<= 0.0342` to `p = 0.0342` (`0.034241...`) and `p = 0.0052` on TVD, and expanded `leave-one-vote-out (LOVO)` and `TVD`. | **Accepted & Verified** |
| `S2-04` | 2 | **Major (Code)** | `experiments/25_.../scripts/train_exp25_lora.py` | Caught GPU/CPU reference-retention regression from Stage 1's `_run_two_step_equiv_batched` helper (where `del full_model` only cleared the helper parameter while `run_sanity` retained `full_model`); split into `_run_two_step_live_prefix` and `_compare_two_step_batched_cached` with `del full_model; torch.cuda.empty_cache()` in `run_sanity`. | **Accepted & Verified** |
| `F04` | 1 | **Major (Code)** | `experiments/{24,25}_.../scripts/verify*.py` | Fixed `FileNotFoundError` when `/tmp` raw bundles are absent by running all repo-local SHA-256, manifest, NPZ, provenance, sanity, L-BFGS-B gradient, and CSV consistency (`< 1e-9`) checks locally while skipping raw vote checks with an explicit notice. | **Accepted & Verified** |
| `F06`, `F07`, `S3-02`, `S3-03` | 1, 3 | **Major / Minor** | `one_pager/README.md`, `10-supplementary.tex`, `9b-appendix-concordance-and-calibration.tex`, `6-case-study.tex` | Fixed `one_pager/README.md` (`10 images` → `13 images` matching `build_bundle.py`), added `exp 15` and `16–22` / `26–28` pointers in HTML, aligned Exp 01 symbol/count wording (`q_\lambda`, `256-replication` across `48` validation/test items), clarified HelpSteer2's official validation split as the held-out test set, expanded `TVD` in `6-case-study.tex`, removed disallowed prose terminology, and made subtraction labels explicit. | **Accepted & Verified** |
| `F08` | 1, 2, 3 | **Major (Build)** | `00_main_aaai_2027.pdf`, `00_main_arxiv.pdf` | Rebuilt venue (`11` pp.), arXiv (`16` pp.), and body-only (`5` pp.) PDFs cleanly with `0` errors, `0` undefined refs/citations, `0` overfull boxes, `0` anonymization leaks, and `Conclusion` starting and finishing on **Page 4** (`References` on Page 5). | **Accepted & Verified** |
| `F09`, `F10`, `S2-05`, `S3-06` | 1, 2, 3 | **Structural** | `experiments/24_.../scripts/*.py`, `experiments/25_.../scripts/*.py` | Standardized AST structural audit: Exp 24 erosion `0.842 -> 0.233` (`-72.3%`, max `CC` `44 -> 15`, max LOC `411 -> 94`, max nest `6 -> 4`, removed 2 dead variables), Exp 25 erosion `0.503 -> 0.114` (`-77.3%`, max `CC` `45 -> 15`, max LOC `271 -> 93`, removed unused `import numpy as np` in `plot_and_report_exp25.py`). Preserved all RNG seeds, optimization math, byte-identical Markdown builder outputs, and `ax.grid(True)` on every Matplotlib axis. | **Accepted & Verified** |

### Open Findings (Unmodified Per Guardrails)

- **`O01` (Major, Read-Only Scope):** `experiments/26_five_new_frontier_judges_on_fresh_homework_and_independent_recheck_start_23Nov2026/analysis/compare_with_codex_exp21.py` fails with `FileNotFoundError: 'work/recheck21/scores.csv'` because it hardcodes remote container paths (`work/recheck21/` and `repo/experiments/21_...`). Left unmodified per `COMMON_11-23-2026.md` read-only boundaries (`scalable-coworker[bot]` directory). Coordinator directly cross-checked `experiments/26_.../results/exp21_recheck/{scores,contrasts}.csv` against `experiments/21_.../short_answers/results/{scores,contrasts}.csv`: `1,170` score fields agree to max abs diff `3.5019e-11` (`0` bad) and `936` contrast fields agree to `2.1458e-11` (`0` bad).
- **`O02` (Minor, Read-Only Scope):** `experiments/26_.../README.md` L3 uses a disallowed grid-entry noun, and `experiments/{24,25}_.../scripts/send_run_email.py` L43 logs `"bcc"` in local JSON receipts despite sending only `"To"`. Left unmodified as read-only / historical utility metadata.
- **`O03` (Info, Intentional Architecture):** Numbered experiments (`24_*` and `25_*`) intentionally duplicate utility helpers rather than sharing a mutable module so historical experiment snapshots remain self-contained.

## Coordinator Post-Stage-3 Verification Summary (All Run After Final Edit)

- **Protocol & Immutable SHA-256 Locks:**
  - Exp 18 `PROTOCOL.md`: `OK` (`MATH_11-21-2026.pdf`: `d929b23e8e397d45966ec663539a8ce1c29e961c5deae7d877b0d579d27edb27`)
  - Exp 24 `PROTOCOL_11-21-2026.md`: `OK`
  - Exp 25 `PROTOCOL_11-21-2026.md`: `OK`
  - Exp 27 `PAPER_11-23-2026.pdf`: `6263b743236557455adad26230edaea3ebcf21626f69725fbc50260edbddf148` (exact match; `git status -s experiments/26_* experiments/27_* experiments/28_* experiments/archive/` clean)
  - Exp 28 `requests.jsonl`: `6004ef74d41bfd99415900a4da7661361c2278c2da22db49217d2960d138ee64` (exact match)
- **Verification Scripts:**
  - `python3 -m py_compile` on Exp 24 & Exp 25 scripts: exit `0`
  - `experiments/24_.../scripts/verify.py`: `VERIFICATION PASSED` (protocol SHA-256, `220/220` updates, `22/22` disk reloads `< 1e-7`, `scores`/`contrasts` `< 1e-9`, zero banned terms; raw bundle skipped gracefully)
  - `experiments/25_.../scripts/verify_exp25.py`: `VERIFICATION PASSED` (`Qwen3-4B` 200 runs + `Llama-3.1-8B` 160 runs, `360/360` L-BFGS-B grad norms `< 1e-6`, `scores`/`contrasts` `< 1e-9`, `ax.grid(True)` verified)
  - Exp 27 `reproduce.py`: `All exact rational assertions passed.` (`TVD=1/5`, `excess RPS=3/200`, `CE=1.064735095 nats`)
  - Exp 27 `verify_identities.py`: `200` random cases, max error `3.0995e-11` (`gradient`), `3.8858e-16` (`rps`)
  - `rps_appendix_examples.py`: max identity error `<= 2.2e-15`
  - `experiments/archive/verify_migration.py`: `PASS` (`195` preserved files)
  - Read-only Exp 21 vs Exp 26 CSV check: `1,170` score fields max diff `3.5019e-11` (`bad 0`), `936` contrast fields max diff `2.1458e-11` (`bad 0`)
- **Paper Dual Build, Page 4 Check, One-Pager, & Repo Hygiene:**
  - `make -C paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL all body-only`: exit `0` (`00_main_aaai_2027.pdf` = `11` pp., `00_main_arxiv.pdf` = `16` pp., `body_only` = `5` pp.; `0` `!` / `Undefined` / `Overfull`)
  - `pdftotext -f 4 -l 5 00_main_aaai_2027.pdf -`: `P4_HAS_CONCLUSION: True`, `P4_ENDS_WITH: 'and equalizing Llama-3.1-8B under high draw variance.'`, `P5_STARTS_WITH: 'References'`, `CONC_ON_P5: False`
  - `build_bundle.py`: `5,522,365` bytes (`index.html` + `13` images)
  - `cmp -s AGENTS.md CLAUDE.md`: exit `0`; `git diff --check`: exit `0`; secret scan: clean; banned-word scan across edited/new docs: clean.

## Round 2 Light QA — Experiment 29 Phase A (Opposite-Company `claude-opus-5-5-high`, `11-23-2026`)

- **Requested & Executed Model:** `claude-opus-5-5-high` via `agy` (quota reset; genuine opposite-company cross-family Light QA completed in fresh context).
- **Brief & Report:** [COMMON_round2_exp29_11-23-2026.md](COMMON_round2_exp29_11-23-2026.md) -> [qa4_round2_light_qa_11-23-2026.md](qa4_round2_light_qa_11-23-2026.md) (`2 P0, 7 P1, 11 P2` findings identified; zero code edits by auditor).
- **Key Verified Fixes Applied:**
  1. **P0-1 (Prior Test Exposure):** Documented exact Experiment 28 prior frontier-judge test exposure (`265/713` `ChaosNLI`, `110/412` `alphaNLI`, `172/300` `DICES`; LoRA weights never trained on test items) across `HOW_TO_READ_RESULTS.md`, `RESULTS_11-23-2026.md`, `STATUS_11-23-2026.md`, `AMENDMENT_1_11-23-2026.md`, `10-supplementary.tex`, `experiments/README.md`, and Version 29 `judge-objectives-status.html`.
  2. **P0-2 (HelpSteer2 Reuse & Parity):** Added explicit disclosure across paper, status, one-pager, `HOW_TO_READ_RESULTS.md`, `RESULTS_11-23-2026.md`, and `AMENDMENT_1_11-23-2026.md` that `hs2_qwen` CE/Pearson predictions reuse Experiment 24 while `161` new `rps`/`mse`/parity runs add Exp 29 trajectories (draw-0 `B=64` CE parity RPS `0.17098` vs `0.16886`, identical `seq_lens`, `max_abs_base_prob_diff = 0.2928`, median `0.0`), cautioning against interpreting cross-codebase gaps `< 0.003` RPS.
  3. **P1-1..P1-7 (Honest Hypothesis Verdicts & Reporting Precision):** Added a full Pre-Registered Hypothesis Verdicts (`H1–H4`) table in `RESULTS_11-23-2026.md`, documented partial outcomes (`K=2` `ce_scalar_cal` still beats `pearson_scalar_cal` `+0.00429 [+0.00058,+0.00800]`; DICES `mse_scalar_cal` edges `ce_scalar_cal` `-0.00191` cond; `ChaosNLI d=5` raw RPS vs MSE crossed `[-0.00094,+0.01816]`; small-budget `RPS` thresholds `B=256` on DICES), created pre-Phase-B `experiments/29_.../AMENDMENT_1_11-23-2026.md` (`chaos3_llama` `200 -> 320` runs), and updated `6-case-study.tex` / `7-discussion.tex` for Brando reporting clarity while keeping `Conclusion` strictly on Page 4 of `00_main_aaai_2027.pdf`.
  4. **P2 Fixes:** Fixed Figure 1/2 axis spelling (`ranked probability score (RPS)`), vote-depth legend (`1 vote/item` / `5 votes/item`), bottom-left title, forest legend placement, `LOVO reference` wording, banned-word scope expansion in `verify_exp29.py`, `.gitignore` rule for `_dist/`, `one_pager/README.md` (`15 images`), and `github.com/brando90/` / Exp 26 file links.

## Pending External / Human Actions (Out of QA Scope)

1. Confirm co-author lists and complete human-only OpenReview sign-in + upload of rebuilt [00_main_aaai_2027.pdf](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/00_main_aaai_2027.pdf).
2. Re-publish the updated Version 29 one-pager (`judge-objectives-status.html` + `build_bundle.py`) to the Claude artifact if desired.
3. Optional Phase B audit (`qa5_round2_phaseB_light_qa_11-23-2026.md`) after `chaos3_llama` + `dices3_llama` (`480` runs) finish on `mercury2`, plus fixing `compare_with_codex_exp21.py` relative paths in Experiment 26.

**TLDR-end:** [judges-corr: mega qa 11-23-2026] Round 1 (`gemini-4-neon-xhigh`, `gemini-pro-orchestrator-neon`, `gemini-4-hydrogen-xhigh`, 21 fixes) plus Round 2 Phase A (`claude-opus-5-5-high` opposite-company Light QA, 2 P0 + 7 P1 + P2 fixes applied & verified) keep all Experiment 18/24/25/29 SHA-256 locks, Page-4 AAAI-27 Conclusion layout, and deterministic verifiers passing.
