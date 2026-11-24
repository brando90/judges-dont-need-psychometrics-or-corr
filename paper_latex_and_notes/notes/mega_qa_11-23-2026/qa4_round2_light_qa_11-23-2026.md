# Light QA Round 2 Phase A (`claude-opus-5-5-high`)

- **Model executed:** `claude-opus-5-5-high`

# Tier 2 Light QA report: Round 2 Phase A, Experiment 29 (11-23-2026)

**Overall: needs fixes.** No files were edited. I found **2 P0, 7 P1 and 11 P2** issues. One of them is new and unrelated to the checks you asked for: many Experiment 29 test items were already evaluated in Experiment 28.

Notes on how I checked:
- I scanned the working tree as of 17:09.
- Another agent was editing files during the audit. `10-supplementary.tex` changed at 17:05:12, root `README.md` at 17:05:14, `judge-objectives-status.html` at 17:05:10, and both PDFs were rebuilt at 17:05–17:06.
- Two banned words were present at 17:04 and fixed by 17:05: "untouched" in `README.md:11` and "cells" in `10-supplementary.tex:275`. Findings below reflect the final state.
- `E` = `experiments/29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026`. All differences are comparator minus target on held-out test RPS. CI = the crossed simultaneous 95% interval (`cross_sim`) unless I say "conditional".

---

## P0 (must fix)

### P0-1. "Zero prior test exposure" is false: Experiment 28 already evaluated many Experiment 29 test items
- **Where:** [HOW_TO_READ_RESULTS.md](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026/HOW_TO_READ_RESULTS.md#L4-L6), lines 4–6 ("Newly split for Experiment 29 (zero prior test exposure)").
- **Evidence:** I matched item IDs in the Experiment 29 bundle (`/tmp/exp29_public_raw/exp29_bundle.json`) against `experiments/28_*/requests.jsonl`. Exp 29 test items that Experiment 28's frontier judges had already scored:

  | Suite | Exp 29 test items also in Exp 28 |
  |---|---|
  | `chaos3` | 265 / 713 |
  | `alpha2` | 110 / 412 |
  | `dices3` | 172 / 300 |

  The project had seen these ratings before. The protocol was also written after Experiment 28's results. This breaks reporting rule 1 (prior exposure).
- **Fix:**
  - Replace the claim with: "Split is new for Exp 29, but 265/713 (ChaosNLI), 110/412 (alphaNLI) and 172/300 (DICES) test items were scored by frontier judges in Exp 28; LoRA weights never saw them."
  - Add the same exposure note to the `RESULTS_11-23-2026.md` header and to the caption of the table at `10-supplementary.tex:281`.
  - Optionally add a sensitivity check on only the items Exp 28 never scored.

### P0-2. HelpSteer2 results combine two experiments' code, and the parity gap is undisclosed outside the results file
- **Evidence:**
  - `calibrate_and_score_exp29.py:17,36` loads the `hs2_qwen` CE-trained and Pearson-trained LoRA predictions from Experiment 24's `predictions.npz`. Only the RPS-trained and MSE-trained runs (161, including one CE parity run) are new.
  - The saved parity check `summary.json → hs2_exp24_parity` shows:
    - draw-0 B=64 CE-trained RPS: 0.17098 (new) vs 0.16886 (Exp 24), a gap of +0.00212;
    - **maximum absolute difference in base-model probabilities: 0.293;**
    - maximum absolute difference in CE-trained probabilities: 0.477.
  - [RESULTS_11-23-2026.md](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026/RESULTS_11-23-2026.md#L80) line 80 blames a "`bfloat16` padded-batching delta". No saved test supports that cause, and a 0.29 base-probability difference is far larger than numerical noise would explain.
  - The HelpSteer2 CE-trained vs RPS-trained difference (+0.00054, CI [−0.00045, +0.00152]) is smaller than this code-path gap.
  - The table at `10-supplementary.tex:279–298`, `STATUS_11-23-2026.md:16`, `6-case-study.tex:14` and the one-pager HTML never mention any of this.
- **Fix:**
  - Footnote every HelpSteer2 row: "CE-/Pearson-trained columns are reused from Exp 24; RPS-/MSE-trained columns are new Exp 29 runs; parity RPS 0.17098 vs 0.16886, maximum base-probability difference 0.29."
  - Remove the bfloat16 explanation unless a test confirms it.
  - Do not treat HelpSteer2 CE-trained vs RPS-trained or MSE-trained differences below about 0.003 as meaningful.

---

## P1 (should fix)

### P1-1. Pre-registered hypotheses that failed are not reported, and summaries say the opposite
The saved contrasts at maximum budget (from `contrasts.csv`):

| Hypothesis | What the protocol predicted | Saved result |
|---|---|---|
| H2 (K=2) | Pearson-trained judge with 1D calibration matches CE-trained | **Fails:** CE-trained wins, `ce_scalar_cal` vs `pearson_scalar_cal` = **+0.00429, CI [+0.00058, +0.00800]**, 10/10 draws. So Pearson's K=2 loss is not only scale drift. |
| H3 (K=3) | Calibrated CE/RPS-trained judges strictly ahead of calibrated Pearson/MSE-trained | **Fails on DICES:** MSE-trained with 1D calibration beats CE-trained (−0.00191, conditional interval [−0.00309, −0.00072]); Pearson-trained ties (+0.00020, CI [−0.00328, +0.00367]). On ChaosNLI d=1 (+0.00132, CI [−0.00023, +0.00288]) and HelpSteer2 (+0.00098, CI [−0.00057, +0.00253]) the crossed interval includes 0. |
| H4, second part | Raw MSE-trained improves on raw Pearson-trained (HelpSteer2) | Not shown: +0.00104, CI [−0.01293, +0.01501]. |

Text that contradicts these results:
- one-pager HTML line 253: "every hypothesis isolates cleanly"
- HTML line 225: "Proves MSE resolves scale…"
- `7-discussion.tex:7`: "raw correlation loses … from affine scale drift (even on binary K=2)"
- `10-supplementary.tex:276` and `STATUS_11-23-2026.md:15`: "raw Pearson still fails from scale drift"

**Fix:**
- Add a "Pre-registered hypothesis verdicts (H1–H4)" table to `RESULTS_11-23-2026.md`, built from these saved rows.
- Remove "isolates cleanly" and "Proves".
- Describe the K=2 Pearson result as "level/scale **plus** residual scalar-quality loss that 1D calibration does not remove".

### P1-2. Equality is claimed from a non-significant difference
- **Where:** `10-supplementary.tex:276`, `experiments/README.md:40`, HTML lines 122 and 253 all say "raw MSE LoRA **equals** raw CE LoRA".
- **Evidence:** the difference is +0.00034, CI [−0.00209, +0.00278]. An interval that includes 0 does not show the two are equal.
- **Fix:** write "is statistically indistinguishable from (interval includes 0)".

### P1-3. "Raw CE and raw RPS beat both MSE and Pearson for every K≥3 suite" is too strong
- **Where:** `10-supplementary.tex:276`, `experiments/README.md:40` ("across every suite"), HTML line 253 ("on every K ≥ 3 benchmark").
- **Evidence:** ChaosNLI d=5, RPS-trained vs MSE-trained = +0.00861, CI [−0.00094, +0.01816]. Only the conditional interval excludes 0.
- **Fix:** add "except raw RPS-trained vs raw MSE-trained at ChaosNLI d=5 (crossed interval includes 0)".

### P1-4. "Consistently, across budgets" is false at small budgets
- **Where:** `README.md:11` ("consistently beats correlation across data budgets"), `7-discussion.tex:7` ("consistently tracks human ratings across budgets"), `STATUS_11-23-2026.md:37` ("wins consistently").
- **Evidence:** the RPS-trained judge does worse than the Pearson-trained judge on:
  - DICES at B=16: −0.01516
  - DICES at B=64: −0.00648
  - alphaNLI at B=16: −0.00204

  CE-trained only ties Pearson-trained on alphaNLI at B=16 (+0.00154, CI [−0.03076, +0.03385]). The budget from which the RPS-trained judge stays ahead of the Pearson-trained judge on DICES is B=256.
- **Fix:** quote those budget thresholds from `summary.json` (CE-trained: 16–64 depending on suite; RPS-trained: up to 256).

### P1-5. The status file switches interval type to claim significance
- **Where:** `STATUS_11-23-2026.md:16`: "Under 1D scalar_cal, CE stays better than Pearson on ChaosNLI and HelpSteer2 (`cond_sim95` excludes 0)". Every other headline number uses the crossed interval.
- **Evidence:** the crossed interval includes 0 for ChaosNLI d=1 and for HelpSteer2 (values in P1-1). It excludes 0 only for ChaosNLI d=5 (+0.00231, CI [+0.00065, +0.00398]).
- **Fix:** report both interval types, or say "crossed intervals include 0 except at ChaosNLI d=5".

### P1-6. The running Llama job differs from the frozen protocol, with no amendment
- **Protocol:** `PROTOCOL_11-23-2026.md:25` fixes `chaos3_llama` budgets at [16, 64, 256, 1024, 2048], which is 200 runs.
- **Code:** `train_exp29_lora.py:768–772` runs all 8 chaos3 d=1 budgets. `verify_exp29.py:25` expects 320 runs, and STATUS line 35 says 320.
- **Effect:** the "480 runs" and "1,401/1,401" totals both depend on this undocumented change.
- **Fix:** add `E/AMENDMENT_1_11-23-2026.md`, written before Phase B is merged, recording the larger budget set and its reason. Do not edit the hashed protocol.

### P1-7. The main-body Exp 29 sentence breaks reporting rules 3 and 4
- **Where:** [6-case-study.tex](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/6-case-study.tex#L14), line 14.
- **Problems:**
  - Bracketed intervals give no type, level, resampling unit or draw count.
  - "$+0.00127$ [$d=1$]" uses brackets for a label, so it reads like an interval.
  - The subtraction order is never written in words, for MSE vs CE-trained or for scalar vs logit calibration.
  - "25/25 suites" for Exp 28 should be "25/25 dataset-by-judge pairs". The same error is at HTML line 224.
- **Fix:**
  - State once: "intervals: crossed (draw × group) simultaneous 95%, 4,000 cluster-bootstrap draws".
  - Write orders in words, e.g. "test RPS of the Pearson-trained judge with 1D calibration minus with (K+2) logit calibration; >0 favours logit".
  - Give the real intervals: d=1 [+0.00021, +0.00232]; d=5 [+0.00209, +0.00526].

---

## P2 (minor)

1. **Figure 2 labels** (`plot_and_report_exp29.py`; PNG/PDF):
   - The bottom-left panel is titled "(H3 Test)". In the protocol, H3 compares calibrated CE/RPS-trained judges with calibrated Pearson/MSE-trained judges, not Pearson-trained 1D vs logit calibration.
   - The top-right legend says "(2/item)" and "(10/item)". At d=1 each item has 1 vote; the 2 is the ratio B_total/N_train.
   - "High-dis/Low-dis" (high- and low-disagreement) is never defined.
2. **Axis labels say "RPS" without spelling it out** in both figures. Use "Held-out test ranked probability score (RPS)". In Figure 1, the bottom-right legend covers the HelpSteer2 Pearson interval.
3. **`HOW_TO_READ_RESULTS.md:13`:**
   - "LOVO **lower bound**" is wrong. Leave-one-vote-out is a reference value, not a bound; the protocol calls it a "reference". With about 3.5 votes per HelpSteer2 item it is also noisy.
   - "leave-one-**voter**-out" here vs "leave-one-**vote**-out" everywhere else.
4. **`RESULTS_11-23-2026.md:9`:** the "95% crossed CI" columns are crossed *simultaneous* (max-t) bands. Label them that way and write the subtraction order in words. The table 2 column needs "> 0 favours logit calibration".
5. **`BUDGET_TABLES_11-23-2026.md`:** the 216 rows match `scores.csv` exactly. The header still needs a direction ("lower is better"), the split (held-out test) and a definition of Low-Dis/High-Dis.
6. **Code-check metric:**
   - `verify_exp29.py:176` measures cyclomatic complexity (CC) without counting `assert`. Under that metric everything passes (`LOC <= 55, CC <= 10`).
   - If asserts count, as in radon, these exceed 10: `verify_sanity_and_provenance` (CC 18), `_assert_disjoint_groups` (`prepare_exp29_bundle.py:141`, CC 12) and `verify_code_and_reports` (CC 11).
7. **Banned-word checker gaps:** `verify_exp29.py:188–190` omits "velocity", "bit flip", "north star" and "killer demo". It never scans `STATUS`, `README.md` or the `.tex` files, which is how "untouched" and "cells" got in. Extend both the word list and the file scope.
8. **`prepare_exp29_bundle.py` changed after the protocol freeze** (`_make_hs2_draws`). The random-number calls happen in the same order, so outputs should be unchanged. Record that in `CKPT_exp29.md`, and confirm by rebuilding (bundle SHA `301869bb…`, Exp 24 SHA `c9163578…`).
9. **One-pager:**
   - The README says "Its 13 images" at line 13 but "15 images" at line 7.
   - `_dist/Correlation_vs_Loss_Judges_bundle.zip` is untracked and not in `.gitignore`. Either ignore it or commit it.
   - "crush raw Pearson" (HTML line 253) is informal and too strong.
10. **Local paths:**
    - `HOW_TO_READ_RESULTS.md:7` and `data_manifest.json` point to `/tmp/exp29_public_raw/…`, which no one else can reproduce.
    - Sanity and provenance JSONs contain `/dfs/scratch0/brando9/…`. This is fine in the repo, since the paper does not link it, but redact them if an anonymized code release is planned.
11. **Paper wording:** the venue PDF says "Llama-3.1-8B running". Use "a Llama-3.1-8B replication is in progress and not reported".

---

## Checks that passed

- **Numbers match the saved CSV/JSON outputs:**
  - Every number in RESULTS tables 1–2, the `10-supplementary.tex:290–294` table (all 5 rows × 10 columns), `6-case-study.tex:14`, `STATUS:15–16,28`, `experiments/README.md:40`, and the HTML forest rows matches `scores.csv` / `contrasts.csv` (`cross_sim` columns), including +0.00293 [+0.00118, +0.00467].
  - All 75 budget-threshold rows in RESULTS §3 match `summary.json`.
  - The HelpSteer2 claim "clearing from B=16 / 8 items" holds: the crossed lower bound is +0.0041 and CE-trained wins 8/10 draws.
- **Run counts:** chaos3 440 + alpha2 160 + dices3 160 + hs2 extra 161 = **921 selected runs**, each with nonzero LoRA and head updates. `verify_exp29.py --allow-missing` prints `PASSED … 921/1401 …`, with RPS/CE/TVD recomputed directly to within 1e-11 and no group overlap between splits.
- **Protocol hash:** `protocol.sha256` matches the protocol file (`e9e3b606…`).
- **Gradient checks:** the finite-difference calibrator check gives a maximum of 1.254e-09, matching RESULTS line 79. Every calibration log has a gradient norm below 1e-6.
- **Experiments 26 and 28** match their source files: 35/35; 0.0660/0.0843, 0.0712/0.0996, 0.0639/0.0675, 0.0757/0.0952, 0.0827/0.0867; $1.34; 2,350 items; 25/25; 19/325; $19.69; Pearson plus calibration ahead on DICES from 64 ratings.
- **Framing:** STATUS, `README.md`, `experiments/README.md`, RESULTS, CKPT, the Exp 29 README, the supplementary file and the HTML all say Qwen3-4B is complete (921/921) and Llama-3.1-8B is running (480). The parity figures 0.17098 vs 0.16886 are kept in RESULTS line 80.
- **Banned words** (final tree): none in any file in scope or in the AAAI PDF text.
- **Anonymization:** no "brando", `/Users/`, "mercury", "Stanford", author names, DOI or GitHub link in `6-case-study.tex`, `7-discussion.tex`, `10-supplementary.tex`, `00_main.bib`, `00_main_aaai_2027.tex` or the PDF text.
- **Bibliography:** the new `nie2020chaosnli` and `aroyo2023dices` entries are correct. I checked DICES online: NeurIPS 36, Datasets and Benchmarks track, pages 53330–53342.
- **Code size limits:** all functions in the Exp 29 scripts and `build_bundle.py` pass the repo's own checker; see P2-6 for the radon-style caveat.
- **Figures:** `ax.grid(True)` is set on every axis, with minor gridlines on log axes. The figure PDFs in the paper folder are byte-identical to the ones in `E/figures/`.
- **AAAI PDF:** 11 pages. "Conclusion" appears on page 4 and References start on page 5. No undefined references or citations in `_build/*.log`. The main body has no forward reference to supplementary labels.
- **Calibrator description is accurate:** `logit_cal` uses the trained judge's own centered log-probabilities (`fit_and_apply_cv`), so the conclusion's "recovers learned shape" is correct. Parameter counts: K+1 for `scalar_cal` and K+2 for `logit_cal`. The 12-candidate grid is 3 learning rates × 4 checkpoints.
