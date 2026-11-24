# Light QA Round 2 Phase B (claude-opus-5-5-high)

- **Model executed:** `claude-opus-5-5-high`

# Tier 2 Light QA report: Round 2 Phase B, Exp 29 complete (`d6930ed` vs working tree, 11-23-2026)

**Verdict: no P0 findings, so this is not a clean PASS.** There are 2 P1 and 6 P2 findings. The P1s are a mislabeled threshold claim that appears in 8 places, and a missing prior-exposure disclosure in the venue PDF. All table numbers match the saved results. I edited no project files. I only read files, re-ran `verify_exp29.py` and ran comparison scripts.

## P1: fix before commit

**P1-1. The "sustained win" thresholds for raw CE/RPS vs raw Pearson are labeled as crossed, but the numbers are mostly the conditional ones.**

What `summary.json` → `sustained_win_thresholds` actually says (B_total):

| Contrast | Conditional 95% band | Crossed 95% band |
|---|---|---|
| `ce_raw_vs_pearson_raw` | 16, 80 (d=5), 64, 16, 16, 64, 16 | 32, 80 (d=5), 64, 64, 16, 64, 64 → **16–80** |
| `rps_raw_vs_pearson_raw` | 16, 80, 64, **256** (Qwen DICES), 64, 64, 16 | 128, **320** (chaos3_qwen d=5), **1024** (alpha2_qwen), 256, 64, 128, 64 → **64–1024** |

Where it goes wrong:
- **Explicitly called crossed but wrong:**
  - [RESULTS_11-23-2026.md:11](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026/RESULTS_11-23-2026.md#L11) says "`64–256` RPS".
  - [STATUS_11-23-2026.md:15](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/STATUS_11-23-2026.md#L15) says "`B=64–256` for RPS".
  - Both leave out alpha2_qwen at 1024 and chaos3_qwen d=5 at 320. The CE range "16–64" also leaves out d=5 at 80.
- **No band type given, and true only under conditional bands, while the surrounding text says "crossed":**
  - "past 16–64 ratings (256 RPS on Qwen DICES)" in [README.md:11](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/README.md#L11), [STATUS_11-23-2026.md:37](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/STATUS_11-23-2026.md#L37), [experiments/README.md:40](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/README.md#L40), [10-supplementary.tex:276](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/10-supplementary.tex#L276) and [judge-objectives-status.html:122](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/judge-objectives-status.html#L122) and line 253.
  - The same phrase is in the **Conclusion** at [7-discussion.tex:7](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/7-discussion.tex#L7), which is on page 4 of the venue PDF.
- **Fix:**
  - In RESULTS and STATUS H1, write "crossed: CE `16–80`, RPS `64–1024` (`1024` alphaNLI, `320` ChaosNLI d=5); conditional: CE `16–80`, RPS `16–256`".
  - In the other files, add "(conditional 95% band)" after "16–64 ratings", or switch to the crossed values.
  - For the Conclusion, a page-neutral option is "CE beats correlation past 16–80 ratings". Rebuild and re-check that the Conclusion stays on page 4.
- **Correct as written:** [6-case-study.tex:14](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/6-case-study.tex#L14). Its claim is CE only, on d=1 panels, where the crossed range really is 16–64.

**P1-2. The venue PDF reports Exp 29 test numbers without saying that some test items had prior exposure (reporting rule 1).**
- [6-case-study.tex:14](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/6-case-study.tex#L14) gives Exp 29 held-out numbers.
- The disclosure (`265/713`, `110/412`, `172/300` test items already scored in Exp 28) is only in `10-supplementary.tex`. That file is excluded from `00_main_aaai_2027.pdf`; `pdftotext` finds no `265/713` in the venue PDF.
- This gap already existed in Phase A (`d6930ed` line 14).
- **Fix:** add a short parenthetical, for example "(Exp. 28 frontier-scored 265/713, 110/412, 172/300 test items; LoRA never trained on them)". Then rebuild and confirm the Conclusion is still on page 4 and References start on page 5.

## P2: nice to fix

1. **"10/10 draws" is true for CE only.** [STATUS_11-23-2026.md:15](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/STATUS_11-23-2026.md#L15) applies it to both CE and RPS. For `rps_raw` vs `pearson_raw`, `draws_target_better = 9` on alpha2_qwen and on chaos3_qwen d=5. Fix: "CE 10/10; RPS ≥9/10".
2. **An untested explanation is stated as cause.** "because Pearson training already compressed non-mean logits" appears in [STATUS_11-23-2026.md:17](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/STATUS_11-23-2026.md#L17) and [10-supplementary.tex:276](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/10-supplementary.tex#L276). No script prints evidence for it. "showing residual scalar-ranking loss" at STATUS:16 has the same problem. Fix: write "possibly because… (untested)" or remove.
3. **"Qwen" should be "Qwen d=1" for the logit-calibration win.** The claim that `ce_logit_cal` beats `pearson_logit_cal` crossed on "Qwen (+0.00293)" is only true for d=1. On d=5 the gain is +0.00104 with crossed interval [−0.00034, +0.00241], so not significant. Affected lines: [RESULTS_11-23-2026.md:13](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026/RESULTS_11-23-2026.md#L13), STATUS:37, experiments/README:40, HTML:122/253 and 6-case-study:14.
4. **Figure captions miss parts of rules 2 and 5.**
   - The supplementary Fig 1 caption (around line 305) has no non-convex optimizer caveat and does not say lower RPS is better.
   - The supplementary Fig 2 caption (around line 312) has no line on how the judge was chosen, no "lower is better" and no exposure note.
   - HTML Figure 1g ([judge-objectives-status.html:257](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/judge-objectives-status.html#L257)) has no line on how the judge was chosen.
5. **The Fig 1 x-axis label is misleading.** It reads "Training rating budget B_total" ([plot_and_report_exp29.py:67](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026/scripts/plot_and_report_exp29.py#L67)). B_total counts train, validation and calibration ratings, which Fig 2 states as `2 * N_train * d` (line 165). Fix: "Total allocated ratings B_total (train+val+cal)". Separately, `HOW_TO_READ_RESULTS.md` never defines a "sustained" threshold. The code's rule is: the smallest budget from which every larger budget also has a lower bound above 0.
6. **Provenance gaps.**
   - [AMENDMENT_1:14](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/experiments/29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026/AMENDMENT_1_11-23-2026.md#L14) credits `max_abs_seq_len_diff = 0`, median `0.0` and `3.87%` to `summary.json`. Those keys are not there. The values are hard-coded strings in `plot_and_report_exp29.py` lines 235 and 368.
   - I recomputed them from the two npz files and all are correct: median 0.0, max 0.29278, 3.870% of items have some class difference above 0.05, sequence-length difference 0. Fix: have `_hs2_parity_check` write them to `summary.json`.
   - `verify_exp29.py` reports "skipped raw-bundle checks (absent)", so the bundle SHA was not re-checked on this machine.
   - The script's banned-word scan does not cover the HTML or the one-pager README. I scanned both by hand and found no banned words.
   - Out of scope and not modified: [1-abstract.tex](file:///Users/brandomiranda/judges-dont-need-psychometrics-or-corr/paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/1-abstract.tex) lines 7, 9 and 19 contain banned terms ("bit flip", "north star") inside `%` comments. They do not appear in the PDF.

## Checks that passed

- **Run counts:**
  - `verify_exp29.py` → `PASSED … 1401/1401 nonzero LoRA/head updates across 6 studies`.
  - The provenance files give `chaos3_llama` 320 runs and `dices3_llama` 160, so Llama has 480.
  - Qwen has 440 + 160 + 160 + 161 = 921. Every run has a nonzero update.
- **Max-budget tables:**
  - RESULTS §1 and §2 match `scores.csv` and `contrasts.csv` in all 7 rows: uniform, CE, RPS, MSE, Pearson, both calibration variants, differences and crossed simultaneous bands. The human leave-one-vote-out (LOVO) references come from the data manifest through the plotting script.
  - The supplementary table at `10-supplementary.tex:290–296` matches in all 7 rows, including the Llama rows (+0.03435 [+0.01927, +0.04944] and +0.04452 [+0.01905, +0.06999]).
- **Threshold table:** all 63 entries in RESULTS §3 match `summary.json`.
- **Budget tables:** all 312 rows in `BUDGET_TABLES_11-23-2026.md` match `scores.csv` (RPS, CE, TVD, Pearson r, low/high-disagreement slices).
- **H1–H4 evidence:**
  - Every number checked: the K=2 control (+0.00034, +0.00119, +0.00429 at 10/10), 5/6 CE>MSE and 5/6 RPS>MSE crossed, chaos3_llama CE minus MSE [−0.00066, +0.00565] with the CE and TVD versions clearing crossed, dices3_llama −0.01610 (conditional interval [−0.02172, −0.01048]), and the logit-calibration effects +0.00127, +0.00368, +0.00073, +0.00006, −0.00011, −0.00075, +0.00205.
  - The HTML forest-plot entries also match.
- **Phase A disclosures:** still present in RESULTS, AMENDMENT, CKPT, HOW_TO_READ, STATUS, supplementary, experiments/README and HTML. These are the Exp 28 exposure counts (265/713, 110/412, 172/300), the HelpSteer2 reuse with parity 0.17098 vs 0.16886, and the 320-run `chaos3_llama` grid.
- **No stale status text:** no "Phase A", "in progress" or "921/921" remains as a current-status claim in any scoped file. The one "Phase B" mention, at AMENDMENT:5, is historical and correct.
- **Banned words:** none in the scoped files.
- **Anonymization:** `\author{Anonymous Submission}`. Neither the venue PDF nor the .tex sources contain an author name, Stanford, mercury2, GitHub or the DOI.
- **Page layout:** the Conclusion ends on page 4 and References begin on page 5. Both PDFs were rebuilt with the new text (`1,401/1,401` appears in each).
- **Reporting rules:** RESULTS and HOW_TO_READ name the data split, metric, subtraction order in words, uncertainty (4,000 draws, crossed and conditional bands) and how the judge was selected (12 AdamW candidates, non-convex objective).
- **Figures:** both regenerated figures show all 7 panels with gridlines (major plus minor on log axes). The difference axes state their order.
