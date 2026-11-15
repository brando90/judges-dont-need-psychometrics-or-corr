# Mega QA of experiment 04 and the reliability appendix: builder notes

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026/qa/mega_qa_11-14-2026/BUILDER_NOTES.md>

**TLDR:** Three sequential reviewers (Carbon, Codex, fresh Carbon) reviewed experiment 04 and the new paper appendix on 11-14-2026; the final verdict is accept with 0 open critical, major or in-scope minor issues. The reviewer reports are kept unedited as records; this note lists the factual slips in them that the builder found while checking their work.

## Chain

| Stage | Reviewer (harness, model, effort, account) | Time (PST) | Findings: critical / major / minor / nit | Verdict | Commit |
| --- | --- | --- | --- | --- | --- |
| Builder | Claude Code, `claude-opus-5-5` | 19:06–19:20 | appendix, bib entry, empty section-reference fix | — | `03ada1e` |
| `qa1_carbon` | Antigravity CLI 1.5.9, Gemini 4 Carbon (High), personal Google AI Ultra | 19:21–19:56 (35 min) | 0 / 1 / 4 / 3 | accept with minor open issues | `9e4b243` |
| `qa2_codex_vals` | Codex CLI 0.169.0, `gpt-6-astra`, ultra, Vals profile | 19:58–20:17 (19 min) | 0 / 2 / 5 / 1 | accept with minor open issues | `a283556` |
| Builder | Claude Code, `claude-opus-5-5` | 20:18–20:22 | closed QA2-F7 (freeze amendment 3), corrected report-generator wording | — | `be39e2a` |
| `qa3_carbon` | Antigravity CLI 1.5.9, Gemini 4 Carbon (High), fresh session | 20:23–20:52 (29 min) | 0 / 0 / 2 / 1 | **accept** | landing commit |

Brando asked for Codex on the Stanford account. Its saved login could not refresh, and the personal Codex account had reached its usage limit until 11-17-2026, so stage 2 ran on the Vals profile with the same model after Brando said: "Any account is fucking fine." Every stage ran with full permissions in an owned detached worktree; each wrote a report in this folder and a marker in `~/.local/state/judges/mega-qa-04-1114/`.

## Builder verification after stage 3

- `FREEZE.json`: the six frozen files match amendment 3 (three amendments in total).
- `python3 scripts/verify_results.py`: 140/140 checks pass; `results/` unchanged by the rerun.
- `python3 scripts/make_report.py` reproduces `RESULTS_reliability_vs_accuracy.md` byte for byte.
- Paper: `make` exits 0; the log has no `!` lines and no undefined references or citations; 6 pages; `latex_sentence_lines.py --check` passes for `9-appendix-reliability.tex`, `4-proposal.tex` and `00_main_aaai_2027.tex`.
- Appendix numbers spot-checked against `results/phase_a.json`, `phase_b.json` and `phase_c.json` (0.446, 0.0167, 0.228, 0.983, 0.273, +0.082 [0.004, 0.168], 0.180, +0.089 [0.085, 0.093], −0.147 [−0.198, −0.097]).

## Factual slips in the reviewer reports (deliverables unaffected)

1. `qa1_carbon.md`, mathematical audit of the shared offset: it quotes a pooled Φ of 0.400 and Eρ² of 1/2 from components 1/2 and 1/2. The saved values are v_u = 9/50, v_e = 27/50 and pooled rater variance 1/4, giving Φ = 18/97 and Eρ² = 1/4. It also calls the per-stratum REML term convex. Stage 2 corrected both (QA2-F4, QA2-F5).
2. `qa3_carbon.md`, reconciliation table: several rows paraphrase or renumber earlier findings inaccurately. Its "QA2-F1" row describes the Monte Carlo explanations, which stage 2 filed as QA2-F3; stage 2's QA2-F1 was the "every moment" and strictly-proper wording. Its "QA2-F4" row quotes offset components 3/16 and 13/24, which do not appear in the saved output (the saved values are 9/50 and 27/50, and only they give Φ = 18/97). Its QA1-F2 and QA1-F6 rows describe changes different from stage 1's actual F2 (a two-sentence line in `4-proposal.tex`) and F6 (status drift). The resolved or open status it assigns matches the stage reports, and the builder checked the deliverables directly.
3. `qa2_codex_vals.md` left the generated report's wording unchanged, reading the common brief's restoration rule as covering it. That rule only governed rerun churn in results files; the builder corrected the generator in `be39e2a`, and stage 3 reviewed that change.

## Open, outside this QA's scope

- The paper directory's `README.md` still describes a five-page PDF with only the RPS appendix; `8-appendix-rps.tex` has multi-sentence lines; a main-text pointer to the reliability appendix remains an author choice.
- Stage 2 noted that the main paper's existing bibliography text includes the companion-project name and a cited author's surname; whole-paper anonymization for submission was not part of this review.

## Corrections owed outside the repository

The Google Doc section added on 11-13-2026 and the first results email repeated the overstatement that design choice "failed only" under the four violations, and gave the +0.082 interval without the same-split selection caveat. The Doc section is corrected after landing; the run-done email states the corrections.

**TLDR-end:** [judge objectives: mega QA of 04_g_theory_reliability_vs_distribution_accuracy_start_13Nov2026] Final verdict accept (fresh Carbon), after Carbon and Codex stages fixed three real overstatements and one unimplemented secondary analysis. Reviewer reports are preserved with their slips listed here.
