# Claude Code prompt: Paper 1 advisor-feedback update (27 Nov 2026)

Paste the fenced block into Claude Code. It layers on top of [claude_code_paper1_prompt_corrected_27Nov2026.md](claude_code_paper1_prompt_corrected_27Nov2026.md).

```text
Read and execute the corrected prompt at
https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/prompts/claude_code_paper1_prompt_corrected_27Nov2026.md
(git pull main first; it supersedes all earlier versions and the addendum). Then apply this ADVISOR FEEDBACK UPDATE (advisor's Nov 26 notes). Where they conflict, this update wins.

GROUND RULES
- Work directly on main (no branches/PRs), per AGENTS.md / CLAUDE.md / .cursor/rules. git pull before every push. Commit in logical groups.
- $0 API spend, CPU only. No paid API or GPU jobs without Brando's explicit yes.
- Never invent numbers; report "not run" honestly. Don't share or send anything to anyone.
- Use repo files as sources (Google Docs may be inaccessible). This update is archived at docs/prompts/claude_code_paper1_advisor_feedback_update_27Nov2026.md.
- Pre-register hypotheses, metrics and paired tests in each new experiment's README BEFORE running. "Beats" = lower mean held-out proper score + significant pre-registered paired test.

SCOPE (Paper 1 = evaluation paper)
- Thesis: correlation, majority-label agreement/accuracy/kappa, and psychometric reliability can't certify a judge as a forecaster of the human rating distribution. Use a strictly proper score matched to the label encoding + a base-rate skill score Skill_S = 1 - S(judge)/S(base rate), stratified by human agreement. Mirage framing: a metric choice producing misleading conclusions about judges. Lean on it only if rank reversals are shown.
- Novelty is NOT "target is a distribution" (credit Pavlick & Kwiatkowski, TACL 2019; Plank, EMNLP 2022) and NOT "use proper scores" (Lyu, Do & Pergola, arXiv 2610.37868, Oct 2026; Li et al., arXiv 2510.00263, Findings of EMNLP 2026). Novelty = the measured practice gap (Exp 43: 43/49 papers have no item-level proper score; 29/30 scored papers report correlation; 13/13 pairwise papers report agreement/accuracy/kappa) + evidence that metric choice changes real judge rankings.
- Training is secondary: ordinal extension of Li et al. Calibration closes most of the gap (Exp 41: raw CE-minus-Pearson up to 0.1282 normalized RPS; after calibration <= 0.0024; regularized MSE + calibration ties CE). Practical rule: evaluate with a proper score; train or at least calibrate with one.
- Bitter Lesson / scaling laws, fixed-budget N-vs-k allocation, end-to-end fine-tuning, and drift go to Discussion/Future Work, not the main claim.

THEORY / WRITING (README, docs, paper_latex/)
1. Define p*(y | c, rater pool, rubric) up front: which raters, eligibility/weighting, rubric. Changing the pool changes p*; that choice IS the construct. On consensus items p* is peaky and a good judge should be too; remove any overstated "impossibility" text.
2. r^2 = 1 - MSE(best affine fit)/Var(Y): correlation = squared error after optimal affine recalibration, so it captures discrimination and ignores calibration (+1 offset example). Include E[(X-Y)^2] = (mu_X-mu_Y)^2 + (sigma_X-sigma_Y)^2 + 2 sigma_X sigma_Y (1-Corr).
3. Frame around ENCODINGS, not first moments: scalar score -> mean only (loses spread; certainty on 2 vs 50/50 on {0,4}); one-hot -> squared error = multiclass Brier, log loss = CE (proper; use for binary/unordered categorical, never RPS there); cumulative 1[y<=k] -> RPS (proper + ordinal distance); continuous -> CRPS. Ordinal Likert = flagship empirical opening (Li et al. leave it to future work).
4. A proper score is unbiased with ONE sampled rating per held-out item. Multiple ratings are needed only for distance-to-p*, per-item divergence, stratification or human ceilings.
5. Answer kappa / Krippendorff's alpha pushback with the base-rate skill score.
6. Accuracy vs. majority label: proper but not strictly proper. Label it as argued analytically unless you add a zero-sorry Lean lemma.
7. Training section: don't strawman. Most teams prompt frontier models or fine-tune with CE on hard labels; "optimizing correlation" in practice means selecting prompts/rubrics/judge models by correlation. Our shift = distributional targets + proper selection/calibration.
8. Lean scope: claim machine-checking only for what compiles with zero sorry. Anti-properness is checked only in squared form (rho^2 < rho); never write "Lean-verified sqrt(rho) > rho".

EXPERIMENTS (new numbered folders with start dates; item/task-clustered bootstrap CIs, B >= 1000)
A. Exp 39 fix + CPU re-run: (a) deterministic tie-break for the RPS oracle; (b) genuinely adaptive D-study selector from pilot variance components; (c) strictly matched total rating budget B (remove hidden pilot leakage in the proper-score selector); (d) Holm-Bonferroni across cells. Update results and write CORRECTION_NOTE_EXP39.md.
B. Exp 44 eval-protocol diagnostics on saved artifacts: Skill_S for CE, Brier, RPS (matched to each dataset's encoding); human-agreement stratification (independent rater splits for stratifying vs scoring where possible); power/MDD curves over (N_test, k_test) showing k_test = 1 is practical.
C. Real-data rank reversal: rank existing judges by correlation vs held-out proper score + skill; report Kendall tau between rankings and any significant reversals. Consolidate Exp 28 (DICES-990; ChaosNLI MNLI-m: use CE/Brier there since labels are unordered, TVD only as a distance), Exps 4 & 41 (reliability vs proper score) and Exp 23 SummEval (raw Pearson 0.877 vs calibrated 0.835 system-level Spearman; report honestly). Say "higher mean Pearson despite worse proper score", not "strictly worse", unless paired CIs exclude zero.
D. Circularity check: a CE/distribution-fit judge must also be evaluated on correlation (Pearson, Spearman, Kendall) and a downstream use (system ranking and/or win-rate prediction), against calibrated-Pearson and regularized-MSE+calibration baselines.
E. Budget split (if feasible on CPU): fixed B = N*k, more items x 1 rating vs fewer items x k. Note Li et al. compare RL on 50K x 1 label vs SFT on 5K x 10, so method and split are confounded there; design ours to separate them. Otherwise list as future work.
F. Drift: if feasible, test a judge calibrated on older models/tasks applied to newer ones; else discuss as the main reason judges stay low-data.

CLEANUP
- Every "N ratings" claim gets a unit (per item vs total). Exp 28's "4-8 ratings" = 4-8 total training ratings at one rating per item in a cached-forecast calibration fit over five frontier judges. Verify against the repo before writing; add a one-paragraph Exp 28 description wherever it's cited.
- Delete the burden-shifting sentence everywhere.
- Repo-wide grep and strip: share.gemini.google, raw AI prompts, advisor names inside prompts, Google Doc share links in appendices.
- Do not cite arXiv 2606.19544 (withdrawn) or OpenReview YawwpIWqTi (withdrawn ICLR 2027 G-theory submission). Gtheory4LLM (CRAN) is OK. Alpha Parity (OpenReview 67Z2nGMzNI): mark as unverified submission. Psychometric turn = "an emerging thread" (1/49), not a rebound.
- Cite Elangovan et al., "Beyond correlation", ICLR 2025, arXiv 2410.03775 (uncertainty stratification, binned JSD, why correlation not accuracy). Report binned JSD as a comparison metric in Exp 44.
- Exp 43: confirm 49-paper counts are canonical, re-run tally.py twice for determinism, and sync README.

PAPER
- Revise paper_latex/ to this scope (title/abstract/intro, construct, theory, empirics wired to committed outputs, discussion/future work, clean .bib). Compile with zero errors, zero [?] and zero ??.
- Update README.md (Goal / Established / Open / Not claimed), including the RPS-centric Thesis section, to the general encoding-matched thesis.

VERIFY + REPORT
- Run lake exe cache get (never rebuild Mathlib) + lake build (zero sorry, no custom axioms), pytest, tally.py, the grep checks, and the LaTeX build.
- Final report: commits, Lean results, Exp 43 counts, Exp 39 corrected results + note path, Exp 44 numbers, rank-reversal results, what was not run and why, PDF page count, and a TL;DR.
```
