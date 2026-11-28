# State of evidence: calibration follow-up experiments 46–52 and spend ledger

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/STATE_OF_EVIDENCE.md>

**TLDR:** One line per follow-up experiment, with its strongest number, its uncertainty and its cumulative API spend. The API ceiling is \$2,000 total, a ceiling and not a target, shared with Claude Code's overnight runs. Claude/Codex subscription credits are never used. Other workers: append your own section and spend rows; do not rewrite these.

**As of:** 11-27-2026 (Friday, Pacific time). Maintained by Grok Bot for Brando Miranda.

## Experiments 46–52 (folder `experiments/NN_*_start_27Nov2026/`)

| exp | question | strongest result (95% CI) | supports thesis? | API spend |
|---|---|---|---|---|
| 46 | Do real LLM judges rank differently under correlation than under proper scores? (interim) | GPT-6 Astra out-correlates GLM 5.3 (Spearman +0.021 [+0.010, +0.034] SNLI; +0.034 [+0.014, +0.056] MNLI) but has lower Brier skill (−0.039 [−0.056, −0.023]; −0.098 [−0.117, −0.080]), Holm-significant. 4/5 frontier judges are worse than the base rate on DICES-350 (skill −0.10 to −0.18, all CIs < 0) at 80–82% top-1 agreement. Calibration removes the reversals (H3). | yes, with the H3 qualification | \$0.00 (no gateway key on box) |
| 47 | Labels needed for calibration to close the gap | At n = 25 single ratings, skill difference −0.066 to −0.116, larger than the CE head's whole skill (≤ 0.071); needs 100–200 labels | yes | \$0 |
| 48 | Does calibration transfer under shift? | Shifted map loses skill vs an in-domain map in 78/82 cases (five frozen frontier judges plus two G-Eval rubric transfers) (CI < 0) and is worse than uncalibrated in 33/82 (e.g. DeepSeek V4.1 Flash, DICES-990 map on SNLI: −0.537 [−0.593, −0.485]); linear heads reopen in 21/22 views | yes | \$0 |
| 49 | Is Exp 23's system-level Spearman drop real? | 0.8765 → 0.8349 reproduced; document bootstrap Δ CI [−0.049, +0.114], p = 0.52; caused by the 4-bin calibrator (a continuous map gives Δ +0.001) | neutral (removes an apparent counterexample) | \$0 |
| 50 | Gaps stratified by human agreement | Frontier skill falls from 0.16–0.91 to −0.38–0.11 on the least-agreement fifth while Pearson stays 0.38–0.86 (NLI); prediction R1 failed (the raw gap concentrates on high-agreement items) | mostly yes, one miss | \$0 |
| 51 | Exp 41 across a fixed ridge grid | Raw gap > 0 in 119/120 cells; matched gap within ±0.003 nRPS in 152/160 | yes (robustness) | \$0 |
| 56 | Labels needed to calibrate the frontier judges themselves | Every judge's mean calibrated skill > base rate at n = 100 single ratings on all 5 sets; at n = 25, 21/25 judge-set cases below uncalibrated; at n = 200 still 0.003–0.022 below full calibration | yes (recalibration has a label cost) | \$0 |
| 52 | Skill-score estimator, one vs many ratings | One-rating estimator unbiased with a known base; at equal rating budget, one rating per item beats five | methods support | \$0 |

## Spend ledger (cumulative, estimated at configured gateway list rates)

| date (Pacific) | experiment | what | USD | cumulative |
|---|---|---|---|---|
| 11-27-2026 | 46–52 | all CPU or frozen receipts; API runner built and mock-tested, not run (no key on box) | 0.00 | 0.00 |

## Abstract/Elangovan–Li QA, 27 Nov 2026 late evening (Grok Bot)

| date (Pacific) | item | what | USD | cumulative |
|---|---|---|---|---|
| 11-27-2026 | Exp 56 | CPU only, frozen forecasts | 0.00 | 0.00 |
| 11-27-2026 | Mega QA reviewers | 'Fable 5.2' and 'Codex Vals Ultra' not run: no gateway key on the box (Stanford gateway returns 401 without a key; Vercel AI Gateway lists anthropic/claude-fable-5.2 but no key is configured; no model named 'Codex Vals Ultra' in any catalog checked; claude/codex CLIs not installed and subscription credits are excluded). QA done by hand: docs/notes/qa_abstract_elangovan_li_27Nov2026.md | 0.00 | 0.00 |

## Reconciliation with Claude Code's overnight commits, and mega QA, 28 Nov 2026 (Grok Bot)

Claude Code commits since 984ba71 (pulled at 929bc19): 60a3ffb, 1e68380, 57a7640, dc81ad9, 929bc19. They are mostly blog-support notes (`docs/notes/blog_support_27Nov2026/`), plus `experiments/46_*/ERRATUM_11-27-2026.md`, a one-line pointer in the Exp 46 README, and a LEAN_TODO update. None of them edit the abstract, the intro or Exps 48/56.

**Conflict.** The ERRATUM says Exp 46's "H1 is supported" is wrong under the frozen H1, which names Pearson: 0 significant Pearson-vs-skill reversals at B=4,000. The 2 Spearman reversals (GPT-6 Astra vs GLM 5.3, SNLI and MNLI) are real but secondary. At B=20,000 the only Pearson reversal is between two readouts of the same Qwen2.5-0.5B. Our abstract and intro (7f06fba) headlined the Spearman reversals without the primary null. The Codex reviewer independently raised the same problem (F08).
**Resolution.** Claude's reading follows the frozen protocol, so it wins. The abstract and intro now lead with the preregistered Pearson null and call the Spearman reversals secondary, with NLI labels coded as ordered (Exp 46 PROTOCOL line 26). The numbers themselves don't conflict; only the framing did. The blog note's always-No baseline (269/350 = 76.9% on DICES-350) is consistent with our DICES base-rate statements.

**Mega QA.** Codex CLI (gpt-6-astra, effort ultra, read-only) on Brando's "vals" subscription reviewed main at 929bc19. It found 20 issues and all 20 were accepted and fixed (triage in `docs/notes/qa_abstract_elangovan_li_27Nov2026.md`). The Fable 5.2 review was not run because every Claude CLI login was logged out or expired (`docs/notes/qa_fable52_28Nov2026.md`). No new numbers were introduced.

| date (Pacific) | item | what | USD | cumulative |
|---|---|---|---|---|
| 11-28-2026 | Mega QA | Codex CLI on Brando's subscription (authorized for this review pass, not experiments); no API calls | 0.00 | 0.00 |

## Experiments 57–61, the abstract against the advisor's notes, and QA, 28 Nov 2026 (Claude)

As of 11-28-2026 (Saturday, Pacific time). The protocols for Experiments 57–60 were committed first (ae58f5c; amendment A1 for Experiment 59 in 973b0a4), the results afterwards (cda2427), the paper and docs in abb1e00. Experiment 61 (post hoc, descriptive; its own protocol) is 3c447d1, and the paper edit pass that answers the advisor's two warnings and the QA findings is 809bb44. "Skill" is one minus the proper-score risk of the judge's forecast over the risk of the base-rate forecast on held-out ratings (higher is better). Intervals are item-clustered bootstrap 95% intervals with 4,000 draws, except Experiment 59, whose intervals are percentile intervals over 200 random pool/test splits, paired across paths (they cover the choice of labeled pool, not item sampling).

| exp | question | strongest result (95% CI) | supports thesis? | API spend |
|---|---|---|---|---|
| 57 | How large are the estimated calibration gain and the calibrated skill for each judge, and which of the two does Pearson r track? | Raw skill equals calibrated skill minus estimated calibration gain by construction (largest residual 3e-17, a floating-point check of the code). Eight cells have raw skill below 0 by point estimate (all on DICES) and all 25 calibrated skills are above 0. DICES-350 Astra: raw −0.143, calibrated skill +0.072, estimated calibration gain +0.215. | partly. H1 (Pearson tracks calibrated skill better than raw skill) held in 1 of 5 sets, so it is falsified. H2 (judges differ more in estimated calibration gain than in calibrated skill) held in 4 of 5. H3 (the DICES-350 rescue is not noise) held. | \$0 |
| 58 | How far is the raw implied event rate from the human rate, and do calibration or counting verdicts fix it? | The raw soft rate misses the human rate by more than 5 points, with the whole interval above 5, in 13 of 25 cells. DICES-350 human unsafe rate 32.7%, raw soft rates 8.6 to 24.0% across the five judges; largest miss −24.1 [−25.9, −22.2] (DeepSeek). Calibrated soft-rate errors are at most 0.1 points with every interval inside ±2.2 (mechanical: the map reproduces the training folds' label shares). Hard verdicts are off by −20.4 to −25.2 points on DICES-350, but a hard-label judge matching the human majority on every item would itself be off by −9.8 [−12.7, −7.1] (DICES-350) and −9.5 [−12.1, −6.9] (DICES-990) (Experiment 61). Raw soft gate flips: 39 of 125, 20 significant (descriptive, grid-dependent). | yes for the level claim; the hard-label part is partly an estimand gap. | \$0 |
| 59 | Choose among the five judges by Pearson r and then recalibrate (path A), or choose by a cross-validated proper score (path B)? | A has lower mean regret than B in all 20 budget-set cells, 17 with the interval below 0. At n = 25 the A minus B difference in points of skill is −3.55, −2.37, −1.13, −6.19, −5.46 (SNLI, MNLI, alphaNLI, DICES-350, DICES-990). H1 and H2 falsified. Against a uniformly random pick (Experiment 61, post hoc) path B has higher mean regret in 11 of 20 cells (9 significant, all at n ≤ 100), A in 1 (not significant), C in 0. A raw-proper-score path (C, not a preregistered decision) has lower regret than A in all 8 DICES cells (+0.21 to +1.74) and is within noise of A on 9 of 12 NLI cells. | **no.** It goes against our recommendation to select by a cross-validated proper score. The introduction and the supplement report it as such. | \$0 |
| 60 | Oracle check, no language model: does the sampled-score theorem hold on real empirical rating laws? | A judge that samples from the empirical rating law has r equal to rho_hat (0.651, 0.192, 0.181 on alphaNLI, DICES-350, DICES-990). The mean judge has r 0.802, 0.424, 0.401, just below sqrt(rho_hat) (0.807, 0.438, 0.425) because each half has finitely many raters. The collapsed forecast's skill depends on the convention (Experiment 61): on DICES-350 rounded mean −0.475, modal label −0.245, two-point −0.152, against +0.167 for the truthful forecast; the truthful forecast beats every collapsed forecast on both DICES sets by 0.32 to 0.64 in skill. | yes, but H1–H3 are near-identities of the algebra and the code, not evidence about any real judge. | \$0 |
| 61 | Post hoc: how do the conventions and comparators behind Experiments 58–60 change the numbers? | Collapsed-forecast skill by convention (above); perfect-judge hard-label gap (above); random-pick comparison for Experiment 59 (above). Nothing here reverses a preregistered decision; it adds a comparator that is unfavorable to path B. | descriptive; unfavorable to the cross-validated proper-score selector. | \$0 |

**Clarification of row 48.** The 82 cases are 80 transfers of the five frozen frontier judges plus 2 G-Eval (GPT-4) rubric transfers on SummEval (README of Experiment 48, rubric row). The frozen rule "gap reopens" holds in 53 of the 82; the transferred map is significantly worse than the raw forecast in 33 of the 82 (21 of them between NLI and DICES, whose label sets differ) and significantly worse than an in-domain map in 78 of the 82.

**Experiment 54 saved output.** `strict_tally_output.txt` was stale. It was saved before 984ba71 corrected the Li et al. coding note and still described Li et al. as Brier and logarithmic scores against a pairwise preference distribution. It is regenerated from the current script (only that line differs; the tallies are unchanged; two runs are identical).

**The advisor's two warnings, as handled in the paper (809bb44).** (1) Elangovan et al. (binned Jensen–Shannon divergence between pooled human and judge labels) and Li et al. (squared error to the mean of ten teacher votes) already compare judges with rating distributions, and so do Guerdan et al. and Chen et al. (Kullback–Leibler divergence and cross-entropy), Wang et al. (Brier-type) and Lyu et al. (ranked probability score). The abstract says that and no longer claims the gap; the audit supports "43 of 49 papers report no item-level proper score or divergence", not "no metric does". (2) The cross-entropy-head versus Pearson-head comparison (Experiments 41, 47, 53) is circular: after calibration the largest five-seed mean gap is 0.0024 normalized RPS (0.0098 in skill over the base rate), consistent with the two heads differing mainly in calibration; the paper reports it as a check of the identity and not as evidence for the evaluation claim, and labels Sections 5b and 6 as training-objective comparisons. Experiment 59 is reported against our own selection recommendation, with the random-pick comparator from Experiment 61.

**Reviews and what each changed.** Opus 5.5 on the paper (21 findings, P1–P21; replacement texts in the QA report), Codex on the paper on Brando's Mac under the Vals subscription (10 findings; the earlier Codex review, F01–F20, was of an earlier state), Fable 5.2 on the literature (18 problems; 17 pinned claims confirmed, one number, 0.7913 / 0.1321, not verifiable) and on the slides. Adopted: the corrected finer-key statement (now checked directly, numerics check 10), the exact Lean status labels, the sign-flip caveat for Pearson r under negative slopes, "estimated calibration gain" and "calibrated skill" in place of Murphy-term names for fitted-map estimates, the convention-dependent collapsed-forecast skill, the Experiment 59 interval type, the stored Experiment 07 restart values. Not adopted wholesale: Codex finding #8 and the 211-word abstract assembled in the Fable literature report; the paper keeps my abstract (248 words by `docs/notes/abstract_wordcount_28Nov2026.py`) with their individual fixes applied, and a cut list toward the advisor's 200 words is in the changelog.

**Not run, and why.** System-level win rates (no system-level judge outputs on the box). Real-LLM sampling at temperature above 0 (no gateway key; Experiment 60 is the CPU oracle check instead). Lean was not compiled in this session (no toolchain on the box); each claim's Lean status is as in `docs/LEAN_TODO.md` and cites the files compiled earlier (Lean 4.36.0, Mathlib fc4f59e5). The always-"safe" agreement baseline was not computed by a script; the paper states only the bound that follows from the printed majority rate (at most about 77 percent for a judge that never answers unsafe on DICES-350).

**Tests this session.** `src/judge_scores`: 24 passed. Experiments 57, 58, 59, 60, 61: 9, 7, 7, 13 and 12 passed. `proper_score_decomposition_numerics_28Nov2026.py`: 28 of 28 PASS, identical across two runs and equal to the saved output. The Experiment 43 and 54 tallies: identical across two runs.

| date (Pacific) | item | what | USD | cumulative |
|---|---|---|---|---|
| 11-28-2026 | 57–61, abstract rewrite, tests | CPU only, on frozen forecasts and empirical rating laws; no API call | 0.00 | 0.00 |
| 11-28-2026 | QA by Claude subagents | Opus 5.5 and Fable 5.2 reviewers at max effort, read-only, run inside this Claude session at Brando's explicit request ("mega qa"); no API calls | 0.00 | 0.00 |
| 11-28-2026 | Codex review ("vals ultra") | Run on Brando's Mac under the Vals subscription at his explicit request; no API dollars attributed here | 0.00 | 0.00 |
