# State of evidence: calibration follow-up experiments 46–52 and spend ledger

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/STATE_OF_EVIDENCE.md>

**TLDR:** One line per follow-up experiment, with its strongest number, its uncertainty and its cumulative API spend. The API ceiling is \$2,000 total, a ceiling and not a target, shared with Claude Code's overnight runs. Claude/Codex subscription credits are never used. Other workers: append your own section and spend rows; do not rewrite these.

**As of:** 11-27-2026 (Friday, Pacific time). Maintained by Grok Bot for Brando Miranda.

## Experiments 46–52 (folder `experiments/NN_*_start_27Nov2026/`)

| exp | question | strongest result (95% CI) | supports thesis? | API spend |
|---|---|---|---|---|
| 46 | Do real LLM judges rank differently under correlation than under proper scores? (interim) | GPT-6 Astra out-correlates GLM 5.3 (Spearman +0.021 [+0.010, +0.034] SNLI; +0.034 [+0.014, +0.056] MNLI) but has lower Brier skill (−0.039 [−0.056, −0.023]; −0.098 [−0.117, −0.080]), Holm-significant. 4/5 frontier judges are worse than the base rate on DICES-350 (skill −0.10 to −0.18, all CIs < 0) at 80–82% top-1 agreement. Calibration removes the reversals (H3). | yes, with the H3 qualification | \$0.00 (no gateway key on box) |
| 47 | Labels needed for calibration to close the gap | At n = 25 single ratings, skill difference −0.066 to −0.116, larger than the CE head's whole skill (≤ 0.071); needs 100–200 labels | yes | \$0 |
| 48 | Does calibration transfer under shift? | Shifted map loses skill vs an in-domain map in 78/82 frontier cases (CI < 0) and is worse than uncalibrated in 33/82 (e.g. DeepSeek V4.1 Flash, DICES-990 map on SNLI: −0.537 [−0.593, −0.485]); linear heads reopen in 21/22 views | yes | \$0 |
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
