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
| 52 | Skill-score estimator, one vs many ratings | One-rating estimator unbiased with a known base; at equal rating budget, one rating per item beats five | methods support | \$0 |

## Spend ledger (cumulative, estimated at configured gateway list rates)

| date (Pacific) | experiment | what | USD | cumulative |
|---|---|---|---|---|
| 11-27-2026 | 46–52 | all CPU or frozen receipts; API runner built and mock-tested, not run (no key on box) | 0.00 | 0.00 |
