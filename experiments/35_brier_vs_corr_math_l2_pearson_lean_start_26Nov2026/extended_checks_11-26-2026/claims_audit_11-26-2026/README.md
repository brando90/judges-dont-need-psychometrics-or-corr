# Claims audit: every mathematical line on the handwritten pages, checked with Lean and numerics (11-26-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/claims_audit_11-26-2026/README.md>

**TLDR:** Brando asked whether the proofs are checked with Lean and quick numerics, because some of what he sent might be wrong. This folder audits the four pages he sent (the 12:03 and 13:01 derivation pages and the annotated 6.437 page) line by line: 24 items, of which 12 are right as written, 3 are right with a note, 4 are wrong or cannot be shown, 2 are imprecise, 1 has an invalid reason for a true conclusion, and 2 are questions. 45 Lean 4 theorems compile with no `sorry` and only the three standard axioms; 36 of 36 deterministic numeric checks pass. Synthetic data and exact arithmetic only; no human ratings.

**Status:** complete (11-26-2026). Part of experiment 35's extended checks ([`../README.md`](../README.md)). The handwriting is transcribed in [`../../assets/`](../../assets/) (`TRANSCRIPTION_DERIVATION_PAGES_11-26-2026.md`, `TRANSCRIPTION_DERIVATION_PAGES_ROUND2_11-26-2026.md`, `TRANSCRIPTION_6437_PAGE_11-26-2026.md`).

## Files

- [`CLAIMS_LEDGER_11-26-2026.md`](CLAIMS_LEDGER_11-26-2026.md): the ledger. One numbered entry per claim (P1.1 to P3.4, S1 to S4) with what is written, the verdict, the corrected statement and the Lean theorem and numeric check that decide it; an honest account of what was and was not machine-checked; errors in my own earlier work that the audit found.
- [`AuditClaims.lean`](AuditClaims.lean): Lean 4.36.0 over Mathlib v4.36.0. Six parts: the identities of pages 1 and 2 for a finite joint law; the degrees-of-freedom count; fixed pairing and the conditional mean; couplings and independent draws; log-loss (Gibbs), one-to-one scores that are improper, and the Bayes rule; and one concrete three-point law showing the hypotheses are satisfiable.
- [`lean_check.txt`](lean_check.txt): toolchain, command, result and the `#print axioms` output for all 45 theorems.
- [`check_claims_audit.py`](check_claims_audit.py): 36 deterministic checks (exact fractions wherever the quantity is rational, numpy and scipy quadrature for the calculus items), each printing PASS or FAIL with its numbers. Claims that are false on the page are tested as failing cases.
- [`results/check_claims_audit.out`](results/check_claims_audit.out): the captured output (36 PASS, 0 FAIL; two runs give identical output).
- [`HOW_TO_READ_RESULTS.md`](HOW_TO_READ_RESULTS.md): data, metrics, tolerances and how instances were chosen.
- [`CLAIMS_AUDIT_11-26-2026.pdf`](CLAIMS_AUDIT_11-26-2026.pdf) and [`CLAIMS_AUDIT_11-26-2026.tex`](CLAIMS_AUDIT_11-26-2026.tex): the typeset version of the corrections with the derivations.

## Reproduce

From this folder: `python3 -I check_claims_audit.py` (about 2 s). Lean: `cd ../lean && lake exe cache get && lake env lean ../claims_audit_11-26-2026/AuditClaims.lean` (about 22 s once Mathlib is built). PDF: `latexmk -pdf CLAIMS_AUDIT_11-26-2026.tex`.
