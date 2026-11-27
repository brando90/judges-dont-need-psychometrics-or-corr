# Audit of the "finalized" real-judging-data plan, and of the Cursor pushes that followed it | 11-26-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/gemini_plan_claims_audit_11-26-2026/README.md>

**TLDR:** On 11-26-2026 (14:53 PST) Brando pasted a plan written by Gemini and a four-stage prompt built from it for Grok and Cursor, and asked me to check their work once it was pushed, to run a maximum-effort second review, and to hold everything to the project's standard: Lean-checked proofs, seeded numerics for every theorem, and real multi-annotator human data. This folder holds the claim-by-claim ledger of the plan (19 claims; the mathematics is sound and the data statements have several errors), the Lean file and seeded scripts behind it, the inventory of real judging data we hold, and the audits of Cursor's Experiments 36, 37 and 38.

**The standard this audit applies:** `CLAUDE.md` and `AGENTS.md` at the repository root (sections "Proofs and conference empirical standard" and "Check every proof with Lean and quick numerics") and `.cursor/rules/proofs-and-real-data-conference-standard.mdc`.

## Contents

1. [`PLAN_CLAIMS_LEDGER_11-26-2026.md`](PLAN_CLAIMS_LEDGER_11-26-2026.md): every checkable claim in the plan, with a verdict and a machine-check label. Of 19 claims: 3 correct as written, 10 correct with a note, 2 imprecise, 2 wrong, 1 not supported by any file I could find, 1 in conflict with the project's primary metric (normalized individual-rating ranked probability score, RPS).
2. [`PlanClaims.lean`](PlanClaims.lean) and [`lean_check.txt`](lean_check.txt): 21 theorems in Lean 4.36.0 with Mathlib (revision `fc4f59e577feb1a4956671bc04ac18aaaf8cd842`). Zero `sorry`; every theorem depends only on `propext`, `Classical.choice` and `Quot.sound`. Compile: `cd <Lean project with this toolchain and Mathlib> && lake env lean <absolute path of PlanClaims.lean>`; `lean_check.txt` is the `#print axioms` output.
3. [`check_plan_claims.py`](check_plan_claims.py) and [`results/check_plan_claims.out`](results/check_plan_claims.out): 51 seeded numeric checks of the same claims (seed 20261126), 51 PASS and 0 FAIL, run twice with byte-identical output ([`results/determinism_check.txt`](results/determinism_check.txt)). It exits nonzero on a failure. These are synthetic finite laws used to check proofs against a known target; they say nothing about human judges.
4. [`dataset_facts.py`](dataset_facts.py) and [`results/dataset_facts.out`](results/dataset_facts.out): counts and paper statements for the public rating sets (SummEval, ChaosNLI, DICES, POPQUORN, HelpSteer2), the Kohli paper and the anonymized FSC files, with the sha256 of every input. 42 PASS and 0 FAIL; run twice, byte-identical ([`results/dataset_facts_determinism.txt`](results/dataset_facts_determinism.txt), which also records a correction to the spelling of FSC). Arguments: `--facts-dir DIR --cache-dir DIR --popquorn-dir DIR --popquorn-paper-txt FILE --summeval-paper-txt FILE --sister-dir DIR --exp36-data-io FILE`. The raw inputs are not committed (DICES alone is about 80 MB); the inventory lists where each came from.
5. [`fsc_counts_11_26.py`](fsc_counts_11_26.py) and [`results/fsc_counts_11_26.out`](results/fsc_counts_11_26.out): counts of the frozen 11-26 FSC bundle in the sister repository (175 artifacts, 50 tasks, 445 ratings) and a check that its `exp27_split` equals the sealed 11-24 split. Usage: `python3 -I fsc_counts_11_26.py <sister repo root>` at commit `7855746` or later. Run twice, identical.
6. [`REAL_JUDGING_DATA_INVENTORY_11-26-2026.md`](REAL_JUDGING_DATA_INVENTORY_11-26-2026.md): which real multi-annotator sets we hold, their raters per item, splits, prior exposure, which experiments used them, and the gaps against the conference standard.
7. [`qa_of_experiment_36_11-26-2026/`](qa_of_experiment_36_11-26-2026/README.md): the second review of Cursor's Experiment 36 by an independent Claude Opus 5.5 agent at maximum effort, the review verbatim, and my verification of it.
8. [`qa_of_experiments_37_38_11-26-2026/`](qa_of_experiments_37_38_11-26-2026/README.md): my audit of Cursor's Experiments 37 and 38 (Lean for every theorem, numerics twice, end-to-end reproduction, FSC splits against the sealed splits, a constant-forecast reference, an item-versus-task leakage check).

## What the audit found, in brief

1. The mathematics in the plan is sound. Every theorem the plan relies on compiles in Lean and passes seeded numerics, with three notes: greedy decoding collapses a judge to its mode, not its mean, so the "square root" anti-properness theorem does not apply to it as written; the plan's `sqrt(RPS)` headline metric conflicts with the project's primary metric (square root only after averaging); and the exact rate `(1 - ||p||^2)/N` is the rate of one context's empirical frequency, not of a trained judge.
2. The data statements in the plan have errors: DICES has 123 raters per item in DICES-350 and 69 to 76 in DICES-990 (not 60 to 112); ChaosNLI's three-class set has 3,113 items (4,645 includes the binary set); the FSC set is 75 plus 100 artifacts with 2 to 4 ratings each (not 343 items with about 5 raters); HelpSteer2's 2-point rule decides whether to collect more annotators and does not drop items.
3. Real judging data: SummEval, ChaosNLI, DICES, HelpSteer2 and the FSC ratings already have real-data runs in this repository; POPQUORN has none and loads from `Jiaxin-Pei/Potato-Prolific-Dataset`.
4. Cursor's Experiments 37 and 38 pass Lean, numerics and tests, but their FSC splits ignore the sealed splits and neither FSC head beats a constant forecast; see the audit note for the six items to fix.

## Limits

1. Grok had pushed nothing I could attribute to it when this was written (16:10 PST); this audit covers Cursor's Experiments 36, 37 and 38 only.
2. The ledger audits claims, not the quality of every experiment's prose. Statements about experiments I did not rerun are labelled as read from their READMEs.
3. The FSC ratings are live: counts here are as of the sister repository's 11-26-2026 15:47 PST freeze.
4. No memory, shared agents-config or sister-repo files were changed by this audit. Proposed rule text is in the final deliverables, not in the repository.
