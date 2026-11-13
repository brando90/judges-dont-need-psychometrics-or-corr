# Historical objective experiment migration checkpoint

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/03_exp06_ce_vs_pearson_matched_budget_start_09Nov2026_end_09Nov2026/CKPT_migration_11-09-2026.md>

**TLDR:** This chat's historical objective experiments and private inputs now live here; transfer checks passed. The repaired CE/Pearson comparison was already complete and has not been rerun.
Created: 11-09-2026 12:16 PST
Last updated: 11-09-2026 12:17 PST

## Goal

- Finish this chat's repository relocation without altering completed scientific conditions.
## What ran

- Verified the actual default working directory and existing experiment03, then imported historical_exp06/ and moved private/historical_exp06/.
- Hash verification, source-freeze verification, Python parsing and both follow-up command-line import checks passed; no model fitting or inference ran.
## Results

- Verified 83/83 public files, 1,521/1,521 private files, 11/11 frozen current files and 6/6 original source imports. All four original images preserved.
- Public evidence: MIGRATION_MANIFEST_11-09-2026.json and results/migration_verification_11-09-2026.json. Scientific results remain in their original reports.
## Stuck / blocked

- Nothing for this scoped migration. Old public files remain frozen provenance; the old private path is a compatibility link for applied-project readers.
## Next steps

- No migration or scientific rerun remains for this chat's objective package. Verify later with: python3 experiments/03_exp06_ce_vs_pearson_matched_budget_start_09Nov2026_end_09Nov2026/scripts/verify_migration.py --with-private
- Broader applied-judge cleanup and VeriBench experiment110 are separate work, documented in the applied project's private migration audit.
## Brando to-do

- None for this migration.

## Log

11-09-2026 12:16 PST -- Confirmed the working-directory move and repaired comparison were already complete; transferred the missing historical package, preserved private compatibility, verified hashes and imports, and updated canonical paths without a scientific rerun.
