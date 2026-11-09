# Historical objective experiments: canonical repository migration

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/03_exp06_ce_vs_pearson_matched_budget/MIGRATION_11-09-2026.md>

**TLDR:** The working directory and completed CE-versus-Pearson comparison were already in this repository. This migration completes the missing historical experiment package, including prompts, protocols, reports, checkpoints, original images and private reconstruction inputs; it does not rerun or amend a scientific condition.

## Canonical locations

| Material | Location relative to this experiment |
| --- | --- |
| Completed repaired cross-entropy (CE) versus Pearson comparison | Existing root protocol, scripts, results and ckpt.md; frozen numerical evidence unchanged |
| Full earlier objective-selection study | historical_exp06/ |
| Distance-selection follow-up, 11-07-2026 | historical_exp06/followup_11-07-2026/ |
| Weighted probability mass function (PMF) training and stable Pearson baseline, 11-08-2026 | historical_exp06/training_followup_11-08-2026/ |
| Raw historical inputs, forecasts, bootstrap draws and operational receipts | private/historical_exp06/ (ignored by Git) |
| Four original handwritten-note/chat images and original transcription | private/historical_exp06/inputs/notes_11-07-2026/ |
| Later transcription with recovered chat context | private/provenance_addenda_11-09-2026/ |
| Transfer manifest and verification | MIGRATION_MANIFEST_11-09-2026.json and results/migration_verification_11-09-2026.json |
| Current relocation state | CKPT_migration_11-09-2026.md |

The complete public tree contains 83 files, including the six source snapshots previously imported separately for the repaired comparison. The distance follow-up has 16 files; weighted training has 32. Preserve the existing six copies under scripts/historical/ because the repaired comparison's source freeze binds those exact paths and bytes.

The private historical store contains 1,521 files. It was moved into this experiment and verified against its per-file hashes. Its former location in trustworthy-theorem-judges/private/06_information_loss_vs_correlation_judge_selection is a compatibility symlink to this single canonical store. It preserves access for applied-study readers without another active private copy. The detailed private receipt remains ignored; only its digest and aggregate file counts are public. The separately saved recovered transcription does not overwrite the older original transcription.

## Provenance and compatibility

The [manifest](MIGRATION_MANIFEST_11-09-2026.json) records the exact source commit, Git blob, SHA-256 hash, size and destination for every public file. Original documents, source freezes, results, timestamps, URLs and captured machine paths inside the historical tree remain byte-for-byte evidence. Read this migration note for current paths and the [criterion correction](EVALUATION_CRITERION_11-09-2026.md) for current study distinctions.

The former public experiment remains a frozen source copy with a relocation notice. Its import_cached.py entry point is retained because the applied project's experiment 07 run.sh still invokes it. That importer is a compatibility dependency, not a second active objective experiment. Future objective-comparison work and checkpoint updates belong here. Applied human-rating collection, prospective judge work and the separate VeriBench experiment-110 migration remain outside this task.

## Verification without rerunning the study

From the repository root:

```sh
python3 experiments/03_exp06_ce_vs_pearson_matched_budget/scripts/verify_migration.py --with-private
```

Without private access, omit --with-private to verify all public historical files, the repaired comparison's freeze, and its six original source imports. The copied historical Python hierarchy is preserved so the two follow-ups can still resolve their parent scripts. Verification does not authorize a new fit or change any saved experiment result.

The working directory is /Users/brandomiranda/judges-dont-need-psychometrics-or-corr, verified with plain pwd on 11-09-2026. This Codex chat already belongs to the corresponding project. The earlier suggestion to switch the working directory or move the completed repaired comparison was outdated; only the missing historical package needed relocation.

**TLDR-end:** [judges-corr: historical experiment migration] This experiment now holds the complete history for this chat's objective comparisons. Frozen evidence stays unchanged; private data remain private and existing readers retain access.
