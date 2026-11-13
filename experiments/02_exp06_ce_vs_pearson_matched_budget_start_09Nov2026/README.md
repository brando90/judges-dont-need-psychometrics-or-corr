# 02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/README.md>

Diagnose historical experiment 06’s Pearson failure and compare cross-entropy with Pearson within a shared, certified probability-mixture family. The comparison is complete; it does not establish CE superiority.

- [Full results and statistics](RESULTS_comparison.md)
- [Frozen protocol](PROTOCOL.md) and [source freeze](FREEZE.json)
- [Checkpoint](ckpt.md) and [profile continuation access](SESSION_ACCESS.md)
- [Independent verification](results/verification.json)
- [Historical experiment migration and canonical paths](MIGRATION_11-09-2026.md), [manifest](MIGRATION_MANIFEST_11-09-2026.json) and [migration checkpoint](CKPT_migration_11-09-2026.md)
- [Evaluation criterion and repository ownership](EVALUATION_CRITERION_11-09-2026.md)

Live dependency: the completed result still needs a paper-integration decision, so this canonical experiment folder remains active. The paper has not been changed.

Regenerate the report using `python build_report.py`. Private ratings and forecasts are withheld. Public bootstrap fit receipts are compressed without loss; `gzip -dk results/bootstrap_fit_receipts.json.gz` restores the file expected by the frozen verifier. The decompressed checksum is in `results/BOOTSTRAP_RECEIPTS_MANIFEST.json`.
