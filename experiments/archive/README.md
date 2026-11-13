# Completed objective-comparison follow-ups

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/archive/README.md>

**TLDR:** These completed studies belong to the position paper's question about training and selecting judges. They form a progression of related conditions, not independent human-rating datasets. Their original experiment numbers and scientific files are preserved; new applied collection remains in trustworthy-theorem-judges.

## What each study adds

| Historical identity and current home | Question or change | Relation to the other studies |
| --- | --- | --- |
| [06_information_loss_vs_correlation_judge_selection](../02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/historical_exp06/README.md), retained within [02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026](../02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/README.md) | Earlier fixed-candidate selection, probability-head fitting, distance comparisons and weighted-loss follow-ups; experiment 02 then compares cross-entropy and Pearson with certified mixture fits. | The complete earlier package was already migrated on 11-09-2026. The redundant source-repository copy is now retired; frozen source snapshots required by experiment 02 remain. |
| [11_formulation4_train_only_ce_mixture_cv_start_10Nov2026_end_10Nov2026](11_formulation4_train_only_ce_mixture_cv_start_10Nov2026_end_10Nov2026/README.md) | Fit cross-entropy mixtures on training tasks only and evaluate with repeated five-fold splits. | Establishes the train-only baseline preceding experiment 12; it does not compare both training objectives. |
| [12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026](12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026/README.md) | Compare correlation and cross-entropy training on repeated 80/20 task splits, plus a separate fixed-split agent comparison. | Extends the baseline with objective comparisons on the same five cached judges and historical ratings. |
| [13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026](13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026/README.md) | Fit flexible judges to nonlinear synthetic rating distributions across training sizes. | A distinct synthetic illustration: unlike experiment 00's marginal-histogram example and experiment 01's fixed-candidate conditional-identification study, it learns a judge. It supplies no human observations or human sample-size guarantee. |
| [14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026](14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/README.md) | Expand experiment 12's judge pool with open-weight model outputs and a larger-pool Pearson solver. | Reuses the same human ratings. Its old-five-judge condition is a reproduction check; its added judge outputs are a distinct condition, not new human data. |

## How to read the archived results

The archived `results.md` files are frozen copies and are not rewritten. Next to each one, a `HOW_TO_READ_RESULTS.md` (added 11-13-2026) states which data each number uses (training tasks only for fitting; held-out test tasks for every reported score, except experiment 14's in-sample per-judge table), the metric definitions and directions, and what each "② minus ④" or "④ minus comparator" difference subtracts: [11](11_formulation4_train_only_ce_mixture_cv_start_10Nov2026_end_10Nov2026/HOW_TO_READ_RESULTS.md), [12](12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026/HOW_TO_READ_RESULTS.md), [13](13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026/HOW_TO_READ_RESULTS.md), [14](14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/HOW_TO_READ_RESULTS.md).

## Why there were so many similar folders

The work progressed through different selection rules, training objectives, split procedures, solver repairs, synthetic constructions and candidate pools. Saving these as separately numbered experiments preserved their different conditions, but the old layout hid that progression. Most empirical follow-ups reuse the same historical ratings; neither repeated splits nor additional judges create independent replications of human evidence.

There was also a real duplication: the full experiment 06 package remained in trustworthy-theorem-judges after its canonical copy moved here. This consolidation removes that redundant public tree. Vendored solver snapshots and experiment 02's six source imports remain because frozen protocols depend on their exact bytes. Similarity alone is not a reason to merge conditions or discard provenance.

The applied repository keeps its original-rating provenance, practical judge study, packet preparation and task/solver/rater size planning. Its paper can cite these objective studies as related evidence without owning another editable copy.

## Migration and preservation — 11-11-2026

The [per-file migration manifest](MIGRATION_11-11-2026.json) records source commit, Git blob, SHA-256 digest, destination and preservation status. Original source revision: [af77a7d](https://github.com/brando90/trustworthy-theorem-judges/tree/af77a7d06e5ea6b0f0a7c3ae68f2120ace676356).

- 112 tracked files from experiments 11–14 were copied byte-for-byte into this archive: 14, 26, 13 and 59 files respectively.
- Experiment 06's 83-file destination package was already present and remains unchanged. Of the retiring source files, 82 match it exactly; the source README adds only a prior relocation notice. The manifest preserves that notice, both digests and the immutable source reference.
- Original numbers are retained across the repository boundary; these are imported historical identities, not newly run experiments.
- Frozen protocols, results, scripts, source hashes and historical README contents are unchanged. Their old paths and document addresses describe the run-time location; use this index and the manifest for current navigation.
- No untracked or ignored files existed inside the moved public trees. Existing private experiment 06 inputs remain in experiment 02's ignored canonical store; its source-repository compatibility link stays because verified consumers and the prior migration check use it. No individual ratings were added to public Git by this move.

## Folder renames — 11-13-2026

Lifecycle dates were added to folder names ([agents-config Rule 88](https://github.com/brando90/agents-config/blob/main/rules/research.md#trigger-rule-88)); see the [experiments index](../README.md#folder-dates) for the git evidence behind each date. The three active experiments were also renumbered to start at 00, so experiment 01 became 00, 02 became 01 and 03 became 02; archived experiments keep their original numbers. For a few hours on 11-13-2026 the active folders briefly carried their old numbers with `_end_` dates (commit `8cd8c57`); that was corrected because only archived experiments have an end date. Folders were moved with `git mv`, and only path text in live files changed. Frozen manifests and migration receipts keep the paths recorded when they were written, and `verify_migration.py` maps those recorded paths to the current folders.

| Previous path | Current path |
| --- | --- |
| `experiments/01_synthetic_pearson_vs_kl_squashed_judge_0to5/` | `experiments/00_synthetic_pearson_vs_kl_squashed_judge_0to5_start_07Nov2026/` |
| `experiments/02_conditional_binomial_distribution_identification/` | `experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/` |
| `experiments/03_exp06_ce_vs_pearson_matched_budget/` | `experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/` |
| `experiments/archive/11_formulation4_train_only_ce_mixture_cv/` | `experiments/archive/11_formulation4_train_only_ce_mixture_cv_start_10Nov2026_end_10Nov2026/` |
| `experiments/archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits/` | `experiments/archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026/` |
| `experiments/archive/13_toy_pearson_vs_ce_nonlinear_ratings/` | `experiments/archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026/` |
| `experiments/archive/14_open_weight_judge_pool_corr_vs_ce_80_20/` | `experiments/archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/` |

## Verification receipt

The transfer check verifies 195 mapped files: 112 newly copied files and the existing 83-file experiment 06 package. All 194 comparable source files match exactly; the one README navigation addition is retained in the manifest. Python syntax checks cover 68 files and JSON parsing covers 51. The earlier migration verifier also passes its 83 public files, 11 frozen files, six original imports and 1,521 private files. No scientific measurement was rerun.

Three original lines in experiment 14 have trailing whitespace (`scripts/assemble.py:21`, `render.py:16`, `run_slot.py:51`). They are preserved byte-for-byte; whitespace checks on newly authored navigation and verification code pass. This is an inherited formatting issue, not an unverified content change.

## Verify and reproduce

From this repository's root:

```sh
python3 experiments/archive/verify_migration.py
python3 experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/scripts/verify_migration.py --with-private
```

Omit `--with-private` on machines without the original private inputs. For source-history and removal verification, add `--source-repo /path/to/trustworthy-theorem-judges --check-source-removed` to the first command. These checks read saved evidence; they do not fit models, regenerate ratings or overwrite results.

Current importer consumers use experiment 03's canonical `historical_exp06/scripts/import_cached.py`; trustworthy-theorem-judges experiment 07 accepts `JUDGE_OBJECTIVES_REPO` for a non-sibling checkout. Commit-pinned citations to earlier source revisions remain valid and are deliberately unchanged.

For numerical reproduction, follow the original experiment's protocol and preserve its inputs, seeds and dependencies. Experiments 11, 12 and 14 contain recorded scratch/remote paths in dataset builders, agent-selection scripts and model launchers; those predate this migration. They are retained as frozen execution evidence, not advertised as portable entry points. Reconstruct their recorded environment from the source revision or supply an explicitly versioned adapter and verify input equivalence before a new run. Experiment 14's documented dataset builder is now in the adjacent experiment 12 directory. Report scripts for 14 expect that experiment as their working directory. No scientific run was launched by this consolidation.

**TLDR-end:** [judge objectives: consolidation] The general objective-comparison history now has one canonical project. The index makes its distinct conditions, shared data and genuine duplicate explicit.
