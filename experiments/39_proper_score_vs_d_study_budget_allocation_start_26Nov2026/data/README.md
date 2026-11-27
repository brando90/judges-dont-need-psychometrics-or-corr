# Data for experiment 39

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/39_proper_score_vs_d_study_budget_allocation_start_26Nov2026/data/README.md>

Public rating tables and the Experiment 28 judge-forecast cache are downloaded at prepare time. They are not committed. Frozen item splits and SHA-256 checksums are committed.

## What is in git

- `splits.json`: frozen 30% test / 70% dev item identifiers, from `SeedSequence([20261126, 1, k])`, on the Experiment 28 subsets reconstructed with `SeedSequence([20261123, k])`.
- `manifest.json`: item counts, rating-depth, and source checksums used in the measured run.
- `SHA256SUMS.txt`: checksums of the six source files.

## What is not in git, and how to fetch it

Run, from this experiment folder:

```bash
python3 src/prepare_data.py
```

That script writes `data/raw/` and `data/processed/study_a_bundle.npz`. Sources:

- ChaosNLI SNLI, MNLI-m, and alphaNLI jsonl from HuggingFace `earino/chaosnli` (the commit pinned in `src/prepare_data.py`).
- DICES-350 and DICES-990 CSVs from the public DICES dataset (the raw GitHub URL pinned in `src/prepare_data.py`).
- Experiment 28 frozen judge probabilities: copy or point to `experiments/28_public_rating_sets_chaosnli_dices_frontier_judges_ce_vs_pearson_start_23Nov2026/` if present, or use the SHA-256 `54fbff95…` file `forecasts_11-23-2026.csv` already used in this run.

Verify `sha256sum -c data/SHA256SUMS.txt` in `data/raw/` before a new analysis run. Study B is fully synthetic and needs no external file.
