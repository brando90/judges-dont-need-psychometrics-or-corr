# Data for experiment 39

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/39_proper_score_vs_d_study_budget_allocation_start_26Nov2026/data/README.md>

The public rating tables are fetched by `src/fetch_raw_data.py` and are not committed; the Experiment 28 judge-forecast cache is already in this repository and is copied, not downloaded. Frozen item splits and SHA-256 checksums are committed.

## What is in git

- `splits.json`: frozen 30% test / 70% dev item identifiers, from `SeedSequence([20261126, 1, k])`, on the Experiment 28 subsets reconstructed with `SeedSequence([20261123, k])`.
- `manifest.json`: item counts, rating-depth, and source checksums used in the measured run.
- `SHA256SUMS.txt`: checksums of the six source files.

## What is not in git, and how to fetch it

Run, from this experiment folder:

```bash
python3 src/fetch_raw_data.py     # downloads the five public files from pinned sources, copies the Experiment 28 forecasts, checks SHA-256
python3 src/prepare_data.py       # reads data/raw/ and writes data/processed/study_a_bundle.npz
sha256sum -c data/SHA256SUMS.txt  # run inside data/raw/ to re-check the six files
```

`src/fetch_raw_data.py` is the only place that knows the URLs. Sources:

- ChaosNLI SNLI, MNLI-m and alphaNLI jsonl from the HuggingFace dataset `earino/chaosnli` at commit `eb7cc18dbfa0f161d93a2b22c7856ca0faafcf2b` (`raw/chaosNLI_snli.jsonl`, `raw/chaosNLI_mnli_m.jsonl`, `raw/chaosNLI_alphanli.jsonl`).
- DICES-350 and DICES-990 CSVs from `google-research-datasets/dices-dataset` on GitHub (`350/diverse_safety_adversarial_dialog_350.csv`, `990/diverse_safety_adversarial_dialog_990.csv`). The branch `main` is not commit-pinned, so the SHA-256 in `SHA256SUMS.txt` is what guarantees the bytes; a mismatch is an error.
- Experiment 28 frozen judge probabilities: `forecasts_11-23-2026.csv` is already in this repository at `experiments/28_public_rating_sets_chaosnli_dices_frontier_judges_ce_vs_pearson_start_23Nov2026/forecasts/` (SHA-256 `54fbff95…`), so the fetch script copies it instead of downloading.

`prepare_data.py` has no download code and reads the six files by the names above. Study B is fully synthetic and needs no external file.
