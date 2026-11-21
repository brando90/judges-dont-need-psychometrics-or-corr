# Reproducing the prospective VeriBench registration | 11-21-2026

Date: 11-21-2026

As of 11-21-2026, [the calibration protocol](CALIBRATION_PROTOCOL_11-21-2026.md) pins the development eligibility, budgets, training draws and tests. [The availability report](STATUS_11-21-2026.md) gives the forecast input commitments and explains how the 100 Set N items were blinded.

**Register predictions.** Recreate the combined 175-row request schedule with `prepare.py`. The parent experiment’s `runners/collect.py` verifies each raw delivery against its message hashes and combines both models’ predictions in the probability CSV. From this directory, the following command consumes that file. The Google Sheets are never inputs. The historical index is verified by SHA-256. Each input line’s artifact ID must belong to the eligible historical train/validation set before its JSON response fields are decoded. The only human values selected are the nonmissing original expert slots p1, p2 and p3. The protocol’s expert cache is derived from that same immutable record, using those same slots and order.

```sh
python3 calibrate.py /tmp/corr-ce-fresh/veribench /tmp/corr-ce-fresh/veribench/registered /path/to/verified_veribench_probabilities.csv
python3 score_release.py /tmp/corr-ce-fresh/veribench/registered /tmp/corr-ce-fresh/veribench/no_released_outcomes
```

The registration keeps three files together in restricted storage. `test_distributions.jsonl` records all 27 six-category predictions of both models for every test artifact, indexed by model order in `commitments.json`. `training_draws.jsonl` records, for each pool, budget, number of observations per artifact, and draw, the CE head indices, Pearson head indices, undefined-Pearson flags and the training histogram. The selected index retrieves the exact registered forecast for any item and draw. The identity head is index zero and the uniform distribution assigns 1/6 to each category. Small-budget histograms are derived from human ratings, so the publicly auditable receipt contains hashes, model identifiers, attempt counts and code fingerprints. Raw receipts and draw-level records stay in restricted storage.

**Score after release.** The dated collection-owner import must have one record per rater/item with fields `source_item`, `candidate_sha256`, `rater_id`, `rating` and boolean `eligible_primary` in JSON Lines format. Retain names, attestation and original source fields in the owning system; the scorer can accept additional metadata. Primary observations must have a numeric integer rating, an eligible non-author rater, a matching packet commitment and one row per primary rater/item. Any records marked ineligible are excluded. This script does not decide who was eligible. When the released data are incomplete, both missingness bounds accompany the point estimate. Use a new output directory for each release.

```sh
python3 score_release.py /tmp/corr-ce-fresh/veribench/registered /path/to/dedicated_results --released-ratings /path/to/owner_released_primary.jsonl
```

The scorer first compares the complete restricted commitments record byte for byte with the public registration receipt. It rejects a substituted manifest before reading registered probabilities or released ratings. A run without a released ratings file verifies the registered hashes and returns `no_released_ratings`, with scheduled counts, zero observed ratings and no estimated loss. The scorer averages individual-draw losses. It never averages a set of calibration heads into a new model.
