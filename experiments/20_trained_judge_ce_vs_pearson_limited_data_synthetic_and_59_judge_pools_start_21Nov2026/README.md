# Experiment 20: a judge trained with cross-entropy (CE) vs one trained on correlation, also with little data (started 11-21-2026) — QA corrections 11-26-2026

**QA correction (11-26-2026):** Read the [dated correction and evidence limits](QA_CORRECTIONS_11-26-2026.md) before the historical text below; original scored tables remain unchanged.

**Status as of 11-21-2026:** done (synthetic sweep, real-data sweep on five judge pools, independent cross-check of experiment 19). [Results with tables](RESULTS_11-21-2026.md).

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/20_trained_judge_ce_vs_pearson_limited_data_synthetic_and_59_judge_pools_start_21Nov2026/README.md>

**TLDR:** Brando's claim: a judge trained with the proper loss (CE) learns the moments correlation cannot see (spread, shape) and is better even with limited data. Tested with two judges identical except for the objective, on synthetic items with a known rating distribution (4 to 512 training items, 200 seeds) and on the real 75 artifacts (1, 2 or all ratings per artifact; 59-judge pool and sub-pools). Result: CE wins above about 16 items (3 ratings each) or 7 training families, learns spread and the mean scale that Pearson does not, and loses at 4 to 8 items or 3 families, where it overfits and a near-uniform Pearson-trained judge scores better; selecting a single judge by Pearson still wins in the historical 3-vote pool.

Built on experiment 18 (math, splits, reconcile run) and next to experiment 19 (Codex: exact small-sample enumeration, four calibrated judges), whose numbers this run reproduces.

## Files
- [RESULTS_11-21-2026.md](RESULTS_11-21-2026.md): bottom line, Codex comparison, one table per set, caveats.
- `fig_synth.png`, `fig_real.png`: Pearson-trained minus CE-trained gaps with intervals.
- `code/`: `lib20.py` (judge, objectives, metrics), `synth20.py` + `agg_synth.py` (set A), `real20.py` + `agg_real.py` (set B), `codex_check20.py` (cross-check), `plot20.py`, `build_results.py`.
- `results/`: full tables (`tables_synth.md`, `tables_real_lam*.md`), aggregates (`summary_synth.json`, `out_real_lam*.json`), `check_codex.json`. Raw per-seed synthetic output (48 MB) is not committed; `python3 code/synth20.py --seeds 200` regenerates it (about 3 minutes with 16 processes). Human ratings stay in their source directories.

Run: set A `python3 synth20.py --seeds 200 --out out_synth.json --procs 16`, then `python3 agg_synth.py out_synth.json tables_synth.md summary_synth.json`; set B `python3 real20.py --lam 0.1 --out out_real_lam0.1.json` (needs the private data files `dataset_pool.json` and experiment 10's `judge_forecasts.json` in the working directory).
