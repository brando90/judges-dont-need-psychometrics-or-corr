# Experiment 16: RPS-selected arm and probability-floor ablation on the historical scalar-vote pools

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/16_rps_selected_arm_and_probability_floor_on_scalar_pools_start_20Nov2026/README.md>

**TLDR:** Sanmi Koyejo's 11-16 point 1 asked for an RPS-selected arm on the human ratings and suggested that the 1e-6 probability floor drives the loss of cross-entropy (CE) selection on 3-sample histograms. This reruns experiments 12 and 14 (same 1,000 task-grouped 20/5 splits, same cached judges, same 75 artifacts and 195 expert ratings) and adds judges picked or mixed by train RPS, a Spearman-picked single judge, and a sweep of the floor. CPU only, no model calls, $0. Retrospective: the same ratings as every other human-rating experiment here.

**Pre-registration (written and pushed before the sweep ran; see git history):** the one exception is `old5` at floor 1e-6, which was run once as a reproduction check before this file was written: it reproduced experiments 12 and 14 (`+0.0056`, `-0.0254`, `+0.0429`) and its RPS-arm numbers were seen. All other cells are blind.

## What is run

- Data: the 75 artifacts (25 tasks), 195 expert ratings (participants 1-3), five cached scalar judges (`old5`), 39 open-weight scalar judges (`open`), both together (`all`, 44 judges). Same dataset and rebuild as [experiment 14](archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/README.md); the ratings are private and not committed.
- Splits: experiment 12's rule and seeds (20 train and 5 test tasks, seeds `20261210 + s`, 1,000 splits).
- Arms, each fitted on train tasks only, one judge or a weighted mixture: constant (train histogram), equal mixture, `corr_single` (best train Pearson), `corr_single_spearman` (best train Spearman, new), `corr_mixture_pearson`, `corr_mixture_spearman`, `ce_single`, `ce_mixture`, **`rps_single`** (best train RPS, new), **`rps_mixture`** (new; convex quadratic program over the simplex, SLSQP, certificate = Frank-Wolfe gap <= 1e-8; failures are counted and kept).
- Floor sweep: each judge's 3-sample histogram is floored at epsilon and renormalized, with epsilon in {1e-6 (experiments 12 and 14), 1e-3, 1e-2, 5e-2}. Epsilon is never selected: all four are reported. It changes the judges' probabilities, so it changes every arm's CE and RPS.
- Metrics: test RPS (primary), CE, TVD, then Spearman and Pearson of judge mean against human mean (secondary columns). Paired differences with a 10,000-draw task bootstrap, as in experiment 12.

## Primary endpoint and predictions (fixed before the blind cells ran)

Primary: test RPS of `corr_single` minus `rps_single`, and of `corr_mixture_pearson` minus `rps_mixture`, per pool at floor 1e-6 (negative means the correlation-picked judge has the lower loss). Everything else, including the floor sweep, is exploratory, with no multiplicity adjustment.

1. At floor 1e-6, `rps_single` is no better than `ce_single` on test RPS (interval of the difference includes 0) on every pool. My guess: 70%.
2. Raising the floor to 5e-2 cuts the test CE of `ce_single` below the constant's CE on every pool, but `ce_single` and `rps_single` still do not beat the constant on test RPS on `all` and `open`. My guess: 65%.
3. `corr_single` stays better than `rps_single` on `all` at floor 1e-6 (the sign of experiment 14's single-judge row). My guess: 60%.
4. `rps_mixture` is better than `corr_mixture_pearson` on every pool at floor 1e-6 (interval excludes 0 on `open` and `all`). My guess: 60%.

## Run record and end condition

- Launch: `scripts/run_all.sh <dataset_pool.json> <out_root> 24` on skampere1 (CPU, Python 3 with numpy and scipy, 24 worker processes), launched 11-20-2026 evening, one `run.log` per cell, `DONE` file per cell, `ALL_DONE` for the sweep. Ends when all 12 cells wrote `DONE`, or the job times out at 3 hours (a cell without `DONE` failed).
- Harness and model: plain Python scripts, no language model is called in this experiment. The experiment was built and launched by the ub9sc assistant (Claude Code harness, Sonnet 5.5).
- Results: `results/<pool>_eps<epsilon>/results.json`, `per_artifact.json`, `run.log`; summary in `results.md` (written after the run).

## Limits

Five test tasks per split and 25 tasks in all; the cached judges' prompts were developed on the original train split; ratings are the same retrospective ones (prior exposure); the scalar-vote judges give 3-sample histograms, which is the reason for the floor in the first place. The floor sweep tests Sanmi's explanation for these judges only. It says nothing about the native-probability judges of experiments 09 and 10.

**TLDR-end:** [judge objectives: experiment 16] Pre-registered rerun of experiments 12 and 14 with RPS-picked judges and a probability-floor sweep; CPU only, $0, retrospective.
