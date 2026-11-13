# 07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026/README.md>

**TLDR:** In experiments 13 and 06, the RPS-, Pearson-, Spearman-, Kendall- and ½ Pearson + ½ Spearman-trained judges were fitted by one L-BFGS-B run from one start with a 400-iteration budget. Those objectives are not convex, so their results could reflect the optimizer. This experiment logs convergence, gives the optimizer 5× the budget, and takes the best of 5 starts (chosen by the training objective), then checks whether any conclusion changes. Synthetic toy that works by construction; not evidence about real ratings.

## Question and prediction (written before measurement, 11-13-2026)

Question: do the experiment 13 and 06 conclusions survive a larger optimizer budget and random restarts? Those conclusions are that the CE-trained judge has lower held-out test RPS and CE excess than every correlation-trained judge from 64 training tasks in scenario A (8 for Kendall) and from 4 tasks in scenario B.

Prediction: restarts and budget may lower the correlation-trained judges' training objective. They will not make those judges competitive on held-out test RPS or CE, because correlation objectives constrain at most the ordering or an affine transform of each task's mean rating. The crossover points move by at most one grid step. The prediction fails if any correlation-trained judge, under the best-of-5 condition, matches the CE-trained judge's held-out test RPS at the largest size run.

## Fixed setup

- **Data, judge family, ridge, objectives, metrics:** identical to experiments [13](../archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026/HOW_TO_READ_RESULTS.md) and [06](../06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026/README.md). The data generator and objective code are imported from their scripts; [scripts/restarts.py](scripts/restarts.py) only adds logging, budgets and starts.
- **Objectives:** CE (convex reference), RPS, Pearson, Pearson + calibrated CE head, smooth Spearman, smooth Kendall, ½ Pearson + ½ smooth Spearman.
- **Three conditions per fit:**
  1. **orig:** the original start (training rating histogram plus a 0.01 seeded perturbation) and budget (400 iterations, 600 evaluations).
  2. **long:** the original start with 2,000 iterations and 4,000 evaluations.
  3. **best:** 5 starts (the original plus 4 random ones, adding N(0, 0.5²) to every weight), each with the long budget. The judge with the lowest *training* objective is kept; test data never enter the choice. The calibrated head is refitted on whichever Pearson judge is chosen.
- **Logged per fit:** training objective, scipy success flag, iterations, whether the budget was hit, held-out test RPS and CE excess, test Pearson and Spearman with the true mean.
- **Size:** 30 seeds (0–29); training sizes n ∈ {4, 8, 16, 32, 64, 128, 256, 512, 1024} (experiment 13's grid without 20 and 2048, to keep the run near an hour); scenarios A and B; 3 ratings per training task; 500 fresh test tasks per seed.
- **Comparisons:** paired by seed, with 95% bootstrap intervals over seeds. (1) long minus orig and best minus orig for each objective. (2) Crossover against the CE-trained judge recomputed under the best condition. (3) Convergence rates under orig.
- **Environment note (found before the sweep):** under orig, this machine (macOS, Python 3.11.6, NumPy 2.2.6, SciPy 1.16.3) reproduces experiment 06 exactly and experiment 13's CE and RPS fits to about 1e-17. Experiment 13's Pearson fits, run on a different Linux machine (Python 3.12), differ slightly (for example 0.009188 vs 0.009176 test RPS excess at scenario A, n = 16, seed 3), and the calibrated head built on them differs more (0.032 vs 0.042). The comparisons here are paired within this run.

## Run

```bash
python3 experiments/07_toy_optimizer_restarts_and_budget_for_nonconvex_objectives_start_13Nov2026/scripts/run_restarts_sweep.py
```

CPU only, no model calls, $0. [results.md](results.md) is the live status.

**TLDR-end:** [judge objectives: experiment 07] Logs convergence and reruns the non-convex toy objectives with 5× budget and best-of-5 restarts to test whether experiments 13 and 06's conclusions depend on the optimizer.
