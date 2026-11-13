# 06_toy_rank_and_mixed_correlation_objectives_vs_ce

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026/README.md>

**TLDR:** Does training a judge with a rank correlation (Spearman or Kendall), or with an equal mix of Pearson and Spearman, fix what training with Pearson misses? This runs those three objectives on experiment 13's synthetic toy, on exactly the same training and test tasks, and compares them with experiment 13's saved cross-entropy-trained and Pearson-trained judges. Synthetic toy that works by construction; not evidence about real ratings.

## Question and prediction (written before measurement, 11-13-2026)

Experiment 13 trained one flexible judge family by cross-entropy (CE), ranked probability score (RPS) and Pearson. A reviewer could object that practitioners also look at rank correlations or at several correlations together. Hypothesis: every correlation objective constrains at most the ordering or an affine transform of the judge's mean rating per task, so none of them learns rater disagreement or calibration. Prediction: all three new objectives have held-out test RPS and CE excess at least as bad as the Pearson-trained judge from the n where the CE-trained judge wins (64 training tasks in scenario A, 4 in scenario B). The prediction fails if any of them matches the CE-trained judge at large n.

## Fixed setup

- **Data, judge family and evaluation:** identical to [experiment 13](../archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026/HOW_TO_READ_RESULTS.md). The data generator, softmax judge on 13 cosine/sine features, ridge (1/n)‖W‖², starting point, L-BFGS-B settings (400 iterations, 600 evaluations) and test metrics are imported unchanged from its frozen `scripts/toy.py`. Same seeds (0–99), training sizes n ∈ {4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048}, 3 ratings per training task, and 500 fresh test tasks scored against the true distribution.
- **New objectives** ([scripts/rank_objectives.py](scripts/rank_objectives.py)): rank correlations are not differentiable, so training uses standard smooth surrogates with temperature τ = 0.1 rating points, fixed before measurement.
  - **Spearman-trained:** maximize Pearson(u, rank(ĥ)), with soft ranks u_i = Σ_j σ((m_i − m_j)/τ) of the judge's expected ratings m and average-tie ranks of the training-task mean human ratings ĥ.
  - **Kendall-trained:** maximize (2/(n(n−1))) Σ_{i<j} tanh((m_i − m_j)/τ) · sign(ĥ_i − ĥ_j).
  - **Pearson+Spearman-trained:** maximize ½ Pearson(m, ĥ) + ½ soft Spearman.
- **Checks before measurement:** analytic gradients match finite differences for all three objectives. Regenerated data reproduce experiment 13's saved CE results exactly; the sweep repeats this check for seed 0 at every (scenario, n).
- **Comparison:** per seed, the new judge's held-out test excess minus the CE-trained (and Pearson-trained) judge's saved excess on the same tasks, with 95% bootstrap intervals over the 100 seeds. Test Spearman and Kendall against the true mean are also reported.

## Run

```bash
python3 experiments/06_toy_rank_and_mixed_correlation_objectives_vs_ce_start_13Nov2026/scripts/run_rank_sweep.py
```

CPU only, no model calls, $0. Results go to `results/raw.jsonl`; [results.md](results.md) is the live status.

**TLDR-end:** [judge objectives: experiment 06] Rank and mixed correlation objectives, trained on experiment 13's exact toy data, against its saved CE-trained and Pearson-trained judges.
