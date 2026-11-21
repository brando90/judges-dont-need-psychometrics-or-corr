# Experiment 15: item signal versus rater severity on the 75 expert-rated artifacts

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/15_rater_severity_vs_item_signal_start_20Nov2026/README.md>

**TLDR:** Sanmi Koyejo's 11-16 point 4 asked whether the ratings measure item signal or just rater severity, with only 2 to 3 raters. The three expert raters are fixed people (slots 1-3: 1 and 2 rated all 75 artifacts, 3 rated 45), so each rater's constant offset is estimable. Result: item signal dominates, rater offsets are small and explain about 2% of within-artifact variance (interval includes 0), and correcting for them changes no judge ranking. CPU only, no model calls, $0. Exploratory, not pre-registered: a 200-draw smoke run of this script was read before this file was written.

## What is run

- Data: 75 artifacts, 25 tasks, 195 expert ratings (slots 1-3; the 0-5 integer ratings). The ratings are private and not committed; inputs were `items.jsonl` (sha256 `ad9ed552...a96c`) from experiment 100 of the private VeriBench repo and the 44-judge cached pool (sha256 `d1924f54...789d`, the one used by experiments 12, 14 and 16). Slots 4 and 5 (other participants, role unverified) are not used.
- Model: rating = artifact effect + rater offset (offsets sum to zero), least squares on the incomplete table. Bootstrap over the 25 tasks, 10,000 draws.
- Reported: offsets, share of within-artifact variance explained by the offsets, ICC consistency and agreement, Lin concordance and Pearson per rater pair, variance components, and whether severity-correcting the human target changes the ranking of the 44 cached judges (Spearman, Pearson, and RPS against the human histogram with floor 0.05).
- Not identifiable: a rater-by-artifact interaction, because no rater rated the same artifact twice.
- Script: `scripts/analyze_raters.py`. Run on skampere1 (Python 3, numpy, scipy), 11-20-2026 22:24 to 22:25, 25 s. Plain Python, no language model; built and launched by the ub9sc assistant (Claude Code harness, Sonnet 5.5). Files: `results/results.json`, `results/run.log`, `results/launcher.log`.

## Results

| Quantity | Estimate | 95% interval (task bootstrap) |
|---|---|---|
| Offset, rater 1 (mean 2.76 over 75) | +0.147 | [-0.045, +0.320] |
| Offset, rater 2 (mean 2.53 over 75) | -0.080 | [-0.286, +0.108] |
| Offset, rater 3 (mean 2.24 over 45) | -0.067 | [-0.327, +0.246] |
| Share of within-artifact variance explained by offsets | 0.021 | [-0.013, +0.208] |
| SD of artifact means / of rater offsets / of residual | 1.44 / 0.13 / 0.70 | n/a |

| Rater pair | Items | ICC consistency | ICC agreement | Lin concordance | Pearson |
|---|---|---|---|---|---|
| 1 vs 2 | 75 | 0.850 [0.771, 0.897] | 0.843 [0.755, 0.890] | 0.841 [0.753, 0.889] | 0.880 |
| 1 vs 3 | 45 | 0.577 [0.136, 0.853] | 0.569 [0.135, 0.769] | 0.563 [0.131, 0.765] | 0.578 |
| 2 vs 3 | 45 | 0.544 [-0.012, 0.819] | 0.550 [-0.012, 0.809] | 0.544 [-0.012, 0.806] | 0.550 |
| All three, complete 45 | 45 | 0.639 [0.297, 0.834] | 0.633 [0.289, 0.791] | n/a | n/a |

Judge ranking, raw versus severity-adjusted human target: Kendall tau 0.996 (Spearman), 0.998 (Pearson), 1.000 (RPS); best judge `H2_opus` in all three; largest change in any judge's Spearman 0.005; median judge Spearman 0.066 raw and 0.063 adjusted.

## Reading

- Artifact differences are far larger than rater differences (SD 1.44 versus 0.13), and consistency and agreement ICC agree (0.85 versus 0.84 for the two full raters), so on this data severity is not what the ratings are measuring.
- Rater 3 agrees less with the other two (ICC 0.54 to 0.58) on a smaller overlap (45 artifacts, 15 tasks), with wide intervals.
- Limits: three raters, 25 tasks, no repeated ratings, offsets are only constants (no per-rater scale or threshold differences, which an ordinal model with rater facets would add, as Sanmi's point 5 lists). The pool of raters is the same retrospective one as the other experiments.

**TLDR-end:** [rater severity: experiment 15] Rater offsets are about 0.1 rating points and explain about 2% of within-artifact variance; severity-correcting the human target changes no judge ranking.
