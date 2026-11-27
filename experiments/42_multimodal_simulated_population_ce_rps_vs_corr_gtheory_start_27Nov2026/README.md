# Experiment 42: a simulated multi-modal population, proper scores versus correlation and G-theory

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/42_multimodal_simulated_population_ce_rps_vs_corr_gtheory_start_27Nov2026/README.md>

**TLDR:** This is a simulated-population testbed, not a certification on real human raters. Four subpopulations on {0, 1, 2, 3, 4, 5} are exponential tilts of frozen VeriBench, DICES-990, and ChaosNLI marginals. On the polarized subpopulation the Pearson binomial forecast has correlation 1 with the bin means and total variation distance 0.615 from the truth. At 256 training ratings per item its expected ranked probability score (RPS) is 0.059517 [0.059452, 0.059583] worse than persona-conditional cross-entropy. A lenient point mass has generalizability coefficient 1 and is selected by that coefficient in 400/400 replications.

**Status:** complete on this machine (11-27-2026). CPU only. No model calls. `$0`.

Number 41 was already the output-space K-sweep on `main`. This folder is 42.

## What is simulated

The base measures are frozen counts in `data/empirical_shapes.json`:

- VeriBench `native_main`, trustworthy-theorem-judges `460466dd`, 250 numeric ratings, counts `[0, 7, 1, 15, 42, 185]` on 0–5. Mean 4.588.
- DICES-990 `Q_overall`, file SHA-256 `14332b6391…`, 64,886 ratings after the experiment 36 quality filter, counts `[45678, 3231, 15977]` on No < Unsure < Yes.
- ChaosNLI SNLI+MNLI, author checksums, 3,113 items and 311,300 votes, counts `[109508, 138882, 62910]`. Half of the items (0.504) have no class above a 0.70 share.

VeriBench `R_v1` at `veribench@d1c47355` returned HTTP 404, the same pin experiment 40 could not byte-verify. Experiment 36's reported `R_v1` mean of 2.97 is cited and is not used as a histogram.

The strict persona is the native_main histogram reflected and tilted. That reflection is a simulation. It is not an observed strict rater pool.

## Reproduce

```bash
python3 experiments/42_multimodal_simulated_population_ce_rps_vs_corr_gtheory_start_27Nov2026/run_experiment.py
python3 -m pytest experiments/42_multimodal_simulated_population_ce_rps_vs_corr_gtheory_start_27Nov2026/tests -q
python3 experiments/42_multimodal_simulated_population_ce_rps_vs_corr_gtheory_start_27Nov2026/checks/quick_numerics.py
cd experiments/42_multimodal_simulated_population_ce_rps_vs_corr_gtheory_start_27Nov2026 && lake build
```

Lean 4.36.0, no Mathlib, no `sorry`. Compiler receipt: `lean_check.txt`. Axioms are `propext`, `Classical.choice`, and `Quot.sound`.

Read `HOW_TO_READ_RESULTS.md` before quoting a number. Tables: `results.md`.
