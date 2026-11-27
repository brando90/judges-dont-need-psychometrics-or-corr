# Experiment 40: G-theory, D-theory, dollar Pareto, and finite-grid CRPS

**Doc link:** https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/README.md

**TLDR:** A finite witness has the same mean, the same variance, and therefore the same generalizability coefficients, while Brier, ranked probability score (RPS), and cross-entropy separate the calibrated law from a distorted one. On real multi-annotator sets the D-study's k and the proper-score allocation agree in held-out cross-entropy once the budget is large, except on VeriBench `native_main`. Read `HOW_TO_READ_RESULTS.md` before quoting a number. Narrative: `REPORT.md`.

**Status:** complete on this machine (11-27-2026). CPU only. No new model calls. `$0`.

Number 39 was already on `main` as the proper-score versus D-study allocation study (`39_proper_score_vs_d_study_budget_allocation_start_26Nov2026`). This folder is 40. That study uses frozen frontier-judge transforms on ChaosNLI and DICES. This one adds the Lean witness, POPQUORN, VeriBench `native_main`, the dollar regimes, and the output-granularity sweep.

## Reproduce

```bash
python3 experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/run_all.py
python3 -m pytest experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026/tests -q
cd experiments/40_g_d_theory_dollar_pareto_crps_start_27Nov2026 && lake build
```

Lean 4.36.0, no Mathlib, no `sorry`. Compiler receipt: `lean_check.txt`.

## What is in the folder

| Path | Role |
| --- | --- |
| `theory/GTheory.lean` | G-theory blindness witness, Brier and RPS decompositions, finite-grid RPS properness and refinement |
| `gstudy.py` | Unbalanced Henderson Method III and the nested one-way estimator |
| `run_all.py` | Real-data G/D-study, dollar frontier, rebin sweep, synthetic CRPS, GPU bridge |
| `PROTOCOL.md` | Thresholds fixed before the run |
| `REPORT.md` | Findings with the measured intervals |
| `tables/`, `figures/` | Allocation, dollars, K-sweep, Pareto plots |
| `provenance/ttj_efdc8c3/` | Byte copies of the trustworthy-theorem-judges experiment 28 files that were bridged |

## Data

Loaders are the experiment 36 loaders. SummEval, POPQUORN (`Jiaxin-Pei/Potato-Prolific-Dataset@dbb118c2`), DICES-990, and ChaosNLI loaded. VeriBench `native_main` loaded from `trustworthy-theorem-judges@460466d` and was scored on the task-pure `exp27_split` from experiment 28 at `efdc8c3`. `veribench@d1c47355` returned HTTP 404 from this machine, so `R_v1` was not byte-verified. The 75-item expert table in the experiment 28 CSV is reported separately, with rater identity already removed.

WMT direct assessment (`RicardoRei/wmt-da-human-evaluation`, commit `301de385`) is reachable and stores a mean plus an annotator count. It was not used as multi-rater microdata. The K-sweep on `{0,…,100}` is the labeled synthetic mixture.
