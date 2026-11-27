# Experiment 36: Sanmi distributional pitch, anti-proper correlation

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/README.md>

**TLDR:** Pearson correlation is affine-invariant and, for a judge sampled independently of the human given the prompt, strictly anti-proper relative to the true conditional distribution. Brier and ranked probability score (RPS) are strictly proper. Real multi-annotator heads are scored across SummEval, ChaosNLI, DICES, POPQUORN (Offensiveness & Politeness), and VeriBench FSC (`R_v1`, live-synced `native_main`, and pooled), alongside real LLM judge panels bridged from CertJudge (Experiments 63, 39, 67). Read `HOW_TO_READ_RESULTS.md` before quoting a number.

**Status:** complete on this machine (11-26-2026) across all downloaded/local multi-annotator sets and CertJudge panels. No GPU (SNAP scripts ready). Cost $0 of new model calls. Raw downloads are cached under `data_cache/` and are not committed.

## Reproduce

From the repository root, with Python 3, NumPy, SciPy, Matplotlib, and pytest:

```bash
python3 experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/eval_power_mdd.py
python3 experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/run_all.py
python3 -m pytest experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/tests -q
```

Lean 4.36.0 (`lean-toolchain`), no Mathlib:

```bash
cd experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026
elan run leanprover/lean4:v4.36.0 lean SanmiPitch.lean
```

Exit code 0, no `sorry`. The captured compiler output is `lean_check.txt`.

## Files

| Path | Role |
| --- | --- |
| `SANMI_PITCH.md` | Four-step pitch, note to Sanmi, two-paper roadmap, transcript coverage |
| `RESULTS_SUMMARY.md` | Proved claims, held-out tables, CertJudge real LLM judge bridge, power & G-theory |
| `HOW_TO_READ_RESULTS.md` | Data, metric, difference, interval, and how the head was chosen |
| `results.md` | Live ledger |
| `results.json`, `results.txt`, `results_power_mdd.json` | Exact metrics & power/MDD grids |
| `PROTOCOL.md` | Pre-registered measurement contract + Amendment 1 (11-26-2026) |
| `SanmiPitch.lean`, `lean_check.txt` | Formal proofs and the compiler receipt |
| `proofs_numeric.py` | Numerical twins of the Lean claims |
| `metrics.py`, `data_io.py`, `fit_eval.py`, `run_all.py`, `eval_power_mdd.py` | Estimators, loaders, heads, drivers |
| `figures/` | Matrices, crossovers, CertJudge real panels, rollout, iso-budget, test-set power |
| `tables/` | Markdown and LaTeX tables copied from the run |
| `scripts/snap_lora_train.py`, `scripts/snap_llm_logit_readout.py` | Ready for SNAP/Sherlock GPU runs (not run here) |
| `tests/test_hygiene.py` | Unbiased risk, prompt/task splits, dataset loaders, metric identities |

## Data that was scored

- **summeval.** downloaded from the SFR jsonl (expert lists of 3 and crowd lists of 5)
- **chaosnli.** downloaded from the author checksums on Hugging Face earino/chaosnli
- **dices350.** downloaded from google-research-datasets/dices-dataset
- **dices990.** downloaded from google-research-datasets/dices-dataset
- **popquorn.** downloaded from Jiaxin-Pei/Potato-Prolific-Dataset@dbb118c2 (offensiveness 1,500 items / 13,036 ratings; politeness_rating 3,718 items / 25,042 ratings; 1-5 scale)
- **veribench_fsc_rv1.** loaded from brando90/veribench@d1c47355 Exp 100 items.jsonl (75 items, 25 tasks, 0-5 scale; all5 345 ratings, expert3 195 ratings)
- **veribench_fsc_native_main.** loaded from brando90/trustworthy-theorem-judges@460466dd Exp 09 live-synced 11-26-2026 (100 items, 25 tasks, 0-5 scale; 250 non-coordinator numeric ratings across 6 active raters; T30 HTTP 401 unverified)
- **certjudge_real_panels.** loaded from brando90/cert-judge@f6047849 Exp 68 Study A (E63 6 Qwen3-8B arms) & Study B (E39 56 LLM configs) + Exp 67 interim (3 judges on 100 native files)

## Verified numerical snapshot

Binary witness, four paired items: Pearson correlation of the calibrated probabilities and of the affine map \(0.1q+0.45\) are both 1.000. Mean Brier 0.0800 versus 0.4418. Mean cross-entropy 0.2231 versus 0.6349 nats. Anti-proper Monte Carlo (n=200000): correlation of the true sampled judge 0.5002 (theory 0.5) and of \(0.1\mu+4\) 0.7074 (theory \(\sqrt{1/2}\approx 0.7071\)). Ordinal witness: both laws have mean 2.0; \(D_B=1.5\); normalized RPS gap 0.2000; cross-entropy of the split forecast on the consensus outcome is infinite.
