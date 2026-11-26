# Experiment 36: Sanmi distributional pitch, anti-proper correlation

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/36_sanmi_distributional_pitch_anti_proper_corr_start_26Nov2026/README.md>

**TLDR:** Pearson correlation is affine-invariant and, for a judge sampled independently of the human given the prompt, strictly anti-proper relative to the true conditional distribution. Brier and ranked probability score (RPS) are strictly proper. The human-data half is a linear head on SummEval, ChaosNLI, and DICES. Read `HOW_TO_READ_RESULTS.md` before quoting a number.

**Status:** run complete on this machine (11-26-2026) for every dataset that could be downloaded. No GPU. Cost $0 of new model calls. Raw downloads are cached under `data_cache/` and are not committed.

## Reproduce

From the repository root, with Python 3, NumPy, SciPy, and Matplotlib:

```bash
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
| `RESULTS_SUMMARY.md` | Proved claims, held-out tables, Kohli positioning, transcript coverage |
| `HOW_TO_READ_RESULTS.md` | Data, metric, difference, interval, and how the head was chosen |
| `results.md` | Live ledger |
| `results.json`, `results.txt` | Exact metrics |
| `PROTOCOL.md` | Measurement contract, written as the pre-registered choices |
| `SanmiPitch.lean`, `lean_check.txt` | Formal proofs and the compiler receipt |
| `proofs_numeric.py` | Numerical twins of the Lean claims |
| `metrics.py`, `data_io.py`, `fit_eval.py`, `run_all.py` | Estimators, loaders, heads, driver |
| `figures/` | Matrix, crossover, high-k, rollout, iso-budget |
| `tables/` | Markdown tables copied from the run |
| `scripts/snap_lora_train.py`, `scripts/snap_llm_logit_readout.py` | Not run here |
| `tests/test_hygiene.py` | Unbiased risk, prompt split, metric identities |

## Data that was scored

- **summeval.** downloaded from the SFR jsonl (expert lists of 3 and crowd lists of 5)
- **chaosnli.** downloaded from the author checksums on Hugging Face earino/chaosnli
- **dices350.** downloaded from google-research-datasets/dices-dataset
- **dices990.** downloaded from google-research-datasets/dices-dataset
- **popquorn.** NOT LOADED. huggingface dataset search returned no POPQUORN card, https://github.com/Jiaxin-Pei/POPQUORN returned 404, and that user's public repos contain Stanford-politeness model code rather than POPQUORN ratings. The Stanford politeness corpus was not substituted.
- **veribench_fsc.** NOT IN THIS REPOSITORY AND NOT ON THIS MACHINE. A search of this repo for FSC / R_v1 / functional semantic correctness ratings found no table. AGENTS.md assigns the shared VeriBench human-rating collection to trustworthy-theorem-judges experiment 09; that repo is not cloned here, so CertJudge / VeriBench FSC R_v1 (0–5, about 5 experts, 343 items) was not scored.

## Verified numerical snapshot

Binary witness, four paired items: Pearson correlation of the calibrated probabilities and of the affine map \(0.1q+0.45\) are both 1.000. Mean Brier 0.0800 versus 0.4418. Mean cross-entropy 0.2231 versus 0.6349 nats. Anti-proper Monte Carlo (n=200000): correlation of the true sampled judge 0.5002 (theory 0.5) and of \(0.1\mu+4\) 0.7074 (theory \(\sqrt{1/2}\approx 0.7071\)). Ordinal witness: both laws have mean 2.0; \(D_B=1.5\); normalized RPS gap 0.2000; cross-entropy of the split forecast on the consensus outcome is infinite.
