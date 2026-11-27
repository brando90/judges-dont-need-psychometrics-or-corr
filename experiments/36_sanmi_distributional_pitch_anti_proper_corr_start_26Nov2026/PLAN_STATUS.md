# Plan and status — 11-26-2026

**Status:** the driver `run_all.py` finished (including Amendment 1 datasets and CertJudge bridge). Smoke flag: False.

## Contract (fixed before test metrics were read)

Predictions, written down before the run:

- Uniform cross-entropy equals \(\log K\).
- A CE-trained softmax should beat a constant uniform forecast on held-out CE when the hash features carry any signal.
- Shuffling train features should not improve held-out CE.
- On a known two-context law, the collapsed judge's correlation should sit near 0.707 and the true sampled judge near 0.50.
- On binary data, scalar L2 and Brier match up to the factor 2. Correlation can still tie while Brier and CE worsen.
- On ordinal data with within-item spread, a mean-matched two-point readout should lose proper scores to a softmax that is allowed to use the full simplex.
- The within-prompt oracle should show a higher correlation for the collapsed affine score than for a draw from the half-sample distribution, and a worse proper score, when within-item variance is positive.

Deviations are whatever `results.json` shows. They are not edited back into this list.

## Availability

All planned multi-annotator datasets (SummEval, ChaosNLI, DICES-350/990, POPQUORN Offensiveness/Politeness, VeriBench FSC `R_v1`, live-synced `native_main`, and pooled) and CertJudge real LLM panels are scored in `results.json`. Only GPU runs (`scripts/snap_*.py`) and `T30` Google Sheet (HTTP 401 without OAuth) remain outside this CPU run.
