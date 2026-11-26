# Plan and status — 11-26-2026

**Status:** the driver `run_all.py` finished. Smoke flag: False.

## Contract (fixed before test metrics were read)

Predictions, written down before the run:

- Uniform cross-entropy equals \(\log K\).
- A CE-trained softmax should beat a constant uniform forecast on held-out CE when the hash features carry any signal. If it does not, the objective comparison is still inside a weak-feature family and the shuffle control should match the real CE head.
- Shuffling train features should not improve held-out CE.
- On a known two-context law, the collapsed judge's correlation should sit near 0.707 and the true sampled judge near 0.50.
- On binary data, scalar L2 and Brier match up to the factor 2. Correlation can still tie while Brier and CE worsen.
- On ordinal data with within-item spread, a mean-matched two-point readout should lose proper scores to a softmax that is allowed to use the full simplex.
- The within-prompt oracle should show a higher correlation for the collapsed affine score than for a draw from the half-sample distribution, and a worse proper score, when within-item variance is positive. This oracle is not a test-set judge.

Deviations are whatever `results.json` shows. They are not edited back into this list.

## Not available

POPQUORN and VeriBench FSC. See the availability block in `results.json`.
