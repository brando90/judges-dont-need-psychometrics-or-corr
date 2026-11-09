# Cross-entropy versus Pearson: complete

**Status, 11-09-2026:** The historical failure was reproduced; the separately frozen matched-mixture comparison and independent primary-prediction verification are complete. All 50 primary fits passed. Two CE certificate failures among 49,204 bootstrap fits were retained, leaving 998/1,000 valid resamples.

CE minus Pearson held-out log loss was -0.260401 [-1.457729, 0.198472], 95% paired task-family percentile interval with refitting, p-val=n/a. The interval includes zero. Both methods had worse observed log loss than the training-human-marginal baseline.

See [RESULTS_comparison.md](RESULTS_comparison.md) for the full table, numerical diagnosis and limitations; [ckpt.md](ckpt.md) for continuation; [SESSION_ACCESS.md](SESSION_ACCESS.md) for cross-profile context. The old affine-softmax optimization problem remains unresolved; this new condition avoids its optimization confound rather than rewriting its results.
