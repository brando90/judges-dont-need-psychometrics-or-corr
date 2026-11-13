# 01_conditional_binomial_distribution_identification_start_08Nov2026

Controlled synthetic study of conditional rating-distribution identification. See [PROTOCOL.md](PROTOCOL.md) for the fixed procedure and predictions, [RESULTS_conditional_identification.md](RESULTS_conditional_identification.md) for the completed report, [results.md](results.md) for status, and [CKPT_conditional_identification.md](CKPT_conditional_identification.md) for continuation.

**How to read the results:** [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md) states which data (train, validation or held-out test), which metric and which comparison every reported number uses.

This experiment remains at its canonical numbered home while the parent completes the requested multi-provider review and integrates the paper. It is separate from theorem-judge experiment06's historical fitting and its completed weighted-probability follow-up.

```bash
python3 -m unittest discover -s experiments/01_conditional_binomial_distribution_identification_start_08Nov2026 -p 'test*.py' -v
timeout 1800s python3 experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/verify_results.py
```

Verification reconstructs every selection and common-metric summary from the saved evidence without drawing new ratings. For a fresh reproduction, supply a new output directory with `timeout 1800s python3 experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/run_study.py --out /path/to/new-results`. The runner verifies the fixed input/source hashes and refuses to overwrite completed evidence. Dependencies are NumPy and SciPy; measured versions are in `results/COMPLETE.json`. Native numerical libraries are capped at eight threads. All data generated here are synthetic and publishable. The external historical source is public aggregate evidence only.

## Review note (11-08-2026, stages 01–02 review)

The prediction table in `PROTOCOL.md` lists "empty/non-PMF input: invalid, not silently normalized". The scorer functions in `run_study.py` do not implement an explicit rejection of such inputs; every forecast in this study is an analytic probability mass function by construction, and `test_study.py` checks that the binomial rows sum to one. `PROTOCOL.md`, `run_study.py` and `test_study.py` are frozen by `FREEZE.json`, so the frozen synthetic runner and measured evidence remain untouched. A separately versioned validation wrapper (`validated_scorer.py`) and deterministic test suite (`test_validated_scorer.py`) enforce rejection of empty, malformed, non-finite or non-normalized inputs for prospective callers while matching `run_study.py` identically on valid inputs.
