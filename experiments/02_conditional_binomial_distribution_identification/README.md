# 02_conditional_binomial_distribution_identification

Controlled synthetic study of conditional rating-distribution identification. See [PROTOCOL.md](PROTOCOL.md) for the fixed procedure and predictions, [results.md](results.md) for live status, and [CKPT_conditional_identification.md](CKPT_conditional_identification.md) for continuation.

This experiment remains at its canonical numbered home while the parent completes the requested multi-provider review and integrates the paper. It is separate from theorem-judge experiment06's historical fitting and its active weighted-probability follow-up.

```bash
python3 -m unittest discover -s experiments/02_conditional_binomial_distribution_identification -p 'test_study.py' -v
timeout 1800s python3 experiments/02_conditional_binomial_distribution_identification/run_study.py
```

The runner verifies the fixed input/source hashes and refuses to overwrite completed evidence. Dependencies are NumPy and SciPy; measured versions are in `results/COMPLETE.json`. All data generated here are synthetic and publishable. The external historical source is public aggregate evidence only.
