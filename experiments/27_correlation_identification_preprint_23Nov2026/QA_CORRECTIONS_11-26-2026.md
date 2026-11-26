# Experiment 27: companion QA note for the frozen deposit (11-26-2026)

As of 11-26-2026. Findings g5-M6 and g5-L4. The deposited preprint and reproduction files remain frozen; this is a separate dated companion.

**g5-L4, presentation correction.** Old frozen table order: “Forecast / Pearson / TVD / Excess RPS / CE.” Corrected presentation 11-26-2026 follows below: ranked probability score (RPS), cross-entropy (CE), total variation distance (TVD), then Pearson correlation. RPS and CE are proper scores, and TVD is a distribution distance; all three are lower-is-better. The exact three-item toy has no train/test split, fitting, sampling or statistical interval. Expected RPS includes the human-distribution variance; excess RPS subtracts the true forecast's expected RPS.

| Forecast | Expected RPS | Excess RPS | CE (nats) | TVD | Pearson correlation of means |
|---|---:|---:|---:|---:|---:|
| True distribution | 49/300 | 0 | 0.9153873296489315 | 0 | 1 |
| Changed shape, same mean | 107/600 | 3/200 | 1.0647350946529013 | 1/5 | 1 |

**g5-M6, deposit evidence limitation.** Old deposit note claims public availability and source ZIP hash `67814aa5b3bd7f360dae68abf75d24b9790d722dc14a69a97f0e813ea8891f96`. Verification 11-26-2026 reproduces the local PDF hash exactly: `6263b743236557455adad26230edaea3ebcf21626f69725fbc50260edbddf148`. The deposited source ZIP is absent from the checkout. Read-only web attempts to open the [Stanford record](https://purl.stanford.edu/bh388th6008) and [source ZIP](https://stacks.stanford.edu/file/bh388th6008/correlation-identification-sources-11-23-2026.zip) returned tool access errors; those errors do not establish that the deposit is unavailable. Publication status and the remote ZIP hash remain unverified in this QA.

Evidence: [verification script](../../docs/qa_11-25-2026/phase2/verify_g5_g6_remaining.py) and [printed output](../../docs/qa_11-25-2026/phase2/verify_g5_g6_remaining.txt). The frozen paper's values are correct; its historical table is preserved.
