# Experiment 24: QA corrections and verification limits (11-26-2026)

As of 11-26-2026. Findings g5-M1 and g5-M4; CPU source/output verification, no model calls.

The saved held-out ranked probability score (RPS), cross-entropy (CE), and total variation distance (TVD) outputs are preserved. The CE definition needs a label correction; independent recomputation from human ratings remains unavailable in this checkout.

- **g5-M4, CE smoothing:** old wording in the frozen protocol and results: probability “floor `6e-6`” and sensitivity “floor `1e-2`.” Corrected 11-26-2026: these are **uniform-mixture weights**, not per-category floors. Both scorer functions form `q_eps = (1-eps)*q + eps/5`; at an originally zero category the probabilities are **0.0000012** and **0.002**, respectively. The old scored tables and protocol remain preserved. This corrects terminology, not recorded CE scores.
- **g5-M1, raw-input limitation:** old headline RPS **0.15200 versus 0.15309** remains consistent with saved scores **0.15199526741822417 versus 0.1530916722240984** for calibrated CE-trained versus calibrated Pearson-trained judges at total budget 2,048. The saved `results/data_manifest.json` reports **1,038 responses, 519 prompts, 3,610 votes**, and the contrast output records the draw wins. Those records do not replace the absent `/tmp/exp24_helpsteer2_raw/prepared_bundle.json`. The prediction archive has probabilities but no human ratings, prompt identities or sampled-vote schedule. Consequently raw-rating scores, bootstrap intervals, draw-win counts and split disjointness remain unverified here; no missing observation was invented.

Evidence: [verification script](../../docs/qa_11-25-2026/phase2/verify_g5_g6_remaining.py) and [printed output](../../docs/qa_11-25-2026/phase2/verify_g5_g6_remaining.txt). Protocol and scored outputs are unchanged.
