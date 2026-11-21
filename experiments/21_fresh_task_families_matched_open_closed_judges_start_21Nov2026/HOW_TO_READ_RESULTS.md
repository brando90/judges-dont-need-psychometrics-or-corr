# How to reproduce the fresh-family judge comparison | 11-21-2026

As of 11-21-2026, the [prespecified protocol](PROTOCOL_11-21-2026.md) defines the decisions and missing-output handling. The two populations have different rubrics. Within a population, the exact system and user message bytes are matched and verified across the two model clients.

## Reproduce the labeled experiment

Download the [authors’ original archive](https://web.eecs.umich.edu/~mihalcea/downloads/ShortAnswerGrading_v2.0.zip). From the experiment directory, the commands below emit the label-free prompts, a matching manifest and a separate local evaluation file. The public forecast table allows all downstream statistics and plots to be reproduced without model calls. The source file checksum is checked by the builder.

```bash
python3 short_answers/prepare.py /path/to/ShortAnswerGrading_v2.0.zip /tmp/unt21 --include-labels
python3 short_answers/metrics.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 short_answers/analyze.py /tmp/unt21/manifest.jsonl /tmp/unt21/labels.jsonl short_answers/forecasts.csv short_answers/results
python3 short_answers/plot.py
python3 short_answers/report.py
```

For inference, the two [runners](runners/) consume only the request manifest, after verifying its hash. A fresh session receives the reference and candidate for a single task. One pinned Qwen3.5-9B Q4_K_M model runs in Ollama and one Claude Sonnet 5 runs under its first-party subscription. Each raw receipt records the actual model, the prompt hash and response. The common parser reproduces the public forecasts with the following command. Full response text and student answers are not needed to replay supervised fitting.

```bash
python3 runners/collect.py /tmp/unt21/requests.jsonl /path/to/qwen_receipts /path/to/claude_receipts.jsonl /tmp/unt21/forecasts.csv
```

## Read the tables

- The external unit of comparison is one homework assignment. Each of ten folds withholds all 16 of its answers, including both original human ratings per answer. Each fit uses exactly the indicated number of individual training ratings. The 200 nested training draws within a fold reuse the same homework pool. The primary condition uses two grades per distinct answer. The one-grade condition has wider answer coverage at the same total number of training ratings.
- The primary ranked probability score (RPS) is individual-grade squared error for the six-category forecast’s cumulative probabilities, averaged over the five thresholds; 0–1, lower is better. Cross-entropy (CE) measures individual-grade negative log probability in nats; total variation distance (TVD) compares the forecast with each answer’s two-grade empirical distribution, from 0 to 1. The unfitted uniform distribution, original smoothed judge and smoothed training histogram give reference values.
- In the paired comparisons, subtract the held-out loss of the CE-fitted judge from that of the named comparator, on identical held-out data. Positive favors the CE fit. [Per-family results](short_answers/results/families.csv) give the ten paired units. The inferential table uses an approximate overlap-adjusted t bound across those ten units; the grade-draw and sign-flip sensitivities have different interpretations. The fixed descending decision sequence concerns the two-grades-per-answer condition. [Full results](short_answers/RESULTS_11-21-2026.md)
- The exhaustive grid fixes the calibration model, optimization budget, and tie rule. No estimates are inferred about any model outside the grid. Judges received the same grading messages, but native chat templates, decoding, architectures and providers differ. The direction of the comparison between training objectives is identified within each judge; a causal effect of making weights public cannot be isolated.
- The prospective VeriBench arm is tracked [separately](veribench/STATUS_11-21-2026.md). Evaluation waits for the collection owner’s authorized scores. Label-free forecasts cannot establish a win on human VeriBench outcomes.
