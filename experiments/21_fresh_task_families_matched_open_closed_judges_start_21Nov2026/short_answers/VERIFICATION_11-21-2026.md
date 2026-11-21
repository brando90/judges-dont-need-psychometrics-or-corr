# Independent replay of the short-answer calibration study | 11-21-2026

As of 11-21-2026. The replay used the forecasts frozen in commit `5e4982f` before the supervised analysis. Codex wrote a separate scorer and tested it on generated data before reading any study grades. [Frozen forecasts](https://github.com/brando90/judges-dont-need-psychometrics-or-corr/commit/5e4982f) · [Independent verifier](verify.py)

Every one of the 320 observed grades matched its original record in the University of North Texas release, using the question ID and answer position. For each of the ten held-out homework assignments and 200 draws, the independent training schedule matched all 144 answer indices in order. Every nine-answer round contained one answer from each training assignment. Each test fold contained the 16 answers from the remaining assignment. [Source archive](https://web.eecs.umich.edu/~mihalcea/downloads/ShortAnswerGrading_v2.0.zip) · [Protocol](../PROTOCOL_11-21-2026.md)

The replay recomputed all 27 calibration transforms of each judge from the frozen probabilities. It fitted cross-entropy (CE) using individual training grades and Pearson using training-answer mean grades. Both use exhaustive search with the declared ties. On each test answer, it scored the two observed grades individually. The normalized ranked probability score (RPS) averages squared cumulative probability error over five thresholds and two grades. Test CE averages the two negative log probabilities in nats. Total variation distance (TVD) is half the summed absolute distance to the two-grader histogram. RPS and TVD range from 0 to 1. Lower is better for all three.

| Independent checks | Two grades per training answer | One grade per training answer |
|---|---:|---:|
| Held-out assignment, judge, budget and method rows (three scores each) | 700 | 600 |
| Absolute score means and both interval endpoints | 210 | 180 |
| Paired score differences and both interval endpoints | 168 | 144 |
| Joint decision or descriptive status rows | 14 | 12 |

For every paired difference the replay also checked the one-sided t-test, all 1,024 fold-sign permutations and the fold win count. Means and bounds agreed within numerical tolerance (absolute and relative tolerance $6\times10^{-12}$). The intervals use ten assignment means, nine degrees of freedom and the prespecified fold-overlap correction. Their coverage is approximate for these dependent homework folds. [Absolute scores](results/scores.csv) · [Paired comparisons](results/contrasts.csv)

In the primary two-grade condition, the Qwen3.5-9B CE fit cleared both comparisons at all seven tested budgets, including four grades on two training answers. At four grades the test RPS of the Pearson fit minus the CE fit was +0.07145 (one-sided 97.5% lower bound +0.05020). Uniform minus CE was +0.09487 (lower bound +0.05412). No passing Qwen budget also cleared the training-histogram lower-bound check. Claude Sonnet 5 stopped at the first formal comparison, 256 grades. Its test RPS difference for Pearson fit minus CE fit was +0.000141 (lower bound −0.000083). Its six smaller budgets had positive local lower bounds against both Pearson and uniform and followed that formal stop. [All budget decisions](results/decisions.csv) · [Results](RESULTS_11-21-2026.md)

Run the commands in this directory after obtaining the source archive and generating the local individual-grade file with `prepare.py --include-labels`. The replay requires NumPy and SciPy.

```sh
python3 verify.py --fixture
python3 verify.py --source /path/to/ShortAnswerGrading_v2.0.zip \
    --labels /path/to/labels.jsonl --manifest design/manifest.jsonl \
    --forecasts forecasts.csv --results results
```
