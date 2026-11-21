# Reading the four-backbone historical calibration study | 11-21-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/19_small_data_moment_recovery_open_and_closed_judges_start_21Nov2026/human_judges/HOW_TO_READ_RESULTS.md>

Date: 11-21-2026

**TLDR:** Start with the absolute test ranked probability score (RPS) values and the first- and second-moment contrasts in [the results](RESULTS_11-21-2026.md). Compare each fitted judge with its original probability vector to see what calibrating on a small sample added to the existing judge.

## Which data and models?

Four named models scored the original 75 VeriBench artifacts. Each supplied a six-category verbal probability output using its study's P0 human-mirror prompt. These are observed, archived forecast vectors, with no synthetic replacement of the human data. Experiment 09 supplies the open-weight native verbal arrays. Experiment 10 supplies closed-weight native JSON probability arrays and the canonical artifact ordering. Experiment 02's historically authorized private archive supplies the 195 original expert ratings. Models, prompts, harness implementations and source years are fixed here. The model IDs' weight classifications describe these checkpoints and service IDs as of 11-21-2026. Open/closed is not a randomized treatment.

- **Open.** Qwen3-4B-Instruct-2507 and Qwen2.5-Coder-14B-Instruct have publisher-distributed downloadable checkpoints. Experiment 09's runner names the exact publisher repositories. The source retains four uniform parse-failure replacements on Qwen3-4B and zero on Qwen2.5-Coder-14B. [Qwen3-4B checkpoint](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/main) · [Qwen2.5-Coder-14B checkpoint](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/tree/main)
- **Closed.** GPT-5.4 mini and GPT-6.1 Sol are provider-hosted OpenAI models, identified in the archive by the API IDs gpt-5.4-mini and gpt-6.1-sol. Neither ID appears in the provider's open-weight distribution catalog. Experiment 10 recorded zero fallback rows for their P0 forecasts. [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini) · [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) · [OpenAI open-weight catalog](https://openai.com/open-models/)

## Which fits and comparisons?

Each training instance searches a grid of 27 calibration heads exhaustively. The heads share three global coefficients controlling probability sharpness, tilt toward high/low ratings and tilt toward extremes. Both primary heads see exactly the same original probabilities, task families and individual ratings. One minimizes individual-rating cross-entropy (CE). The other maximizes Pearson correlation with artifact mean ratings. Each fit is a global optimum within the declared grid up to the 10⁻¹² objective tolerance. The first tie rule favors the smallest parameter departure from the identity transform. A secondary arm breaks equally optimal Pearson ties by training RPS. A training-histogram constant ignores the native input. The identity head retains the original probabilities following the shared normalization and floor.

Training uses 3, 7 or 15 task families and 1 or 2 ratings per artifact. Each of the 250 previously used splits holds out four disjoint task families, with two or three test ratings per artifact. Test ratings, raters and families are historically familiar from earlier studies. This study performs no test-based selection. Rater identity is never held out. The constant is refit and each grid is searched again for every family-weight draw. No individual split is an independent replicate of the whole dataset.

## Metric and result files

- [scores.csv](scores.csv) reports all 2,700 absolute metric means and valid-split counts across the six inputs, six budgets and five procedures. Metric m1 is squared error of the estimated normalized mean versus a future rating divided by five, m2 through m5 are squared error of the corresponding raw powers; each ranges from zero to one. The centered-variance diagnostic compares to the sample variance with divisor k−1; its squared loss can reach 0.25 for a two-rating sample. Each loss weights artifacts equally. Estimates include the irreducible noise in one observed rating or sample variance, so a nonzero score does not imply a wrong population moment. The metrics have different scales; compare two methods on the same metric.
- In [contrasts.csv](contrasts.csv), subtract identifies the method whose test metric is subtracted from the method column. Positive means higher loss for method; lower is better. The 95% intervals are pointwise Bayesian bootstrap intervals obtained by 1,000 common exponential reweightings across the 19 original families; they incorporate refitting on reweighted train labels and reweighted test averages. They hold the observed models, raters and labels fixed. Comparing fitted models to their original outputs is marked exploratory because the explicit interval calculation was added after inspection of the prespecified identity baseline. No p-values or multiplicity adjustment.
- [accounting.csv](accounting.csv) gives annotation budgets, test counts, fully undefined Pearson fits and ties. [choices.csv](choices.csv) records frequencies of each chosen coefficient triple among unweighted fits; unlisted configurations were selected zero times. The [model manifest](models.json) gives source fingerprints, exact names and supplier links. The [protocol](PROTOCOL_11-21-2026.md) was committed before the sweep.

## Reproduce

From this directory with Python 3, NumPy, SciPy and Matplotlib and authorized access to the historical ratings archive, run the following commands. The analysis reads the original data in place and writes aggregate tables.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 analyze.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 verify.py
python3 report.py
```

The verifier reconstructs the grid using direct powers of the input probabilities, computes losses from individual observed ratings, reconstructs the fitted grid choices using centered correlation and raw-rating baseline counts, and checks all 1,440 mean losses and 1,728 paired differences, plus 54 weighted settings across six inputs and three fitting policies. The first deterministic check detected spurious near-ties from subtracting raw second moments of nearly constant predictions. Centering and scaling predictions before computing weighted Pearson fixed the discrepancy. All six forecast rows per artifact include every predeclared model and pooled input.
