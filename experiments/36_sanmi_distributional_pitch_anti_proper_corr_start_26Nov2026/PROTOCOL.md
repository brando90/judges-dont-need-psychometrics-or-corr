# Measurement contract — 11-26-2026

Written before `run_all.py` printed a held-out test metric.

## Estimator

For a proper loss L in {cross-entropy, one-hot Brier, ranked probability score}, the test risk is

R(q) = (1/|D_test|) sum_c (1/k_c) sum_j L(q(c), y_{c,j}).

Plug-in D_B of the test histogram is not this number. Its bias against the truth is (1/k)(1-||p*||^2) in expectation.

## Split

Prompt groups, seed 0, fractions 0.6 / 0.2 / 0.2. SummEval groups are source articles. ChaosNLI and DICES groups are item ids. Validation does not choose hyperparameters.

## Interval

Item-clustered percentile bootstrap, B=1000, seed 1. The interval is for the fixed trained head.

## Head

Hash dimension 256. 24 principal components on train prompts only, plus an intercept. L2 penalty 1e-2 on non-intercept weights. L-BFGS-B from zero, at most 35 iterations, for cross-entropy, Brier, RPS, and the Pearson sample-correlation surrogate. MSE is ridge, closed form, two-point readout. Cross-entropy is convex in the linear logits. The other softmax objectives are not, and are reported as this start and this budget.

## Predictions

- Uniform CE equals log K.
- Shuffling train features does not lower held-out CE.
- Known two-context law: true sampled correlation near 0.50, collapsed 0.1*mean+4 near 0.707.
- Binary: Brier = 2 * scalar squared error, and an affine forecast can tie correlation while Brier and CE worsen.
- Ordinal consensus-on-2 versus 50/50 on {0,4}: same mean, D_B = 1.5.
- Within-prompt oracle (labeled, not a held-out judge): collapsed affine correlation above the sampled-distribution correlation when within-item variance is positive.

## Dollars

Top axis uses $0.05 per rating (POPQUORN's published crowd rate) and $5 per rating (assumed expert rate). Neither is a measured wage for SummEval, ChaosNLI, or DICES. Experiment 23 found no verified per-grade dollar for those releases.

## Not in the contract because the files are absent

POPQUORN ratings. VeriBench FSC / CertJudge R_v1. Live LLM logits. LoRA.
