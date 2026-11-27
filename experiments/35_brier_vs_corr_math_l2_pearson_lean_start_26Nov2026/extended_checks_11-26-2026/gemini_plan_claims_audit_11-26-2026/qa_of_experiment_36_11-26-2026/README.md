# QA of Experiment 36 with Claude Opus 5.5 at maximum effort (11-26-2026)

**TLDR:** Brando asked for a QA with Opus 5.5 max of the Gemini plan, of Cursor's Experiment 36 and of my experiment-35 audit. An independent agent audited commit `036cafa` and found that the committed Pearson-trained head is crippled by its penalty, that its features carry almost no signal, and that matched calibration shrinks the proper-score gap by 40 to 100 times or removes it. I then reproduced or read-confirmed each finding myself; [`VERIFICATION_STATUS_11-26-2026.md`](VERIFICATION_STATUS_11-26-2026.md) says which, finding by finding. Nothing in Experiment 36 was changed by this QA.

**Data behind the numbers:** real multi-annotator ratings (SummEval expert, ChaosNLI SNLI+MNLI, DICES-990 and two binary views), held-out test split by prompt group (60/20/20, seed 0), the committed linear heads on 24 hashed-text components, metrics and intervals as defined in `VERIFICATION_STATUS_11-26-2026.md`. These are not human-judge results for large language models.

## Contents

1. [`QA_OPUS55_MAX_REPORT_11-26-2026.md`](QA_OPUS55_MAX_REPORT_11-26-2026.md): the review as written by the agent (not edited). Its labels are CONFIRMED (reproduced by the agent) and PLAUSIBLE (could not reproduce). It reads the Gemini plan, Experiment 36 and my audit.
2. [`VERIFICATION_STATUS_11-26-2026.md`](VERIFICATION_STATUS_11-26-2026.md): what I checked myself, and two corrections to the review.
3. [`my_scripts/`](my_scripts): three scripts of mine and their saved outputs. `qa_exp36_fairness_check.py` (penalty sweep for the Pearson-trained head), `qa_exp36_matched_calibration.py` (the same validation-fitted temperature and class offsets for both heads) and `chaos_premise_overlap.py` (premise leakage). Usage: `python3 -I <script> <path to Experiment 36 folder> <data directory>`; run against the code at commit `036cafa`.
4. [`agent_scripts/`](agent_scripts) and [`agent_outputs/`](agent_outputs): the review's scripts (verbatim, with `/tmp` paths from its sandbox) and outputs.
5. [`tally_fair_json.py`](tally_fair_json.py) and [`tally_fair_json.out`](tally_fair_json.out): counts which of the review's saved paired intervals exclude 0.

## How to reproduce

1. Download the inputs pinned by sha256 in Experiment 36's `data_io.py` (the same bytes are fingerprinted in `../results/dataset_facts.out`).
2. Run the three scripts in `my_scripts/` with the paths in point 3 above; the saved `.out` files are what I got.
3. `python3 -I tally_fair_json.py agent_outputs` reproduces `tally_fair_json.out` without any data.

## Limits

The review ran no language model and did not look at the figures. It also compiled `SanmiPitch.lean` (exit 0); I recompiled it myself with the same result (`my_scripts/sanmi_lean_recompile_11-26-2026.out`, identical axioms to Cursor's committed `lean_check.txt`). The review's optimizer experiments (multi-start, analytic gradients, 3,000 iterations) are in `agent_scripts/fairness.py` and I did not re-run them.
