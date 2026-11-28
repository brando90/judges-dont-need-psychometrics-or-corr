# Support files for the evidence-led blog post (11-27-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/docs/notes/blog_support_27Nov2026/README.md>

**TLDR:** Every number in the preview post is recomputed from committed result files by `blog_numbers_check.py` (51 claims, 0 FAIL, two byte-identical runs), and `scan_post_numbers.py` fails if the post contains a number that no passed claim registers.

**What this is.** Support for the preview post `_posts/2026-11-27-bitter-judges-lesson-evidence.md` in the website repository ([brando90/brandomiranda](https://github.com/brando90/brandomiranda)), an evidence-led rewrite of "The Bitter Judges Lesson". Nothing here is a new experiment. The inputs are committed results of experiments 41, 43, 44, 45, 46, 47, 48, 49, 53, 54 and 55 (and experiment 31 through experiment 41's `verify_blog_numbers.py`), plus exact-fraction arithmetic for the toy example and the square-root identity.

## Files

1. [`blog_numbers_check.py`](blog_numbers_check.py): the claim checker. Run `python3 -I blog_numbers_check.py` from this folder. It re-derives each quoted number from a committed CSV, JSON or log, or by exact fractions (seed 20261127 for the random-law checks), prints PASS or FAIL per claim, then prints the registry of numeric tokens the post may use, and exits 1 on any FAIL.
2. [`blog_numbers_check_output_run1_27Nov2026.txt`](blog_numbers_check_output_run1_27Nov2026.txt) and [`blog_numbers_check_output_run2_27Nov2026.txt`](blog_numbers_check_output_run2_27Nov2026.txt): two runs of the checker on `main` at the time of writing. `cmp` reports no difference.
3. [`scan_post_numbers.py`](scan_post_numbers.py): run `python3 -I scan_post_numbers.py POST.md [CHECK_OUTPUT.txt]`. It extracts every number in the post's prose and requires an exact match with a registered token (rounded, year, 95% and single-digit tokens are listed for review). It also scans for banned patterns (provenance notes, internal-folder links, appendix or sources lists, advisor names, share links) and estimates the read time the website layout shows. It exits 1 on any unregistered token or banned pattern.
4. [`make_blog_figure.py`](make_blog_figure.py) and [`dices350_agreement_vs_skill.png`](dices350_agreement_vs_skill.png): the figure in section 4 of the post. It reads only `experiments/46_*/results/judge_skill_ci.json`: for five frozen frontier judges on DICES-350, top-1 agreement with the human majority rating (higher is better) and normalized ranked probability score (RPS) skill over the cross-fitted human base rate (higher is better; zero is the base rate), with 95% item-clustered bootstrap intervals, B = 4,000. No new computation; every axis has `ax.grid(True)`.

## What is not machine-checked

1. That the per-sample proper risk is unbiased: argued in the paper and listed in [`docs/LEAN_TODO.md`](../../LEAN_TODO.md) as not yet Lean-checked.
2. The squared-correlation identity (one minus the squared error of the best affine refit, over the variance of the rating): exact-fraction numerics on about two thousand random finite laws, not Lean-checked. The comparison of the square root of rho with rho for finite joint laws is Lean-checked (`PlanAudit.Model.anti_proper`, experiment 35).
3. What the post says about other papers (Pavlick and Kwiatkowski 2019, Plank 2022, Elangovan et al., Guerdan et al., Chen et al., Li et al., Wang, Zhang and Choi, Lyu, Do and Pergola, Murphy 1988, Kumar and Hoerling 2000, ChaosNLI, DICES): read from each paper's abstract page or search snippet on 11-27-2026, not by script. The census numbers (experiment 45) describe 40 sources chosen on purpose, not the field.
4. The Experiment 46 README says "H1 is supported". The frozen H1 is about Pearson and is not supported at the frozen B = 4,000; see [`ERRATUM_11-27-2026.md`](../../../experiments/46_real_llm_judge_rank_reversal_corr_vs_proper_scores_start_27Nov2026/ERRATUM_11-27-2026.md).
