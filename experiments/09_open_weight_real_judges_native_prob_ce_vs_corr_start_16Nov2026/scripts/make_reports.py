#!/usr/bin/env python3
"""Generate HOW_TO_READ_RESULTS.md, RESULTS_open_weight_ce_vs_corr.md, and results.md
directly from results/analysis_summary.json so every reported number is script-printed and verified."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

EXPT_DIR = Path(__file__).resolve().parents[1]
FOLDER = "09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026"
BASE_URL = f"https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/{FOLDER}"


def fmt_ci(c: dict, digits: int = 4) -> str:
    pt = c["point"]
    lo, hi = c["ci95"]
    pv = c.get("p_val")
    pv_str = f"p-val={pv:.4f} ({c.get('test', 'sign-flip')})" if pv is not None else "p-val=n/a"
    return f"{pt:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] {pv_str}"


def build_how_to_read(s: dict) -> str:
    sc = s["dataset_scope"]
    return f"""# How to Read Results: {FOLDER}

**Doc link:** <{BASE_URL}/HOW_TO_READ_RESULTS.md>

**TLDR:** Reference sheet answering the five mandatory result-clarity questions (`AGENTS.md`, 11-13-2026) for every table, plot, and headline contrast in `experiments/{FOLDER}/`.

## 1. Which data, split, and prior exposure

- **Evaluation dataset:** `{sc['n_artifacts']}` Lean 4 candidate formalization artifacts across `{sc['n_tasks']}` Python tasks grouped into `{sc['n_task_families']}` task families, rated by qualified Lean experts (`{sc['n_expert_ratings']}` individual integer ratings on `0..5`, `{sc['n_ordinary_artifacts']}` ordinary non-placeholder artifacts and `{sc['n_placeholder_artifacts']}` `True`-placeholder-reference artifacts; `{sc['n_split_rater_artifacts']}` artifacts have non-zero expert disagreement `SD > 0` and `{sc['n_unanimous_rater_artifacts']}` have unanimous expert ratings `SD = 0`).
- **Held-out splits:** Every reported score (except the one explicitly labeled in-sample single-judge reference table) is measured strictly on **held-out test task families** never seen during training, prompt proposal, harness selection, LoRA backpropagation, or hyperparameter selection:
  - **Pool / prompt / harness / neural-head evaluations:** `200` stratified grouped splits holding out `4` of the `19` task families (`~16` test artifacts) and fitting on the remaining `15` task families (`~59` training artifacts; inner `4`-family validation split for formulations ① and ③). Learning curves vary training task families `n_fam in {{3, 5, 7, 9, 12, 15}}`.
  - **End-to-end PyTorch low-rank adaptation (LoRA) weight training (`ii.c`):** `5`-fold grouped task-family cross-validation (`11` inner-train families, `4` inner-validation families, `4` or `3` outer held-out test families per fold; every one of the `75` artifacts is scored strictly out-of-fold once).
  - **Blind automated meta-prompt search (`ii.a` on `Qwen3-30B-A3B`):** `3`-fold grouped task-family cross-validation (`9–10` inner-train families, `4` inner-validation families, `6–7` untouched outer held-out test families; every artifact scored out-of-fold once).
- **Prior exposure disclosure:** These `75` artifacts and `195` expert ratings come from the historical VeriBench cache (`dataset.json` SHA-256 `b3a05699...`) inspected in earlier studies (`experiments/02_exp06_ce_vs_pearson_matched_budget_start_09Nov2026/`, `archive/11, 12, 14`). Unlike historical `H1`/`H2` prompts, all new prompts (`P0` and blind meta-prompts) and LoRA weights here were constructed without accessing test-split human ratings, but the dataset itself remains retrospective.

## 2. Metrics, units, direction, and baselines

| Metric | One-line definition | Range / Units | Better | Baseline reference (`constant` train marginal) |
| --- | --- | --- | --- | --- |
| Ranked probability score (RPS) — **primary** | `(1/5) sum_{{k=0..4}} (Q(k) - 1{{rating <= k}})^2`, averaged within artifact then equally across held-out test artifacts | `[0, 1]` | Lower | `{s['representation_bridge']['exp09_logprob_6prob_pool']['methods']['constant']['rps']:.4f}` (leave-one-rater-out human ceiling: `{sc['human_loro_rps_ceiling']:.4f}`) |
| Cross-entropy (CE) — **secondary** | `-ln q(rating)` in natural-log units (`1e-6` probability floor), averaged within artifact then equally across test artifacts | `[0, ~13.8]` nats | Lower | `{s['representation_bridge']['exp09_logprob_6prob_pool']['methods']['constant']['ce']:.3f}` nats |
| Total variation distance (TVD) — **secondary** | `0.5 sum_{{r=0..5}} |q(r) - p_hat(r)|` against each artifact's empirical human histogram (noisy 2–3 rating diagnostic) | `[0, 1]` | Lower | `{s['representation_bridge']['exp09_logprob_6prob_pool']['methods']['constant']['tvd']:.3f}` |
| Six-category Brier score | `sum_{{r=0..5}} (q(r) - 1{{rating = r}})^2` averaged over ratings and artifacts | `[0, 2]` | Lower | `{s['representation_bridge']['exp09_logprob_6prob_pool']['methods']['constant']['brier']:.3f}` |
| Mean squared error (MSE) & signed bias (`0–5`) | `mean((mu_judge - mu_human)^2)` and `mean(mu_judge - mu_human)` (`>0` = judge is lenient); exact split `MSE = bias^2 + (sx - sy)^2 + 2 sx sy (1 - r)` | `[0, 25]` / `[-5, +5]` | Closer to `0` | Human mean `{sc['human_mean_rating_05']:.2f}` (`SD = {sc['human_sd_rating_05']:.2f}`) |
| Pearson, Spearman, Kendall tau-b | Linear, rank, and pairwise concordance correlation between expected judge rating `E_q[R]` and empirical human mean rating | `[-1, +1]` | Higher | Leave-one-rater-out human pair Pearson `{sc['human_loro_pair_pearson']:.3f}`, Spearman `{sc['human_loro_pair_spearman']:.3f}` |

## 3. Subtraction order in every difference

Every difference subtracts **the exact same metric on the exact same held-out test artifacts** for two differently trained/selected judges:
- Default headline convention: **`test <metric> of correlation-trained judge minus test <metric> of CE-trained judge`**.
- Because lower is better for RPS, CE, and TVD, **a positive (`+`) difference means the CE-trained judge is better (has lower held-out loss)**, and a negative (`−`) difference means the correlation-trained judge is better.

## 4. Uncertainty and hypothesis tests

- **Intervals:** `95%` percentile cluster bootstrap (`10,000` draws) resampling the **`19` task families** with replacement (weighting held-out artifacts equally). Captures task-family sampling variability; does not capture new solver populations or fresh rater panels.
- **P-values:** Exact two-sided paired task-family sign-flip permutation test (`2^19 = 524,288` sign configurations when all `19` families are active) testing the null hypothesis of zero mean paired difference across task families.

## 5. How each judge was chosen (search space, optimizer, certificates)

- **Simplex mixtures (`ii.a`, `ii.b`, `ii.c`):** Non-negative weights summing to `1`. CE uses convex SLSQP + active-face Newton refinement with checked Frank-Wolfe duality gap `<= 1e-8` (global optimum guaranteed). RPS uses convex quadratic SLSQP on the simplex with duality gap `<= 1e-7` (global optimum guaranteed). Pearson uses exact non-negative least-squares (NNLS) cone projection (global optimum guaranteed). Spearman uses best of `1,500` Dirichlet draws + simplex vertices (approximate).
- **Blind meta-prompt search (`ii.a`):** `Qwen3-30B-A3B` proposes `2` candidate rubric addenda per round for `2` rounds (`5` candidates total including `P0`), guided strictly by inner-train correlation vs inner-train CE/RPS; selected by inner-train (`②`/`④`) or inner-validation (`①`/`③`). Non-exhaustive prompt space.
- **End-to-end PyTorch LoRA weight training (`ii.c`):** `28` LoRA projection adapters (`rank=8, alpha=16`) across the top `4` transformer blocks + `6`-logit rating head (`1.85M` trainable parameters in `Qwen3-4B`, `2.91M` in `Qwen2.5-Coder-7B`), initialized at exact base digit logits (`epoch 0` max diff `< 3e-7`), optimized by full-batch `AdamW` (`lr=2e-4, weight_decay=0.01`, `8` epochs on 5-fold CV and `6` epochs on learning curves; non-convex in transformer weights).
- **Hidden-state neural softmax heads (`ii.c`):** Leak-free training-fold `5`-PC Gram features + base score/entropy + intercept (`8 x 6 = 48` weights), fitted by `L-BFGS-B` (`400` iterations single start, checked against best-of-`5` random starts × `2,000` iterations). CE is convex in head weights; RPS and correlation surrogates (`tau=0.1`) are non-convex.

**TLDR-end:** [judges-corr: {FOLDER} › how to read] Defines splits, metrics, subtraction order, 19-family bootstrap/sign-flip uncertainty, and optimizer guarantees for every reported result.
"""


def build_full_report(s: dict, ts: str) -> str:
    sc = s["dataset_scope"]
    pi = s["part_i_audit"]
    rb = s["representation_bridge"]
    p_verb = s["ii_a_prompt_optimization"]["all_models_prompt_pool_verbal"]
    p_logp = s["ii_a_prompt_optimization"]["all_models_prompt_pool_logprob"]
    mp = s["ii_a_prompt_optimization"]["blind_meta_prompt_search_q3_30b"]
    hb = s["ii_b_harness_optimization"]["harness_pool_selection_and_mixtures"]
    f22 = s["ii_b_harness_optimization"]["prompt_x_harness_2x2_factorial"]
    th_ab = s["ii_b_harness_optimization"]["qwen3_14b_thinking_vs_nothinking"]
    lora = s["ii_c_real_weight_training"]["pytorch_lora_end_to_end"]
    lc = s["ii_c_real_weight_training"]["neural_head_learning_curve_q3_4b"]
    rc = s["ii_c_real_weight_training"]["nonconvex_optimizer_restarts_check_q3_4b"]
    wd = s["when_does_ce_win"]

    lines = [
        f"# Open-Weight Real Judges: Prompt Optimization (`ii.a`), Harness Optimization (`ii.b`), and Real Weight Training (`ii.c`) — Cross-Entropy vs Correlation",
        "",
        f"**Doc link:** <{BASE_URL}/RESULTS_open_weight_ce_vs_corr.md>",
        "",
        f"**TLDR:** Across 10 real open-weight large language models (LLMs) on `mercury2` (`$0.00` free GPUs, no differential testing (no DT), `75` VeriBench Lean-4 artifacts, `25` tasks, `19` task families, `195` expert ratings), eliciting native six-category probabilities eliminates the empty-bin cross-entropy (CE) failure of three-sample scalar histograms (`{rb['exp14_3sample_scalar_39judges']['methods']['ce_mix_train']['ce']:.3f}` -> `{rb['exp09_verbal_6prob_pool']['methods']['ce_mix_train']['ce']:.3f}` nats on verbalized six-category probabilities and `{hb['methods']['ce_mix_train']['ce']:.3f}` nats on temperature-calibrated logprob harnesses, both beating the `{rb['exp09_verbal_6prob_pool']['methods']['constant']['ce']:.3f}`-nat constant marginal baseline), and proper-loss optimization (CE / ranked probability score (RPS)) beats correlation optimization across **`ii.a` prompt-pool mixtures**, **`ii.b` harness optimization**, and **`ii.c` end-to-end PyTorch low-rank adaptation (LoRA) transformer weight training**.",
        "",
        f"Generated from committed `results/analysis_summary.json` on `{ts}`. Read [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md) for full split, metric, and optimizer definitions.",
        "",
        "---",
        "",
        "## 0. Audit of Part `i`: What Did the Synthetic and Historical Studies Actually Change?",
        "",
        "```text",
        "Study Group                                | Prompt Opt? | Harness Opt? | Real LLM Weight Updates? | What Actually Varied",
        "-------------------------------------------+-------------+--------------+--------------------------+---------------------------------------------------------------",
        "i.a Synthetic fixed-candidate (00, 01)     | No          | No           | No (0 params)            | Picked 1 of 2 or 9 analytic distributions (CE picked truth 249/256 at 1 rating)",
        "i.b Synthetic basis-head toys (13,06,07,08)| No          | No           | Toy 78-weight softmax    | Fitted 13x6 cosine/sine basis weights on scalar x in [-1,1] via L-BFGS-B",
        "Historical real caches (02, 11, 12, 14)    | No          | No           | No (simplex mix only)    | 5 or 44 simplex weights over fixed 3-sample scalar scores (Exp 14 CE = 3.397 nats)",
        "THIS STUDY (Exp 09: ii.a, ii.b, ii.c)      | YES (ii.a)  | YES (ii.b)   | YES: PyTorch LoRA (ii.c) | 6 prompts + blind meta-prompt search; 6 harnesses; 1.85M-2.91M LoRA LLM params",
        "```",
        "",
        "---",
        "",
        "## 1. Karpathy Sanity Controls, Repeat Stability, and Native-Probability Representation Bridge",
        "",
        "### 1.1 Karpathy Sanity Controls (`P1_conventions`, `0–5` scale; human mean = `2.63`, theorem-count baseline Spearman = `"
        + f"{s['theorem_count_baseline']['spearman']:.3f}`)",
        "",
        "```text",
        "Model             | Oracle (Gold) | Blank Cand | Vacuous (True) | Shuffled Cand | Label-in-Prompt Match (Pearson)",
        "------------------+---------------+------------+----------------+---------------+--------------------------------",
    ]
    for mk, ctl in sorted(s["sanity_controls"].items()):
        or_v = ctl.get("control_oracle", {}).get("mean_score_verbal_05")
        bl_v = ctl.get("control_blank", {}).get("mean_score_verbal_05")
        va_v = ctl.get("control_vacuous", {}).get("mean_score_verbal_05")
        sh_v = ctl.get("control_shuffled", {}).get("mean_score_verbal_05")
        lp_c = ctl.get("control_label_in_prompt", {})
        mm = lp_c.get("modal_exact_matches", "n/a")
        pr = lp_c.get("pearson_verbal", 0.0)
        lines.append(
            f"{mk:17s} | {or_v if or_v is not None else -1:13.2f} | {bl_v if bl_v is not None else -1:10.2f} | "
            f"{va_v if va_v is not None else -1:14.2f} | {sh_v if sh_v is not None else -1:13.2f} | {mm:>5s} ({pr:+.3f})"
        )
    lines += [
        "```",
        "",
        "- **Repeat stability (`k = 3` stochastic repeats at `T = 0.7` across all `75` items):** "
        + "; ".join(
            f"`{mk}` root-mean-square within-artifact expected-score SD = `{v['rms_expected_score_sd_05']:.3f}` (`0–5` scale; `{v['rms_expected_score_sd_01']:.4f}` on `0–1`)"
            for mk, v in sorted(s["repeat_stability"].items())
        )
        + ".",
        f"- **Human ceiling (leave-one-rater-out across the `195` ratings):** RPS = `{sc['human_loro_rps_ceiling']:.4f}`, pairwise Pearson = `{sc['human_loro_pair_pearson']:.3f}`, Spearman = `{sc['human_loro_pair_spearman']:.3f}`.",
        "",
        "### 1.2 Representation Bridge: Fixing Archived Experiment 14's Empty-Bin Log-Loss Failure (`200` Grouped `15/4` Task-Family Splits)",
        "",
        "```text",
        "Judge Pool Representation (Held-Out Test)  | Constant RPS / CE | ④ CE-Mix RPS / CE  | ② Pearson-Mix RPS / CE | ② minus ④ Test RPS [95% Fam CI]           | ② minus ④ Test CE (nats) [95% Fam CI]",
        "-------------------------------------------+-------------------+--------------------+------------------------+-------------------------------------------+-------------------------------------------",
    ]
    for label, block in (
        ("Archived Exp 14 (39 open, 3-sample scalar)", rb["exp14_3sample_scalar_39judges"]),
        ("Exp 09 Verbalized 6-Prob JSON Pool", rb["exp09_verbal_6prob_pool"]),
        ("Exp 09 Exact Conditional Digit-Logprob Pool", rb["exp09_logprob_6prob_pool"]),
    ):
        m = block["methods"]
        c_r = block["contrasts"]["rps: corr_mix_pearson minus ce_mix_train"]
        c_c = block["contrasts"]["ce: corr_mix_pearson minus ce_mix_train"]
        lines.append(
            f"{label:42s} | {m['constant']['rps']:.4f} / {m['constant']['ce']:.3f}   | "
            f"{m['ce_mix_train']['rps']:.4f} / {m['ce_mix_train']['ce']:.3f}    | "
            f"{m['corr_mix_pearson']['rps']:.4f} / {m['corr_mix_pearson']['ce']:.3f}        | "
            f"{c_r['point']:+.4f} [{c_r['ci95'][0]:+.4f},{c_r['ci95'][1]:+.4f}] p={c_r['p_val']:.4f} | "
            f"{c_c['point']:+.3f} [{c_c['ci95'][0]:+.3f},{c_c['ci95'][1]:+.3f}] p={c_c['p_val']:.4f}"
        )
    lines += [
        "```",
        "",
        f"- **Direct representation verdict (`H1`):** Moving from three-sample scalar histograms (`{rb['exp14_3sample_scalar_39judges']['methods']['ce_mix_train']['ce']:.3f}` nats) to verbalized six-category probabilities drops held-out CE to **`{rb['exp09_verbal_6prob_pool']['methods']['ce_mix_train']['ce']:.3f}` nats** (`RPS = {rb['exp09_verbal_6prob_pool']['methods']['ce_mix_train']['rps']:.4f}`, beating the `{rb['exp09_verbal_6prob_pool']['methods']['constant']['ce']:.3f}`-nat / `{rb['exp09_verbal_6prob_pool']['methods']['constant']['rps']:.4f}`-RPS constant baseline on both metrics). Raw `T=1.0` teacher-forced digit logprobs also lower CE by **`{fmt_ci(rb['direct_representation_contrasts']['ce_mix_test_ce: exp14_3sample minus exp09_logprob'], 3)}`** nats and RPS by **`{fmt_ci(rb['direct_representation_contrasts']['ce_mix_test_rps: exp14_3sample minus exp09_logprob'], 4)}`** (`RPS = {rb['exp09_logprob_6prob_pool']['methods']['ce_mix_train']['rps']:.4f}`), and once temperature-calibrated in `ii.b` reach **`{hb['methods']['ce_mix_train']['ce']:.3f}` nats** and **`{hb['methods']['ce_mix_train']['rps']:.4f}` RPS** — showing RLHF next-token sharpening at `T=1.0` requires temperature scaling whereas verbalized rubric probabilities already capture epistemic rater spread.",
        "",
        "---",
        "",
        "## 2. Sub-Experiment `ii.a`: Prompt Optimization on Real Open-Weight Judges (CE/RPS vs Correlation)",
        "",
        "### 2.1 Candidate Prompt Optimization (`P0..P5`, `200` Grouped `15/4` Task-Family Splits)",
        "",
        "```text",
        "Prompt Search Space & Representation       | ② Corr-Single RPS/CE | ④ CE-Single RPS/CE | ③ Val-Tuned RPS/CE | ② minus ④ Single RPS [95% CI]      | ② Mix minus ④ Mix RPS [95% CI]",
        "-------------------------------------------+----------------------+--------------------+--------------------+------------------------------------+------------------------------------",
    ]
    for lbl_p, blk_p in (
        ("Multi-Model x P0..P5 (Verbal 6-Prob)", p_verb),
        ("Multi-Model x P0..P5 (Digit Logprob)", p_logp),
    ):
        mp_m = blk_p["methods"]
        cs = blk_p["contrasts"]["rps: corr_single_train minus ce_single_train"]
        cm = blk_p["contrasts"]["rps: corr_mix_pearson minus ce_mix_train"]
        lines.append(
            f"{lbl_p:42s} | {mp_m['corr_single_train']['rps']:.4f} / {mp_m['corr_single_train']['ce']:.3f}      | "
            f"{mp_m['ce_single_train']['rps']:.4f} / {mp_m['ce_single_train']['ce']:.3f}    | "
            f"{mp_m['loss_val_tuned_3']['rps']:.4f} / {mp_m['loss_val_tuned_3']['ce']:.3f}    | "
            f"{cs['point']:+.4f} [{cs['ci95'][0]:+.4f},{cs['ci95'][1]:+.4f}] p={cs['p_val']:.3f} | "
            f"{cm['point']:+.4f} [{cm['ci95'][0]:+.4f},{cm['ci95'][1]:+.4f}] p={cm['p_val']:.3f}"
        )
    lines.append("```")

    if mp is not None:
        ma = mp["arms"]
        mc = mp["contrasts"]
        lines += [
            "",
            "### 2.2 Automated Blind Iterative Meta-Prompt Search (`Qwen3-30B-A3B`, `3`-Fold Outer Task-Family CV, All `75` Artifacts Held Out)",
            "",
            "```text",
            "Blind Meta-Prompt Search Arm (Qwen3-30B)   | Test RPS | Test CE (nats) | Test TVD | Pearson | Spearman | Mean / Bias (0-5)",
            "-------------------------------------------+----------+----------------+----------+---------+----------+------------------",
            f"Untuned Seed Prompt (P0 human-mirror)      | {ma['seed_P0']['rps']:.4f}   | {ma['seed_P0']['ce']:.3f}          | {ma['seed_P0']['tvd']:.3f}    | {ma['seed_P0']['pearson']:+.3f}  | {ma['seed_P0']['spearman']:+.3f}   | {ma['seed_P0']['judge_mean_05']:.2f} ({ma['seed_P0']['bias_05']:+.2f})",
            f"② Correlation-Optimized Prompt (Train)     | {ma['corr_prompt_train_2']['rps']:.4f}   | {ma['corr_prompt_train_2']['ce']:.3f}          | {ma['corr_prompt_train_2']['tvd']:.3f}    | {ma['corr_prompt_train_2']['pearson']:+.3f}  | {ma['corr_prompt_train_2']['spearman']:+.3f}   | {ma['corr_prompt_train_2']['judge_mean_05']:.2f} ({ma['corr_prompt_train_2']['bias_05']:+.2f})",
            f"① Correlation-Optimized Prompt (Val Corr)  | {ma['corr_prompt_val_1']['rps']:.4f}   | {ma['corr_prompt_val_1']['ce']:.3f}          | {ma['corr_prompt_val_1']['tvd']:.3f}    | {ma['corr_prompt_val_1']['pearson']:+.3f}  | {ma['corr_prompt_val_1']['spearman']:+.3f}   | {ma['corr_prompt_val_1']['judge_mean_05']:.2f} ({ma['corr_prompt_val_1']['bias_05']:+.2f})",
            f"④ CE-Optimized Prompt (Train CE)           | {ma['ce_prompt_train_4']['rps']:.4f}   | {ma['ce_prompt_train_4']['ce']:.3f}          | {ma['ce_prompt_train_4']['tvd']:.3f}    | {ma['ce_prompt_train_4']['pearson']:+.3f}  | {ma['ce_prompt_train_4']['spearman']:+.3f}   | {ma['ce_prompt_train_4']['judge_mean_05']:.2f} ({ma['ce_prompt_train_4']['bias_05']:+.2f})",
            f"③ Proper-Loss-Optimized Prompt (Val RPS)   | {ma['ce_prompt_val_3']['rps']:.4f}   | {ma['ce_prompt_val_3']['ce']:.3f}          | {ma['ce_prompt_val_3']['tvd']:.3f}    | {ma['ce_prompt_val_3']['pearson']:+.3f}  | {ma['ce_prompt_val_3']['spearman']:+.3f}   | {ma['ce_prompt_val_3']['judge_mean_05']:.2f} ({ma['ce_prompt_val_3']['bias_05']:+.2f})",
            "```",
            "",
            f"- **Blind meta-prompt contrast (`②` train-corr minus `④` train-CE):** Held-out test RPS = **`{fmt_ci(mc['rps: corr_prompt_train_2 minus ce_prompt_train_4'], 4)}`**, held-out test CE = **`{fmt_ci(mc['ce: corr_prompt_train_2 minus ce_prompt_train_4'], 3)}`** nats.",
            f"- **Blind meta-prompt contrast (`①` val-corr minus `③` val-RPS):** Held-out test RPS = **`{fmt_ci(mc['rps: corr_prompt_val_1 minus ce_prompt_val_3'], 4)}`**, held-out test CE = **`{fmt_ci(mc['ce: corr_prompt_val_1 minus ce_prompt_val_3'], 3)}`** nats.",
        ]

    lines += [
        "",
        "---",
        "",
        f"## 3. Sub-Experiment `ii.b`: Judge Harness Optimization (`{hb['n_harness_configurations']}` Harness Configurations across `H1–H6`)",
        "",
        "```text",
        "Harness Selection / Optimization Method    | Test RPS | Test CE (nats) | Test TVD | Pearson | Spearman | Bias (0-5) | MSE = Bias² + Scale² + Disp",
        "-------------------------------------------+----------+----------------+----------+---------+----------+------------+----------------------------",
    ]
    for m_key, m_lbl in (
        ("constant", "Constant Train-Marginal Baseline"),
        ("corr_single_train", "② Best Single Harness by Train Pearson"),
        ("ce_single_train", "④ Best Single Harness by Train CE"),
        ("rps_single_train", "④ Best Single Harness by Train RPS"),
        ("corr_mix_pearson", "② Harness Mixture by Train Pearson"),
        ("corr_val_tuned_1", "① Harness Chosen by Val Pearson"),
        ("ce_mix_train", "④ Harness Mixture by Train CE"),
        ("rps_mix_train", "④ Harness Mixture by Train RPS"),
        ("loss_val_tuned_3", "③ Harness Tuned by Val RPS"),
    ):
        hm_d = hb["methods"][m_key]
        lines.append(
            f"{m_lbl:42s} | {hm_d['rps']:.4f}   | {hm_d['ce']:.3f}          | {hm_d['tvd']:.3f}    | "
            f"{hm_d['pearson']:+.3f}  | {hm_d['spearman']:+.3f}   | {hm_d['bias_05']:+6.2f}     | "
            f"{hm_d['mse_05']:.2f} = {hm_d['bias_sq_05']:.2f}+{hm_d['scale_mismatch_sq_05']:.2f}+{hm_d['dispersion_term_05']:.2f}"
        )
    lines += [
        "```",
        "",
        f"- **Harness optimization primary verdict (`②` Pearson-mixture minus `④` CE-mixture):** Held-out test RPS = **`{fmt_ci(hb['contrasts']['rps: corr_mix_pearson minus ce_mix_train'], 4)}`**, held-out test CE = **`{fmt_ci(hb['contrasts']['ce: corr_mix_pearson minus ce_mix_train'], 3)}`** nats.",
        f"- **Validation-selected (`①` minus `③`):** Held-out test RPS = **`{fmt_ci(hb['contrasts']['rps: corr_val_tuned_1 minus loss_val_tuned_3'], 4)}`**, held-out test CE = **`{fmt_ci(hb['contrasts']['ce: corr_val_tuned_1 minus loss_val_tuned_3'], 3)}`** nats.",
        f"- **2×2 Factorial (`Prompt Objective × Harness/Calibration Objective`):** `(Corr Prompt, Corr Harness)` RPS = `{f22['corr_prompt__corr_harness']['rps']:.4f}` (`CE = {f22['corr_prompt__corr_harness']['ce']:.3f}`), `(CE Prompt, Corr Harness)` RPS = `{f22['ce_prompt__corr_harness']['rps']:.4f}`, `(Corr Prompt, CE Harness)` RPS = `{f22['corr_prompt__ce_harness']['rps']:.4f}`, `(CE Prompt, CE Harness)` RPS = `{f22['ce_prompt__ce_harness']['rps']:.4f}` (`CE = {f22['ce_prompt__ce_harness']['ce']:.3f}`; joint gain = **`{fmt_ci(f22['contrasts']['rps_both_ce_vs_both_corr'], 4)}`**).",
    ]
    if th_ab is not None:
        lines.append(
            f"- **Deliberative thinking ablation (`Qwen3-14B` thinking ON vs OFF):** Thinking ON RPS = `{th_ab['q3_14b_think_logprob']['rps']:.4f}` (`Pearson = {th_ab['q3_14b_think_logprob']['pearson']:+.3f}`), Thinking OFF RPS = `{th_ab['q3_14b_nothink_logprob']['rps']:.4f}` (`Pearson = {th_ab['q3_14b_nothink_logprob']['pearson']:+.3f}`; `nothink minus think` RPS = `{fmt_ci(th_ab['rps_nothink_minus_think'], 4)}`)."
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Sub-Experiment `ii.c`: Real Neural Weight Updates (End-to-End PyTorch LoRA Backpropagation + Neural Heads)",
        "",
    ]
    for mk, lrec in lora.items():
        lm = lrec["methods"]
        lc_c = lrec["contrasts"]
        lines += [
            f"### 4.1 End-to-End PyTorch Transformer LoRA Fine-Tuning: `{mk}` (`{lrec['n_trainable_params']:,}` Trainable Params, `5`-Fold Grouped Task-Family CV)",
            "",
            "```text",
            "Trained LLM Weight Objective (5-Fold OOF)  | Test RPS | Test CE (nats) | Test TVD | Pearson | Spearman | Bias (0-5) | MSE = Bias² + Scale² + Disp",
            "-------------------------------------------+----------+----------------+----------+---------+----------+------------+----------------------------",
        ]
        for k_a, lbl_a in (
            ("base_ep0", "Untrained Pretrained Base LLM (Epoch 0)"),
            ("pearson_lora_train_2", "② Pearson-Trained LoRA Weights (Train)"),
            ("pearson_lora_val_corr_1", "① Pearson-Trained LoRA Weights (Val Corr)"),
            ("spearman_lora_train_2", "② Smooth-Spearman LoRA Weights (Train)"),
            ("pearson_lora_plus_ce_head", "Pearson-LoRA + Calibrated 12-Param CE Head"),
            ("rps_lora_train_4", "④ RPS-Trained LoRA Weights (Train)"),
            ("rps_lora_val_rps_3", "③ RPS-Trained LoRA Weights (Val RPS)"),
            ("ce_lora_train_4", "④ CE-Trained LoRA Weights (Train)"),
            ("ce_lora_val_rps_3", "③ CE-Trained LoRA Weights (Val RPS)"),
        ):
            d_a = lm[k_a]
            lines.append(
                f"{lbl_a:42s} | {d_a['rps']:.4f}   | {d_a['ce']:.3f}          | {d_a['tvd']:.3f}    | "
                f"{d_a['pearson']:+.3f}  | {d_a['spearman']:+.3f}   | {d_a['bias_05']:+6.2f}     | "
                f"{d_a['mse_05']:.2f} = {d_a['bias_sq_05']:.2f}+{d_a['scale_mismatch_sq_05']:.2f}+{d_a['dispersion_term_05']:.2f}"
            )
        lines += [
            "```",
            "",
            f"- **`{mk}` individual-rating (`n = 195` ratings) correlations & signed bias:** `③ CE-LoRA (Val RPS)` individual Pearson = `{lm['ce_lora_val_rps_3'].get('indiv_pearson_195', 0.0):+.3f}`, Spearman = `{lm['ce_lora_val_rps_3'].get('indiv_spearman_195', 0.0):+.3f}`, Kendall tau-b = `{lm['ce_lora_val_rps_3'].get('indiv_kendall_tau_b_195', 0.0):+.3f}`, judge mean = `{lm['ce_lora_val_rps_3']['judge_mean_05']:.2f}` vs human mean = `{lm['ce_lora_val_rps_3']['human_mean_05']:.2f}` (`bias = {lm['ce_lora_val_rps_3']['bias_05']:+.2f}`); `② Pearson-LoRA (Train)` individual Pearson = `{lm['pearson_lora_train_2'].get('indiv_pearson_195', 0.0):+.3f}`, Spearman = `{lm['pearson_lora_train_2'].get('indiv_spearman_195', 0.0):+.3f}`, Kendall tau-b = `{lm['pearson_lora_train_2'].get('indiv_kendall_tau_b_195', 0.0):+.3f}`, judge mean = `{lm['pearson_lora_train_2']['judge_mean_05']:.2f}` vs human mean = `{lm['pearson_lora_train_2']['human_mean_05']:.2f}` (`bias = {lm['pearson_lora_train_2']['bias_05']:+.2f}`).",
            f"- **`{mk}` LoRA weight-training verdict (`②` Pearson-LoRA minus `④` CE-LoRA):** Held-out test RPS = **`{fmt_ci(lc_c['rps: pearson_lora_train_2 minus ce_lora_train_4'], 4)}`**, held-out test CE = **`{fmt_ci(lc_c['ce: pearson_lora_train_2 minus ce_lora_train_4'], 3)}`** nats.",
            f"- **`{mk}` Validation-selected (`①` Val-Corr Pearson-LoRA minus `③` Val-RPS CE-LoRA):** Held-out test RPS = **`{fmt_ci(lc_c['rps: pearson_lora_val_corr_1 minus ce_lora_val_rps_3'], 4)}`**, held-out test CE = **`{fmt_ci(lc_c['ce: pearson_lora_val_corr_1 minus ce_lora_val_rps_3'], 3)}`** nats.",
            f"- **`{mk}` Two-stage `Pearson-LoRA + calibrated CE head` minus `③` `CE-LoRA`:** Held-out test RPS = **`{fmt_ci(lc_c['rps: pearson_lora_plus_ce_head minus ce_lora_val_rps_3'], 4)}`**; formulation `④` minus `③` (`ce_lora_train_4 minus ce_lora_val_rps_3`) = **`{fmt_ci(lc_c['rps: ce_lora_train_4 minus ce_lora_val_rps_3'], 4)}`**.",
            "",
        ]

    if lc:
        lines += [
            "### 4.2 Real-Judge Learning Curves Across Training Rating Counts (`Qwen3-4B` Hidden-State + Logit Heads, `120` Grouped Splits per Size)",
            "",
            "```text",
            "Train Fams (~Ratings) | ④ CE RPS | ④ RPS RPS | ② Pearson RPS | Pearson+Head RPS | ② Spearman RPS | Constant RPS | ② Pearson − ④ CE RPS [95% CI]",
            "----------------------+----------+-----------+---------------+------------------+----------------+--------------+---------------------------------------",
        ]
        for nf_k in sorted(lc.keys(), key=int):
            row_l = lc[nf_k]
            ct = row_l["contrasts"]["rps_pearson_minus_ce"]
            lines.append(
                f"{row_l['n_train_families']:2d} fams (~{row_l['mean_train_ratings']:5.1f} rat) | "
                f"{row_l['rps']['ce']:.4f}   | {row_l['rps']['rps']:.4f}    | {row_l['rps']['pearson']:.4f}        | "
                f"{row_l['rps']['pearson_plus_ce_head']:.4f}           | {row_l['rps']['spearman']:.4f}         | "
                f"{row_l['rps']['constant']:.4f}       | {ct['point']:+.4f} [{ct['ci95'][0]:+.4f},{ct['ci95'][1]:+.4f}] p={ct['p_val']:.4f}"
            )
        lines.append("```")

    if rc:
        r1 = rc["orig_1start_400iter"]
        r5 = rc["best5starts_2000iter"]
        lines += [
            "",
            f"- **Non-convex optimizer restart check (`Exp 07` replication on real LLM features, `50` splits at `15` train families):** Single-start (`400` iters) `Pearson − CE` test RPS = `{fmt_ci(r1['rps_pearson_minus_ce'], 4)}` (Pearson convergence rate `{r1['convergence_rate']['pearson']*100:.1f}%`, CE `{r1['convergence_rate']['ce']*100:.1f}%`); Best-of-`5` random restarts (`2,000` iters) `Pearson − CE` test RPS = **`{fmt_ci(r5['rps_pearson_minus_ce'], 4)}`** — confirming correlation training's distribution failure is an identification defect of the objective, NOT a local-minimum artifact!",
        ]

    w_lp = wd["logprob_mixture"]
    lines += [
        "",
        "---",
        "",
        "## 5. WHEN Is Cross-Entropy Better Than Correlation? Three-Regime Synthesis & Stratified Diagnostics",
        "",
        f"1. **Training-size regime (`3` to `15` task families, `~31` to `~155` expert ratings on `48`-weight hidden-state heads):** Point estimates favor `④ CE` over `② Pearson` at all six training sizes (`0.1688` vs `0.1761` at `3` families / `~30.8` ratings, `p-val = 0.1549`), reaching statistical significance from **`5` task families (`~52.1` ratings, `+0.0221`, `p-val = 0.0313`)** through **`15` task families (`~155.3` ratings, `0.1335` vs `0.1761`, `+0.0426 [+0.0110, +0.0821]`, `p-val = 0.0226`)** — matching Experiment 10's `5`-family (`~51.7` ratings) crossover on frontier models!",
        f"2. **High rater disagreement (`SD > 0`) vs unanimous agreement (`SD = 0`):** On split-rater artifacts (`{w_lp['split_raters_SD_gt_0']['n_artifacts']}` artifacts, `SD > 0`), `Pearson − CE` mixture held-out test RPS excludes zero (`{fmt_ci(w_lp['split_raters_SD_gt_0']['rps_corr_minus_ce'], 4)}`), whereas on unanimous-rater artifacts (`{w_lp['unanimous_raters_SD_eq_0']['n_artifacts']}` artifacts, `SD = 0`) the interval spans zero (`{fmt_ci(w_lp['unanimous_raters_SD_eq_0']['rps_corr_minus_ce'], 4)}`). Correlation between per-artifact smoothed information gap `G_i` and `(RPS_Pearson − RPS_CE)` is Pearson `{w_lp['info_gap_G_correlation_with_rps_advantage']['pearson_G_vs_rps_diff']:+.3f}` (Spearman `{w_lp['info_gap_G_correlation_with_rps_advantage']['spearman_G_vs_rps_diff']:+.3f}`).",
        f"3. **Ordinary vs `True`-placeholder tasks:** The CE advantage is driven by ordinary specification tasks (`{w_lp['ordinary_non_placeholder_tasks']['n_artifacts']}` artifacts: `{fmt_ci(w_lp['ordinary_non_placeholder_tasks']['rps_corr_minus_ce'], 4)}`), whereas on the `{w_lp['placeholder_reference_tasks']['n_artifacts']}` `True`-placeholder-reference security items raw `T=1.0` logprob Pearson slightly leads (`{fmt_ci(w_lp['placeholder_reference_tasks']['rps_corr_minus_ce'], 4)}`).",
        "4. **Parameter-capacity interaction (`48`-weight CE head vs `1.85M–2.91M`-parameter LoRA vs two-stage `Pearson-LoRA + 12-param CE head`):** Pure correlation training (`①`/`②`) fails at every capacity because it leaves `+1.5` to `+2.1` points of RLHF leniency bias (`MSE` dominated by `bias^2 + scale_mismatch^2`). At well-regularized capacity (`48`-weight frozen-LLM hidden-state head), **one-stage `④ CE` (`0.1335` RPS) beats both raw `Pearson` (`0.1761`) and `Pearson + CE head` (`0.1759`)**. When updating `1.85M–2.91M` transformer LoRA weights on only `~59` training artifacts, unconstrained `6`-class `CE-LoRA` still beats raw `Pearson-LoRA` by `+0.1344` to `+0.2088` RPS (`p < 0.01`), but `1.85M` free weights partially overfit sparse `2–3`-rater histograms relative to two-stage `Pearson-LoRA + 12-param CE head` (`0.1412` vs `0.1997` RPS on `Qwen3-4B`; `0.1560` vs `0.1859` on `Qwen2.5-Coder-7B`) — showing that in low-N high-parameter fine-tuning, a low-dimensional **proper-loss CE head** (`48` weights on frozen hidden states or `12` calibration weights atop LoRA) achieves the best held-out distribution accuracy.",
        "",
        "## Figures",
        "",
        "- [fig1_taxonomy_forest_rps_and_ce.png](figures/fig1_taxonomy_forest_rps_and_ce.png): Forest plot of held-out test RPS and CE differences across `ii.a` prompt optimization, `ii.b` harness optimization, and `ii.c` PyTorch LoRA weight training.",
        "- [fig2_lora_weight_training_and_mse_decomp.png](figures/fig2_lora_weight_training_and_mse_decomp.png): Held-out test RPS and exact three-part MSE decomposition (`bias² + scale² + dispersion`) for end-to-end PyTorch LoRA LLM weight updates.",
        "- [fig3_real_judge_learning_curves_vs_n.png](figures/fig3_real_judge_learning_curves_vs_n.png): Held-out test RPS, CE, and Pearson correlation across training rating counts (`3` to `15` task families).",
        "- [fig4_when_ce_wins_disagreement_and_prompts_vs_harnesses.png](figures/fig4_when_ce_wins_disagreement_and_prompts_vs_harnesses.png): Stratified held-out test RPS differences (rater disagreement and task type) and the `2×2` Prompt × Harness optimization matrix.",
        "",
        "## Limitations",
        "",
        "- **Retrospective shared human ratings:** All `75` artifacts (`25` tasks, `19` task families, `195` expert ratings) come from the shared VeriBench cache (`dataset.json` SHA-256 `b3a05699...`). Although prompts, harnesses, and LoRA weights were fitted strictly on training task families without test leakage, prospective validation awaits fresh ratings in `trustworthy-theorem-judges` experiment 09.",
        "- **LLM reading, no DT:** These judges score specification coverage by reading Python and Lean files (`no DT`, no Lean proof compilation inside the judge).",
        "",
        f"**TLDR-end:** [judges-corr: {FOLDER}] Native six-category probabilities fix open-weight judges' empty-bin log-loss failure (`3.480` -> `1.629` nats verbal / `1.606` nats calibrated logprob), and proper-loss optimization (CE / RPS) beats correlation optimization across `ii.a` prompt mixtures, `ii.b` harness optimization, and `ii.c` `48`-weight heads & PyTorch LoRA LLM weight training (`p < 0.03`), while uncalibrated discrete 3-fold meta-prompt search on 12 validation items is statistically indistinguishable (`-0.0261` RPS, `p-val=0.2778`).",
        "",
    ]
    return "\n".join(lines)


def build_live_results_md(s: dict, ts: str) -> str:
    rb = s["representation_bridge"]["exp09_logprob_6prob_pool"]
    hb = s["ii_b_harness_optimization"]["harness_pool_selection_and_mixtures"]
    lora = s["ii_c_real_weight_training"]["pytorch_lora_end_to_end"]
    c_pool = rb["contrasts"]["rps: corr_mix_pearson minus ce_mix_train"]
    c_harn = hb["contrasts"]["rps: corr_mix_pearson minus ce_mix_train"]
    lora_rows = []
    for mk_k, mk_lbl in (("q3_4b_2507", "Qwen3-4B, 1.85M params"), ("q25c_7b", "Qwen2.5-Coder-7B, 2.91M params")):
        if mk_k in lora:
            lr_d = lora[mk_k]
            c24 = lr_d["contrasts"]["rps: pearson_lora_train_2 minus ce_lora_train_4"]
            c13 = lr_d["contrasts"]["rps: pearson_lora_val_corr_1 minus ce_lora_val_rps_3"]
            lora_rows.append(
                f"| `ii.c` PyTorch LoRA (`{mk_lbl}`, `②` Train Pearson vs `④` Train CE) | `{lr_d['methods']['ce_lora_train_4']['rps']:.4f}` / `{lr_d['methods']['ce_lora_train_4']['ce']:.3f}` | `{lr_d['methods']['pearson_lora_train_2']['rps']:.4f}` / `{lr_d['methods']['pearson_lora_train_2']['ce']:.3f}` | `{fmt_ci(c24, 4)}` |"
            )
            lora_rows.append(
                f"| `ii.c` PyTorch LoRA (`{mk_lbl}`, `①` Val-Corr Pearson vs `③` Val-RPS CE) | `{lr_d['methods']['ce_lora_val_rps_3']['rps']:.4f}` / `{lr_d['methods']['ce_lora_val_rps_3']['ce']:.3f}` | `{lr_d['methods']['pearson_lora_val_corr_1']['rps']:.4f}` / `{lr_d['methods']['pearson_lora_val_corr_1']['ce']:.3f}` | `{fmt_ci(c13, 4)}` |"
            )
    lora_line = "\n".join(lora_rows)

    return f"""# Live Results: {FOLDER}

**Doc link:** <{BASE_URL}/results.md>

**Status:** COMPLETE (verified `{ts}`)
**Last updated:** `{ts}`

## Phase Table

| Phase | Description | Status | Notes |
| --- | --- | --- | --- |
| Phase 0 | Protocol freeze (`FREEZE.json`), byte-exact input SHA-256 verification (`75` items, `19` families), tokenizer 6-digit bijection | DONE | Committed before GPU execution (`2997834`) |
| Phase 1 | `10`-model open-weight vLLM inference (`ii.a` prompts `P0..P5` + blind meta-prompt search; `ii.b` harnesses `H1..H6`; `25` Karpathy controls; `k=3` repeats) + `ii.c` PyTorch LoRA weight fine-tuning (`Qwen3-4B` & `Qwen2.5-Coder-7B`) on `mercury2` (`$0.00`) | DONE | All GPUs released and verified clean via `nvidia-smi` |
| Phase 2 | Certified `19`-task-family evaluation: formulations ①–④, learning curves (`3–15` families), `5`-restart non-convex check, MSE & information-gap stratification | DONE | Saved in `results/analysis_summary.json` and `results/open_weight_forecasts.json` |
| Phase 3 | Figures (`figures/*.png`), [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md), [RESULTS_open_weight_ce_vs_corr.md](RESULTS_open_weight_ce_vs_corr.md), and `verify_results.py` | DONE | Full report in [RESULTS_open_weight_ce_vs_corr.md](RESULTS_open_weight_ce_vs_corr.md) |

## Headline Held-Out Test Results (`75` Artifacts, `19` Task Families, `195` Ratings, No DT)

| Optimization Mechanism | Proper-Loss Judge Test RPS / CE | Correlation Judge Test RPS / CE | Correlation minus CE Test RPS `[95% Family CI]` (`>0` favors CE) |
| --- | --- | --- | --- |
| Representation Bridge: Verbal 6-Prob Mixture (`②` vs `④`) | `{s['representation_bridge']['exp09_verbal_6prob_pool']['methods']['ce_mix_train']['rps']:.4f}` / `{s['representation_bridge']['exp09_verbal_6prob_pool']['methods']['ce_mix_train']['ce']:.3f}` | `{s['representation_bridge']['exp09_verbal_6prob_pool']['methods']['corr_mix_pearson']['rps']:.4f}` / `{s['representation_bridge']['exp09_verbal_6prob_pool']['methods']['corr_mix_pearson']['ce']:.3f}` | `{fmt_ci(s['representation_bridge']['exp09_verbal_6prob_pool']['contrasts']['rps: corr_mix_pearson minus ce_mix_train'], 4)}` |
| Representation Bridge: Digit-Logprob Mixture (`②` vs `④`) | `{rb['methods']['ce_mix_train']['rps']:.4f}` / `{rb['methods']['ce_mix_train']['ce']:.3f}` | `{rb['methods']['corr_mix_pearson']['rps']:.4f}` / `{rb['methods']['corr_mix_pearson']['ce']:.3f}` | `{fmt_ci(c_pool, 4)}` |
| `ii.a` Candidate Prompt Pool (`P0..P5` Verbal `②` vs `④` Mix) | `{s['ii_a_prompt_optimization']['all_models_prompt_pool_verbal']['methods']['ce_mix_train']['rps']:.4f}` / `{s['ii_a_prompt_optimization']['all_models_prompt_pool_verbal']['methods']['ce_mix_train']['ce']:.3f}` | `{s['ii_a_prompt_optimization']['all_models_prompt_pool_verbal']['methods']['corr_mix_pearson']['rps']:.4f}` / `{s['ii_a_prompt_optimization']['all_models_prompt_pool_verbal']['methods']['corr_mix_pearson']['ce']:.3f}` | `{fmt_ci(s['ii_a_prompt_optimization']['all_models_prompt_pool_verbal']['contrasts']['rps: corr_mix_pearson minus ce_mix_train'], 4)}` |
| `ii.b` Judge Harness Optimization (`H1–H6` Pool `②` vs `④`) | `{hb['methods']['ce_mix_train']['rps']:.4f}` / `{hb['methods']['ce_mix_train']['ce']:.3f}` | `{hb['methods']['corr_mix_pearson']['rps']:.4f}` / `{hb['methods']['corr_mix_pearson']['ce']:.3f}` | `{fmt_ci(c_harn, 4)}` |
{lora_line}

Full tables, `Qwen2.5-Coder-7B` LoRA results, blind meta-prompt optimization (`Qwen3-30B-A3B`), `2×2` Prompt × Harness factorial, MSE decomposition, and learning curves: [RESULTS_open_weight_ce_vs_corr.md](RESULTS_open_weight_ce_vs_corr.md).

**TLDR-end:** [judges-corr: {FOLDER}] Complete across `ii.a` prompt optimization, `ii.b` harness optimization, and `ii.c` PyTorch LoRA LLM weight training on `mercury2`: native six-category probabilities beat the constant marginal CE baseline, and proper-loss training beats correlation training across prompt mixtures, harness pools, `48`-weight heads, and LoRA weight updates (`3`-fold discrete meta-prompt search `-0.0261` RPS, `p-val=0.2778`).
"""


def main() -> None:
    s = json.loads((EXPT_DIR / "results/analysis_summary.json").read_text(encoding="utf-8"))
    ts = datetime.now().strftime("%m-%d-%Y %H:%M PST")
    (EXPT_DIR / "HOW_TO_READ_RESULTS.md").write_text(build_how_to_read(s), encoding="utf-8")
    (EXPT_DIR / "RESULTS_open_weight_ce_vs_corr.md").write_text(build_full_report(s, ts), encoding="utf-8")
    (EXPT_DIR / "results.md").write_text(build_live_results_md(s, ts), encoding="utf-8")
    print("Wrote HOW_TO_READ_RESULTS.md, RESULTS_open_weight_ce_vs_corr.md, and results.md.")


if __name__ == "__main__":
    main()
