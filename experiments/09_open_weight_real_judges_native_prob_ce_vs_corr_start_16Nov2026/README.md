# 09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026/README.md>

**TLDR:** Systematically tests whether and when cross-entropy (CE) and ranked probability score (RPS) optimization beat correlation optimization on **real open-weight large language model (LLM) judges** across all three optimization mechanisms requested for Sanmi (`ii.a` prompt optimization, `ii.b` harness optimization, and `ii.c` real neural weight updates via PyTorch low-rank adaptation (LoRA) backpropagation and hidden-state heads) on `mercury2` (`10x NVIDIA RTX A4000`, `$0.00` free GPUs, no differential testing (no DT)), alongside a complete audit of `i.` (what the synthetic studies changed). Companion Experiment 10 (`experiments/10_frontier_ai_judges_native_prob_and_prompt_search_ce_vs_corr_start_16Nov2026/`) tests `iii.a` prompt optimization and `iii.b` harness optimization on frontier AI judges.

## Scientific Taxonomy (`i`, `ii.a–c`, `iii.a–b`)

Origin: Brando Miranda's Vals AI weekly update (`11-13 to 11-16-2026`, item 3) and the meeting brief `Correlation_vs_Loss_Judges.zip` (`paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager/`):

- **`i. Synthetic / toy / simulated` (answered from verified repo history):**
  - **Prompt or harness changes?** **None.** No LLM, prompt, or harness ran in any synthetic experiment.
  - **What changed?** `00` and `01` did **fixed-candidate selection** over analytic distributions (`0` trained weights); `archive/13`, `06`, `07`, and `08` updated **`78` linear-softmax weights** ($W \in \mathbb{R}^{13 \times 6}$ on 13 cosine/sine features of scalar $x \in [-1, 1]$, plus a `12`-weight CE calibration head) via `L-BFGS-B`. Historical human-rating studies (`02`, `archive/11`, `12`, `14`) only fitted simplex mixture weights over fixed three-sample scalar outputs — **never** optimizing prompts, harnesses, or real LLM neural weights.
- **`ii. Real open-weight judges` (THIS EXPERIMENT, `09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026`):**
  - **`ii.a. Prompt optimization`:** Six zero-leakage prompt designs (`P0_human_mirror`, `P1_conventions`, `P2_disagreement_calibrated`, `P3_strict_anti_leniency`, `P4_rank_contrastive`, `P5_cdf_threshold`) across 10 open-weight models, **plus** an automated blind meta-prompt optimization loop (`Qwen3-30B-A3B`) proposing and selecting prompt revisions on training/validation task families under **Correlation (`①`/`②`)** versus **CE/RPS (`③`/`④`)**.
  - **`ii.b. Harness optimization`:** Six real judge harness architectures (`H1_single_verbal`, `H2_single_logprob`, `H3_ast_obligation_scaffold` with programmatic Python/Lean AST theorem extraction, `H4_two_call_critic_aggregator` theorem-by-theorem decomposition pipeline, `H5_deliberative_thinking` reasoning traces, and `H6_calibrated_ensemble` multi-sample + temperature/prior harness), optimized on training/validation task families under **Correlation (`①`/`②`)** versus **CE/RPS (`③`/`④`)**.
  - **`ii.c. Weight updates, real training`:**
    1. **End-to-end PyTorch LoRA + six-logit head backpropagation** inside pretrained open-weight LLMs (`Qwen3-4B-Instruct-2507` and `Qwen2.5-Coder-7B-Instruct` in `bfloat16` on `mercury2`) across `5` grouped task-family folds and training-size subsets under **CE**, **RPS**, **exact batch Pearson**, **smooth Spearman**, and **two-stage `Pearson-trained LLM + calibrated CE head`** (formulations ①, ②, ③, ④).
    2. **LLM hidden-state neural head training** (`200` grouped task-family splits × `6` training sizes `n_fam` × `5` restarts across all `7` objectives from `experiments/13, 06, 07, 08`) and **certified simplex mixtures** (bridging directly to `archive/14`).

## Files

- [PROTOCOL.md](PROTOCOL.md): pre-registered hypotheses, measurement contract, Karpathy sanity controls, GPU allocation plan, and decision rules across `ii.a`, `ii.b`, and `ii.c`.
- [FREEZE.json](FREEZE.json): SHA-256 hashes of the pre-registered protocol and scripts committed before measurement.
- [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md): splits, metrics, subtraction order, uncertainty, and optimization details for every table and plot.
- [results.md](results.md): live status and headline results.
- [RESULTS_open_weight_ce_vs_corr.md](RESULTS_open_weight_ce_vs_corr.md): comprehensive report covering `i.`, `ii.a`, `ii.b`, and `ii.c`.
- [CKPT_open_weight_ce_vs_corr.md](CKPT_open_weight_ce_vs_corr.md): resumable checkpoint.
- `scripts/`: `build_prompts.py`, `run_mercury2_open_judges.py` (`ii.a` & `ii.b`), `train_mercury2_lora_judges.py` (`ii.c`), `analyze_open_judges.py`, `make_plots.py`, `verify_results.py`.

**TLDR-end:** [judges-corr: 09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026] Tests CE and RPS versus correlation across prompt optimization (`ii.a`), harness optimization (`ii.b`), and real PyTorch LoRA LLM weight training (`ii.c`) on open-weight judges over 19 task families.
