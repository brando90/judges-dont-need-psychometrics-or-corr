# GPU work is not the next step for the generalizability question

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/NEEDS_GPU.md>

**TLDR:** Nicole's question, whether generalizability theory or a decision study is needed once cross-entropy is available, is answered by the known-distribution simulation in experiment 34. No open-weight GPU run is the next step, and no Stanford Network Analysis Project (SNAP) script is queued.

**As of:** 11-25-2026.

The question needs a conditional distribution whose higher moments are known, so that a reliability coefficient and a proper score can be compared with the truth. Experiment `34_g_d_theory_vs_ce_where_reliability_agrees_and_fails_start_25Nov2026` does that on CPU in 32.2 seconds. A low-rank adaptation run would estimate a judge, not the population identities those coefficients miss.

Open-weight prompt, harness, and weight comparisons that already exist, and do not need to be rerun for this question:

- `09_open_weight_real_judges_native_prob_ce_vs_corr_start_16Nov2026`: prompt, harness, and low-rank adaptation on ten models.
- `24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026` and `25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026`: matched low-rank adaptation on HelpSteer2, including Llama-3.1-8B.
- `29_four_objectives_lora_and_shape_control_chaosnli_dices_helpsteer2_start_23Nov2026`: cross-entropy, ranked probability score, mean squared error, and Pearson, on Qwen3-4B and Llama-3.1-8B.

Do not launch a job on skampere2 GPUs 4–7 for this question. There is no `sbatch` script because there is no GPU command to run.

**TLDR-end:** [judge objectives: GPU] Experiment 34 answers the generalizability question on CPU. Existing open-weight runs stay where they are. No SNAP job.
