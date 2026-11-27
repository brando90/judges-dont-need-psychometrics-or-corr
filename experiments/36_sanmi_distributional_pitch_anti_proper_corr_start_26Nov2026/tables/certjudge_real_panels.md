## CertJudge real LLM judge panels (`cert-judge@f6047849`, `$0` extra API spend)

These are real open-weight and frontier LLM judges evaluated on VeriBench functional semantic correctness (FSC, 0–5 integer scale), bridged directly from verified `cert-judge` Experiment 67 & 68 artifacts:

### 1. CertJudge Experiment 63 (`Qwen3-8B` native 6-class probabilities on VeriBench FSC `R_v1` held-out test, 30 items / 10 tasks)

| Training / prompting arm | `all5` CE [95% CI] | `all5` RPS [95% CI] | `all5` Spearman | `expert3` CE | `expert3` RPS | `expert3` Spearman |
| --- | --- | --- | --- | --- | --- | --- |
| `zero_shot_base` | 4.178 [2.896, 5.635] | 0.264 [0.187, 0.348] | -0.162 | 5.071 | 0.312 | -0.165 |
| `prompt_opt_arm1` | 3.964 [2.772, 5.314] | 0.292 [0.194, 0.404] | -0.181 | 4.710 | 0.338 | -0.184 |
| `rl_grpo` | 5.556 [3.740, 7.464] | 0.213 [0.151, 0.288] | 0.543 | 6.260 | 0.237 | 0.565 |
| `teacher_only_sft` | 1.741 [1.404, 2.056] | 0.157 [0.118, 0.195] | 0.710 | 1.658 | 0.149 | 0.772 |
| `human_mix_sft` | 1.620 [1.277, 1.937] | 0.142 [0.100, 0.184] | 0.746 | 1.392 | 0.114 | 0.828 |
| `shuffled_control` | 1.831 [1.679, 1.957] | 0.185 [0.155, 0.220] | -0.039 | 1.962 | 0.198 | 0.024 |

Key finding on real `Qwen3-8B` weights: `rl_grpo` reaches Spearman `0.543` (`all5`) / `0.565` (`expert3`), yet blows up held-out CE to `5.556` / `6.260` (worse than `shuffled_control` CE `1.831` / `1.962`). Training with cross-entropy (`human_mix_sft`) achieves the lowest CE (`1.620` / `1.392`), lowest RPS (`0.142` / `0.114`), and highest Spearman (`0.746` / `0.828`).

### 2. CertJudge Experiment 39 (56 LLM judge configs on 45 held-out VeriBench FSC `R_v1` items / 15 tasks, `all5`)

- **All 56 LLM configs:** mean raw CE `1.9128`, `CE-fit` mean CE `1.6428` vs `Pearson-fit` mean CE `1.6621` (`CE-fit` beats `Pearson-fit` on CE in **48/56** configs and on RPS in **40/56** configs: mean RPS `0.1536` vs `0.1547`).
- **Strictly held-out `test` split (18 LLM configs):** `CE-fit` mean CE `1.6175` vs `Pearson-fit` `1.6400` (`CE-fit` wins CE in **16/18**, RPS in **12/18**).

### 3. CertJudge Experiment 67 Interim (3 headline judges × 3 repeats on 100 `native_main` VeriBench artifacts / 25 tasks, 250 ratings)

| Judge family | `dRPS` (CE-fit minus Pearson-fit) [95% CI] | `dCE` (CE-fit minus Pearson-fit) [95% CI] | `dTVD` (CE-fit minus Pearson-fit) [95% CI] |
| --- | --- | --- | --- |
| Codex | -0.0077 [-0.0172, 0.0031] | -0.1437 [-0.2756, -0.0012] | -0.0585 [-0.1081, -0.0021] |
| Claude Opus | -0.0077 [-0.0167, 0.0023] | -0.1483 [-0.2816, -0.0075] | -0.0478 [-0.1003, 0.0108] |
| Qwen3-Coder | -0.0073 [-0.0164, 0.0029] | -0.1403 [-0.2691, 0.0021] | -0.0517 [-0.1043, 0.0049] |
