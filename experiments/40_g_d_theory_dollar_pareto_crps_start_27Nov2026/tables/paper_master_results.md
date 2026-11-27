# Paper master numbers

CPU rows are computed in this folder. GPU rows are quoted from trustworthy-theorem-judges experiment 28 at `efdc8c3`.

## CPU allocation, one budget per dataset (largest B in the grid)

| Dataset | Held-out test items | D-study k | Oracle k for test CE | D-study minus oracle test CE |
|---|---:|---:|---:|---|
| summeval_expert | 1280 | 2 | 2 | 0.0000 [0.0000, 0.0000] |
| summeval_crowd | 1280 | 5 | 3 | 0.0010 [-0.0006, 0.0025] |
| popquorn_offensiveness | 300 | 10 | 10 | 0.0000 [0.0000, 0.0000] |
| popquorn_politeness | 745 | 6 | 2 | 0.0018 [-0.0012, 0.0049] |
| dices990 | 198 | 17 | 8 | 0.0000 [-0.0034, 0.0034] |
| chaosnli_snli_mnli | 624 | 5 | 8 | 0.0015 [-0.0004, 0.0036] |
| veribench_fsc_native_main | 40 | 2 | 3 | 0.0813 [0.0128, 0.1406] |

## GPU bridge, quoted, not recomputed

Proper-score means and intervals in the quoted rows are copied from RESULTS_11-26-2026.md at efdc8c3. They were not recomputed: the JSON stores probabilities, not the human ratings those scores were computed against. The JSON checks are recomputed here.

| `Qwen3-4B` LoRA — `CE` (`exp27_test`, 20 tasks) | **0.1183** [0.0699, 0.1796] | **1.313** [0.891, 1.812] | **0.441** | **0.5623** | +0.362 (4.062 vs 3.700) | **0.557** | **0.749** | **-0.0662** [-0.1152, -0.0257] (`0.0002`) / **-2.813** |
| `Qwen3-4B` LoRA — `RPS` (`exp27_test`, 20 tasks) | 0.1914 [0.1068, 0.2928] | 4.144 [2.355, 6.282] | 0.514 | 0.8229 | +0.796 (4.496 vs 3.700) | 0.214 | 0.722 | `RPS - Pear`: +0.0069 [-0.0048, +0.0188] (`0.2524`) |
| `Qwen3-4B` LoRA — `Pearson` (`exp27_test`) | 0.1845 [0.1008, 0.2858] | 4.126 [2.294, 6.328] | 0.459 | 0.7758 | +0.874 (4.574 vs 3.700) | 0.278 | 0.723 | reference |
| `Qwen3-4B` LoRA — `CE` (`cv5_oof`, 50 tasks) | **0.1131** [0.0896, 0.1389] | **1.286** [1.073, 1.511] | **0.475** | **0.6100** | **+0.017** (3.759 vs 3.742) | **0.618** | **0.669** | **-0.0597** [-0.0920, -0.0307] (`0.0002`) / **-3.700** |
| `Qwen3-4B` LoRA — `RPS` (`cv5_oof`, 50 tasks) | 0.1389 [0.1055, 0.1762] | 2.409 [1.730, 3.174] | 0.491 | 0.7259 | +0.247 (3.989 vs 3.742) | 0.487 | 0.571 | `RPS - Pear`: **-0.0338** [-0.0572, -0.0115] (`0.0026`) |
| `Qwen3-4B` LoRA — `Pearson` (`cv5_oof`) | 0.1728 [0.1277, 0.2232] | 4.986 [3.784, 6.272] | 0.499 | 0.9090 | +0.661 (4.403 vs 3.742) | 0.387 | 0.551 | reference (`cv5` post-cal quarantined) |
| `Qwen3-4B-Instruct-2507` single-pass logits (`6144` tok) | **0.1482** [0.1144, 0.1859] | **3.589** [2.747, 4.505] | **0.499** | **0.8852** | +0.369 (4.111 vs 3.742) | 0.568 | 0.673 |
| `Qwen3-4B` `M=1` argmax one-hot (`6144` tok) | 0.1648 [0.1293, 0.2046] | 7.421 [6.209, 8.652] | 0.537 | 1.0743 | +0.378 (4.120 vs 3.742) | 0.570 | 0.653 |
| `Qwen3-8B` single-pass logits (2-GPU split, `6144` tok) | **0.2145** [0.1816, 0.2517] | **5.375** [4.441, 6.402] | **0.695** | **1.2167** | -0.637 (3.105 vs 3.742) | 0.413 | 0.514 |
| `Qwen3-8B` `M=1` argmax one-hot (`6144` tok) | 0.2470 [0.2129, 0.2848] | 10.408 [9.550, 11.244] | 0.753 | 1.5067 | -0.679 (3.063 vs 3.742) | 0.381 | 0.469 |

## JSON integrity, recomputed here

- `qwen3_4b_logit_6144.json`: 175 items, max |sum(probs)-1| = 2.000e-06, max |dot(probs, 0..5) - expected_score| = 6.000e-06.
- `qwen3_8b_logit_6144.json`: 175 items, max |sum(probs)-1| = 2.000e-06, max |dot(probs, 0..5) - expected_score| = 5.000e-06.
- LoRA `ce`: 70 exp27 test predictions, 175 OOF predictions, selected hp lr=0.0004 steps=24.
- LoRA `rps`: 70 exp27 test predictions, 175 OOF predictions, selected hp lr=0.0004 steps=12.
- LoRA `pearson`: 70 exp27 test predictions, 175 OOF predictions, selected hp lr=0.0001 steps=24.
