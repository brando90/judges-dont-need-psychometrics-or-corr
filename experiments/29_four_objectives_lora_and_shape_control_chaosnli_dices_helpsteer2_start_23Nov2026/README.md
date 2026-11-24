# Experiment 29: Four-Objective LoRA Training & Shape Control

Tests **cross-entropy (CE)**, **ranked probability score (RPS)**, **mean squared error (MSE)**, and **Pearson correlation (r)** LoRA fine-tuning across `ChaosNLI SNLI+MNLI` ($K=3$, $d \in \{1, 5\}$), `ChaosNLI alphaNLI` ($K=2$ binary control), `DICES-350/990` ($K=3$), and `HelpSteer2` ($K=5$).

- **Status:** Complete (`1,401/1,401` Qwen3-4B + Llama-3.1-8B runs verified).
- **Protocol & Amendment:** [PROTOCOL_11-23-2026.md](PROTOCOL_11-23-2026.md) and [AMENDMENT_1_11-23-2026.md](AMENDMENT_1_11-23-2026.md)
- **Results & Methodology:** [RESULTS_11-23-2026.md](RESULTS_11-23-2026.md) and [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md)
