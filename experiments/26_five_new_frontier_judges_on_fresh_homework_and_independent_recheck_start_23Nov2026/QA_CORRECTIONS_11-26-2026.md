# Experiment 26: QA verification limits (11-26-2026)

As of 11-26-2026. Finding g5-M3; CPU source/output verification, no model calls.

The GPT-6 Astra headline at 256 training ratings and two ratings per answer reports held-out ranked probability score (RPS) **0.0660 versus 0.0843**, CE-fit versus Pearson-fit. Corrected interpretation 11-26-2026: the saved score means are **0.066027006287224 versus 0.0843181067819244**, agreeing to rounding; this is verification of saved summaries, not a raw-rating replication.

`analysis/recheck.py` requires an ID-to-human-ratings JSONL file. That input is absent, and the new-five results do not supply per-fold losses. Raw labeled scores, interval widths and fold-win counts therefore remain unverified here. The experiment-21 recheck's separate evidence does not fill the missing new-five inputs. No numbers or frozen protocols have been changed.

Evidence: [verification script](../../docs/qa_11-25-2026/phase2/verify_g5_g6_remaining.py) and [printed output](../../docs/qa_11-25-2026/phase2/verify_g5_g6_remaining.txt).
