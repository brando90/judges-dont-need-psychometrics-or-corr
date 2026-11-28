"""judge_scores: score an LLM judge against a human rating distribution with a strictly proper, base-rate-referenced skill score."""
from .bootstrap import boot_mean, boot_p, boot_ratio_skill, boot_weights, boot_weights_grouped, pct_ci, summarize
from .calibration import apply_calibrator, calibration_split, crossfit_calibrate, fit_calibrator
from .core import (FLOOR, RULES, base_rate_forecast, crossfit_base_rate, floor_probs, human_dist, item_scores, make_folds,
                   make_group_folds, matched_rule, ratings_to_counts, risk, scores_vs_distribution, skill)
from .report import agreement_vs_skill_report, mode_weights

__version__ = "0.1.0"
__all__ = [n for n in dir() if not n.startswith("_")]
