#!/usr/bin/env python3
"""Independent verification of Experiment 24 splits, LoRA provenance, algebraic RPS/CE, and reports."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = Path("/tmp/exp24_helpsteer2_raw/prepared_bundle.json")
RESULTS_DIR = EXP_DIR / "results"
GRID = np.arange(5, dtype=np.float64)


def literal_rps_vote_by_vote(q_item: np.ndarray, votes: list[int]) -> float:
    """Explicit vote-by-vote RPS formula without any vectorization shortcut."""
    f = np.cumsum(q_item)[:4]
    losses = []
    for y in votes:
        g = np.array([1.0 if y <= k else 0.0 for k in range(4)], dtype=np.float64)
        losses.append(float(np.mean((f - g) ** 2)))
    return float(np.mean(losses))


def literal_ce_vote_by_vote(q_item: np.ndarray, votes: list[int], eps: float = 6e-6) -> float:
    q_eps = (1.0 - eps) * q_item + eps / 5.0
    return float(np.mean([-np.log(q_eps[y]) for y in votes]))


def literal_tvd(q_item: np.ndarray, votes: list[int]) -> float:
    h = np.bincount(votes, minlength=5).astype(np.float64) / len(votes)
    return float(0.5 * np.sum(np.abs(q_item - h)))


def main() -> None:
    # 1. Frozen protocol hash check
    expected_proto = (RESULTS_DIR / "protocol.sha256").read_text(encoding="utf-8").split()[0]
    actual_proto = hashlib.sha256((EXP_DIR / "PROTOCOL_11-21-2026.md").read_bytes()).hexdigest()
    assert expected_proto == actual_proto, f"PROTOCOL hash changed: {expected_proto} vs {actual_proto}"

    # 2. Bundle split disjointness & label accounting
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    items = bundle["items"]
    splits = bundle["splits"]
    prompt_sets = []
    for key in ("train_ids", "val_ids", "cal_ids", "test_ids"):
        ids = splits[key]
        pset = {items[i]["prompt"] for i in ids}
        prompt_sets.append(pset)
        for i in ids:
            assert len(items[i]["votes"]) >= 2
            assert all(0 <= v <= 4 for v in items[i]["votes"])
    for a in range(len(prompt_sets)):
        for b in range(a + 1, len(prompt_sets)):
            assert len(prompt_sets[a] & prompt_sets[b]) == 0, f"Prompt overlap between split {a} and {b}"

    # 3. Sanity checks verification
    sanity = json.loads((RESULTS_DIR / "sanity_checks.json").read_text(encoding="utf-8"))
    assert sanity["gradient_verification"]["pearson_two_pass_rel_error"] < 1e-9
    assert sanity["epoch0_full_vs_h32_max_prob_diff"] < 1e-4
    for obj in ("ce", "pearson"):
        assert sanity["two_step_full_vs_h32_equivalence"][obj]["max_prob_diff_after_2_steps"] < 1e-4
    assert sanity["karpathy_controls"]["label_in_prompt_hits"] >= 4
    assert sanity["karpathy_controls"]["blank_response_mean_grade"] < 1.5

    # 4. LoRA provenance verification
    prov = json.loads((RESULTS_DIR / "training_provenance.json").read_text(encoding="utf-8"))
    n_runs = prov["n_selected_runs"]
    assert n_runs == 10 * (len(bundle["budgets_d1"]) + len(bundle["budgets_d2"])) * 2 == 220
    assert prov["nonzero_delta_lora_count"] == n_runs, "Found LoRA run with zero weight delta!"
    assert prov["nonzero_delta_head_count"] == n_runs, "Found head run with zero weight delta!"
    draw0_reloads = [r["reload_val_rps_diff"] for r in prov["runs"] if r["draw_id"] == 0]
    assert len(draw0_reloads) == 22 and all(d is not None and d < 1e-7 for d in draw0_reloads)

    # 5. Probability simplex + literal vote-by-vote metric check on predictions.npz vs scores.csv
    npz = np.load(RESULTS_DIR / "predictions.npz")
    for k in npz.files:
        arr = npz[k]
        if arr.ndim == 2 and arr.shape[1] == 5 and "logits" not in k:
            assert np.isfinite(arr).all(), f"Non-finite in {k}"
            assert (arr >= 0.0).all(), f"Negative prob in {k}"
            assert np.max(np.abs(arr.sum(axis=1) - 1.0)) < 1e-4, f"Prob sum mismatch in {k}"

    with (RESULTS_DIR / "scores.csv").open(encoding="utf-8") as f:
        scores = list(csv.DictReader(f))
    score_lookup = {(int(r["depth"]), int(r["b_total"]), r["method"]): r for r in scores}

    test_ids = splits["test_ids"]
    test_votes = [items[i]["votes"] for i in test_ids]
    for depth, b_tot in [(1, 16), (1, 256), (1, 2048), (2, 1024)]:
        for obj in ("ce", "pearson"):
            rps_draws, ce_draws, tvd_draws = [], [], []
            for draw_id in range(bundle["n_draws"]):
                q_te = npz[f"draw{draw_id}_d{depth}_B{b_tot}_{obj}_test"].astype(np.float64)
                rps_draws.append(np.mean([literal_rps_vote_by_vote(q_te[i], test_votes[i]) for i in range(len(test_ids))]))
                ce_draws.append(np.mean([literal_ce_vote_by_vote(q_te[i], test_votes[i]) for i in range(len(test_ids))]))
                tvd_draws.append(np.mean([literal_tvd(q_te[i], test_votes[i]) for i in range(len(test_ids))]))
            row = score_lookup[(depth, b_tot, f"{obj}_raw")]
            np.testing.assert_allclose(float(row["rps"]), np.mean(rps_draws), atol=1e-11, rtol=0)
            np.testing.assert_allclose(float(row["ce_nats"]), np.mean(ce_draws), atol=1e-11, rtol=0)
            np.testing.assert_allclose(float(row["tvd"]), np.mean(tvd_draws), atol=1e-11, rtol=0)

    # 6. Check calibration logs convergence
    with (RESULTS_DIR / "calibration_logs.csv").open(encoding="utf-8") as f:
        cal_logs = list(csv.DictReader(f))
    assert len(cal_logs) == 220
    assert all(float(r["grad_inf_norm"]) < 1e-6 for r in cal_logs)

    # 7. Check Markdown terminology & report artifacts exist
    banned = re.compile(r"\b(arms?|cells?|contracts?|untouched|holdouts?|compact)\b|CE minus Pearson")
    for md_file in sorted(EXP_DIR.glob("*.md")):
        txt = md_file.read_text(encoding="utf-8")
        m = banned.search(txt)
        assert m is None, f"Banned terminology {m.group(0)!r} found in {md_file.name}"

    plot_src = (EXP_DIR / "scripts" / "plot_and_report.py").read_text(encoding="utf-8")
    assert "ax.grid(True" in plot_src, "Missing ax.grid(True) in plot_and_report.py"

    print(
        f"VERIFICATION PASSED: protocol SHA-256 intact, {n_runs}/{n_runs} non-zero LoRA/head updates, "
        f"22/22 disk reloads < 1e-7, literal vote-by-vote RPS/CE/TVD matched scores.csv (< 1e-11), "
        f"zero prompt overlap, zero banned terminology."
    )


if __name__ == "__main__":
    main()
