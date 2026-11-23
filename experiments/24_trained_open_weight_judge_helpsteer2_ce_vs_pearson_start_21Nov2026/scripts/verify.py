#!/usr/bin/env python3
"""Independent verification of Experiment 24 splits, LoRA provenance, algebraic RPS/CE, and reports."""
from __future__ import annotations

import argparse
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
EXPECTED_POOL_LENS = {"train_pool": 1600, "val_pool": 800, "cal_pool": 800, "test_set": 1038}
METRIC_SCORE_COL = {"rps": "rps", "ce_nats": "ce_nats", "tvd": "tvd", "brier": "brier"}


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


def verify_protocol_hash() -> None:
    expected_proto = (RESULTS_DIR / "protocol.sha256").read_text(encoding="utf-8").split()[0]
    actual_proto = hashlib.sha256((EXP_DIR / "PROTOCOL_11-21-2026.md").read_bytes()).hexdigest()
    assert expected_proto == actual_proto, f"PROTOCOL hash changed: {expected_proto} vs {actual_proto}"


def verify_bundle_splits(bundle: dict) -> None:
    items = bundle["items"]
    splits = bundle["splits"]
    prompt_sets = []
    for key in ("train_ids", "val_ids", "cal_ids", "test_ids"):
        ids = splits[key]
        prompt_sets.append({items[i]["prompt"] for i in ids})
        for i in ids:
            votes = items[i]["votes"]
            assert len(votes) >= 2 and all(0 <= v <= 4 for v in votes)
    for a, p_a in enumerate(prompt_sets):
        for b_off, p_b in enumerate(prompt_sets[a + 1:], start=a + 1):
            assert not (p_a & p_b), f"Prompt overlap between split {a} and {b_off}"


def verify_manifest_and_contrasts(score_lookup: dict) -> None:
    manifest = json.loads((RESULTS_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_test_items"] == 1038 and summary["n_test_prompts"] == 519 and summary["n_test_votes"] == 3610
    for pool_key, exp_n in EXPECTED_POOL_LENS.items():
        assert manifest["pool_statistics"][pool_key]["n_responses"] == exp_n
    with (RESULTS_DIR / "contrasts.csv").open(encoding="utf-8") as f:
        contrasts = list(csv.DictReader(f))
    for row in contrasts:
        col = METRIC_SCORE_COL.get(row["metric"])
        if col is None:
            continue
        t_val = float(score_lookup[(int(row["depth"]), int(row["b_total"]), row["target"])][col])
        c_val = float(score_lookup[(int(row["depth"]), int(row["b_total"]), row["comparator"])][col])
        np.testing.assert_allclose(float(row["diff_comparator_minus_target"]), c_val - t_val, atol=1e-9, rtol=0)


def verify_sanity() -> None:
    sanity = json.loads((RESULTS_DIR / "sanity_checks.json").read_text(encoding="utf-8"))
    assert sanity["gradient_verification"]["pearson_two_pass_rel_error"] < 1e-9
    assert sanity["epoch0_full_vs_h32_max_prob_diff"] < 1e-4
    for obj in ("ce", "pearson"):
        assert sanity["two_step_full_vs_h32_equivalence"][obj]["max_prob_diff_after_2_steps"] < 1e-4
    assert sanity["karpathy_controls"]["label_in_prompt_hits"] >= 4
    assert sanity["karpathy_controls"]["blank_response_mean_grade"] < 1.5


def verify_provenance() -> int:
    prov = json.loads((RESULTS_DIR / "training_provenance.json").read_text(encoding="utf-8"))
    n_runs = prov["n_selected_runs"]
    assert n_runs == 220
    assert prov["nonzero_delta_lora_count"] == n_runs, "Found LoRA run with zero weight delta!"
    assert prov["nonzero_delta_head_count"] == n_runs, "Found head run with zero weight delta!"
    draw0_reloads = [r["reload_val_rps_diff"] for r in prov["runs"] if r["draw_id"] == 0]
    assert len(draw0_reloads) == 22 and all(d is not None and d < 1e-7 for d in draw0_reloads)
    return n_runs


def verify_npz_simplex(npz) -> None:
    for k in npz.files:
        arr = npz[k]
        if arr.ndim != 2 or arr.shape[1] != 5 or "logits" in k:
            continue
        assert np.isfinite(arr).all(), f"Non-finite in {k}"
        assert (arr >= 0.0).all(), f"Negative prob in {k}"
        assert np.max(np.abs(arr.sum(axis=1) - 1.0)) < 1e-4, f"Prob sum mismatch in {k}"
        if k.endswith("_test"):
            assert arr.shape == (1038, 5), f"Unexpected test shape in {k}: {arr.shape}"


def verify_literal_metrics(bundle: dict, npz, score_lookup: dict) -> None:
    items = bundle["items"]
    test_ids = bundle["splits"]["test_ids"]
    test_votes = [items[i]["votes"] for i in test_ids]
    n_draws = int(bundle["n_draws"])
    for depth, b_tot in [(1, 16), (1, 256), (1, 2048), (2, 1024)]:
        for obj in ("ce", "pearson"):
            rps_draws, ce_draws, tvd_draws = [], [], []
            for draw_id in range(n_draws):
                q_te = npz[f"draw{draw_id}_d{depth}_B{b_tot}_{obj}_test"].astype(np.float64)
                rps_draws.append(np.mean([literal_rps_vote_by_vote(q_te[i], test_votes[i]) for i in range(len(test_ids))]))
                ce_draws.append(np.mean([literal_ce_vote_by_vote(q_te[i], test_votes[i]) for i in range(len(test_ids))]))
                tvd_draws.append(np.mean([literal_tvd(q_te[i], test_votes[i]) for i in range(len(test_ids))]))
            row = score_lookup[(depth, b_tot, f"{obj}_raw")]
            np.testing.assert_allclose(float(row["rps"]), np.mean(rps_draws), atol=1e-11, rtol=0)
            np.testing.assert_allclose(float(row["ce_nats"]), np.mean(ce_draws), atol=1e-11, rtol=0)
            np.testing.assert_allclose(float(row["tvd"]), np.mean(tvd_draws), atol=1e-11, rtol=0)


def verify_calibration_and_reports() -> None:
    with (RESULTS_DIR / "calibration_logs.csv").open(encoding="utf-8") as f:
        cal_logs = list(csv.DictReader(f))
    assert len(cal_logs) == 220
    assert all(float(r["grad_inf_norm"]) < 1e-6 for r in cal_logs)

    banned = re.compile(r"\b(arms?|cells?|contracts?|untouched|holdouts?|compact)\b|CE minus Pearson", re.IGNORECASE)
    for md_file in sorted(EXP_DIR.glob("*.md")):
        txt = md_file.read_text(encoding="utf-8")
        m = banned.search(txt)
        assert m is None, f"Banned terminology {m.group(0)!r} found in {md_file.name}"

    plot_src = (EXP_DIR / "scripts" / "plot_and_report.py").read_text(encoding="utf-8")
    assert ".grid(True" in plot_src, "Missing ax.grid(True) in plot_and_report.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify Experiment 24 provenance, metrics, and reports.")
    ap.add_argument("--bundle", type=Path, default=BUNDLE_PATH, help="Optional raw bundle path in /tmp for vote-by-vote checks.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    verify_protocol_hash()
    verify_sanity()
    n_runs = verify_provenance()

    with (RESULTS_DIR / "scores.csv").open(encoding="utf-8") as f:
        scores = list(csv.DictReader(f))
    score_lookup = {(int(r["depth"]), int(r["b_total"]), r["method"]): r for r in scores}
    verify_manifest_and_contrasts(score_lookup)

    npz = np.load(RESULTS_DIR / "predictions.npz")
    verify_npz_simplex(npz)

    bundle = json.loads(args.bundle.read_text(encoding="utf-8")) if args.bundle.exists() else None
    if bundle is not None:
        verify_bundle_splits(bundle)
        verify_literal_metrics(bundle, npz, score_lookup)

    verify_calibration_and_reports()
    bundle_note = (
        "literal vote-by-vote RPS/CE/TVD matched scores.csv (< 1e-11), zero prompt overlap"
        if bundle is not None
        else f"repo-local scores/contrasts arithmetic matched (< 1e-9); skipped raw-bundle checks ({args.bundle} absent)"
    )
    print(
        f"VERIFICATION PASSED: protocol SHA-256 intact, {n_runs}/{n_runs} non-zero LoRA/head updates, "
        f"22/22 disk reloads < 1e-7, {bundle_note}, zero banned terminology."
    )


if __name__ == "__main__":
    main()
