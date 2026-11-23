#!/usr/bin/env python3
"""Independent verification of Experiment 25 splits, LoRA provenance, algebraic RPS/CE/TVD, and reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = Path("/tmp/exp25_helpsteer2_raw/exp25_bundle.json")
RESULTS_DIR = EXP_DIR / "results"

EXPECTED_STUDIES = {
    "qwen3_4b": (20, 5, 200, (4096, 2048, 2048, 1038)),
    "llama31_8b": (10, 8, 160, (1600, 800, 800, 1038)),
}
METRIC_SCORE_COL = {"rps": "rps", "ce_nats": "ce_nats", "tvd": "tvd", "brier": "brier"}


def literal_rps_vote_by_vote(q_item: np.ndarray, votes: list[int]) -> float:
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


def _extract_split_prompts(model_key: str, split_name: str, ids: list[int], items: list[dict]) -> set[str]:
    prompts = set()
    for i in ids:
        votes = items[i]["votes"]
        assert len(votes) >= 2
        assert min(votes) >= 0 and max(votes) <= 4
        prompts.add(items[i]["prompt"])
    return prompts


def _assert_pairwise_disjoint(model_key: str, psets: list[set[str]]) -> None:
    for a, p_a in enumerate(psets):
        for b_off, p_b in enumerate(psets[a + 1:], start=a + 1):
            assert not (p_a & p_b), f"{model_key} prompt overlap between split {a} and {b_off}"


def verify_study_splits(
    model_key: str,
    study: dict,
    items: list[dict],
    exp_draws: int,
    exp_budgets: int,
    exp_split_lens: tuple[int, int, int, int],
) -> None:
    assert int(study["n_draws"]) == exp_draws
    assert len(study["budgets"]) == exp_budgets
    splits = study["splits"]
    psets = []
    for idx_s, key in enumerate(("train_ids", "val_ids", "cal_ids", "test_ids")):
        ids = splits[key]
        assert len(ids) == exp_split_lens[idx_s], f"{model_key} {key}: {len(ids)} != {exp_split_lens[idx_s]}"
        psets.append(_extract_split_prompts(model_key, key, ids, items))
    _assert_pairwise_disjoint(model_key, psets)


def verify_manifest_and_summary(score_lookup: dict) -> None:
    manifest = json.loads((RESULTS_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_test_items"] == 1038 and summary["n_test_prompts"] == 519 and summary["n_test_votes"] == 3610
    for model_key, (exp_draws, exp_budgets, _, exp_lens) in EXPECTED_STUDIES.items():
        m_study = manifest["studies"][model_key]
        assert m_study["n_draws"] == exp_draws and len(m_study["budgets"]) == exp_budgets
        actual_lens = tuple(m_study[k]["n_responses"] for k in ("train_pool", "val_pool", "cal_pool", "test_set"))
        assert actual_lens == exp_lens, f"{model_key} manifest split mismatch: {actual_lens} != {exp_lens}"
    with (RESULTS_DIR / "contrasts.csv").open(encoding="utf-8") as f:
        contrasts = list(csv.DictReader(f))
    for row in contrasts:
        col = METRIC_SCORE_COL.get(row["metric"])
        if col is None:
            continue
        t_val = float(score_lookup[(row["study"], int(row["b_total"]), row["target"])][col])
        c_val = float(score_lookup[(row["study"], int(row["b_total"]), row["comparator"])][col])
        np.testing.assert_allclose(float(row["diff_comparator_minus_target"]), c_val - t_val, atol=1e-9, rtol=0)


def verify_study_sanity(model_key: str) -> None:
    san = json.loads((RESULTS_DIR / f"{model_key}_sanity.json").read_text(encoding="utf-8"))
    assert san["gradient_verification"]["pearson_two_pass_rel_error"] < 1e-9
    assert san["epoch0_unbatched_vs_left_padded_max_prob_diff"] < 2e-2
    for obj in ("ce", "pearson"):
        assert san["two_step_full_vs_batched_equivalence"][obj]["max_prob_diff_after_2_steps"] < 1e-5
    assert san["karpathy_controls"]["label_in_prompt_hits"] >= 4
    assert san["karpathy_controls"]["blank_response_mean_grade"] < 1.2


def verify_study_provenance(model_key: str, exp_runs: int, exp_budgets: int) -> None:
    prov = json.loads((RESULTS_DIR / f"{model_key}_provenance.json").read_text(encoding="utf-8"))
    assert prov["n_selected_runs"] == exp_runs
    assert prov["nonzero_delta_lora_count"] == exp_runs
    assert prov["nonzero_delta_head_count"] == exp_runs
    draw0_reloads = [r["reload_val_rps_diff"] for r in prov["runs"] if r["draw_id"] == 0]
    assert len(draw0_reloads) == exp_budgets * 2 and all(d is not None and d < 1e-7 for d in draw0_reloads)


def verify_npz_simplex(model_key: str, npz) -> None:
    for k in npz.files:
        arr = npz[k]
        if arr.ndim != 2 or arr.shape[1] != 5 or "logits" in k:
            continue
        assert np.isfinite(arr).all(), f"Non-finite in {model_key}:{k}"
        assert (arr >= 0.0).all(), f"Negative prob in {model_key}:{k}"
        if "base_probs" not in k or np.any(arr.sum(axis=1) > 0):
            # base_probs only populated on study items; non-study rows are softmax([0]*5)=0.2
            assert np.max(np.abs(arr.sum(axis=1) - 1.0)) < 1e-4, f"Prob sum mismatch in {model_key}:{k}"
        if k.endswith("_test"):
            assert arr.shape == (1038, 5), f"Unexpected test shape in {model_key}:{k}: {arr.shape}"


def _mean_draw_metrics(
    npz,
    exp_draws: int,
    b_tot: int,
    obj: str,
    test_ids: list[int],
    test_votes: list[list[int]],
) -> tuple[float, float, float]:
    rps_draws, ce_draws, tvd_draws = [], [], []
    n_test = len(test_ids)
    for draw_id in range(exp_draws):
        q_te = npz[f"draw{draw_id}_d1_B{b_tot}_{obj}_test"].astype(np.float64)
        rps_draws.append(np.mean([literal_rps_vote_by_vote(q_te[i], test_votes[i]) for i in range(n_test)]))
        ce_draws.append(np.mean([literal_ce_vote_by_vote(q_te[i], test_votes[i]) for i in range(n_test)]))
        tvd_draws.append(np.mean([literal_tvd(q_te[i], test_votes[i]) for i in range(n_test)]))
    return float(np.mean(rps_draws)), float(np.mean(ce_draws)), float(np.mean(tvd_draws))


def verify_literal_metrics(
    model_key: str,
    study: dict,
    items: list[dict],
    exp_draws: int,
    npz,
    score_lookup: dict,
) -> None:
    test_ids = study["splits"]["test_ids"]
    test_votes = [items[i]["votes"] for i in test_ids]
    check_budgets = [study["budgets"][0], study["budgets"][-1]]
    for b_tot in check_budgets:
        for obj in ("ce", "pearson"):
            mean_rps, mean_ce, mean_tvd = _mean_draw_metrics(npz, exp_draws, int(b_tot), obj, test_ids, test_votes)
            row = score_lookup[(model_key, int(b_tot), f"{obj}_raw")]
            np.testing.assert_allclose(float(row["rps"]), mean_rps, atol=1e-11, rtol=0)
            np.testing.assert_allclose(float(row["ce_nats"]), mean_ce, atol=1e-11, rtol=0)
            np.testing.assert_allclose(float(row["tvd"]), mean_tvd, atol=1e-11, rtol=0)


def verify_calibration_and_reports() -> None:
    with (RESULTS_DIR / "calibration_logs.csv").open(encoding="utf-8") as f:
        cal_logs = list(csv.DictReader(f))
    assert len(cal_logs) == 200 + 160
    assert all(float(r["grad_inf_norm"]) < 1e-6 for r in cal_logs)

    banned = re.compile(r"\b(arms?|cells?|contracts?|untouched|holdouts?|compact)\b|CE minus Pearson", re.IGNORECASE)
    for md_file in sorted(EXP_DIR.glob("*.md")):
        txt = md_file.read_text(encoding="utf-8")
        m = banned.search(txt)
        assert m is None, f"Banned terminology {m.group(0)!r} in {md_file.name}"

    plot_src = (EXP_DIR / "scripts" / "plot_and_report_exp25.py").read_text(encoding="utf-8")
    assert plot_src.count(".grid(True") >= 3, "Missing ax.grid(True) across figure subplots"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify Experiment 25 provenance, metrics, and reports.")
    ap.add_argument("--bundle", type=Path, default=BUNDLE_PATH, help="Optional raw bundle path in /tmp for vote-by-vote checks.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    verify_protocol_hash()
    with (RESULTS_DIR / "scores.csv").open(encoding="utf-8") as f:
        scores = list(csv.DictReader(f))
    score_lookup = {(r["study"], int(r["b_total"]), r["method"]): r for r in scores}
    verify_manifest_and_summary(score_lookup)

    bundle = json.loads(args.bundle.read_text(encoding="utf-8")) if args.bundle.exists() else None
    items = bundle["items"] if bundle is not None else None

    for model_key, (exp_draws, exp_budgets, exp_runs, exp_split_lens) in EXPECTED_STUDIES.items():
        verify_study_sanity(model_key)
        verify_study_provenance(model_key, exp_runs, exp_budgets)
        npz_path = RESULTS_DIR / f"{model_key}_predictions.npz"
        assert npz_path.stat().st_size < 50_000_000, f"{npz_path.name} exceeds 50 MB GitHub limit"
        npz = np.load(npz_path)
        verify_npz_simplex(model_key, npz)
        if bundle is not None and items is not None:
            study = bundle["studies"][model_key]
            verify_study_splits(model_key, study, items, exp_draws, exp_budgets, exp_split_lens)
            verify_literal_metrics(model_key, study, items, exp_draws, npz, score_lookup)

    verify_calibration_and_reports()
    bundle_note = (
        "literal vote-by-vote RPS/CE/TVD matched scores.csv (< 1e-11), zero prompt overlap"
        if bundle is not None
        else f"repo-local scores/contrasts arithmetic matched (< 1e-9); skipped raw-bundle checks ({args.bundle} absent)"
    )
    print(
        "VERIFICATION PASSED: protocol SHA-256 intact, Qwen3-4B (200 runs) + Llama-3.1-8B (160 runs) non-zero LoRA/head updates, "
        f"all draw-0 disk reloads < 1e-7, {bundle_note}, "
        "360/360 L-BFGS-B calibrator gradient norms < 1e-6, zero banned terminology."
    )


if __name__ == "__main__":
    main()
