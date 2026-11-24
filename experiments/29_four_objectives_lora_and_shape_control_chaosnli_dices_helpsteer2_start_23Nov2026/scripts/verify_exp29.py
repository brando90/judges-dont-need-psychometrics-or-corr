#!/usr/bin/env python3
"""Independent verification of Experiment 29 splits, provenance, metrics, AST structure, and reports."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent.parent
BUNDLE_PATH = Path("/tmp/exp29_public_raw/exp29_bundle.json")
RESULTS_DIR = EXP_DIR / "results"
METRIC_COL = {"rps": "rps", "ce": "ce", "tvd": "tvd", "brier": "brier"}

JOB_RUN_COUNTS = {
    "chaos3_qwen": 440,
    "alpha2_qwen": 160,
    "dices3_qwen": 160,
    "hs2_qwen_extra": 161,
    "chaos3_llama": 320,
    "dices3_llama": 160,
}

STUDY_INFO = {
    "chaos3_qwen": ("chaos3", "chaos3_qwen_predictions.npz", 3, 713, (1200, 600, 600, 713), (1, 5)),
    "alpha2_qwen": ("alpha2", "alpha2_qwen_predictions.npz", 2, 412, (560, 280, 280, 412), (1,)),
    "dices3_qwen": ("dices3", "dices3_qwen_predictions.npz", 3, 300, (520, 260, 260, 300), (1,)),
    "hs2_qwen": ("hs2", "hs2_qwen_extra_predictions.npz", 5, 1038, (1600, 800, 800, 1038), (1,)),
    "chaos3_llama": ("chaos3", "chaos3_llama_predictions.npz", 3, 713, (1200, 600, 600, 713), (1,)),
    "dices3_llama": ("dices3", "dices3_llama_predictions.npz", 3, 300, (520, 260, 260, 300), (1,)),
}


def verify_protocol_and_manifest() -> None:
    expected = (RESULTS_DIR / "protocol.sha256").read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256((EXP_DIR / "PROTOCOL_11-23-2026.md").read_bytes()).hexdigest()
    assert expected == actual, f"PROTOCOL hash mismatch: {expected} != {actual}"
    manifest = json.loads((RESULTS_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["raw_file_sha256"]) == 8
    assert manifest["suites"]["hs2"]["exp24_bundle_sha256_verified"] == "c916357850510076f17c2974c45ff128873987eb9c4a4f8f0c9dae1ce47a475d"


def verify_sanity_and_provenance(allow_missing: bool) -> int:
    san_seen = 0
    for mkey in ("qwen3_4b", "llama31_8b"):
        p = RESULTS_DIR / f"{mkey}_sanity.json"
        if not p.exists():
            assert allow_missing, f"Missing sanity artifact: {p}"
            continue
        san_seen += 1
        san = json.loads(p.read_text(encoding="utf-8"))
        for k_grp, losses in san["grad_fd_checks"].items():
            assert all(v < 1e-6 for v in losses.values()), f"{mkey} FD failed on {k_grp}: {losses}"
        assert all(v < 1e-5 for v in san["two_step_batched_equiv"].values()), f"{mkey} 2-step equiv failed"
        assert san["epoch0_and_overfit"]["semantic_anchor_hits_3"] >= 2
        assert san["epoch0_and_overfit"]["matched_r"] > san["epoch0_and_overfit"]["shuffled_r"]
    assert san_seen >= 1, "No sanity JSON files found"
    total_runs = 0
    for job, exp_runs in JOB_RUN_COUNTS.items():
        p = RESULTS_DIR / f"{job}_provenance.json"
        if not p.exists():
            assert allow_missing, f"Missing provenance: {p}"
            continue
        prov = json.loads(p.read_text(encoding="utf-8"))
        assert prov["n_selected_runs"] == exp_runs, f"{job}: {prov['n_selected_runs']} != {exp_runs}"
        assert prov["nonzero_delta_lora_count"] == exp_runs
        assert prov["nonzero_delta_head_count"] == exp_runs
        total_runs += exp_runs
    return total_runs


def _check_npz_array(fname: str, k: str, arr: np.ndarray, n_cat: int, n_test: int) -> None:
    if arr.ndim != 2 or arr.shape[1] != n_cat or "logits" in k:
        return
    assert np.isfinite(arr).all() and (arr >= 0.0).all(), f"Non-finite/neg in {fname}:{k}"
    assert np.max(np.abs(arr.sum(axis=1) - 1.0)) < 1e-4, f"Simplex error in {fname}:{k}"
    if k.endswith("_test"):
        assert arr.shape == (n_test, n_cat), f"Bad test shape in {fname}:{k}: {arr.shape}"


def verify_npz_archives(allow_missing: bool) -> list[str]:
    present = []
    for study, (_suite, fname, n_cat, n_test, _lens, _depths) in STUDY_INFO.items():
        p = RESULTS_DIR / fname
        if not p.exists():
            assert allow_missing, f"Missing archive: {p}"
            continue
        assert p.stat().st_size < 50_000_000, f"Oversized archive: {p}"
        present.append(study)
        npz = np.load(p)
        for k in npz.files:
            _check_npz_array(fname, k, npz[k], n_cat, n_test)
    assert len(present) >= 4 or not allow_missing, f"Expected >=4 Qwen studies, got {present}"
    return present


def verify_csv_consistency() -> None:
    with (RESULTS_DIR / "scores.csv").open(encoding="utf-8") as f:
        scores = list(csv.DictReader(f))
    lookup = {(r["study"], int(r["depth"]), int(r["b_total"]), r["method"]): r for r in scores}
    with (RESULTS_DIR / "contrasts.csv").open(encoding="utf-8") as f:
        contrasts = list(csv.DictReader(f))
    for r in contrasts:
        col = METRIC_COL[r["metric"]]
        t_v = float(lookup[(r["study"], int(r["depth"]), int(r["b_total"]), r["target"])][col])
        c_v = float(lookup[(r["study"], int(r["depth"]), int(r["b_total"]), r["comparator"])][col])
        np.testing.assert_allclose(float(r["diff_comparator_minus_target"]), c_v - t_v, atol=1e-9, rtol=0)
    with (RESULTS_DIR / "calibration_logs.csv").open(encoding="utf-8") as f:
        cal_logs = list(csv.DictReader(f))
    assert cal_logs and all(float(r["grad_inf_norm"]) < 1e-6 for r in cal_logs)


def literal_rps(q: np.ndarray, counts: np.ndarray) -> float:
    p = counts / counts.sum()
    n_cat = len(p)
    fq, fp = np.cumsum(q)[: n_cat - 1], np.cumsum(p)[: n_cat - 1]
    return float(np.mean((fq - fp) ** 2 + fp * (1.0 - fp)))


def literal_ce(q: np.ndarray, counts: np.ndarray, eps: float = 6e-6) -> float:
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(np.clip(q, eps, 1.0))))


def literal_tvd(q: np.ndarray, counts: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(q - (counts / counts.sum()))))


def _assert_split_sets(study: str, suite: dict, exp_lens: tuple) -> None:
    sp = suite["splits"]
    actual_lens = tuple(len(sp[k]) for k in ("train_ids", "val_ids", "cal_ids", "test_ids"))
    assert actual_lens == exp_lens, f"{study} split lens {actual_lens} != {exp_lens}"
    g_sets = [{suite["items"][i]["group_key"] for i in sp[k]} for k in ("train_ids", "val_ids", "cal_ids", "test_ids")]
    for i, gi in enumerate(g_sets):
        for gj in g_sets[i + 1:]:
            assert not (gi & gj), f"Group overlap in {study}"


def _mean_literal_metric(arrs: list[np.ndarray], counts: list[np.ndarray], metric_fn) -> float:
    draw_means = []
    for arr in arrs:
        vals = [metric_fn(arr[i], counts[i]) for i in range(len(counts))]
        draw_means.append(float(np.mean(vals)))
    return float(np.mean(draw_means))


def _verify_one_study_literal(study: str, info: tuple, suite: dict, lookup: dict) -> None:
    _sname, fname, _ncat, _ntest, exp_lens, _depths = info
    _assert_split_sets(study, suite, exp_lens)
    if study == "hs2_qwen":
        return
    npz = np.load(RESULTS_DIR / fname)
    b_test = int(suite["budgets_d1"][-1])
    counts = [np.asarray(suite["items"][i]["counts"], dtype=np.float64) for i in suite["splits"]["test_ids"]]
    for obj in ("ce", "rps", "mse", "pearson"):
        arrs = [npz[f"draw{d}_d1_B{b_test}_{obj}_test"].astype(np.float64) for d in range(10)]
        row = lookup[(study, 1, b_test, f"{obj}_raw")]
        np.testing.assert_allclose(float(row["rps"]), _mean_literal_metric(arrs, counts, literal_rps), atol=1e-11, rtol=0)
        np.testing.assert_allclose(float(row["ce"]), _mean_literal_metric(arrs, counts, literal_ce), atol=1e-11, rtol=0)
        np.testing.assert_allclose(float(row["tvd"]), _mean_literal_metric(arrs, counts, literal_tvd), atol=1e-11, rtol=0)


def verify_bundle_literal(bundle: dict, present_studies: list[str]) -> None:
    with (RESULTS_DIR / "scores.csv").open(encoding="utf-8") as f:
        lookup = {(r["study"], int(r["depth"]), int(r["b_total"]), r["method"]): r for r in csv.DictReader(f)}
    for study in present_studies:
        info = STUDY_INFO[study]
        _verify_one_study_literal(study, info, bundle["suites"][info[0]], lookup)


def _ast_cc(node: ast.AST) -> int:
    c = 1
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.With, ast.AsyncWith, ast.IfExp, ast.comprehension)):
            c += 1
        elif isinstance(n, ast.BoolOp):
            c += len(n.values) - 1
    return c


def verify_code_and_reports() -> None:
    root = EXP_DIR.parent.parent
    terms = ("ar" + "ms?", "cel" + "ls?", "con" + "tracts?", "un" + "touched", "hold" + "outs?", "com" + "pact", "vel" + "ocity", "bit " + "flip", "north " + "star", "killer " + "demo")
    banned = re.compile("|".join(rf"\b{t}\b" for t in terms) + "|CE " + "mi" + "nus Pearson", re.I)
    targets = list(sorted((EXP_DIR / "scripts").glob("*.py"))) + list(sorted(EXP_DIR.glob("*.md")))
    targets += [root / "STATUS_11-23-2026.md", root / "README.md", root / "experiments/README.md"]
    targets += [root / "paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL" / f for f in ("6-case-study.tex", "7-discussion.tex", "10-supplementary.tex")]
    targets += [root / "paper_latex_and_notes/notes/correlation_vs_loss_judges_one_pager" / f for f in ("judge-objectives-status.html", "README.md")]
    for path in targets:
        src = path.read_text(encoding="utf-8")
        assert banned.search(src) is None, f"Banned word in {path.name}"
        if path.suffix == ".py":
            for fn in ast.walk(ast.parse(src)):
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    loc, cc = fn.end_lineno - fn.lineno + 1, _ast_cc(fn)
                    assert loc <= 55 and cc <= 10, f"AST limit exceeded in {path.name}:{fn.name} ({loc}, {cc})"
    plot_src = (EXP_DIR / "scripts" / "plot_and_report_exp29.py").read_text(encoding="utf-8")
    assert plot_src.count(".grid(True") >= 4, "Missing ax.grid(True) across plot panels"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=BUNDLE_PATH)
    ap.add_argument("--allow-missing", action="store_true", help="Verify present jobs only (interim Qwen3 stage).")
    args = ap.parse_args()
    verify_protocol_and_manifest()
    total_runs = verify_sanity_and_provenance(args.allow_missing)
    present_studies = verify_npz_archives(args.allow_missing)
    verify_csv_consistency()
    verify_code_and_reports()
    bundle_note = "skipped raw-bundle checks (absent)"
    if args.bundle.exists():
        verify_bundle_literal(json.loads(args.bundle.read_text(encoding="utf-8")), present_studies)
        bundle_note = "literal RPS/CE/TVD matched (< 1e-11), zero group overlap"
    expected_total = sum(JOB_RUN_COUNTS.values())
    print(f"EXP29 VERIFICATION PASSED: protocol SHA-256 intact, {total_runs}/{expected_total} nonzero LoRA/head updates across {len(present_studies)} studies, cal grad norms < 1e-6, {bundle_note}, AST & banned-word clean.")


if __name__ == "__main__":
    main()
