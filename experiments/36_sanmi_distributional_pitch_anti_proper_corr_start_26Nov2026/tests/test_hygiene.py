"""Unit tests for the unbiased risk, the prompt split, and the metric identities."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_io import (
    assert_no_prompt_leakage,
    load_certjudge_real_panels,
    load_popquorn,
    load_veribench_fsc_native_main,
    load_veribench_fsc_pooled,
    load_veribench_fsc_rv1,
    split_by_group,
)
from metrics import (
    brier_one,
    ce_one,
    expected_plugin_bias,
    item_mean_losses,
    plugin_l2,
    rps_one,
    unbiased_risk,
)
from proofs_numeric import check_all


def test_unbiased_risk_matches_the_prompt_average():
    # Two prompts. Losses 1 and 3 on the first (mean 2), loss 4 on the second.
    # The estimator weights prompts equally: (2 + 4) / 2 = 3, not (1+3+4)/3.
    q = np.array([0.2, 0.8])
    # Build losses explicitly rather than through a forecast, by checking the formula.
    item_means = np.array([np.mean([1.0, 3.0]), np.mean([4.0])])
    assert unbiased_risk(item_means) == 3.0
    # And the per-prompt helper averages ratings, not prompts.
    # y=1 has CE -log(0.8); y=0 has CE -log(0.2).
    got = item_mean_losses(q, np.array([1, 0]))
    assert abs(got["ce"] - 0.5 * (ce_one(q, 1) + ce_one(q, 0))) < 1e-12
    assert abs(got["brier"] - 0.5 * (brier_one(q, 1) + brier_one(q, 0))) < 1e-12
    assert abs(got["rps"] - 0.5 * (rps_one(q, 1) + rps_one(q, 0))) < 1e-12


def test_plugin_bias_formula():
    p = np.array([0.2, 0.5, 0.3])
    n = 10
    theory = (1 - np.dot(p, p)) / n
    assert abs(expected_plugin_bias(p, n) - theory) < 1e-12
    q = np.array([0.5, 0.25, 0.25])
    # Monte Carlo: E[||freq - q||^2] = ||p-q||^2 + theory
    rng = np.random.default_rng(0)
    plugins = []
    for _ in range(3000):
        freq = np.bincount(rng.choice(3, size=n, p=p), minlength=3) / n
        plugins.append(plugin_l2(freq, q))
    assert abs(np.mean(plugins) - (plugin_l2(p, q) + theory)) < 0.01


def test_binary_brier_is_twice_scalar_l2():
    for q in (0.2, 0.5, 0.9):
        for y in (0, 1):
            dist = np.array([1 - q, q])
            assert abs(brier_one(dist, y) - 2 * (q - y) ** 2) < 1e-12


def test_prompt_split_has_no_rater_leakage():
    groups = ["a", "a", "b", "c", "c", "d", "e", "f", "g", "h"]
    rater_ids = [[f"{g}-{j}" for j in range(3)] for g in groups]
    train, val, test = split_by_group(groups, seed=0)
    assert_no_prompt_leakage(groups, rater_ids, train, val, test)
    # No group id on two sides.
    def sides(idx):
        return {groups[i] for i in idx}
    assert sides(train).isdisjoint(sides(val))
    assert sides(train).isdisjoint(sides(test))
    assert sides(val).isdisjoint(sides(test))


def test_numeric_proof_suite():
    check_all(seed=0)


def test_popquorn_loaders_and_splits():
    for task, exp_items, exp_ratings in (("offensiveness", 1500, 13036), ("politeness", 3718, 25042)):
        ds = load_popquorn(task)
        s = ds.summary()
        assert s["n_items"] == exp_items
        assert s["n_ratings"] == exp_ratings
        assert s["n_classes"] == 5
        assert s["class_values"] == [1.0, 2.0, 3.0, 4.0, 5.0]
        for r in ds.ratings:
            assert np.all((r >= 0) & (r < 5))
        tr, va, te = split_by_group(ds.groups, seed=0)
        assert_no_prompt_leakage(ds.groups, ds.rater_ids, tr, va, te)


def test_veribench_loaders_and_task_disjoint_splits():
    rv1 = load_veribench_fsc_rv1("all5")
    assert rv1.n_items == 75
    assert rv1.summary()["n_ratings"] == 345
    assert len(set(rv1.groups)) == 25
    tr, va, te = split_by_group(rv1.groups, seed=0)
    assert_no_prompt_leakage(rv1.groups, rv1.rater_ids, tr, va, te)

    rv1_exp3 = load_veribench_fsc_rv1("expert3")
    assert rv1_exp3.n_items == 75
    assert rv1_exp3.summary()["n_ratings"] == 195

    nat = load_veribench_fsc_native_main()
    assert nat.n_items == 100
    assert nat.summary()["n_ratings"] == 250
    assert len(set(nat.groups)) == 25
    for rats in nat.rater_ids:
        assert "A1" not in rats
    tr_n, va_n, te_n = split_by_group(nat.groups, seed=0)
    assert_no_prompt_leakage(nat.groups, nat.rater_ids, tr_n, va_n, te_n)

    pooled = load_veribench_fsc_pooled()
    assert pooled.n_items == 175
    assert pooled.summary()["n_ratings"] == 595
    assert len(set(pooled.groups)) == 50
    # Raw task names (without the rv1::/native:: prefix) must not be shared across the two pools.
    assert not set(rv1.groups) & set(nat.groups)
    tr_p, va_p, te_p = split_by_group(pooled.groups, seed=0)
    assert_no_prompt_leakage(pooled.groups, pooled.rater_ids, tr_p, va_p, te_p)


def test_certjudge_real_panels_integrity():
    cj = load_certjudge_real_panels()
    arms = cj["e63_study_a"]["all5"]["arms"]
    assert arms["human_mix_sft"]["ce"] < arms["rl_grpo"]["ce"]
    assert arms["rl_grpo"]["ce"] > arms["shuffled_control"]["ce"]
    e39 = cj["e39_study_b"]
    assert e39["all_llm"]["n_configs"] == 56
    assert e39["all_llm"]["ce_cal_beats_pe_cal_ce"] == 48
    assert e39["test_llm"]["n_configs"] == 18
    assert e39["test_llm"]["ce_cal_beats_pe_cal_ce"] == 16


def test_reports_have_no_stale_unavailability():
    exp_dir = Path(__file__).resolve().parents[1]
    for name in ("README.md", "RESULTS_SUMMARY.md", "SANMI_PITCH.md", "PLAN_STATUS.md"):
        txt = (exp_dir / name).read_text()
        assert "NOT LOADED" not in txt
        assert "NOT IN THIS REPOSITORY" not in txt
        assert "POPQUORN | not scored" not in txt
        assert "CertJudge / VeriBench FSC, 0–5 | not scored" not in txt


def test_experiments_index_row36_matches_results_json():
    """Every held-out CE quoted in experiments/README.md Row 36 is the 3-decimal value in results.json."""
    import json

    exp_dir = Path(__file__).resolve().parents[1]
    mats = json.loads((exp_dir / "results.json").read_text())["matrices"]
    row = next(
        line for line in (exp_dir.parent / "README.md").read_text().splitlines() if line.startswith("| 36 |")
    )
    pairs = {
        "popquorn_offensiveness": "Offensiveness CE {ce} vs Pearson {pe}",
        "popquorn_politeness": "Politeness CE {ce} vs {pe}",
        "veribench_fsc_rv1": "CE {ce} vs {pe}",
        "veribench_fsc_native_main": "CE {ce} vs {pe}",
        "veribench_fsc_pooled": "CE {ce} vs {pe}",
    }
    for name, fmt in pairs.items():
        rows = mats[name]["rows"]
        text = fmt.format(ce=f"{float(rows['ce']['point']['ce']):.3f}", pe=f"{float(rows['pearson']['point']['ce']):.3f}")
        assert text in row, (name, text)


def test_resolve_sibling_text_rejects_dirty_working_tree_and_bad_sha(tmp_path, monkeypatch):
    """Pinned git bytes win over dirty working-tree edits, and mismatched SHAs are rejected."""
    import hashlib
    import subprocess
    import pytest
    import data_io

    repo = tmp_path / "sibling_repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "qa@example.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "QA"], check=True, capture_output=True)
    rel = "sub/data.txt"
    (repo / "sub").mkdir()
    committed = "pinned-clean-content\n"
    (repo / rel).write_text(committed, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", rel], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True)
    rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    good_sha = hashlib.sha256(committed.encode("utf-8")).hexdigest()

    # Dirty the working tree; _resolve_sibling_text must still return the committed bytes.
    (repo / rel).write_text("DIRTY-LOCAL-EDIT\n", encoding="utf-8")
    monkeypatch.setenv("TEST_SIB_ROOT", str(repo))
    got = data_io._resolve_sibling_text("TEST_SIB_ROOT", "nonexistent", "invalid/invalid", rev, rel, expected_sha=good_sha)
    assert got == committed

    # A wrong expected_sha must raise rather than silently accept bytes.
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        data_io._resolve_sibling_text("TEST_SIB_ROOT", "nonexistent", "invalid/invalid", rev, rel, expected_sha="0" * 64)

    # On a non-git checkout with wrong local bytes and an expected_sha, local file is rejected.
    nongit = tmp_path / "nongit_repo"
    (nongit / "sub").mkdir(parents=True)
    (nongit / rel).write_text("TAMPERED\n", encoding="utf-8")
    monkeypatch.setenv("TEST_SIB_ROOT", str(nongit))
    monkeypatch.setattr(data_io, "CACHE", tmp_path / "cache")
    with pytest.raises(Exception):
        data_io._resolve_sibling_text("TEST_SIB_ROOT", "nonexistent", "invalid/invalid", rev, rel, expected_sha=good_sha)


def test_rollout_figure_uses_finite_mc_means():
    """Every rollout row plotted in single_pass_vs_rollout.png has a finite CE value."""
    import json
    import math

    exp_dir = Path(__file__).resolve().parents[1]
    rows = json.loads((exp_dir / "results.json").read_text())["rollout"]["rows"]
    assert len(rows) == 4
    for r in rows:
        val = float(r.get("ce_mean_over_reps", r["ce"]))
        assert math.isfinite(val) and val > 0, r
