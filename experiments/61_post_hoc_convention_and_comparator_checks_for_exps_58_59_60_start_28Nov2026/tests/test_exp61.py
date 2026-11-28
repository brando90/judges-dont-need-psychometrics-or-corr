"""Tests for Experiment 61 (run: python -m pytest experiments/61_*/tests -q).

Synthetic tests are CODE CHECKS of definitions on toy laws, not evidence about real judges or raters. Tests that read results/ check that the
committed numbers are internally consistent and that the statements quoted in the README and the paper (counts, ranges, signs) hold in them."""
from __future__ import annotations

import csv
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "scripts"))
import run_exp61 as R  # noqa: E402
import frozen as Z  # noqa: E402  (found through run_exp61's path set-up)

RES = EXP / "results"
LAB3 = np.arange(3.0)


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ C1: collapsed-forecast conventions (code checks on toy laws)
def test_two_point_forecast_is_mean_preserving_and_uses_two_adjacent_labels():
    rng = np.random.default_rng(20261128)
    for K in (3, 5):
        pA = rng.dirichlet(np.ones(K), size=40)
        mu = pA @ np.arange(K)
        q = R.collapsed_forecasts(pA, mu, K)["mean_preserving_two_point"]
        assert np.allclose(q.sum(1), 1.0, atol=1e-9)
        assert np.allclose(q @ np.arange(K), mu, atol=1e-5)  # the 1e-6 probability floor moves the mean by at most K * 1e-6
        support = q > 1e-4
        assert (support.sum(1) <= 2).all()
        for row in support:
            idx = np.where(row)[0]
            assert len(idx) == 1 or idx[1] - idx[0] == 1


def test_two_label_two_point_forecast_equals_the_truthful_forecast():
    rng = np.random.default_rng(1)
    pA = rng.dirichlet(np.ones(2), size=30)
    mu = pA[:, 1]
    q = R.collapsed_forecasts(pA, mu, 2)["mean_preserving_two_point"]
    assert np.allclose(q, R.floor_probs(pA), atol=1e-12)


def test_modal_median_and_rounded_mean_conventions_on_known_laws():
    pA = np.array([[0.5, 0.2, 0.3],    # mean 0.8: rounded mean 1, mode 0, median 0
                   [0.2, 0.3, 0.5],    # mean 1.3: rounded mean 1, mode 2, median 1
                   [0.5, 0.0, 0.5]])   # mean 1.0: tie in the mode goes to the lowest label; median 0
    mu = pA @ LAB3
    cf = R.collapsed_forecasts(pA, mu, 3)
    label = lambda q: q.argmax(1)  # noqa: E731
    assert label(cf["rounded_mean"]).tolist() == [1, 1, 1]
    assert label(cf["modal_label"]).tolist() == [0, 2, 0]
    assert label(cf["median_label"]).tolist() == [0, 1, 0]
    for k in R.CONVENTIONS:
        assert np.allclose(cf[k].sum(1), 1.0, atol=1e-9)


def test_collapsed_forecast_is_strictly_worse_than_the_truth_on_a_strictly_proper_score():
    pA = np.array([[0.1, 0.6, 0.3], [0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
    counts = pA * 100  # the item's own rating law as expected counts
    mu = pA @ LAB3
    cf = R.collapsed_forecasts(pA, mu, 3)
    for rule in ("brier", "nrps"):
        truth = R.item_scores(R.floor_probs(pA), counts, rule)
        for k in R.CONVENTIONS:
            assert (R.item_scores(cf[k], counts, rule) > truth - 1e-12).all()  # propriety: no forecast beats the truth in expectation
        assert R.item_scores(cf["rounded_mean"], counts, rule).sum() > truth.sum() + 1e-6  # and the point mass is strictly worse here


def test_conventions_for_set_is_deterministic_and_reproduces_experiment_60():
    a = R.conventions_for_set("dices350")
    b = R.conventions_for_set("dices350")
    assert a == b
    ref = {r["dataset"]: r for r in _rows(R.E60.OUT / "sampled_vs_mean.csv")}["dices350"]
    mine = next(r for r in a if r["convention"] == "rounded_mean")
    assert abs(mine["skill"] - float(ref["skill_collapsed"])) < 1e-8


# ------------------------------------------------------------------ C2: the gap between the rater-share rate and the majority rate (code checks on toy laws)
def test_perfect_judge_gap_on_toy_distributions():
    f = R.E58.hard_label_share
    soft = lambda h, labels: h[:, labels].sum(1)  # noqa: E731
    h = np.array([[0.4, 0.35, 0.25]])
    assert f(h, [2]).tolist() == [0.0] and abs(soft(h, [2])[0] - 0.25) < 1e-12           # majority says not-event, raters say 25%
    h = np.array([[0.5, 0.0, 0.5]])
    assert f(h, [2]).tolist() == [0.5] and abs(soft(h, [2])[0] - 0.5) < 1e-12            # a tie is split, so no gap
    h = np.array([[0.2, 0.3, 0.5]])
    assert f(h, [2]).tolist() == [1.0] and abs(soft(h, [2])[0] - 0.5) < 1e-12            # majority says event, raters say 50%
    h = np.array([[0.4, 0.35, 0.25], [0.2, 0.3, 0.5]])
    gap = f(h, [2]).mean() - soft(h, [2]).mean()
    assert abs(gap - (Fraction(1, 2) - Fraction(3, 8))) < 1e-12                           # (0 + 1)/2 - (0.25 + 0.5)/2 = 1/8
    unanimous = np.eye(3)
    assert abs(f(unanimous, [2]).mean() - soft(unanimous, [2]).mean()) < 1e-12           # no disagreement, no gap


# ------------------------------------------------------------------ C3: the random-pick comparator (code checks)
def test_random_pick_regret_is_the_mean_regret_over_the_judges():
    cal = np.array([0.30, 0.25, 0.20, 0.40, 0.35])
    base = 0.5
    got = R.random_pick_regret(cal, base)
    want = sum(Fraction(str(c)) - Fraction("0.20") for c in cal) / 5 / Fraction("0.5")
    assert abs(got - float(want)) < 1e-12 and abs(got - 0.2) < 1e-12
    assert R.random_pick_regret(np.array([0.2, 0.2, 0.2, 0.2, 0.2]), base) == 0.0  # judges that tie: nothing to lose by a random pick
    assert R.random_pick_regret(cal, base) >= 0.0


def test_recomputed_draws_equal_the_committed_experiment_59_draws():
    ref = _rows(R.E59.OUT / "draws.csv")
    for name in Z.DATASETS:
        for ni in (0, 3):
            for draw in (0, 1):
                mine = R.draw_with_risks(name, ni, draw)
                theirs = next(r for r in ref if r["dataset"] == name and int(r["n_grid"]) == R.E59.GRID[ni] and int(r["draw"]) == draw)
                for k in ("n_used", "n_pool", "n_test", "pick_A", "pick_B", "pick_C", "pick_oracle"):
                    assert mine[k] == int(theirs[k])
                for k in ("regret_A", "regret_B", "regret_C"):
                    assert mine[k] == pytest.approx(float(theirs[k]), rel=1e-7, abs=1e-12)


# ------------------------------------------------------------------ committed results: internal consistency and the statements quoted elsewhere
def test_committed_draw_risks_reproduce_the_committed_regrets():
    short = [Z.SHORT[j] for j in Z.JUDGES]
    for r in _rows(RES / "exp59_draw_risks.csv"):
        cal = np.array([float(r[f"cal_test_risk_{s}"]) for s in short])
        base = float(r["base_test_risk"])
        # The CSV stores every float at 10 significant digits (frozen.write_csv), so a regret rebuilt from the stored risks can differ from the stored regret by
        # about 1e-10; the named tolerance is an absolute 1e-9 on regrets (units: skill, a 0-to-1 scale).
        assert float(r["regret_random"]) == pytest.approx(R.random_pick_regret(cal, base), abs=1e-9)
        for t in "ABC":
            assert float(r[f"regret_{t}"]) == pytest.approx((cal[int(r[f"pick_{t}"])] - cal.min()) / base, abs=1e-9)
        assert cal[int(r["pick_oracle"])] == pytest.approx(cal.min(), abs=1e-9)


def test_committed_exp60_conventions_statements():
    rows = _rows(RES / "exp60_conventions.csv")
    ref = {r["dataset"]: r for r in _rows(R.E60.OUT / "sampled_vs_mean.csv")}
    for name in R.ORDERED_SETS:
        by = {r["convention"]: r for r in rows if r["dataset"] == name}
        assert float(by["rounded_mean"]["skill"]) == pytest.approx(float(ref[name]["skill_collapsed"]), abs=1e-8)
        assert float(by["truthful"]["skill"]) == pytest.approx(float(ref[name]["skill_truthful"]), abs=1e-8)
        assert float(by["base"]["skill"]) == pytest.approx(0.0, abs=1e-12)
        for k in R.CONVENTIONS:  # the truthful forecast beats every collapsed forecast, interval excluding 0 (propriety)
            assert float(by[k]["diff_lo"]) > -1e-9 or name == "chaosnli_alphanli" and k == "mean_preserving_two_point"
        if name != "chaosnli_alphanli":  # on both DICES sets every collapsed forecast is BELOW the base rate and the truthful one is above it
            assert float(by["truthful"]["skill"]) > 0.1
            for k in R.CONVENTIONS:
                assert float(by[k]["skill_hi"]) < 0.0
    a = {r["convention"]: float(r["skill"]) for r in rows if r["dataset"] == "chaosnli_alphanli"}
    assert a["mean_preserving_two_point"] == pytest.approx(a["truthful"], abs=1e-12)


def test_committed_exp58_estimand_gap_statements():
    gap = _rows(RES / "exp58_estimand_gap.csv")
    for r in gap:
        assert float(r["perfect_judge_hard_label_error_pp"]) == pytest.approx(float(r["human_majority_rate_pct"]) - float(r["human_rater_share_rate_pct"]), abs=1e-8)
        assert float(r["gap_lo"]) <= float(r["perfect_judge_hard_label_error_pp"]) <= float(r["gap_hi"])
    prim = {r["dataset"]: r for r in gap if r["event"] == "primary"}
    assert round(float(prim["dices350"]["perfect_judge_hard_label_error_pp"]), 1) == -9.8
    assert round(float(prim["dices990"]["perfect_judge_hard_label_error_pp"]), 1) == -9.5
    cells = [r for r in _rows(RES / "exp58_hard_vs_majority.csv") if r["event"] == "primary"]
    assert len(cells) == 25
    for r in cells:
        assert float(r["error_vs_rater_share_pp"]) - float(r["error_vs_majority_pp"]) == pytest.approx(float(r["human_majority_rate_pct"]) - float(r["human_rater_share_rate_pct"]), abs=1e-8)


def test_committed_exp59_random_baseline_counts():
    rows = _rows(RES / "exp59_random_baseline.csv")
    assert len(rows) == 20

    def count(tag):
        worse = [r for r in rows if float(r[f"{tag}_minus_random"]) > 0]
        return len(worse), sum(float(r[f"{tag}_minus_random_lo"]) > 0 for r in worse), sum(float(r[f"{tag}_minus_random_hi"]) < 0 for r in rows)

    assert count("A") == (1, 0, 17)
    assert count("B") == (11, 9, 8)
    assert count("C") == (0, 0, 19)
    text = (RES / "summary.md").read_text()
    assert "mean regret above the random pick's in 11 of 20 cells (9 with the whole interval above 0)" in text
