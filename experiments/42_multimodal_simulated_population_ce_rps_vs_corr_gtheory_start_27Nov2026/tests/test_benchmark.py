"""Exact identities, leakage, and one-seed repeatability. No 400-rep Monte Carlo."""

from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import (  # noqa: E402
    BIN_MEANS,
    LAW_A,
    LAW_B,
    M1,
    M2,
    PERSONAS,
    SHIFT_WEIGHTS,
    TRAIN_ITEMS,
    TEST_ITEMS,
    VAL_ITEMS,
    build_population,
    expected_rps_rows,
    leakage_report,
    mix,
    moment_panels,
    panel_coeffs,
    pearson,
    rng_for,
    row_means,
    row_vars,
    score_forecast,
    tilt_to_mean,
    tvd_rows,
)


def test_m1_m2_moments_and_rps_gap():
    assert np.isclose(M1 @ np.arange(6), 2.5)
    assert np.isclose(M2 @ np.arange(6), 2.5)
    assert np.isclose(np.sum(M1 * (np.arange(6) - 2.5) ** 2), 4.0)
    assert np.isclose(np.sum(M2 * (np.arange(6) - 2.5) ** 2), 4.0)
    gap = expected_rps_rows(M2[None, :], M1[None, :])[0] - expected_rps_rows(M1[None, :], M1[None, :])[0]
    assert abs(gap - float(F(9, 512))) < 1e-12
    assert abs(float(tvd_rows(M2[None, :], M1[None, :])[0]) - float(F(9, 16))) < 1e-12


def test_moment_panels_share_erho_and_differ_in_rps():
    panels = moment_panels()
    assert abs(panels["coeffs_U"]["erho2"] - panels["coeffs_V"]["erho2"]) < 1e-12
    assert abs(panels["coeffs_U"]["erho2"] - 0.2) < 1e-12
    gap = panels["rps_V_on_U"] - panels["rps_U"]
    assert abs(gap - float(F(167, 23040))) < 1e-12
    assert panels["tvd"] > 0.25
    assert np.allclose(panels["means_U"], panels["means_V"])
    assert np.allclose(panels["vars_U"], panels["vars_V"])


def test_population_means_and_strict_shift_keeps_pearson():
    pop = build_population()
    laws = pop["laws"]
    assert abs(row_means(laws[0]).mean() - (BIN_MEANS.mean() - 0.8)) < 1e-8
    assert abs(row_means(laws[1]).mean() - (BIN_MEANS.mean() + 0.8)) < 1e-8
    assert np.allclose(row_means(laws[2]), BIN_MEANS, atol=1e-8)
    assert np.allclose(row_means(laws[3]), BIN_MEANS, atol=1e-8)
    shift = mix(SHIFT_WEIGHTS, laws)
    assert np.allclose(row_means(shift), BIN_MEANS, atol=1e-8)
    assert pearson(row_means(laws[2]), row_means(laws[0])) > 0.999
    scored = score_forecast(laws[2], laws[2])
    assert scored["tvd"] == 0.0
    other = score_forecast(laws[0], laws[2])
    assert other["tvd"] > 0.2
    assert other["pearson"] > 0.999


def test_tilt_hits_a_target_inside_the_support():
    template = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    q = tilt_to_mean(template, 2.25)
    assert abs(float(q @ np.arange(6)) - 2.25) < 1e-8
    assert abs(q.sum() - 1) < 1e-12


def test_leakage_and_split_ids():
    report = leakage_report()
    assert report["fit_ignores_test_array"]
    assert report["rater_ids_disjoint_across_splits"]
    assert report["item_ids_disjoint_across_splits"]
    assert set(TRAIN_ITEMS).isdisjoint(VAL_ITEMS)
    assert set(TRAIN_ITEMS).isdisjoint(TEST_ITEMS)
    assert set(VAL_ITEMS).isdisjoint(TEST_ITEMS)


def test_one_draw_repeats_and_the_next_replication_differs():
    pop = build_population()
    a = rng_for(1, 16, 0, 1).multinomial(20, pop["laws"][2, 0])
    b = rng_for(1, 16, 0, 1).multinomial(20, pop["laws"][2, 0])
    c = rng_for(1, 16, 1, 1).multinomial(20, pop["laws"][2, 0])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_law_a_and_b_match_moments():
    y = np.arange(6)
    for law in (LAW_A, LAW_B):
        assert abs(law.sum() - 1) < 1e-12
        assert abs(float(law @ y) - 1.5) < 1e-12
        assert abs(float(law @ (y - 1.5) ** 2) - 2) < 1e-12


def test_constant_shift_preserves_pearson():
    means = np.array([1.6, 2.0, 2.4, 2.8, 3.2, 3.6])
    assert abs(pearson(means, means - 0.8) - 1.0) < 1e-12


def test_point_mass_reliability_is_one_when_locations_differ():
    from benchmark import point_rows

    q = point_rows(np.array([0, 1, 2, 3, 4, 5]))
    coef = panel_coeffs(q)
    assert coef["erho2"] == 1.0
    assert coef["ve"] == 0.0
    assert row_vars(q).sum() == 0.0


def test_persona_names_cover_four_groups():
    assert PERSONAS == ("strict", "lenient", "polarized", "chaos")
