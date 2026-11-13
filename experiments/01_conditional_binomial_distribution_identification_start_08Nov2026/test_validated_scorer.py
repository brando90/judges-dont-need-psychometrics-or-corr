#!/usr/bin/env python3
"""Deterministic unit tests for the prospective validated_scorer wrapper."""
import json
from pathlib import Path
import unittest
import numpy as np
import run_study
import validated_scorer

ROOT = Path(__file__).resolve().parent


class ValidatedScorerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'protocol.json').read_text())
        self.rows = run_study.items(self.cfg)
        self.t = np.array([r['t'] for r in self.rows[:16]])
        self.p = run_study.population(self.t)
        self.mu = 5.0 * self.t
        rng = np.random.default_rng(20261108)
        self.ratings = rng.binomial(5, self.t[:, None], size=(len(self.t), 3))

    def test_exact_equality_with_frozen_scorer_on_valid_inputs(self):
        train_ratings = self.ratings[:8]
        q_all, mu_all = run_study.candidates(self.t, train_ratings, self.cfg)
        for q, mu in zip(q_all, mu_all):
            expected_frozen = run_study.expected_metrics(self.p, q, mu)
            expected_valid = validated_scorer.validated_expected_metrics(self.p, q, mu)
            self.assertEqual(expected_frozen, expected_valid)
            empirical_frozen = run_study.empirical_metrics(q, mu, self.ratings)
            empirical_valid = validated_scorer.validated_empirical_metrics(q, mu, self.ratings)
            self.assertEqual(empirical_frozen, empirical_valid)

    def test_rejects_empty_and_wrong_shape_pmfs(self):
        with self.assertRaisesRegex(ValueError, 'must not be empty'):
            validated_scorer.validated_expected_metrics(
                np.empty((0, 6)), np.empty((0, 6)), np.empty((0,))
            )
        with self.assertRaisesRegex(ValueError, 'must have shape'):
            validated_scorer.validated_expected_metrics(
                self.p[:, :5], self.p[:, :5], self.mu
            )
        with self.assertRaisesRegex(ValueError, 'must have shape'):
            validated_scorer.validated_expected_metrics(
                self.p[0], self.p[0], self.mu[:1]
            )

    def test_rejects_nonfinite_negative_zero_and_unnormalized_pmfs(self):
        bad_nan = self.p.copy()
        bad_nan[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            validated_scorer.validated_expected_metrics(self.p, bad_nan, self.mu)

        bad_inf = self.p.copy()
        bad_inf[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            validated_scorer.validated_expected_metrics(self.p, bad_inf, self.mu)

        bad_neg = self.p.copy()
        bad_neg[0, 0] = -0.1
        bad_neg[0, 1] += 0.1
        with self.assertRaisesRegex(ValueError, 'strictly positive'):
            validated_scorer.validated_expected_metrics(self.p, bad_neg, self.mu)

        bad_zero = self.p.copy()
        bad_zero[0, 1] += bad_zero[0, 0]
        bad_zero[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, 'strictly positive'):
            validated_scorer.validated_empirical_metrics(bad_zero, self.mu, self.ratings)

        unnormalized = self.p * 2.0
        with self.assertRaisesRegex(ValueError, 'do not sum to 1'):
            validated_scorer.validated_expected_metrics(self.p, unnormalized, self.mu)

    def test_rejects_malformed_means_and_ratings(self):
        with self.assertRaisesRegex(ValueError, 'must have shape'):
            validated_scorer.validated_empirical_metrics(self.p, self.mu[:-1], self.ratings)

        bad_mu = self.mu.copy()
        bad_mu[0] = 5.5
        with self.assertRaisesRegex(ValueError, 'must lie in \\[0, 5\\]'):
            validated_scorer.validated_empirical_metrics(self.p, bad_mu, self.ratings)

        with self.assertRaisesRegex(ValueError, 'must have shape'):
            validated_scorer.validated_empirical_metrics(
                self.p, self.mu, np.empty((len(self.t), 0), dtype=int)
            )

        bad_ratings_range = self.ratings.copy()
        bad_ratings_range[0, 0] = 6
        with self.assertRaisesRegex(ValueError, 'entries must lie in'):
            validated_scorer.validated_empirical_metrics(self.p, self.mu, bad_ratings_range)

        bad_ratings_float = self.ratings.astype(float)
        bad_ratings_float[0, 0] = 2.5
        with self.assertRaisesRegex(ValueError, 'exact integers'):
            validated_scorer.validated_empirical_metrics(self.p, self.mu, bad_ratings_float)


if __name__ == '__main__':
    unittest.main()
