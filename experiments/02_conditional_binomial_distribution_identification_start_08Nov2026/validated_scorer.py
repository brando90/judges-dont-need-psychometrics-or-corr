#!/usr/bin/env python3
"""Separately versioned input-validation wrapper for the conditional-distribution scorer.

Preserves frozen run_study.py and results/COMPLETE.json unchanged while enforcing
the PROTOCOL.md contract ("Empty/non-PMF input: invalid, not silently normalized")
for prospective callers.
"""
import numpy as np
import run_study


def validate_pmf(pmf, name='pmf', *, require_positive=False, sum_tol=1e-9):
    """Validate an (N, 6) categorical probability mass function without normalizing."""
    if pmf is None:
        raise ValueError(f'{name} must not be None')
    arr = np.asarray(pmf)
    if arr.dtype.kind not in ('f', 'i', 'u'):
        raise ValueError(f'{name} must be numeric, got dtype {arr.dtype}')
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(run_study.SCORES):
        raise ValueError(f'{name} must have shape (N, 6), got {arr.shape}')
    if arr.shape[0] == 0:
        raise ValueError(f'{name} must not be empty')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} contains non-finite values')
    if require_positive:
        if np.any(arr <= 0.0):
            raise ValueError(f'{name} must be strictly positive for log-loss evaluation')
    elif np.any(arr < 0.0):
        raise ValueError(f'{name} contains negative probabilities')
    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=sum_tol, rtol=0.0):
        max_dev = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f'{name} rows do not sum to 1 within {sum_tol} (max deviation {max_dev:.3e})')
    return arr


def validate_means(mu, n_items, name='mu'):
    """Validate an (N,) array of forecast conditional means on [0, 5]."""
    if mu is None:
        raise ValueError(f'{name} must not be None')
    arr = np.asarray(mu)
    if arr.dtype.kind not in ('f', 'i', 'u'):
        raise ValueError(f'{name} must be numeric, got dtype {arr.dtype}')
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n_items:
        raise ValueError(f'{name} must have shape ({n_items},), got {arr.shape}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} contains non-finite values')
    if np.any((arr < -1e-9) | (arr > 5.0 + 1e-9)):
        raise ValueError(f'{name} must lie in [0, 5]')
    return arr


def validate_ratings(ratings, n_items, name='ratings'):
    """Validate an (N, m) integer rating matrix with entries in {0, 1, 2, 3, 4, 5}."""
    if ratings is None:
        raise ValueError(f'{name} must not be None')
    arr = np.asarray(ratings)
    if arr.dtype.kind == 'b' or arr.dtype.kind not in ('i', 'u', 'f'):
        raise ValueError(f'{name} must be numeric integer ratings, got dtype {arr.dtype}')
    if arr.ndim != 2 or arr.shape[0] != n_items or arr.shape[1] == 0:
        raise ValueError(f'{name} must have shape ({n_items}, m) with m >= 1, got {arr.shape}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} contains non-finite values')
    int_arr = np.asarray(arr, dtype=int)
    if not np.array_equal(arr, int_arr):
        raise ValueError(f'{name} must contain exact integers in 0..5')
    if np.any((int_arr < 0) | (int_arr > 5)):
        raise ValueError(f'{name} entries must lie in {{0, 1, 2, 3, 4, 5}}')
    return int_arr


def validated_empirical_metrics(q, mu, ratings, *, sum_tol=1e-9):
    """Validate inputs and compute empirical individual-rating metrics."""
    q_arr = validate_pmf(q, 'q', require_positive=True, sum_tol=sum_tol)
    mu_arr = validate_means(mu, q_arr.shape[0], 'mu')
    ratings_arr = validate_ratings(ratings, q_arr.shape[0], 'ratings')
    return run_study.empirical_metrics(q_arr, mu_arr, ratings_arr)


def validated_expected_metrics(p, q, mu, *, sum_tol=1e-9):
    """Validate inputs and compute exact population expected metrics."""
    p_arr = validate_pmf(p, 'p', require_positive=True, sum_tol=sum_tol)
    q_arr = validate_pmf(q, 'q', require_positive=True, sum_tol=sum_tol)
    if p_arr.shape != q_arr.shape:
        raise ValueError(f'p and q shapes must match, got {p_arr.shape} vs {q_arr.shape}')
    mu_arr = validate_means(mu, p_arr.shape[0], 'mu')
    return run_study.expected_metrics(p_arr, q_arr, mu_arr)
