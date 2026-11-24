"""Unit tests for src/statistics.py: Newey-West inference and the stationary
bootstrap, on synthetic series with a known mean."""

from __future__ import annotations

import numpy as np
import pytest

from src.statistics import newey_west_tstat, romano_wolf_stepdown, stationary_bootstrap_mean_ci


def test_newey_west_tstat_large_mean_is_significant():
    rng = np.random.default_rng(0)
    x = 0.01 + rng.normal(0, 0.001, size=500)
    mean, t = newey_west_tstat(x, lag=10)
    assert mean == pytest.approx(0.01, abs=5e-4)
    assert t > 5


def test_newey_west_tstat_zero_mean_not_significant():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, size=500)
    mean, t = newey_west_tstat(x, lag=10)
    assert abs(t) < 3


def test_newey_west_requires_1d_series():
    with pytest.raises(AssertionError):
        newey_west_tstat(np.zeros((3, 3)), lag=2)


def test_stationary_bootstrap_ci_contains_true_mean():
    rng = np.random.default_rng(2)
    x = 0.02 + rng.normal(0, 0.01, size=300)
    lo, hi = stationary_bootstrap_mean_ci(x, n_boot=500, expected_block=10, seed=0)
    assert lo < 0.02 < hi


def test_stationary_bootstrap_ci_excludes_zero_for_strong_signal():
    rng = np.random.default_rng(3)
    x = 0.05 + rng.normal(0, 0.005, size=300)
    lo, hi = stationary_bootstrap_mean_ci(x, n_boot=500, expected_block=10, seed=1)
    assert lo > 0.0


def test_romano_wolf_stepdown_orders_pvalues_with_statistics():
    rng = np.random.default_rng(4)
    n = 300
    series = {
        "strong": 0.05 + rng.normal(0, 0.01, size=n),
        "weak": 0.001 + rng.normal(0, 0.01, size=n),
        "none": rng.normal(0, 0.01, size=n),
    }
    adjusted = romano_wolf_stepdown(series, nw_lag=5, n_boot=300, expected_block=10, seed=5)
    assert set(adjusted) == set(series)
    assert all(0.0 <= p <= 1.0 for p in adjusted.values())
    assert adjusted["strong"] <= adjusted["none"]
