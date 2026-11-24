"""Unit tests for src/empirical.py on synthetic return panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.empirical import iterate_estimation_windows, l1_turnover, oos_daily_returns, r0_mean_return, rent_stats
from src.portfolio import n_eff


def _synthetic_panel(t_days: int, n_assets: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0003, 0.01, size=(t_days, n_assets))
    return pd.DataFrame(data, columns=[f"asset_{i}" for i in range(n_assets)])


def test_iterate_estimation_windows_uses_only_past_data():
    panel = _synthetic_panel(160, 10)
    windows = iterate_estimation_windows(panel, window=60, hold=10)
    assert len(windows) == (160 - 60) // 10
    for w, mu_hat, sigma_hat, test in windows:
        assert mu_hat.shape == (10,)
        assert sigma_hat.shape == (10, 10)
        assert test.shape == (10, 10)
        # The estimation window must end exactly where the holding block begins.
        assert w.test_start == w.train_end


def test_iterate_estimation_windows_mu_matches_manual_mean():
    panel = _synthetic_panel(130, 5, seed=1)
    windows = iterate_estimation_windows(panel, window=60, hold=10)
    w, mu_hat, _, _ = windows[0]
    manual = panel.to_numpy()[w.train_start : w.train_end].mean(axis=0)
    assert np.allclose(mu_hat, manual)


def test_oos_daily_returns_shape_and_formula():
    weights = np.array([0.5, 0.5])
    returns_oos = np.array([[0.01, -0.02], [0.03, 0.0]])
    out = oos_daily_returns(weights, returns_oos)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(0.5 * 0.01 + 0.5 * -0.02)


def test_l1_turnover_zero_for_first_rebalance():
    assert l1_turnover(np.array([0.5, 0.5]), None) == 0.0


def test_l1_turnover_matches_manual_sum():
    w_new = np.array([0.6, 0.4])
    w_old = np.array([0.5, 0.5])
    assert l1_turnover(w_new, w_old) == pytest.approx(0.2)


def test_rent_stats_significant_for_clear_positive_signal():
    rng = np.random.default_rng(2)
    diff = 0.01 + rng.normal(0, 0.001, size=300)
    stats = rent_stats(diff, nw_lag=10, boot_b=300, boot_block=10, seed=0)
    assert stats["mean"] == pytest.approx(0.01, abs=5e-4)
    assert stats["tstat"] > 2
    assert stats["significant"] is True


def test_rent_stats_not_significant_for_noise():
    rng = np.random.default_rng(3)
    diff = rng.normal(0, 0.01, size=300)
    stats = rent_stats(diff, nw_lag=10, boot_b=300, boot_block=10, seed=1)
    assert abs(stats["tstat"]) < 5  # weak/no signal should not produce an extreme t-stat


def test_r0_mean_return_matches_manual_average():
    mu_hat = np.array([0.01, 0.02, -0.01])
    assert r0_mean_return(mu_hat) == pytest.approx(mu_hat.mean())


def test_n_eff_consistent_with_empirical_weights():
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    assert n_eff(weights) == pytest.approx(4.0)
