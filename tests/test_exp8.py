"""Unit tests for exp/exp8.py on synthetic return panels (no network calls)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exp.exp8 import BENCHMARKS, STRATEGIES, evaluate_window, net_daily_returns
from src.empirical import iterate_estimation_windows


def _synthetic_panel(t_days: int, n_assets: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0005, 0.01, size=(t_days, n_assets))
    return pd.DataFrame(data, columns=[f"asset_{i}" for i in range(n_assets)])


def test_net_daily_returns_matches_gross_when_turnover_zero():
    gross = np.array([0.01, -0.02, 0.0, 0.005, -0.001])
    net = net_daily_returns(gross, turnover=0.0, tc_bps=30, hold=5)
    assert np.array_equal(net, gross)


def test_net_daily_returns_amortised_cost_matches_manual_value():
    gross = np.array([0.01, -0.02, 0.0, 0.005, -0.001])
    net = net_daily_returns(gross, turnover=1.0, tc_bps=30, hold=5)
    expected_daily_cost = 0.0030 * 1.0 / 5
    assert expected_daily_cost == pytest.approx(0.0006)
    assert np.allclose(gross - net, expected_daily_cost)
    assert np.allclose(net, gross - expected_daily_cost)


def test_net_daily_returns_scales_with_tc_bps_and_turnover():
    gross = np.zeros(5)
    net = net_daily_returns(gross, turnover=2.0, tc_bps=30, hold=5)
    # turnover doubles the cost relative to the single-turnover case above.
    assert np.allclose(net, -0.0012)


def test_evaluate_window_first_window_has_zero_turnover_for_every_strategy():
    panel = _synthetic_panel(130, 8, seed=0)
    windows = iterate_estimation_windows(panel, window=60, hold=5)
    assert len(windows) >= 3

    prev_weights: dict[str, np.ndarray | None] = {name: None for name in STRATEGIES}
    _w0, mu_hat, sigma_hat, test = windows[0]
    results, prev_weights = evaluate_window(
        mu_hat, sigma_hat, test, lam=10.0, delta=1e-6, hold=5, tc_bps=30, prev_weights=prev_weights
    )

    for name in STRATEGIES:
        assert results[name]["turnover"] == 0.0
        assert np.allclose(results[name]["net"], results[name]["gross"])


def test_evaluate_window_later_window_has_nonzero_turnover():
    panel = _synthetic_panel(130, 8, seed=0)
    windows = iterate_estimation_windows(panel, window=60, hold=5)

    prev_weights: dict[str, np.ndarray | None] = {name: None for name in STRATEGIES}
    turnovers_by_window = []
    for _w, mu_hat, sigma_hat, test in windows[:3]:
        results, prev_weights = evaluate_window(
            mu_hat, sigma_hat, test, lam=10.0, delta=1e-6, hold=5, tc_bps=30, prev_weights=prev_weights
        )
        turnovers_by_window.append({name: results[name]["turnover"] for name in STRATEGIES})

    # First window: no prior position anywhere.
    assert all(t == 0.0 for t in turnovers_by_window[0].values())
    # A later window should generally show nonzero turnover for at least one
    # strategy once a prior position exists (the equally weighted portfolio
    # is identical every window by construction, so it alone may stay at 0).
    later_turnovers = [t for window in turnovers_by_window[1:] for name, t in window.items() if name != "equally_weighted"]
    assert any(t > 0.0 for t in later_turnovers)


def test_evaluate_window_net_equals_gross_minus_constant_amortised_cost():
    panel = _synthetic_panel(130, 8, seed=1)
    windows = iterate_estimation_windows(panel, window=60, hold=5)

    prev_weights: dict[str, np.ndarray | None] = {name: None for name in STRATEGIES}
    _w0, mu_hat, sigma_hat, test = windows[0]
    results, prev_weights = evaluate_window(
        mu_hat, sigma_hat, test, lam=10.0, delta=1e-6, hold=5, tc_bps=30, prev_weights=prev_weights
    )
    _w1, mu_hat, sigma_hat, test = windows[1]
    results, _prev_weights = evaluate_window(
        mu_hat, sigma_hat, test, lam=10.0, delta=1e-6, hold=5, tc_bps=30, prev_weights=prev_weights
    )

    for name in STRATEGIES:
        turnover = results[name]["turnover"]
        daily_cost = (30 / 10_000.0) * turnover / 5
        assert np.allclose(results[name]["gross"] - results[name]["net"], daily_cost)


def test_benchmarks_and_strategies_consistency():
    assert set(BENCHMARKS) <= set(STRATEGIES)
    assert "robust" in STRATEGIES
    assert len(STRATEGIES) == len(BENCHMARKS) + 1
