"""Unit tests for exp/exp7.py on a small synthetic panel (no network calls)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exp.exp7 import evaluate_window
from src import portfolio as pf
from src.empirical import iterate_estimation_windows


def _synthetic_panel(t_days: int = 130, n_assets: int = 8, seed: int = 0) -> pd.DataFrame:
    # Per-asset drift gives the cross-sectional dispersion in mu_hat that the
    # return-floored problem (K) needs to be feasible (its return floor
    # phi'mu - sqrt(delta)*||phi||_2 >= r0 is otherwise unattainable when
    # every asset shares (close to) the same mean return).
    rng = np.random.default_rng(seed)
    drift = rng.uniform(-0.001, 0.002, size=n_assets)
    data = drift + rng.normal(0.0, 0.01, size=(t_days, n_assets))
    return pd.DataFrame(data, columns=[f"asset_{i}" for i in range(n_assets)])


# A delta small enough to keep problem (K) feasible on the synthetic panel
# above (see module docstring note); EXP7_DELTA_GRID's smallest radius.
TEST_DELTA = 1e-6


@pytest.fixture
def one_window():
    panel = _synthetic_panel()
    windows = iterate_estimation_windows(panel, window=60, hold=5)
    assert len(windows) >= 1
    return windows[0]


def test_evaluate_window_returns_valid_simplex_weights(one_window):
    w, mu_hat, sigma_hat, test = one_window
    res = evaluate_window(mu_hat, sigma_hat, test, lam=10.0, delta=TEST_DELTA)

    for key in ("phi_robust", "phi_bm", "phi_rm", "phi_mv", "phi_ew"):
        weights = res[key]
        assert weights.shape == (sigma_hat.shape[0],)
        assert np.all(weights >= -1e-9), f"{key} has a negative weight"
        assert weights.sum() == pytest.approx(1.0, abs=1e-6), f"{key} does not sum to 1"


def test_ordinary_mv_dominates_equally_weighted_in_sample(one_window):
    # Ordinary mean-variance is the in-sample optimum of U(phi), so it must
    # weakly dominate any other feasible portfolio, including equal weight,
    # when both are evaluated on the SAME mu_hat/Sigma_hat.
    w, mu_hat, sigma_hat, test = one_window
    res = evaluate_window(mu_hat, sigma_hat, test, lam=10.0, delta=TEST_DELTA)

    util_mv = pf.utility(mu_hat, sigma_hat, res["phi_mv"], lam=10.0)
    util_ew = pf.utility(mu_hat, sigma_hat, res["phi_ew"], lam=10.0)
    assert util_mv >= util_ew - 1e-8


def test_breadth_matched_ridge_norm_matches_h_star(one_window):
    # Theorem 5.2/Corollary 5.3: the breadth-matched ridge benchmark solved
    # with h_star = ||phi_robust||_2 should realise (approximately) that same
    # Euclidean norm.
    w, mu_hat, sigma_hat, test = one_window
    res = evaluate_window(mu_hat, sigma_hat, test, lam=10.0, delta=TEST_DELTA)

    h_star = res["h_star"]
    realised_norm = float(np.linalg.norm(res["phi_bm"], 2))
    assert realised_norm == pytest.approx(h_star, abs=1e-3)


def test_evaluate_window_utility_and_return_keys_are_finite(one_window):
    w, mu_hat, sigma_hat, test = one_window
    res = evaluate_window(mu_hat, sigma_hat, test, lam=10.0, delta=TEST_DELTA)

    for key in ("ret_robust", "ret_bm", "ret_rm", "ret_mv", "ret_ew"):
        assert np.isfinite(res[key])
    for key in ("util_robust", "util_bm", "util_rm", "util_mv", "util_ew"):
        assert np.isfinite(res[key])
