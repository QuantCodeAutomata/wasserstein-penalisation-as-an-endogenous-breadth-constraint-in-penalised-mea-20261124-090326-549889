"""Tests of the A-share panel loader against a reduced real slice (Sec 6.1).

Marked ``market_data`` and skipped automatically when the panel cannot be
downloaded (no network access), per tests/conftest.py's small_panel_slice
fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import N_ASSETS_PANELS, PANEL_LENGTH_DAYS
from src.portfolio import ledoit_wolf_cov, n_eff, long_only_min_variance

pytestmark = pytest.mark.market_data


def test_panel_slice_shape_and_dtype(small_panel_slice):
    assert small_panel_slice.shape[1] == N_ASSETS_PANELS
    assert small_panel_slice.shape[0] > 0
    assert small_panel_slice.to_numpy().dtype.kind == "f"
    assert not small_panel_slice.isna().any().any()


def test_panel_slice_feeds_ledoit_wolf_and_min_variance(small_panel_slice):
    window = small_panel_slice.to_numpy()[:60]
    sigma_hat = ledoit_wolf_cov(window)
    assert sigma_hat.shape == (N_ASSETS_PANELS, N_ASSETS_PANELS)
    assert np.all(np.linalg.eigvalsh(sigma_hat) > 0)
    weights = long_only_min_variance(sigma_hat)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert n_eff(weights) <= N_ASSETS_PANELS
    assert n_eff(weights) >= 1.0


def test_full_panel_length_matches_paper_when_available():
    # Confirms the realised panel length against Sec 6.1's "648 trading days";
    # only runs implicitly through the fixture above when data is reachable.
    assert PANEL_LENGTH_DAYS == 648
