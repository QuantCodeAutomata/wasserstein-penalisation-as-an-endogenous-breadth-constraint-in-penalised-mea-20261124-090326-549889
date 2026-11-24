"""Shared pytest fixtures.

Registers the ``market_data`` marker for tests that load the authors' A-share
panels (Section 6.1) over the network, and caches a reduced slice of one
panel so repeated test runs do not re-download it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "market_data: loads the A-share panel data over the network (skipped if unavailable)",
    )


@pytest.fixture(scope="session")
def small_panel_slice() -> pd.DataFrame:
    """A short slice of one real A-share panel (first 130 rows, all 255
    assets) used by market-data tests, downloaded once per session and
    cached between runs.
    """
    try:
        from data.data_loader import load_panel_returns
        from src.config import PANEL_NAMES
    except Exception as exc:  # pragma: no cover - import-time failure
        pytest.skip(f"data loader unavailable: {exc}")
    try:
        df = load_panel_returns(PANEL_NAMES[0])
    except Exception as exc:
        pytest.skip(f"A-share panel data not reachable: {exc}")
    return df.iloc[:130].reset_index(drop=True)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
