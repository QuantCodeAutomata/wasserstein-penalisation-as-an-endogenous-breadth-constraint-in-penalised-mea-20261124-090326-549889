"""Unit tests for exp/exp6.py on synthetic return panels.

No network calls are made here: ``run()``'s call to ``load_all_panels`` is
monkeypatched to return small synthetic panels, matching the pattern used by
``tests/test_empirical.py`` for the shared rolling-window pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import exp.exp6 as exp6
from src.empirical import iterate_estimation_windows


def _synthetic_panel(t_days: int, n_assets: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0005, 0.01, size=(t_days, n_assets))
    return pd.DataFrame(data, columns=[f"asset_{i}" for i in range(n_assets)])


# --------------------------------------------------------------------------
# compute_tracked_cap: the ex ante tracking-rule helper
# --------------------------------------------------------------------------


def test_compute_tracked_cap_previous():
    assert exp6.compute_tracked_cap([10.0], "previous") == pytest.approx(10.0)


def test_compute_tracked_cap_mean_past():
    assert exp6.compute_tracked_cap([10.0, 20.0], "mean_past") == pytest.approx(15.0)


def test_compute_tracked_cap_median_past():
    assert exp6.compute_tracked_cap([1.0, 2.0, 3.0, 4.0], "median_past") == pytest.approx(2.5)


def test_compute_tracked_cap_last5_uses_only_last_five():
    history = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    expected = np.mean(history[-5:])  # drops the oldest entry (5.0)
    assert exp6.compute_tracked_cap(history, "last5") == pytest.approx(expected)
    assert expected == pytest.approx(20.0)


def test_compute_tracked_cap_last10_with_fewer_than_ten_past_windows():
    history = [1.0, 2.0, 3.0]
    assert exp6.compute_tracked_cap(history, "last10") == pytest.approx(np.mean(history))


def test_compute_tracked_cap_empty_history_raises():
    # Invariant: there must be at least one past window before a tracking
    # cap can be formed; window 0 of a panel has no history and must be
    # skipped by the caller rather than calling this with an empty list.
    with pytest.raises(ValueError):
        exp6.compute_tracked_cap([], "previous")


def test_compute_tracked_cap_unknown_rule_raises():
    with pytest.raises(ValueError):
        exp6.compute_tracked_cap([1.0], "not_a_real_rule")


# --------------------------------------------------------------------------
# Structural no-look-ahead test: the tracking rules must never see the
# current window's own realised N_eff.
# --------------------------------------------------------------------------


def test_tracking_rules_never_use_current_window_n_eff(monkeypatch):
    panel = _synthetic_panel(t_days=130, n_assets=6, seed=1)

    recorded_histories: list[list[float]] = []
    original_compute = exp6.compute_tracked_cap

    def spy_compute(history, rule):
        recorded_histories.append(list(history))
        return original_compute(history, rule)

    n_eff_calls: list[float] = []
    original_n_eff = exp6.pf.n_eff

    def spy_n_eff(weights):
        value = original_n_eff(weights)
        n_eff_calls.append(value)
        return value

    monkeypatch.setattr(exp6, "compute_tracked_cap", spy_compute)
    monkeypatch.setattr(exp6.pf, "n_eff", spy_n_eff)
    monkeypatch.setattr(exp6, "EXP6_DELTA_GRID", [1e-6])
    monkeypatch.setattr(exp6, "EXP6_NEFF_TARGETS", [2])
    monkeypatch.setattr(exp6, "ABSOLUTE_NAMES", ["Neff=2"])
    monkeypatch.setattr(exp6, "EXP6_TRACKING_RULES", ["previous"])
    monkeypatch.setattr(exp6, "COMPARISON_NAMES", ["Neff=2", "previous"])

    exp6._process_panel("synthetic", panel)

    n_windows = len(iterate_estimation_windows(panel, 60, 10))
    # One tracking-rule call per window t=1..n_windows-1 (single delta, single rule).
    assert len(recorded_histories) == n_windows - 1
    for i, hist in enumerate(recorded_histories):
        t = i + 1
        # hist must equal exactly the n_eff values of windows 0..t-1, i.e. it
        # must never contain n_eff_calls[t], the current window's own value.
        assert hist == pytest.approx(n_eff_calls[:t])
        assert len(hist) == t


# --------------------------------------------------------------------------
# End-to-end smoke test of _process_panel / run() / summarize() on a
# reduced synthetic panel, with load_all_panels monkeypatched so no network
# call is made.
# --------------------------------------------------------------------------


@pytest.fixture
def _fast_exp6_config(monkeypatch):
    """Shrink the breadth-cap grids and the bootstrap/Newey-West settings so
    the full pipeline runs quickly on a tiny synthetic panel (6 assets),
    while keeping the N_eff targets small enough to stay feasible for the
    norm-capped QP (the simplex's minimum achievable norm is 1/sqrt(N))."""
    monkeypatch.setattr(exp6, "EXP6_DELTA_GRID", [1e-6])
    monkeypatch.setattr(exp6, "EXP6_NEFF_TARGETS", [2, 3])
    monkeypatch.setattr(exp6, "ABSOLUTE_NAMES", ["Neff=2", "Neff=3"])
    monkeypatch.setattr(exp6, "EXP6_TRACKING_RULES", ["previous", "mean_past"])
    monkeypatch.setattr(exp6, "COMPARISON_NAMES", ["Neff=2", "Neff=3", "previous", "mean_past"])
    monkeypatch.setattr(exp6, "EXP6_NW_LAG", 2)
    monkeypatch.setattr(exp6, "EXP6_BOOTSTRAP_B", 50)
    monkeypatch.setattr(exp6, "EXP6_BOOTSTRAP_BLOCK", 2)


def test_process_panel_smoke(_fast_exp6_config):
    panel = _synthetic_panel(t_days=100, n_assets=6, seed=2)
    rows = exp6._process_panel("synthetic", panel)
    assert len(rows) > 0
    df = pd.DataFrame(rows)
    expected_cols = {
        "panel",
        "delta",
        "comparison",
        "mean_diff_1e4",
        "tstat",
        "ci_lo_1e4",
        "ci_hi_1e4",
        "significant",
        "romano_wolf_adj_p",
    }
    assert expected_cols == set(df.columns)
    assert set(df["comparison"]) == {"Neff=2", "Neff=3", "previous", "mean_past"}
    assert (df["panel"] == "synthetic").all()
    assert df["romano_wolf_adj_p"].between(0.0, 1.0).all()


def test_run_end_to_end_without_network(monkeypatch, _fast_exp6_config):
    synthetic_panels = {
        "panel_a": _synthetic_panel(t_days=100, n_assets=6, seed=3),
        "panel_b": _synthetic_panel(t_days=100, n_assets=6, seed=4),
    }

    def fake_load_all_panels(panel_names):
        return synthetic_panels

    monkeypatch.setattr(exp6, "load_all_panels", fake_load_all_panels)

    raw = exp6.run()
    assert isinstance(raw, pd.DataFrame)
    assert not raw.empty
    assert set(raw["panel"]) == set(synthetic_panels)

    summary = exp6.summarize(raw)
    assert not summary.empty
    assert {"delta", "comparison", "median_mean_diff_1e4", "n_significant", "n_panels"} == set(summary.columns)
    assert (summary["n_panels"] == len(synthetic_panels)).all()
