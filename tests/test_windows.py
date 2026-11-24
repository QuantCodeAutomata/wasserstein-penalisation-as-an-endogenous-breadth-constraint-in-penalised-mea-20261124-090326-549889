"""Unit tests for src/windows.py: window counts and absence of look-ahead."""

from __future__ import annotations

from src.windows import non_overlapping_windows, rolling_windows


def test_non_overlapping_windows_count_table_4():
    # Sec 6.1: 648 trading days / 60-day windows = 10 non-overlapping windows
    # per panel, 14 panels -> 140 windows for Table 4.
    windows = non_overlapping_windows(648, 60)
    assert len(windows) == 10
    assert windows[0].train_start == 0
    assert windows[0].train_end == 60
    assert windows[-1].train_end == 600


def test_non_overlapping_windows_do_not_overlap():
    windows = non_overlapping_windows(200, 60)
    for a, b in zip(windows, windows[1:]):
        assert a.train_end == b.train_start


def test_rolling_windows_hold_blocks_are_strictly_after_training():
    windows = rolling_windows(648, 60, 10)
    for w in windows:
        assert w.test_start == w.train_end  # no look-ahead: trade only after the estimation window ends
        assert w.test_end - w.test_start == 10
        assert w.train_end - w.train_start == 60


def test_rolling_windows_step_by_holding_period():
    windows = rolling_windows(648, 60, 10)
    assert windows[1].train_start - windows[0].train_start == 10
    assert len(windows) == (648 - 60) // 10


def test_rolling_windows_never_exceed_sample():
    windows = rolling_windows(100, 60, 10)
    assert all(w.test_end <= 100 for w in windows)
