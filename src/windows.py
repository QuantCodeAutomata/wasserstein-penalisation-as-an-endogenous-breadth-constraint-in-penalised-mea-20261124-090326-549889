"""Rolling and non-overlapping window construction shared by every experiment
that uses the A-share panels (Table 4-7, Figures 1-2), so that estimation
windows and out-of-sample holding periods are built identically everywhere
and the temporal ordering (estimate on the past, trade in the strictly
following days) is enforced in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EstimationWindow:
    """A single 60-day estimation window with an optional out-of-sample
    holding block of the following ``hold`` trading days (Sec 6.1/6.3)."""

    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive


def non_overlapping_windows(t_days: int, window: int) -> list[EstimationWindow]:
    """floor(T / window) non-overlapping estimation windows with no holding
    block, used where only in-sample structural quantities are needed
    (Table 4, Figure 2: "60-day rolling windows yield 140 non-overlapping
    windows" across fourteen panels, i.e. 10 per panel)."""
    n_windows = t_days // window
    return [
        EstimationWindow(i * window, (i + 1) * window, (i + 1) * window, (i + 1) * window)
        for i in range(n_windows)
    ]


def rolling_windows(t_days: int, window: int, hold: int) -> list[EstimationWindow]:
    """Estimation windows that step forward by the holding period ``hold``,
    so consecutive out-of-sample blocks are back to back and never overlap,
    while consecutive estimation windows may share history (Sec 6.1: "a
    sixty-day estimation window using only the trailing sixty trading days,
    followed by a ten-day [or five-day] holding period")."""
    n_windows = (t_days - window) // hold
    windows = []
    for i in range(n_windows):
        start = i * hold
        train_end = start + window
        test_end = train_end + hold
        if test_end > t_days:
            break
        windows.append(EstimationWindow(start, train_end, train_end, test_end))
    return windows
