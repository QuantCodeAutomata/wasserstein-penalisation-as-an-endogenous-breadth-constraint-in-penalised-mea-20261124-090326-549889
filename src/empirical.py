"""Shared rolling-window pipeline for the A-share panel experiments (Tables
1, 5, 6, 7). Every experiment estimates mu_hat and Sigma_hat on a strictly
preceding 60-day window, trades the following holding period, and measures
the "rent" of one portfolio over a benchmark with Newey-West HAC inference
and a stationary-bootstrap confidence interval (Sec 6.1-6.3). Putting the
window loop and the inference step here keeps every table's estimation
window, holding period, and significance test identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import ledoit_wolf_cov
from src.statistics import newey_west_tstat, stationary_bootstrap_mean_ci
from src.windows import EstimationWindow, rolling_windows


def iterate_estimation_windows(
    returns: pd.DataFrame, window: int, hold: int
) -> list[tuple[EstimationWindow, np.ndarray, np.ndarray, np.ndarray]]:
    """For each rolling window: the window bounds, mu_hat (sample mean over
    the trailing `window` days), Sigma_hat (Ledoit-Wolf over the same days),
    and the out-of-sample return matrix of the following `hold` days.

    mu_hat and Sigma_hat use only ``returns[train_start:train_end]``; the
    out-of-sample block ``returns[test_start:test_end]`` is never used to
    estimate anything, which is what prevents look-ahead here.
    """
    arr = returns.to_numpy(dtype=float)
    t_days = arr.shape[0]
    windows = rolling_windows(t_days, window, hold)
    out = []
    for w in windows:
        train = arr[w.train_start : w.train_end]
        test = arr[w.test_start : w.test_end]
        mu_hat = train.mean(axis=0)
        sigma_hat = ledoit_wolf_cov(train)
        out.append((w, mu_hat, sigma_hat, test))
    return out


def oos_daily_returns(weights: np.ndarray, returns_oos: np.ndarray) -> np.ndarray:
    """Daily portfolio returns phi' r_t over the holding block, shape (hold,)."""
    return returns_oos @ weights


def l1_turnover(w_new: np.ndarray, w_old: np.ndarray | None) -> float:
    """sum(|phi_t - phi_{t-1}|); 0 for the first rebalance of a panel (no
    prior position), consistent with charging costs only on actual trades."""
    if w_old is None:
        return 0.0
    return float(np.sum(np.abs(w_new - w_old)))


def rent_stats(
    diff_series: np.ndarray,
    nw_lag: int,
    boot_b: int,
    boot_block: int,
    seed: int | None = None,
) -> dict:
    """Mean, Newey-West t-statistic, and stationary-bootstrap 95% CI of a
    series of daily return differences (e.g. robust minus a benchmark),
    with a flag for "Newey-West |t| > 2 and the bootstrap CI excludes zero"
    (Sec 6.1-6.3, the panel-level significance criterion of Tables 1, 6, 7).
    """
    diff_series = np.asarray(diff_series, dtype=float)
    assert diff_series.ndim == 1 and diff_series.size > 1
    mean, tstat = newey_west_tstat(diff_series, nw_lag)
    ci_lo, ci_hi = stationary_bootstrap_mean_ci(diff_series, boot_b, boot_block, seed=seed)
    significant = bool(abs(tstat) > 2.0 and (ci_lo > 0.0 or ci_hi < 0.0))
    return {
        "mean": mean,
        "tstat": tstat,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "significant": significant,
    }


def r0_mean_return(mu_hat: np.ndarray) -> float:
    """r0 = 1'mu_hat / N, the return-floored problem's floor (Tables 5-7)."""
    return float(np.mean(mu_hat))
