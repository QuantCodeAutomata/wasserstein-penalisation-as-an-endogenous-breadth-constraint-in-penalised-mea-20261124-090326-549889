"""Inference tools used in the empirical sections of the paper: Newey-West
HAC standard errors (Newey and West, 1987), the stationary bootstrap (Politis
and Romano, 1994), and the Romano-Wolf step-down multiple-testing correction
(Romano and Wolf, 2005). statsmodels provides Newey-West HAC covariance
estimation; the stationary bootstrap and the step-down correction are not
implemented by any installed library and are coded here directly from the
cited papers.
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.sandwich_covariance import cov_hac
import statsmodels.api as sm


def newey_west_tstat(x: np.ndarray, lag: int) -> tuple[float, float]:
    """Mean and Newey-West t-statistic of a series x (e.g. a daily return
    difference), testing H0: mean(x) = 0, with HAC lag `lag`.

    # statsmodels sm.OLS + cov_hac: Newey-West (1987) HAC covariance matrix
    """
    x = np.asarray(x, dtype=float)
    assert x.ndim == 1 and x.size > 1
    exog = np.ones((x.size, 1))
    model = sm.OLS(x, exog).fit()
    cov = cov_hac(model, nlags=lag)
    se = float(np.sqrt(cov[0, 0]))
    mean = float(x.mean())
    tstat = mean / se if se > 0 else np.inf * np.sign(mean)
    return mean, tstat


def _stationary_bootstrap_indices(n: int, expected_block: int, rng: np.random.Generator) -> np.ndarray:
    """One resample of indices 0..n-1 via the stationary bootstrap of Politis
    and Romano (1994): blocks of geometric length with mean `expected_block`,
    wrapping circularly, concatenated until length n is reached.
    """
    p = 1.0 / expected_block
    idx = np.empty(n, dtype=int)
    pos = 0
    while pos < n:
        start = rng.integers(0, n)
        block_len = rng.geometric(p)
        block_len = min(block_len, n - pos)
        for k in range(block_len):
            idx[pos + k] = (start + k) % n
        pos += block_len
    return idx


def stationary_bootstrap_mean_ci(
    x: np.ndarray,
    n_boot: int,
    expected_block: int,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Stationary-bootstrap percentile confidence interval for mean(x)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = x.size
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, expected_block, rng)
        means[b] = x[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def stationary_bootstrap_max_stat_draws(
    series_by_name: dict[str, np.ndarray],
    n_boot: int,
    expected_block: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Joint stationary-bootstrap resamples of several equal-length series,
    used to build the Romano-Wolf null distribution of the maximum
    studentised statistic across contrasts sharing the same window index.
    Returns, for each name, the array of bootstrap means (length n_boot).
    """
    names = list(series_by_name)
    n = len(series_by_name[names[0]])
    for name in names:
        assert len(series_by_name[name]) == n, "all series must share the same length to be resampled jointly"
    rng = np.random.default_rng(seed)
    out = {name: np.empty(n_boot) for name in names}
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, expected_block, rng)
        for name in names:
            out[name][b] = series_by_name[name][idx].mean()
    return out


def romano_wolf_stepdown(
    series_by_name: dict[str, np.ndarray],
    nw_lag: int,
    n_boot: int,
    expected_block: int,
    seed: int | None = None,
) -> dict[str, float]:
    """Romano-Wolf (2005) step-down correction for a family of one-sided
    (H0: mean <= 0) contrasts sharing the same window index within one panel.

    For each contrast the observed t-statistic is studentised with the
    Newey-West standard error; the null distribution of the maximum
    studentised statistic over the surviving contrasts is approximated by
    centering stationary-bootstrap resamples at zero and re-studentising them
    with the same (bootstrap) standard error, then stepping down from the
    largest observed statistic. Returns adjusted p-values keyed by name.
    """
    names = list(series_by_name)
    observed_t = {}
    observed_mean = {}
    for name in names:
        mean, t = newey_west_tstat(series_by_name[name], nw_lag)
        observed_mean[name] = mean
        observed_t[name] = t

    boot_means = stationary_bootstrap_max_stat_draws(series_by_name, n_boot, expected_block, seed)
    boot_se = {name: float(np.std(boot_means[name], ddof=1)) for name in names}
    boot_t = {
        name: (boot_means[name] - boot_means[name].mean()) / boot_se[name] if boot_se[name] > 0 else np.zeros(n_boot)
        for name in names
    }

    remaining = sorted(names, key=lambda nm: observed_t[nm], reverse=True)
    adjusted_p = {}
    active = list(remaining)
    while active:
        max_boot = np.max(np.column_stack([boot_t[nm] for nm in active]), axis=1)
        current = active[0]  # largest observed statistic among the still-active contrasts
        adjusted_p[current] = float(np.mean(max_boot >= observed_t[current]))
        active = active[1:]
    # Enforce monotonicity of the step-down procedure (adjusted p-values are non-decreasing
    # as the observed statistic decreases down the ranked list).
    ranked = sorted(names, key=lambda nm: observed_t[nm], reverse=True)
    running_max = 0.0
    for name in ranked:
        running_max = max(running_max, adjusted_p[name])
        adjusted_p[name] = running_max
    return adjusted_p
