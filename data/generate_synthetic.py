"""Synthetic data generators for the simulation experiments of the paper.

Two generators are used:

* ``simulate_factor_returns`` draws returns from a five-factor model
  r_t = mu + B f_t + eps_t with a sparse cross-sectional alpha, the
  known-truth design of Section 6.4 ("Known-truth simulation"), reused
  without the alpha term for the radius-dial illustration of Figure 1
  (Section 4, "simulated data with N = 255, T = 60, lambda = 10").
* ``student_t_innovations`` rescales Student-t(3) draws to unit variance for
  the heavy-tailed robustness check of Section 6.4.
"""

from __future__ import annotations

import numpy as np


def simulate_factor_returns(
    n_assets: int,
    t_obs: int,
    n_factors: int,
    alpha_fraction: float = 0.0,
    alpha_scale_c: float = 0.0,
    heavy_tailed: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a panel of returns from r_t = mu + B f_t + eps_t.

    Factor loadings B (N x K) are drawn N(0, 1) and scaled so that the factor
    block of the covariance has a realistic magnitude; factor returns f_t are
    N(0, Sigma_f) with Sigma_f = 0.01^2 * I (1% daily factor volatility);
    idiosyncratic variances sigma_i^2 are drawn uniformly in [0.01^2, 0.03^2]
    (1%-3% daily idiosyncratic volatility). None of these magnitudes is given
    numerically in the paper's Section 6.4; they are a documented, internally
    consistent choice recorded here and in results/RESULTS.md.

    Sparse alpha: on a fraction ``alpha_fraction`` of assets (chosen at
    random), mu_i = alpha_scale_c * sigma_i / sqrt(T) (Sec 6.4, "sparse alpha
    of magnitude c*sigma_i/sqrt(T) on 20% of assets"); mu_i = 0 elsewhere.

    Returns ``(returns, mu_true, sigma_true)`` where ``returns`` has shape
    (T, N), and ``mu_true``, ``sigma_true`` are the population parameters used
    to generate the data (the "known truth" used to compute attainable
    utility U(phi*) in Section 6.4).
    """
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0.0, 1.0, size=(n_assets, n_factors)) * 0.02
    factor_vol = 0.01
    idio_vol = rng.uniform(0.01, 0.03, size=n_assets)

    n_alpha = int(round(alpha_fraction * n_assets))
    alpha_support = rng.choice(n_assets, size=n_alpha, replace=False)
    mu_true = np.zeros(n_assets)
    mu_true[alpha_support] = alpha_scale_c * idio_vol[alpha_support] / np.sqrt(t_obs)

    sigma_true = loadings @ loadings.T * factor_vol**2 + np.diag(idio_vol**2)

    if heavy_tailed:
        factor_draws = student_t_innovations((t_obs, n_factors), rng) * factor_vol
        idio_draws = student_t_innovations((t_obs, n_assets), rng) * idio_vol
    else:
        factor_draws = rng.normal(0.0, factor_vol, size=(t_obs, n_factors))
        idio_draws = rng.normal(0.0, 1.0, size=(t_obs, n_assets)) * idio_vol

    returns = mu_true[None, :] + factor_draws @ loadings.T + idio_draws
    return returns, mu_true, sigma_true


def student_t_innovations(size: tuple[int, ...], rng: np.random.Generator, df: int = 3) -> np.ndarray:
    """Student-t(df) draws rescaled to unit variance: Var[t_df] = df/(df-2)
    for df > 2, so dividing by sqrt(df/(df-2)) gives unit-variance innovations
    (Sec 6.4, "t3 distribution rescaled to unit variance")."""
    assert df > 2, "variance of Student-t is undefined for df <= 2"
    draws = rng.standard_t(df, size=size)
    return draws / np.sqrt(df / (df - 2))
