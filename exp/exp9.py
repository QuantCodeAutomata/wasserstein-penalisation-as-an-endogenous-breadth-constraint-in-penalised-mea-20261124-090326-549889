"""Experiment 9: known-truth factor model simulation (Section 6.4).

Returns are simulated from a five-factor model r_t = mu + B f_t + eps_t with
a sparse cross-sectional alpha of magnitude c*sigma_i/sqrt(T) on 20% of
N = 255 assets (``data/generate_synthetic.py::simulate_factor_returns``).
Because the population parameters (mu_true, Sigma_true) are known by
construction, the true mean-variance optimum phi* and its "attainable
utility" U_true(phi*) can be computed directly and used as the yardstick for
every estimated strategy's regret U_true(phi*) - U_true(phi_hat), reported as
a percentage of U_true(phi*) (Sec 6.4).

Eight strategies are compared, all evaluated at the *true* mu, Sigma so that
only estimation error -- not population mis-specification -- drives the
regret:

  1. breadth-matched minimum variance: the Theorem 5.2 ridge benchmark at the
     exact robust solution's own realised norm h*;
  2. exact robust portfolio: problem (2) solved at the canonical radius with
     the window's own mean as the return floor;
  3. equally weighted;
  4. robust + EW: an equal blend of (2) and the equally weighted portfolio;
  5. mean shrinkage: ordinary mean-variance on a mean vector shrunk halfway
     to its cross-sectional average (shrinkage weight not specified in the
     paper; see results/RESULTS.md);
  6. penalised robust (P) at a wide radius;
  7. penalised robust (P) at a radius chosen on a held-out validation split
     of the estimation window;
  8. ordinary mean-variance on the sample covariance (no shrinkage).

The design grid varies (N, T, lambda); a heavy-tailed (Student-t3) robustness
check is run on the canonical configuration only (see results/RESULTS.md for
the reduction from the paper's full 16-configuration grid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.generate_synthetic import simulate_factor_returns
from src import portfolio as pf
from src.config import (
    EXP9_ALPHA_FRACTION,
    EXP9_ALPHA_SCALE_C,
    EXP9_CANONICAL_DELTA,
    EXP9_DESIGN_GRID_RUN,
    EXP9_N_FACTORS,
    EXP9_N_REPS_RUN,
    EXP9_SEED,
    EXP9_VALIDATION_DELTA_GRID,
    EXP9_VALIDATION_HOLD_FRACTION,
    EXP9_WIDE_DELTA,
)

RESULTS_DIR = "results"

STRATEGIES = [
    "breadth_matched_mv",
    "exact_robust",
    "equally_weighted",
    "robust_plus_ew",
    "mean_shrinkage",
    "penalized_wide",
    "penalized_validation",
    "sample_cov_mv",
]


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Plain sample covariance (ddof=1), the "sample covariance MV" baseline
    of Sec 6.4, contrasted throughout the paper with Ledoit-Wolf shrinkage."""
    return np.cov(returns, rowvar=False, ddof=1)


def mean_shrinkage_estimate(mu_hat: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """Shrink the sample mean halfway towards its cross-sectional average:
    mu_shrunk = (1-w)*mu_hat + w*mean(mu_hat). The paper's Sec 6.4 names a
    "mean shrinkage" strategy without giving its shrinkage weight; w = 0.5 is
    a documented, symmetric choice recorded in results/RESULTS.md."""
    return (1.0 - weight) * mu_hat + weight * np.mean(mu_hat)


def select_validation_delta(
    returns: np.ndarray, lam: float, delta_grid: list[float], hold_fraction: float
) -> float:
    """Split the T-day window into a train and a validation half, solve the
    penalised robust portfolio (P) on the train half for every candidate
    radius, and pick the radius whose portfolio attains the highest
    mean-variance utility evaluated on the validation half's own mean and
    Ledoit-Wolf covariance (a conventional, documented validation split; the
    paper names "validation-selected radius" without specifying the
    procedure, Sec 6.4)."""
    t_obs = returns.shape[0]
    n_train = int(round(t_obs * (1.0 - hold_fraction)))
    n_train = max(n_train, 5)
    train, val = returns[:n_train], returns[n_train:]
    assert val.shape[0] >= 5, "validation split too small"

    mu_train = train.mean(axis=0)
    sigma_train = pf.ledoit_wolf_cov(train)
    mu_val = val.mean(axis=0)
    sigma_val = pf.ledoit_wolf_cov(val)

    best_delta, best_utility = delta_grid[0], -np.inf
    for delta in delta_grid:
        phi = pf.penalized_robust_portfolio(mu_train, sigma_train, lam, delta)
        utility = pf.utility(mu_val, sigma_val, phi, lam)
        if utility > best_utility:
            best_delta, best_utility = delta, utility
    return best_delta


def one_replication(
    n_assets: int, t_obs: int, lam: float, heavy_tailed: bool, seed: int
) -> dict[str, float]:
    """Simulate one factor-model draw and return every strategy's regret
    U_true(phi*) - U_true(phi_hat), in units of attainable utility U_true(phi*).
    """
    returns, mu_true, sigma_true = simulate_factor_returns(
        n_assets,
        t_obs,
        EXP9_N_FACTORS,
        alpha_fraction=EXP9_ALPHA_FRACTION,
        alpha_scale_c=EXP9_ALPHA_SCALE_C,
        heavy_tailed=heavy_tailed,
        seed=seed,
    )
    mu_hat = returns.mean(axis=0)
    sigma_lw = pf.ledoit_wolf_cov(returns)
    sigma_sample = sample_covariance(returns)
    r0 = float(np.mean(mu_hat))

    phi_true = pf.long_only_mean_variance(mu_true, sigma_true, lam)
    u_true_star = pf.utility(mu_true, sigma_true, phi_true, lam)
    assert u_true_star >= pf.utility(mu_true, sigma_true, pf.equally_weighted(n_assets), lam) - 1e-8, (
        "the true optimum must weakly dominate the equally weighted portfolio under the true parameters"
    )

    phi_robust = pf.exact_robust_portfolio(mu_hat, sigma_lw, lam, EXP9_CANONICAL_DELTA, r0)
    # Theorem 5.2 / Corollary 5.3 match the breadth-matched ridge benchmark to
    # the *penalised* problem (P)'s own realised norm h*, not to the exact
    # problem (2) used for the separate "exact robust" strategy below.
    phi_p_canonical = pf.penalized_robust_portfolio(mu_hat, sigma_lw, lam, EXP9_CANONICAL_DELTA)
    h_star = float(np.linalg.norm(phi_p_canonical, 2))
    phi_bm = pf.breadth_matched_ridge(mu_hat, sigma_lw, lam, EXP9_CANONICAL_DELTA, h_star)
    phi_ew = pf.equally_weighted(n_assets)
    phi_blend = 0.5 * phi_robust + 0.5 * phi_ew
    phi_shrink = pf.long_only_mean_variance(mean_shrinkage_estimate(mu_hat), sigma_lw, lam)
    phi_wide = pf.penalized_robust_portfolio(mu_hat, sigma_lw, lam, EXP9_WIDE_DELTA)
    delta_val = select_validation_delta(
        returns, lam, EXP9_VALIDATION_DELTA_GRID, EXP9_VALIDATION_HOLD_FRACTION
    )
    phi_val = pf.penalized_robust_portfolio(mu_hat, sigma_lw, lam, delta_val)
    phi_sample = pf.long_only_mean_variance(mu_hat, sigma_sample, lam)

    weights = {
        "breadth_matched_mv": phi_bm,
        "exact_robust": phi_robust,
        "equally_weighted": phi_ew,
        "robust_plus_ew": phi_blend,
        "mean_shrinkage": phi_shrink,
        "penalized_wide": phi_wide,
        "penalized_validation": phi_val,
        "sample_cov_mv": phi_sample,
    }
    out = {"u_true_star": u_true_star, "selected_delta_validation": delta_val}
    for name, phi in weights.items():
        regret = u_true_star - pf.utility(mu_true, sigma_true, phi, lam)
        out[f"regret_pct_{name}"] = 100.0 * regret / u_true_star
    return out


def run_grid(design_grid: list[tuple[int, int, float]], n_reps: int, heavy_tailed: bool) -> pd.DataFrame:
    rows = []
    for n_assets, t_obs, lam in design_grid:
        for rep in range(n_reps):
            seed = EXP9_SEED + 1000 * design_grid.index((n_assets, t_obs, lam)) + rep
            result = one_replication(n_assets, t_obs, lam, heavy_tailed, seed)
            result.update({"n_assets": n_assets, "t_obs": t_obs, "lam": lam, "heavy_tailed": heavy_tailed, "rep": rep})
            rows.append(result)
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n_assets, t_obs, lam, heavy_tailed), group in raw.groupby(
        ["n_assets", "t_obs", "lam", "heavy_tailed"], sort=False
    ):
        row = {"n_assets": n_assets, "t_obs": t_obs, "lam": lam, "heavy_tailed": heavy_tailed, "n_reps": len(group)}
        for name in STRATEGIES:
            row[f"median_regret_pct_{name}"] = group[f"regret_pct_{name}"].median()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    raw_light = run_grid(EXP9_DESIGN_GRID_RUN, EXP9_N_REPS_RUN, heavy_tailed=False)
    canonical = EXP9_DESIGN_GRID_RUN[0]
    raw_heavy = run_grid([canonical], EXP9_N_REPS_RUN, heavy_tailed=True)
    raw = pd.concat([raw_light, raw_heavy], ignore_index=True)
    raw.to_csv(f"{RESULTS_DIR}/exp9_section64_raw.csv", index=False)

    summary = summarize(raw)
    summary.to_csv(f"{RESULTS_DIR}/exp9_section64_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"saved exp9_section64_raw.csv, exp9_section64_summary.csv in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
