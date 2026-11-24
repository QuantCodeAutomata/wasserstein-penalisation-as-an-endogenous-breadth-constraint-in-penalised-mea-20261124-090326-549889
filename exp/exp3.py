"""Experiment 3: the Wasserstein radius as a breadth dial (Figure 1).

Solves the penalised robust portfolio (P) of Section 2.2,
    max_phi  phi'mu_hat - lambda*phi'Sigma_hat*phi - sqrt(delta)*||phi||_2
    s.t.     1'phi = 1, phi >= 0,
across the paper's grid of radii delta, on one simulated panel with
N = 255 assets, T = 60 observations and lambda = 10 (Figure 1 caption). By
Proposition 4.2 the optimal N_eff = 1/HHI(phi*) increases monotonically in
delta; this script verifies the monotonicity and reproduces the table
printed under Figure 1.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.generate_synthetic import simulate_factor_returns
from src import portfolio as pf
from src.config import EXP3_DELTA_GRID, EXP3_LAMBDA, EXP3_N_ASSETS, EXP3_N_FACTORS, EXP3_SEED, EXP3_T_OBS

RESULTS_DIR = "results"


def run() -> pd.DataFrame:
    returns, _, _ = simulate_factor_returns(
        n_assets=EXP3_N_ASSETS,
        t_obs=EXP3_T_OBS,
        n_factors=EXP3_N_FACTORS,
        alpha_fraction=0.0,  # Figure 1 uses a constant-mean-free DGP; no sparse alpha is mentioned
        seed=EXP3_SEED,
    )
    mu_hat = returns.mean(axis=0)
    sigma_hat = np.cov(returns, rowvar=False)
    sigma_hat += 1e-8 * np.eye(EXP3_N_ASSETS)  # numerical floor: T < N makes the sample covariance singular

    rows = []
    for delta in EXP3_DELTA_GRID:
        phi = pf.penalized_robust_portfolio(mu_hat, sigma_hat, EXP3_LAMBDA, delta)
        rows.append({"delta": delta, "n_eff": pf.n_eff(phi), "herfindahl": float(np.sum(phi**2))})
    table = pd.DataFrame(rows)
    diffs = np.diff(table["n_eff"].to_numpy())
    assert np.all(diffs >= -1e-6), "N_eff must increase monotonically with delta (Proposition 4.2)"
    return table


def plot(table: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["delta"], table["n_eff"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel(r"Wasserstein radius $\delta$")
    ax.set_ylabel(r"Optimal $N_{eff} = 1/\mathrm{HHI}(\phi^*)$")
    ax.set_title("Effective number of holdings vs. Wasserstein radius (Figure 1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    table = run()
    print(table.to_string(index=False))
    table.to_csv(f"{RESULTS_DIR}/exp3_figure1_table.csv", index=False)
    plot(table, f"{RESULTS_DIR}/exp3_figure1.png")
    print(f"saved {RESULTS_DIR}/exp3_figure1_table.csv and {RESULTS_DIR}/exp3_figure1.png")


if __name__ == "__main__":
    main()
