"""Experiment 5: comparison of three Wasserstein-robust scalarisations on the
same sixty-day estimation windows (Table 5).

For each window of the first six A-share panels (Sec 6.3), this script solves
the penalised problem (P), the exact problem (2), and the return-floored
problem (K) at the same radius delta, together with the breadth-matched ridge
benchmark of Theorem 5.2/Corollary 5.3 evaluated at the penalised solution's
own norm h* = ||phi_P||_2. It reports the median Euclidean norm and implied
N_eff of each solution, the return-floor slack of (2) and (K), the ten-day
out-of-sample gain against the equally weighted portfolio, the pairwise
Euclidean distances between the three scalarisations, and the residual
||phi_P - phi_breadth_matched||_2 that verifies Theorem 5.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.data_loader import load_all_panels
from src import portfolio as pf
from src.config import EXP5_DELTA, EXP5_HOLD_DAYS, EXP5_LAMBDA, EXP5_N_PANELS, PANEL_NAMES, ROLLING_WINDOW_DAYS
from src.empirical import iterate_estimation_windows, oos_daily_returns, r0_mean_return

RESULTS_DIR = "results"


def run() -> pd.DataFrame:
    panel_names = PANEL_NAMES[:EXP5_N_PANELS]
    panels = load_all_panels(panel_names)

    rows = []
    for panel_name, returns_df in panels.items():
        windows = iterate_estimation_windows(returns_df, ROLLING_WINDOW_DAYS, EXP5_HOLD_DAYS)
        for w, mu_hat, sigma_hat, test in windows:
            r0 = r0_mean_return(mu_hat)

            phi_p = pf.penalized_robust_portfolio(mu_hat, sigma_hat, EXP5_LAMBDA, EXP5_DELTA)
            phi_exact = pf.exact_robust_portfolio(mu_hat, sigma_hat, EXP5_LAMBDA, EXP5_DELTA, r0)
            phi_k = pf.return_floored_portfolio(mu_hat, sigma_hat, EXP5_LAMBDA, EXP5_DELTA, r0)
            phi_ew = pf.equally_weighted(sigma_hat.shape[0])

            h_star = float(np.linalg.norm(phi_p, 2))
            phi_bm = pf.breadth_matched_ridge(mu_hat, sigma_hat, EXP5_LAMBDA, EXP5_DELTA, h_star)

            tau = np.sqrt(EXP5_DELTA)
            slack_exact = float(mu_hat @ phi_exact - tau * np.linalg.norm(phi_exact, 2) - r0)
            slack_k = float(mu_hat @ phi_k - tau * np.linalg.norm(phi_k, 2) - r0)

            ret_p = oos_daily_returns(phi_p, test).mean()
            ret_exact = oos_daily_returns(phi_exact, test).mean()
            ret_k = oos_daily_returns(phi_k, test).mean()
            ret_ew = oos_daily_returns(phi_ew, test).mean()

            rows.append(
                {
                    "panel": panel_name,
                    "window_start": w.train_start,
                    "norm2_P": h_star,
                    "norm2_D": float(np.linalg.norm(phi_exact, 2)),
                    "norm2_K": float(np.linalg.norm(phi_k, 2)),
                    "n_eff_P": pf.n_eff(phi_p),
                    "n_eff_D": pf.n_eff(phi_exact),
                    "n_eff_K": pf.n_eff(phi_k),
                    "slack_D": slack_exact,
                    "slack_K": slack_k,
                    "gain_P_vs_EW": (ret_p - ret_ew) * 1e4,
                    "gain_D_vs_EW": (ret_exact - ret_ew) * 1e4,
                    "gain_K_vs_EW": (ret_k - ret_ew) * 1e4,
                    "dist_P_D": float(np.linalg.norm(phi_p - phi_exact, 2)),
                    "dist_P_K": float(np.linalg.norm(phi_p - phi_k, 2)),
                    "dist_D_K": float(np.linalg.norm(phi_exact - phi_k, 2)),
                    "breadth_matched_residual": float(np.linalg.norm(phi_p - phi_bm, 2)),
                }
            )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    scalarisations = {
        "P (penalised)": ("norm2_P", "n_eff_P", None, "gain_P_vs_EW"),
        "D (exact, eq. 2)": ("norm2_D", "n_eff_D", "slack_D", "gain_D_vs_EW"),
        "K (return-floored)": ("norm2_K", "n_eff_K", "slack_K", "gain_K_vs_EW"),
    }
    rows = []
    for label, (norm_col, neff_col, slack_col, gain_col) in scalarisations.items():
        rows.append(
            {
                "scalarisation": label,
                "median_norm2": table[norm_col].median(),
                "median_n_eff": table[neff_col].median(),
                "median_slack": table[slack_col].median() if slack_col else np.nan,
                "median_gain_vs_EW_1e4": table[gain_col].median(),
            }
        )
    summary = pd.DataFrame(rows)
    distances = pd.Series(
        {
            "median_dist_P_D": table["dist_P_D"].median(),
            "median_dist_P_K": table["dist_P_K"].median(),
            "median_dist_D_K": table["dist_D_K"].median(),
            "median_breadth_matched_residual": table["breadth_matched_residual"].median(),
        }
    )
    return summary, distances


def main() -> None:
    table = run()
    print(f"windows computed: {len(table)} (expected {EXP5_N_PANELS * (648 // 60 - 1)} with a 10-day holding step)")
    table.to_csv(f"{RESULTS_DIR}/exp5_table5_windows.csv", index=False)

    summary, distances = summarize(table)
    print(summary.to_string(index=False))
    print(distances.to_string())
    summary.to_csv(f"{RESULTS_DIR}/exp5_table5_summary.csv", index=False)
    distances.to_csv(f"{RESULTS_DIR}/exp5_table5_distances.csv", header=["value"])
    print(f"saved exp5_table5_windows.csv, exp5_table5_summary.csv, exp5_table5_distances.csv in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
