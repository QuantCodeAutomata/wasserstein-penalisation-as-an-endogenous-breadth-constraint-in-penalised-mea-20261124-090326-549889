"""Experiment 2: structural portfolio quantities on fourteen A-share panels
(Table 4).

For each of the 140 non-overlapping 60-day windows (14 panels x 10 windows,
Sec 6.1), this script computes the Ledoit-Wolf covariance estimate and
reports: the affine minimum-variance portfolio q_aff and its N_eff, the
long-only minimum-variance portfolio and its N_eff and support size, the
condition number and lambda_min(Sigma_hat), and kappa_Delta (the smallest
Rayleigh quotient of Sigma_hat restricted to the simplex's tangent space),
verifying kappa_Delta/lambda_min = 1 in every window (Sec 3.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.data_loader import load_all_panels
from src import portfolio as pf
from src.config import EXP2_N_PANELS_RUN, PANEL_NAMES, ROLLING_WINDOW_DAYS, SUPPORT_SIZE_TOL
from src.windows import non_overlapping_windows

RESULTS_DIR = "results"


def run() -> pd.DataFrame:
    panel_names = PANEL_NAMES[:EXP2_N_PANELS_RUN]
    panels = load_all_panels(panel_names)

    rows = []
    for panel_name, returns_df in panels.items():
        returns = returns_df.to_numpy()
        t_days = returns.shape[0]
        windows = non_overlapping_windows(t_days, ROLLING_WINDOW_DAYS)
        for w in windows:
            window_returns = returns[w.train_start : w.train_end]
            sigma_hat = pf.ledoit_wolf_cov(window_returns)

            q_aff = pf.affine_min_variance(sigma_hat)
            q_long = pf.long_only_min_variance(sigma_hat)

            lam_min = pf.lambda_min(sigma_hat)
            kappa_delta = pf.kappa_delta(sigma_hat)

            rows.append(
                {
                    "panel": panel_name,
                    "window_start": w.train_start,
                    "norm2_q_aff": float(np.linalg.norm(q_aff, 2)),
                    "n_eff_q_aff": pf.n_eff(q_aff),
                    "n_eff_long_only": pf.n_eff(q_long),
                    "support_size": int(np.sum(q_long > SUPPORT_SIZE_TOL)),
                    "condition_number": pf.condition_number(sigma_hat),
                    "lambda_min": lam_min,
                    "kappa_delta": kappa_delta,
                    "kappa_delta_over_lambda_min": kappa_delta / lam_min,
                }
            )
    table = pd.DataFrame(rows)
    ratio = table["kappa_delta_over_lambda_min"]
    assert np.allclose(ratio, 1.0, atol=1e-6), "kappa_Delta must equal lambda_min(Sigma_hat) in every window (Sec 3.2)"
    return table


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "norm2_q_aff", "n_eff_q_aff", "n_eff_long_only", "support_size",
        "condition_number", "lambda_min", "kappa_delta_over_lambda_min",
    ]
    summary = table[cols].quantile([0.10, 0.50, 0.90]).T
    summary.columns = ["p10", "median", "p90"]
    return summary


def main() -> None:
    table = run()
    print(f"windows computed: {len(table)} (expected 140 at full panel count)")
    table.to_csv(f"{RESULTS_DIR}/exp2_table4_windows.csv", index=False)
    summary = summarize(table)
    print(summary.to_string())
    summary.to_csv(f"{RESULTS_DIR}/exp2_table4_summary.csv")
    print(f"saved {RESULTS_DIR}/exp2_table4_windows.csv and {RESULTS_DIR}/exp2_table4_summary.csv")


if __name__ == "__main__":
    main()
