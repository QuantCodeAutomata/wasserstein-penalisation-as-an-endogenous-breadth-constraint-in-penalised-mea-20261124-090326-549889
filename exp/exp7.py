"""Experiment 7: the rent of distributional robustness under four benchmarks
(Table 6).

Across all fourteen A-share panels (Sec 6.3), for each sixty-day estimation
window followed by a five-day holding period, this script solves the
Wasserstein-penalised robust portfolio (P) at three radii delta and compares
its out-of-sample daily return against four benchmarks evaluated on the same
window:

    (i)   a breadth-matched ridge portfolio with the same realised Euclidean
          norm h* = ||phi_P||_2 (Theorem 5.2/Corollary 5.3);
    (ii)  the return-floored portfolio (K), which imposes the same worst-case
          return guarantee as (P) without penalising the norm directly;
    (iii) the ordinary (non-robust) mean-variance portfolio;
    (iv)  the equally weighted portfolio.

For each panel, delta, and benchmark, the per-window series of "rent"
(robust minus benchmark daily return) is summarised with a Newey-West
t-statistic and a stationary-bootstrap confidence interval (Sec 6.1-6.3,
the panel-level significance criterion). Results are aggregated across
panels as the median of the per-panel mean rent (in units of 1e-4/day) and
the count of panels for which the rent is statistically significant, along
with the median in-sample utility difference between the robust solution
and each benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.data_loader import load_all_panels
from src import portfolio as pf
from src.config import (
    EXP7_BOOTSTRAP_B,
    EXP7_BOOTSTRAP_BLOCK,
    EXP7_DELTA_GRID,
    EXP7_HOLD_DAYS,
    EXP7_LAMBDA,
    EXP7_N_PANELS,
    EXP7_NW_LAG,
    PANEL_NAMES,
    ROLLING_WINDOW_DAYS,
)
from src.empirical import iterate_estimation_windows, oos_daily_returns, r0_mean_return, rent_stats

RESULTS_DIR = "results"

# Maps a benchmark's label to the dict keys produced by evaluate_window for
# its weights, out-of-sample daily return, and in-sample utility.
BENCHMARKS: dict[str, tuple[str, str, str]] = {
    "breadth_matched_ridge": ("phi_bm", "ret_bm", "util_bm"),
    "return_floored": ("phi_rm", "ret_rm", "util_rm"),
    "ordinary_mv": ("phi_mv", "ret_mv", "util_mv"),
    "equally_weighted": ("phi_ew", "ret_ew", "util_ew"),
}


def evaluate_window(
    mu_hat: np.ndarray, sigma_hat: np.ndarray, test: np.ndarray, lam: float, delta: float
) -> dict:
    """Solve the robust portfolio (P) and all four Table 6 benchmarks on one
    estimation window, and evaluate their out-of-sample daily return (mean
    over the holding block) and in-sample utility.
    """
    r0 = r0_mean_return(mu_hat)

    phi_robust = pf.penalized_robust_portfolio(mu_hat, sigma_hat, lam, delta)
    h_star = float(np.linalg.norm(phi_robust, 2))
    phi_bm = pf.breadth_matched_ridge(mu_hat, sigma_hat, lam, delta, h_star)
    phi_rm = pf.return_floored_portfolio(mu_hat, sigma_hat, lam, delta, r0)
    phi_mv = pf.long_only_mean_variance(mu_hat, sigma_hat, lam)
    phi_ew = pf.equally_weighted(sigma_hat.shape[0])

    return {
        "r0": r0,
        "h_star": h_star,
        "phi_robust": phi_robust,
        "phi_bm": phi_bm,
        "phi_rm": phi_rm,
        "phi_mv": phi_mv,
        "phi_ew": phi_ew,
        "ret_robust": float(oos_daily_returns(phi_robust, test).mean()),
        "ret_bm": float(oos_daily_returns(phi_bm, test).mean()),
        "ret_rm": float(oos_daily_returns(phi_rm, test).mean()),
        "ret_mv": float(oos_daily_returns(phi_mv, test).mean()),
        "ret_ew": float(oos_daily_returns(phi_ew, test).mean()),
        "util_robust": pf.utility(mu_hat, sigma_hat, phi_robust, lam),
        "util_bm": pf.utility(mu_hat, sigma_hat, phi_bm, lam),
        "util_rm": pf.utility(mu_hat, sigma_hat, phi_rm, lam),
        "util_mv": pf.utility(mu_hat, sigma_hat, phi_mv, lam),
        "util_ew": pf.utility(mu_hat, sigma_hat, phi_ew, lam),
    }


def run() -> pd.DataFrame:
    """Per-window rent and in-sample utility difference against every
    benchmark, for every (panel, delta) combination. One row per window.
    """
    panel_names = PANEL_NAMES[:EXP7_N_PANELS]
    panels = load_all_panels(panel_names)

    rows = []
    for panel_name, returns_df in panels.items():
        for delta in EXP7_DELTA_GRID:
            windows = iterate_estimation_windows(returns_df, ROLLING_WINDOW_DAYS, EXP7_HOLD_DAYS)
            for w, mu_hat, sigma_hat, test in windows:
                res = evaluate_window(mu_hat, sigma_hat, test, EXP7_LAMBDA, delta)
                row = {"panel": panel_name, "delta": delta, "window_start": w.train_start}
                for benchmark, (_, ret_key, util_key) in BENCHMARKS.items():
                    row[f"rent_{benchmark}"] = res["ret_robust"] - res[ret_key]
                    row[f"abs_util_diff_{benchmark}"] = abs(res["util_robust"] - res[util_key])
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two Table 6 outputs:

    - raw: one row per (panel, delta, benchmark) with the Newey-West mean,
      t-statistic, bootstrap CI, significance flag, and median absolute
      in-sample utility difference restricted to that panel/delta/benchmark.
    - summary: one row per (delta, benchmark), aggregating across panels as
      the median of the per-panel mean rent and the count of significant
      panels, plus the median absolute utility difference across ALL windows
      of ALL panels at that delta (raw units, not scaled by 1e4).
    """
    raw_rows = []
    for panel in table["panel"].unique():
        for delta in table["delta"].unique():
            sub = table[(table["panel"] == panel) & (table["delta"] == delta)]
            for benchmark in BENCHMARKS:
                series = sub[f"rent_{benchmark}"].to_numpy(dtype=float)
                stats = rent_stats(
                    series, nw_lag=EXP7_NW_LAG, boot_b=EXP7_BOOTSTRAP_B, boot_block=EXP7_BOOTSTRAP_BLOCK, seed=0
                )
                raw_rows.append(
                    {
                        "panel": panel,
                        "delta": delta,
                        "benchmark": benchmark,
                        "mean_rent_1e4": stats["mean"] * 1e4,
                        "tstat": stats["tstat"],
                        "ci_lo_1e4": stats["ci_lo"] * 1e4,
                        "ci_hi_1e4": stats["ci_hi"] * 1e4,
                        "significant": stats["significant"],
                        "median_abs_utility_diff": sub[f"abs_util_diff_{benchmark}"].median(),
                    }
                )
    raw_df = pd.DataFrame(raw_rows)

    summary_rows = []
    for delta in table["delta"].unique():
        for benchmark in BENCHMARKS:
            sub = raw_df[(raw_df["delta"] == delta) & (raw_df["benchmark"] == benchmark)]
            all_windows = table[table["delta"] == delta][f"abs_util_diff_{benchmark}"]
            summary_rows.append(
                {
                    "delta": delta,
                    "benchmark": benchmark,
                    "median_rent_1e4": sub["mean_rent_1e4"].median(),
                    "n_significant": int(sub["significant"].sum()),
                    "n_panels": int(len(sub)),
                    "median_abs_utility_diff": all_windows.median(),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    return raw_df, summary_df


def main() -> None:
    table = run()
    raw_df, summary_df = summarize(table)

    raw_df.to_csv(f"{RESULTS_DIR}/exp7_table6_raw.csv", index=False)
    summary_df.to_csv(f"{RESULTS_DIR}/exp7_table6_summary.csv", index=False)

    print(summary_df.to_string(index=False))
    print(f"saved exp7_table6_raw.csv, exp7_table6_summary.csv in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
