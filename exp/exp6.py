"""Experiment 6: ex ante audit of breadth-cap recovery of the robust
portfolio's gain (Table 1).

For each sixty-day estimation window of the first six A-share panels (Sec
6.3), this script fixes a breadth cap on the ordinary mean-variance problem
*before* that window's own data is used to pick it: either an absolute
N_eff target (available from the very first window) or a tracking rule
built only from the realised N_eff of the robust portfolio (P) on *past*
windows of the same panel (never the current window's own breadth). The
norm-capped mean-variance solution is then compared, out of sample over the
following holding period, against the Wasserstein-penalised robust
portfolio (P) solved at the same (lambda, delta). Significance of the
per-panel mean return difference is assessed with the Newey-West /
stationary-bootstrap criterion of Sec 6.1, and the eight breadth-cap rules
of Table 1 are jointly Romano-Wolf corrected within each (panel, delta).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.data_loader import load_all_panels
from src import portfolio as pf
from src.config import (
    EXP6_BOOTSTRAP_B,
    EXP6_BOOTSTRAP_BLOCK,
    EXP6_DELTA_GRID,
    EXP6_HOLD_DAYS,
    EXP6_LAMBDA,
    EXP6_N_PANELS,
    EXP6_NEFF_TARGETS,
    EXP6_NW_LAG,
    EXP6_TRACKING_RULES,
    PANEL_NAMES,
    ROLLING_WINDOW_DAYS,
)
from src.empirical import iterate_estimation_windows, oos_daily_returns, rent_stats
from src.statistics import romano_wolf_stepdown

RESULTS_DIR = "results"

ABSOLUTE_NAMES = [f"Neff={n_target}" for n_target in EXP6_NEFF_TARGETS]
COMPARISON_NAMES = ABSOLUTE_NAMES + list(EXP6_TRACKING_RULES)


def compute_tracked_cap(history: list[float], rule: str) -> float:
    """h_bar_t for one of the Table 1 tracking rules, built only from the
    realised N_eff,s of the robust portfolio (P) on *past* windows s < t of
    the same panel (``history``). The caller must never include the current
    window's own N_eff in ``history``: this is what makes the resulting cap
    "ex ante".

    Raises ``ValueError`` if ``history`` is empty (there is no past window
    yet, so no tracking cap can be formed) or if ``rule`` is unknown.
    """
    if not history:
        raise ValueError("compute_tracked_cap requires at least one past window of history")
    if rule == "mean_past":
        return float(np.mean(history))
    if rule == "median_past":
        return float(np.median(history))
    if rule == "previous":
        return float(history[-1])
    if rule == "last5":
        return float(np.mean(history[-5:]))
    if rule == "last10":
        return float(np.mean(history[-10:]))
    raise ValueError(f"unknown tracking rule: {rule!r}")


def _process_panel(panel_name: str, returns_df: pd.DataFrame) -> list[dict]:
    """Per-panel rows of Table 1: for each delta, the Newey-West/bootstrap
    rent statistics and the Romano-Wolf adjusted p-value of each of the
    eight breadth-cap comparisons, computed on this panel's own windows.
    """
    windows = iterate_estimation_windows(returns_df, ROLLING_WINDOW_DAYS, EXP6_HOLD_DAYS)
    rows: list[dict] = []

    for delta in EXP6_DELTA_GRID:
        history: list[float] = []
        diffs: dict[str, list[float]] = {name: [] for name in COMPARISON_NAMES}

        for t, (_w, mu_hat, sigma_hat, test) in enumerate(windows):
            phi_robust = pf.penalized_robust_portfolio(mu_hat, sigma_hat, EXP6_LAMBDA, delta)
            n_eff_t = pf.n_eff(phi_robust)
            ret_robust_t = float(oos_daily_returns(phi_robust, test).mean())

            for n_target, name in zip(EXP6_NEFF_TARGETS, ABSOLUTE_NAMES):
                h_cap = 1.0 / np.sqrt(n_target)
                phi_cap = pf.norm_capped_mv(mu_hat, sigma_hat, EXP6_LAMBDA, h_cap)
                ret_cap = float(oos_daily_returns(phi_cap, test).mean())
                diffs[name].append(ret_robust_t - ret_cap)

            if t > 0:
                for rule in EXP6_TRACKING_RULES:
                    h_bar = compute_tracked_cap(history, rule)
                    h_cap = 1.0 / np.sqrt(h_bar)
                    phi_track = pf.norm_capped_mv(mu_hat, sigma_hat, EXP6_LAMBDA, h_cap)
                    ret_track = float(oos_daily_returns(phi_track, test).mean())
                    diffs[rule].append(ret_robust_t - ret_track)

            history.append(n_eff_t)

        # Romano-Wolf jointly corrects the eight comparisons, which requires
        # equal-length series: the absolute-target series are one window
        # longer than the tracking-rule series (which skip window 0), so
        # their first element is dropped only for this joint test.
        rw_series = {
            name: (np.asarray(diffs[name][1:]) if name in ABSOLUTE_NAMES else np.asarray(diffs[name]))
            for name in COMPARISON_NAMES
        }
        adjusted_p = romano_wolf_stepdown(
            rw_series,
            nw_lag=EXP6_NW_LAG,
            n_boot=EXP6_BOOTSTRAP_B,
            expected_block=EXP6_BOOTSTRAP_BLOCK,
            seed=0,
        )

        for name in COMPARISON_NAMES:
            diff_series = np.asarray(diffs[name])
            stats = rent_stats(
                diff_series,
                nw_lag=EXP6_NW_LAG,
                boot_b=EXP6_BOOTSTRAP_B,
                boot_block=EXP6_BOOTSTRAP_BLOCK,
                seed=0,
            )
            rows.append(
                {
                    "panel": panel_name,
                    "delta": delta,
                    "comparison": name,
                    "mean_diff_1e4": stats["mean"] * 1e4,
                    "tstat": stats["tstat"],
                    "ci_lo_1e4": stats["ci_lo"] * 1e4,
                    "ci_hi_1e4": stats["ci_hi"] * 1e4,
                    "significant": stats["significant"],
                    "romano_wolf_adj_p": adjusted_p[name],
                }
            )

    return rows


def run() -> pd.DataFrame:
    panel_names = PANEL_NAMES[:EXP6_N_PANELS]
    panels = load_all_panels(panel_names)

    rows: list[dict] = []
    for panel_name, returns_df in panels.items():
        rows.extend(_process_panel(panel_name, returns_df))
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (delta, comparison), group in raw.groupby(["delta", "comparison"], sort=False):
        rows.append(
            {
                "delta": delta,
                "comparison": comparison,
                "median_mean_diff_1e4": group["mean_diff_1e4"].median(),
                "n_significant": int(group["significant"].sum()),
                "n_panels": int(group["panel"].nunique()),
            }
        )
    summary = pd.DataFrame(rows)
    # Keep a stable (delta, comparison) order matching Table 1's layout.
    comparison_order = {name: i for i, name in enumerate(COMPARISON_NAMES)}
    summary["_order"] = summary["comparison"].map(comparison_order)
    summary = summary.sort_values(["delta", "_order"]).drop(columns="_order").reset_index(drop=True)
    return summary


def main() -> None:
    raw = run()
    raw.to_csv(f"{RESULTS_DIR}/exp6_table1_raw.csv", index=False)

    summary = summarize(raw)
    print(summary.to_string(index=False))
    summary.to_csv(f"{RESULTS_DIR}/exp6_table1_summary.csv", index=False)
    print(f"saved exp6_table1_raw.csv, exp6_table1_summary.csv in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
