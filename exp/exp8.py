"""Experiment 8: disjoint subperiods with transaction costs (Table 7).

Repeats the Table-6-style rent comparison of the robust penalised portfolio
(problem (P)) against the breadth-matched ridge, return-floored, and equally
weighted benchmarks, separately on the two disjoint continuous A-share panels
named in Sec 6.3 (the first and the last of the fourteen panel directories),
now also charging a transaction cost of thirty basis points on L1 portfolio
turnover at every rebalance, amortised over the five-day holding period and
netted from the daily out-of-sample returns before the rent is computed
(Sec 6.3, "disjoint subperiods with transaction costs").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.data_loader import load_all_panels
from src import portfolio as pf
from src.config import (
    EXP8_BOOTSTRAP_B,
    EXP8_BOOTSTRAP_BLOCK,
    EXP8_DELTA_GRID,
    EXP8_HOLD_DAYS,
    EXP8_LAMBDA,
    EXP8_NW_LAG,
    EXP8_PANEL_A,
    EXP8_PANEL_B,
    EXP8_TC_BPS,
    ROLLING_WINDOW_DAYS,
)
from src.empirical import iterate_estimation_windows, l1_turnover, oos_daily_returns, r0_mean_return, rent_stats

RESULTS_DIR = "results"

# Benchmarks compared against the robust penalised portfolio, Table 6/7.
BENCHMARKS = ["breadth_matched", "return_matched", "equally_weighted"]
STRATEGIES = ["robust", *BENCHMARKS]


def net_daily_returns(gross: np.ndarray, turnover: float, tc_bps: float, hold: int) -> np.ndarray:
    """Net a constant daily transaction cost off a block of gross OOS daily
    returns: the one-time L1-turnover cost ``tc_bps/10_000 * turnover`` is
    amortised (divided) evenly over the ``hold`` holding days and subtracted
    from every day in the block (Sec 6.3, "amortized over the five-day
    holding period"). ``turnover == 0`` (e.g. the first rebalance of a panel,
    with no prior position) leaves the gross returns unchanged.
    """
    gross = np.asarray(gross, dtype=float)
    tc_fraction = (tc_bps / 10_000.0) * turnover
    daily_cost = tc_fraction / hold
    return gross - daily_cost


def evaluate_window(
    mu_hat: np.ndarray,
    sigma_hat: np.ndarray,
    test: np.ndarray,
    lam: float,
    delta: float,
    hold: int,
    tc_bps: float,
    prev_weights: dict[str, np.ndarray | None],
) -> tuple[dict[str, dict[str, object]], dict[str, np.ndarray]]:
    """Solve the four strategies on one estimation window and net transaction
    costs off their out-of-sample daily returns.

    Returns a per-strategy results dict (weights, gross and net daily return
    arrays of shape (hold,), and the realised L1 turnover versus
    ``prev_weights``) and the updated ``prev_weights`` to carry into the next
    window's call.
    """
    r0 = r0_mean_return(mu_hat)
    phi_robust = pf.penalized_robust_portfolio(mu_hat, sigma_hat, lam, delta)
    h_star = float(np.linalg.norm(phi_robust, 2))
    phi_bm = pf.breadth_matched_ridge(mu_hat, sigma_hat, lam, delta, h_star)
    try:
        phi_rm = pf.return_floored_portfolio(mu_hat, sigma_hat, lam, delta, r0)
    except pf.InfeasibleProblem:
        # Sec 4 (Figure 2): the return-floored constraint phi'mu - sqrt(delta)
        # ||phi||_2 >= r0 can become infeasible on the simplex once the radius
        # is wide relative to the window's own return spread. Such windows are
        # dropped from the return-matched comparison and counted (see
        # results/RESULTS.md) rather than masked by a fallback portfolio.
        phi_rm = None
    phi_ew = pf.equally_weighted(sigma_hat.shape[0])

    weights_by_strategy = {
        "robust": phi_robust,
        "breadth_matched": phi_bm,
        "return_matched": phi_rm,
        "equally_weighted": phi_ew,
    }

    results: dict[str, dict[str, object]] = {}
    new_prev_weights: dict[str, np.ndarray] = {}
    for name, weights in weights_by_strategy.items():
        if weights is None:
            continue  # return-matched window dropped: infeasible return floor
        gross = oos_daily_returns(weights, test)
        turnover = l1_turnover(weights, prev_weights.get(name))
        net = net_daily_returns(gross, turnover, tc_bps, hold)
        results[name] = {"weights": weights, "gross": gross, "net": net, "turnover": turnover}
        new_prev_weights[name] = weights
    if "return_matched" not in results:
        new_prev_weights["return_matched"] = prev_weights.get("return_matched")
    return results, new_prev_weights


def run() -> pd.DataFrame:
    panel_labels = {EXP8_PANEL_A: "A", EXP8_PANEL_B: "B"}
    panels = load_all_panels([EXP8_PANEL_A, EXP8_PANEL_B])

    rows = []
    for panel_name, label in panel_labels.items():
        returns_df = panels[panel_name]
        for delta in EXP8_DELTA_GRID:
            windows = iterate_estimation_windows(returns_df, ROLLING_WINDOW_DAYS, EXP8_HOLD_DAYS)

            prev_weights: dict[str, np.ndarray | None] = {name: None for name in STRATEGIES}
            gross_rents: dict[str, list[float]] = {bench: [] for bench in BENCHMARKS}
            net_rents: dict[str, list[float]] = {bench: [] for bench in BENCHMARKS}
            n_windows = 0
            n_infeasible_return_matched = 0

            for _w, mu_hat, sigma_hat, test in windows:
                n_windows += 1
                results, prev_weights = evaluate_window(
                    mu_hat, sigma_hat, test, EXP8_LAMBDA, delta, EXP8_HOLD_DAYS, EXP8_TC_BPS, prev_weights
                )
                if "return_matched" not in results:
                    n_infeasible_return_matched += 1
                robust_gross_mean = float(np.mean(results["robust"]["gross"]))
                robust_net_mean = float(np.mean(results["robust"]["net"]))
                for bench in BENCHMARKS:
                    if bench not in results:
                        continue
                    gross_rents[bench].append(robust_gross_mean - float(np.mean(results[bench]["gross"])))
                    net_rents[bench].append(robust_net_mean - float(np.mean(results[bench]["net"])))

            if n_infeasible_return_matched:
                print(
                    f"panel {label} delta={delta}: {n_infeasible_return_matched}/{n_windows} "
                    "windows dropped for return_matched (infeasible return floor, Sec 4)"
                )

            for bench in BENCHMARKS:
                for cost_regime, series in (("gross", gross_rents[bench]), ("net", net_rents[bench])):
                    stats = rent_stats(
                        np.asarray(series),
                        nw_lag=EXP8_NW_LAG,
                        boot_b=EXP8_BOOTSTRAP_B,
                        boot_block=EXP8_BOOTSTRAP_BLOCK,
                        seed=0,
                    )
                    rows.append(
                        {
                            "panel_label": label,
                            "delta": delta,
                            "benchmark": bench,
                            "cost_regime": cost_regime,
                            "mean_rent_1e4": stats["mean"] * 1e4,
                            "tstat": stats["tstat"],
                            "significant": stats["significant"],
                        }
                    )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long gross/net rows into one row per (panel_label, delta,
    benchmark) with side-by-side gross and net columns, convenient for
    comparison to the paper's Table 7."""
    index_cols = ["panel_label", "delta", "benchmark"]
    pivot_mean = raw.pivot_table(index=index_cols, columns="cost_regime", values="mean_rent_1e4")
    pivot_tstat = raw.pivot_table(index=index_cols, columns="cost_regime", values="tstat")
    summary = pd.DataFrame(
        {
            "gross_mean_rent_1e4": pivot_mean["gross"],
            "gross_tstat": pivot_tstat["gross"],
            "net_mean_rent_1e4": pivot_mean["net"],
            "net_tstat": pivot_tstat["net"],
        }
    ).reset_index()
    return summary.sort_values(index_cols).reset_index(drop=True)


def main() -> None:
    raw = run()
    raw.to_csv(f"{RESULTS_DIR}/exp8_table7_raw.csv", index=False)

    summary = summarize(raw)
    summary.to_csv(f"{RESULTS_DIR}/exp8_table7_summary.csv", index=False)

    print(summary.to_string(index=False))
    # Paper Sec 6.3 explicitly reports the breadth-matched rent net of costs
    # for the later panel (B); highlight it alongside the general summary.
    panel_b_bm_net = summary[(summary["panel_label"] == "B") & (summary["benchmark"] == "breadth_matched")]
    print("\nPanel B breadth-matched rent, net of transaction costs:")
    print(panel_b_bm_net[["delta", "net_mean_rent_1e4", "net_tstat"]].to_string(index=False))
    print(f"saved exp8_table7_raw.csv, exp8_table7_summary.csv in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
