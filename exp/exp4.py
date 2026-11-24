"""Experiment 4: calibration of the Wasserstein radius (Figure 2).

For every 60-day non-overlapping estimation window of the fourteen A-share
panels, computes the dimensionless ratio

    a = sqrt(delta) / s(mu_hat),      s(mu_hat) = max_i mu_hat_i - mean(mu_hat),

for delta in {1e-8, 1e-6, 1e-5, 1e-4} (Section 4). The dashed line a = 1 is
the feasibility boundary of the return-floored problem (K) with p = 1: the
constraint phi'mu_hat - sqrt(delta) >= r0 can only be met by some feasible
phi on the simplex when s(mu_hat) >= sqrt(delta), i.e. a <= 1.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_all_panels
from src.config import EXP4_DELTA_GRID, PANEL_NAMES, ROLLING_WINDOW_DAYS
from src.windows import non_overlapping_windows

RESULTS_DIR = "results"


def run() -> pd.DataFrame:
    panels = load_all_panels(PANEL_NAMES)
    rows = []
    for panel_name, df in panels.items():
        arr = df.to_numpy(dtype=float)
        windows = non_overlapping_windows(arr.shape[0], ROLLING_WINDOW_DAYS)
        for w_idx, w in enumerate(windows):
            mu_hat = arr[w.train_start : w.train_end].mean(axis=0)
            s_mu = float(np.max(mu_hat) - np.mean(mu_hat))
            assert s_mu >= 0.0, "s(mu_hat) must be non-negative by construction"
            for delta in EXP4_DELTA_GRID:
                a_ratio = np.sqrt(delta) / s_mu if s_mu > 0 else np.inf
                rows.append(
                    {"panel": panel_name, "window": w_idx, "delta": delta, "s_mu_hat": s_mu, "a": a_ratio}
                )
    table = pd.DataFrame(rows)
    n_windows = table.groupby("delta").size().iloc[0]
    expected = len(PANEL_NAMES) * (panels[PANEL_NAMES[0]].shape[0] // ROLLING_WINDOW_DAYS)
    assert n_windows == expected, f"expected {expected} windows per delta, got {n_windows}"
    return table


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Lower whisker (min), Q1, median, Q3, upper whisker (max) of a per delta,
    matching the box-plot statistics quoted under Figure 2."""
    grouped = table.groupby("delta")["a"]
    return pd.DataFrame(
        {
            "lower_whisker": grouped.min(),
            "q1": grouped.quantile(0.25),
            "median": grouped.median(),
            "q3": grouped.quantile(0.75),
            "upper_whisker": grouped.max(),
        }
    ).reset_index()


def plot(table: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    deltas = sorted(table["delta"].unique())
    data = [table.loc[table["delta"] == d, "a"].to_numpy() for d in deltas]
    ax.boxplot(data, labels=[f"{d:.0e}" for d in deltas], showfliers=False)
    ax.axhline(1.0, linestyle="--", color="black", label="feasibility boundary a = 1")
    ax.set_xlabel(r"Wasserstein radius $\delta$")
    ax.set_ylabel(r"$a = \sqrt{\delta}/s(\hat{\mu})$")
    ax.set_title("Calibration ratio across fourteen A-share panels (Figure 2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    table = run()
    stats = summary(table)
    print(stats.to_string(index=False))
    table.to_csv(f"{RESULTS_DIR}/exp4_figure2_raw.csv", index=False)
    stats.to_csv(f"{RESULTS_DIR}/exp4_figure2_summary.csv", index=False)
    plot(table, f"{RESULTS_DIR}/exp4_figure2.png")
    print(f"saved exp4_figure2_raw.csv, exp4_figure2_summary.csv, exp4_figure2.png in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
