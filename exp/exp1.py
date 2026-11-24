"""Experiment 1: digital verification of Theorem 3.1.

Theorem 3.1 states that positive definiteness alone does not bound the
breadth of the long-only mean-variance optimum: for any spectral floor
kappa_0 > 0 and any target breadth L > 1 there is a positive definite
covariance matrix with lambda_min >= kappa_0 whose optimum has N_eff <= L.
This script builds the diagonal witness Sigma = diag(kappa_0, v, ..., v) of
the theorem's proof for N = 20 assets, a constant mean (so the mean-variance
optimum coincides with the minimum-variance portfolio, Proposition 2.1), and
reports lambda_min, the condition number, and N_eff(phi*) as v grows with
kappa_0 = 1e-4 held fixed (Table 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import portfolio as pf
from src.config import EXP1_KAPPA0, EXP1_N_ASSETS, EXP1_V_GRID

RESULTS_DIR = "results"


def run() -> pd.DataFrame:
    rows = []
    for v in EXP1_V_GRID:
        sigma = np.diag([EXP1_KAPPA0] + [v] * (EXP1_N_ASSETS - 1))
        assert np.all(np.linalg.eigvalsh(sigma) > 0), "witness matrix must be positive definite"

        # Constant mean => alpha = 0 => the mean-variance optimum equals the
        # long-only minimum-variance portfolio regardless of the risk-aversion
        # coefficient lambda (Theorem 3.1 setup; lambda is not specified
        # because it drops out of the optimum).
        phi_closed_form = pf.diag_min_variance_closed_form(EXP1_KAPPA0, v, EXP1_N_ASSETS)
        phi_solver = pf.long_only_min_variance(sigma)

        closed_form_n_eff = pf.n_eff(phi_closed_form)
        solver_n_eff = pf.n_eff(phi_solver)
        assert abs(closed_form_n_eff - solver_n_eff) < 1e-3, (
            "closed-form (Proposition 2.1) and numerical solve must agree"
        )

        rows.append(
            {
                "v": v,
                "lambda_min": pf.lambda_min(sigma),
                "condition_number": pf.condition_number(sigma),
                "n_eff_closed_form": closed_form_n_eff,
                "n_eff_solver": solver_n_eff,
            }
        )
    table = pd.DataFrame(rows)
    assert table["lambda_min"].nunique() == 1 and np.isclose(table["lambda_min"].iloc[0], EXP1_KAPPA0), (
        "lambda_min must stay fixed at kappa_0 across the whole grid (Theorem 3.1)"
    )
    assert table["n_eff_closed_form"].is_monotonic_decreasing, (
        "N_eff must fall monotonically as v/kappa_0 grows (Table 3)"
    )
    return table


def main() -> None:
    table = run()
    print(table.to_string(index=False))
    out_path = f"{RESULTS_DIR}/exp1_table3.csv"
    table.to_csv(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
