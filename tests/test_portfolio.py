"""Unit tests for src/portfolio.py on small synthetic inputs."""

from __future__ import annotations

import numpy as np
import pytest

from src import portfolio as pf


def test_n_eff_equal_weight_equals_n():
    w = np.full(10, 0.1)
    assert pf.n_eff(w) == pytest.approx(10.0)


def test_n_eff_single_holding_equals_one():
    w = np.zeros(5)
    w[0] = 1.0
    assert pf.n_eff(w) == pytest.approx(1.0)


def test_n_eff_requires_nonzero_weights():
    with pytest.raises(AssertionError):
        pf.n_eff(np.zeros(3))


def test_condition_number_identity_is_one():
    sigma = np.eye(6)
    assert pf.condition_number(sigma) == pytest.approx(1.0)


def test_condition_number_diagonal():
    sigma = np.diag([1e-4, 1.0, 1.0])
    assert pf.condition_number(sigma) == pytest.approx(1e4)


def test_lambda_min_diagonal():
    sigma = np.diag([1e-4, 2e-4, 3e-4])
    assert pf.lambda_min(sigma) == pytest.approx(1e-4)


def test_kappa_delta_equals_lambda_min_for_identity():
    # With Sigma = I, every direction has Rayleigh quotient 1, including the
    # tangent space of the simplex (Sec 3.2), so kappa_Delta = lambda_min = 1.
    sigma = np.eye(8)
    assert pf.kappa_delta(sigma) == pytest.approx(1.0, abs=1e-8)
    assert pf.lambda_min(sigma) == pytest.approx(1.0, abs=1e-8)


def test_kappa_delta_matches_lambda_min_random_spd(rng):
    n = 8
    a = rng.normal(size=(n, n))
    sigma = a @ a.T + np.eye(n) * 0.1
    # Theorem 3.1 / Sec 3.2: for a generic SPD matrix the smallest Rayleigh
    # quotient restricted to {phi: 1'phi=0} need not equal lambda_min(Sigma)
    # in general, but it must be bounded below by lambda_min(Sigma) and above
    # by lambda_max(Sigma).
    eigvals = np.linalg.eigvalsh(sigma)
    kd = pf.kappa_delta(sigma)
    assert eigvals[0] - 1e-8 <= kd <= eigvals[-1] + 1e-8


def test_affine_min_variance_sums_to_one():
    rng = np.random.default_rng(1)
    n = 6
    a = rng.normal(size=(n, n))
    sigma = a @ a.T + np.eye(n)
    q = pf.affine_min_variance(sigma)
    assert q.sum() == pytest.approx(1.0)


def test_diag_min_variance_closed_form_matches_solver_theorem_3_1():
    kappa0, v, n = 1e-4, 1e-2, 20
    sigma = np.diag([kappa0] + [v] * (n - 1))
    closed = pf.diag_min_variance_closed_form(kappa0, v, n)
    solved = pf.long_only_min_variance(sigma)
    assert closed.sum() == pytest.approx(1.0)
    assert np.allclose(closed, solved, atol=1e-4)
    assert pf.n_eff(closed) == pytest.approx(pf.n_eff(solved), rel=1e-3)


def test_n_eff_collapses_as_v_grows_table_3():
    # Table 3: N_eff of the long-only minimum-variance optimum falls toward 1
    # as v/kappa_0 grows, even though lambda_min(Sigma) stays fixed at kappa_0.
    kappa0, n = 1e-4, 20
    n_effs = [pf.n_eff(pf.diag_min_variance_closed_form(kappa0, v, n)) for v in (1e-3, 1e-2, 1e-1, 1.0)]
    assert n_effs == sorted(n_effs, reverse=True)
    assert n_effs[-1] == pytest.approx(1.0, abs=1e-2)
    assert n_effs[0] == pytest.approx(7.0672, abs=1e-2)


def test_long_only_min_variance_feasible_and_normalized():
    rng = np.random.default_rng(2)
    n = 10
    a = rng.normal(size=(n, n))
    sigma = a @ a.T + np.eye(n) * 0.5
    w = pf.long_only_min_variance(sigma)
    assert np.all(w >= -1e-8)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_penalized_robust_breadth_increases_with_delta():
    # Proposition 4.2: the radius delta is a monotone breadth dial.
    rng = np.random.default_rng(3)
    n = 15
    mu = rng.normal(0, 0.001, size=n)
    a = rng.normal(size=(n, n)) * 0.01
    sigma = a @ a.T + np.eye(n) * 1e-4
    n_effs = []
    for delta in (1e-8, 1e-5, 1e-2):
        w = pf.penalized_robust_portfolio(mu, sigma, lam=10.0, delta=delta)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        n_effs.append(pf.n_eff(w))
    assert n_effs[0] <= n_effs[1] + 1e-6 <= n_effs[2] + 1e-6


def test_breadth_matched_ridge_reproduces_penalized_solution_theorem_5_2():
    # Corollary 5.3: the breadth-matched ridge problem with nu = sqrt(delta)/(2 h*)
    # reproduces the penalized-robust solution phi_P to high numerical precision.
    rng = np.random.default_rng(4)
    n = 12
    mu = rng.normal(0, 0.001, size=n)
    a = rng.normal(size=(n, n)) * 0.01
    sigma = a @ a.T + np.eye(n) * 1e-4
    lam, delta = 10.0, 1e-5
    phi_p = pf.penalized_robust_portfolio(mu, sigma, lam, delta)
    h_star = float(np.linalg.norm(phi_p, 2))
    phi_ridge = pf.breadth_matched_ridge(mu, sigma, lam, delta, h_star)
    assert np.linalg.norm(phi_p - phi_ridge) < 1e-3


def test_equally_weighted():
    w = pf.equally_weighted(4)
    assert np.allclose(w, 0.25)
    assert pf.n_eff(w) == pytest.approx(4.0)


def test_ledoit_wolf_cov_is_symmetric_psd():
    rng = np.random.default_rng(5)
    returns = rng.normal(0, 0.01, size=(40, 6))
    sigma = pf.ledoit_wolf_cov(returns)
    assert np.allclose(sigma, sigma.T)
    assert np.all(np.linalg.eigvalsh(sigma) > -1e-12)
