"""Portfolio construction primitives for the Wasserstein-penalisation study.

Every optimisation problem solved here appears explicitly in the paper:
Proposition 2.1 (affine and long-only minimum variance), Theorem 3.1 (the
diagonal witness), problems (P), (2), (K) of Section 2.2 and Theorem 5.2 /
Corollary 5.3 (the breadth-matched ridge benchmark). Quadratic and
second-order-cone programs are solved with cvxpy (CLARABEL solver); the
diagonal closed form of Proposition 2.1 and the effective-number-of-holdings
statistic are implemented directly from the formulas in the paper.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from sklearn.covariance import LedoitWolf

_SOLVER_KWARGS = {"solver": cp.CLARABEL}


def utility(mu: np.ndarray, sigma: np.ndarray, phi: np.ndarray, lam: float) -> float:
    """Mean-variance utility U(phi) = phi'mu - lam*phi'Sigma*phi (Sec 2.1),
    evaluated either in-sample (mu_hat, Sigma_hat) or at the population
    parameters (mu_true, Sigma_true) to obtain the attainable utility of
    Section 6.4's known-truth simulation."""
    phi = np.asarray(phi, dtype=float)
    return float(phi @ mu - lam * phi @ sigma @ phi)


def n_eff(weights: np.ndarray) -> float:
    """Effective number of holdings N_eff(phi) = 1 / sum(phi_i^2) = 1/||phi||_2^2."""
    weights = np.asarray(weights, dtype=float)
    ssq = float(np.sum(weights**2))
    assert ssq > 0, "weights must not be identically zero"
    return 1.0 / ssq


def condition_number(sigma: np.ndarray) -> float:
    """lambda_max(Sigma) / lambda_min(Sigma)."""
    eigvals = np.linalg.eigvalsh(sigma)
    assert eigvals[0] > 0, "Sigma must be positive definite"
    return float(eigvals[-1] / eigvals[0])


def lambda_min(sigma: np.ndarray) -> float:
    """Smallest eigenvalue of Sigma."""
    return float(np.linalg.eigvalsh(sigma)[0])


def kappa_delta(sigma: np.ndarray) -> float:
    """Smallest Rayleigh quotient of Sigma restricted to the hyperplane {phi: 1'phi = 0}
    (paper Table 4 / Sec 3.2). Built via an orthonormal basis P of 1^perp and the
    smallest eigenvalue of P'Sigma P, an (N-1)x(N-1) reduced problem.
    """
    n = sigma.shape[0]
    ones = np.ones(n) / np.sqrt(n)
    # Orthonormal complement of span(1) via QR on [ones | identity].
    basis = np.eye(n)[:, 1:]
    basis = basis - np.outer(ones, ones) @ basis
    q, _ = np.linalg.qr(np.column_stack([ones, basis]))
    p = q[:, 1:]  # (N, N-1) orthonormal basis of 1^perp
    reduced = p.T @ sigma @ p
    return float(np.linalg.eigvalsh(reduced)[0])


def affine_min_variance(sigma: np.ndarray) -> np.ndarray:
    """q = Sigma^{-1} 1 / (1' Sigma^{-1} 1), Proposition 2.1 eq. (5)."""
    n = sigma.shape[0]
    ones = np.ones(n)
    inv_ones = np.linalg.solve(sigma, ones)
    return inv_ones / (ones @ inv_ones)


def diag_min_variance_closed_form(kappa0: float, v: float, n: int) -> np.ndarray:
    """Closed-form long-only minimum-variance portfolio on the diagonal witness
    Sigma = diag(kappa0, v, ..., v), Theorem 3.1 proof: q_1 = v/(v+(N-1)kappa0),
    q_j = kappa0/(v+(N-1)kappa0) for j >= 2.
    """
    denom = v + (n - 1) * kappa0
    q = np.full(n, kappa0 / denom)
    q[0] = v / denom
    return q


def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance estimate (Ledoit and Wolf, 2004).

    # skfolio / scikit-learn LedoitWolf: verified via Context7 that
    # sklearn.covariance.LedoitWolf implements the Ledoit-Wolf (2004) shrinkage
    # estimator the paper cites; used directly rather than re-derived.
    """
    return LedoitWolf().fit(returns).covariance_


def long_only_min_variance(sigma: np.ndarray) -> np.ndarray:
    """min phi'Sigma phi s.t. 1'phi = 1, phi >= 0."""
    n = sigma.shape[0]
    phi = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(phi, cp.psd_wrap(sigma))), [cp.sum(phi) == 1])
    prob.solve(**_SOLVER_KWARGS)
    return _finalize(phi, prob)


def long_only_mean_variance(mu: np.ndarray, sigma: np.ndarray, lam: float) -> np.ndarray:
    """Ordinary mean-variance: max phi'mu - lam*phi'Sigma*phi s.t. simplex."""
    n = sigma.shape[0]
    phi = cp.Variable(n, nonneg=True)
    objective = mu @ phi - lam * cp.quad_form(phi, cp.psd_wrap(sigma))
    prob = cp.Problem(cp.Maximize(objective), [cp.sum(phi) == 1])
    prob.solve(**_SOLVER_KWARGS)
    return _finalize(phi, prob)


def equally_weighted(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def penalized_robust_portfolio(mu: np.ndarray, sigma: np.ndarray, lam: float, delta: float) -> np.ndarray:
    """Problem (P): max phi'mu - lam*phi'Sigma*phi - sqrt(delta)*||phi||_2
    s.t. 1'phi = 1, phi >= 0 (paper Sec 2.2, eq. (3) with p = 2)."""
    n = sigma.shape[0]
    tau = np.sqrt(delta)
    phi = cp.Variable(n, nonneg=True)
    objective = mu @ phi - lam * cp.quad_form(phi, cp.psd_wrap(sigma)) - tau * cp.norm(phi, 2)
    prob = cp.Problem(cp.Maximize(objective), [cp.sum(phi) == 1])
    prob.solve(**_SOLVER_KWARGS)
    return _finalize(phi, prob)


def exact_robust_portfolio(mu: np.ndarray, sigma: np.ndarray, lam: float, delta: float, r0: float) -> np.ndarray:
    """Problem (2): min (sqrt(phi'Sigma phi) + sqrt(delta)*||phi||_2)^2
    s.t. 1'phi = 1, phi >= 0, phi'mu - sqrt(delta)*||phi||_2 >= r0.

    The objective is minimised via an epigraph variable s >= sqrt(phi'Sigma phi)
    + sqrt(delta)*||phi||_2 (both terms are convex norms of phi, so s is convex
    in phi), squared monotonically for s >= 0: minimising s^2 is equivalent to
    minimising s because squaring is increasing on s >= 0. lambda plays no
    role in (2): the objective combines only the Wasserstein-robust variance
    and norm terms (paper eq. (2)).
    """
    n = sigma.shape[0]
    tau = np.sqrt(delta)
    phi = cp.Variable(n, nonneg=True)
    sigma_sqrt = _psd_sqrt(sigma)
    s = cp.norm(sigma_sqrt @ phi, 2) + tau * cp.norm(phi, 2)
    constraints = [
        cp.sum(phi) == 1,
        mu @ phi - tau * cp.norm(phi, 2) >= r0,
    ]
    prob = cp.Problem(cp.Minimize(s), constraints)
    prob.solve(**_SOLVER_KWARGS)
    return _finalize(phi, prob)


def return_floored_portfolio(mu: np.ndarray, sigma: np.ndarray, lam: float, delta: float, r0: float) -> np.ndarray:
    """Problem (K): max phi'mu - lam*phi'Sigma*phi s.t. 1'phi=1, phi>=0,
    phi'mu - sqrt(delta)*||phi||_2 >= r0 (paper eq. (4))."""
    n = sigma.shape[0]
    tau = np.sqrt(delta)
    phi = cp.Variable(n, nonneg=True)
    objective = mu @ phi - lam * cp.quad_form(phi, cp.psd_wrap(sigma))
    constraints = [
        cp.sum(phi) == 1,
        mu @ phi - tau * cp.norm(phi, 2) >= r0,
    ]
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(**_SOLVER_KWARGS)
    return _finalize(phi, prob)


def breadth_matched_ridge(mu: np.ndarray, sigma: np.ndarray, lam: float, delta: float, h_star: float) -> np.ndarray:
    """Breadth-matched benchmark of Theorem 5.2 via Corollary 5.3's ridge form:
    max phi'mu - phi'(lam*Sigma + nu*I)phi s.t. simplex, nu = sqrt(delta)/(2*h_star).
    """
    n = sigma.shape[0]
    nu = np.sqrt(delta) / (2.0 * h_star)
    phi = cp.Variable(n, nonneg=True)
    penalty = lam * cp.psd_wrap(sigma) + nu * np.eye(n)
    objective = mu @ phi - cp.quad_form(phi, cp.psd_wrap(penalty))
    prob = cp.Problem(cp.Maximize(objective), [cp.sum(phi) == 1])
    prob.solve(**_SOLVER_KWARGS)
    return _finalize(phi, prob)


def norm_capped_mv(mu: np.ndarray, sigma: np.ndarray, lam: float, h_cap: float) -> np.ndarray:
    """Direct norm-constrained ordinary mean-variance: max phi'mu - lam*phi'Sigma*phi
    s.t. 1'phi=1, phi>=0, ||phi||_2 <= h_cap (paper Sec 5.2/6.2, the norm-cap solve
    used both for Table 1's breadth caps and as an independent check of Theorem 5.2).
    """
    n = sigma.shape[0]
    phi = cp.Variable(n, nonneg=True)
    objective = mu @ phi - lam * cp.quad_form(phi, cp.psd_wrap(sigma))
    constraints = [cp.sum(phi) == 1, cp.norm(phi, 2) <= h_cap]
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(**_SOLVER_KWARGS)
    return _finalize(phi, prob)


def _psd_sqrt(sigma: np.ndarray) -> np.ndarray:
    """Symmetric square root of a positive semidefinite matrix, used to write
    sqrt(phi'Sigma phi) = ||Sigma^{1/2} phi||_2 as a second-order cone term."""
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


class InfeasibleProblem(RuntimeError):
    """Raised when a constrained portfolio problem has no feasible point.

    Section 4 shows the return-floored problem's feasibility depends on the
    ratio a = sqrt(delta)/s(mu_hat) (Figure 2): at a large enough radius the
    return floor and the simplex constraint can be jointly infeasible. This is
    a property of the problem, not a solver defect, so callers that can meet
    an infeasible window (e.g. return-floored or exact-robust solves at a wide
    radius) must catch it explicitly rather than have it masked.
    """


def _finalize(phi: cp.Variable, prob: cp.Problem) -> np.ndarray:
    if prob.status in ("infeasible", "infeasible_inaccurate"):
        raise InfeasibleProblem(f"solver status: {prob.status}")
    assert prob.status in ("optimal", "optimal_inaccurate"), f"solver failed: {prob.status}"
    weights = np.asarray(phi.value, dtype=float)
    weights = np.clip(weights, 0.0, None)  # clip tiny negative solver noise
    total = weights.sum()
    assert total > 0
    return weights / total
