"""Maximum Entropy Nash Equilibrium (MENE) solver.

Implementation based on Zun Li et al. (2024) using MILP formulation
with piecewise linear entropy approximation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

from iterative_game_analysis.solvers.base import register_solver
from iterative_game_analysis.utils import simplex_projection

if TYPE_CHECKING:
    from numpy.typing import NDArray

EPSILON = 1e-6


@register_solver("mene")
class MENESolver:
    """Maximum Entropy Nash Equilibrium solver.

    Uses Mixed Integer Linear Programming (MILP) to find the Nash equilibrium
    that maximizes entropy, providing a unique selection among potentially
    multiple equilibria.

    Args:
        discrete_factors: Number of breakpoints for piecewise linear
            approximation of entropy. Higher values give better approximation
            but slower computation. Default is 100.
    """

    def __init__(self, discrete_factors: int = 2000):
        self.discrete_factors = discrete_factors

    def solve(self, payoff_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute maximum entropy Nash equilibrium.

        Tries MILP with progressively relaxed regret tolerances (1e-6, 1e-5,
        1e-4). If all fail, falls back to entropy-regularized replicator
        dynamics.

        Args:
            payoff_matrix: Square payoff matrix for symmetric 2-player game.

        Returns:
            Equilibrium mixed strategy.

        Raises:
            RuntimeError: If both MILP and replicator dynamics fail.
        """
        # First try the strict tolerance (runs the full MILP)
        try:
            return milp_max_entropy_ne(
                payoff_matrix, self.discrete_factors, regret_tolerance=EPSILON
            )
        except RuntimeError as e:
            milp_err = e

        # MILP failed at 1e-6 — try relaxed tolerances without re-solving
        # by calling with the most relaxed tolerance (the MILP result is
        # deterministic, so we only need to re-solve once)
        for tol in [1e-5, 1e-4]:
            try:
                result = milp_max_entropy_ne(
                    payoff_matrix, self.discrete_factors, regret_tolerance=tol
                )
                warnings.warn(
                    f"MENE solution accepted with relaxed tolerance {tol} "
                    f"(original failure: {milp_err})."
                )
                return result
            except RuntimeError:
                continue

        # Fall back to replicator dynamics with progressive tolerances
        for rd_tol in [1e-5, 1e-4]:
            try:
                result = _replicator_dynamics_mene(
                    payoff_matrix, regret_tolerance=rd_tol
                )
                warnings.warn(
                    f"MILP failed ({milp_err}). "
                    f"Replicator dynamics succeeded at tolerance {rd_tol}."
                )
                return result
            except RuntimeError:
                continue

        raise RuntimeError(
            f"Both MILP ({milp_err}) and replicator dynamics failed at all tolerances."
        )


def milp_max_entropy_ne(
    game_matrix: NDArray[np.floating],
    discrete_factors: int = 2000,
    regret_tolerance: float = EPSILON,
) -> NDArray[np.floating]:
    """Compute maximum entropy Nash equilibrium using MILP.

    Implements the formulation from Zun Li et al.:
        min  Σ σ(π) log σ(π)
        s.t. u* ≥ u_π for all π
             u* - u_π ≤ U · b_π  (big-M constraint)
             σ(π) ≤ 1 - b_π
             σ(π) ≥ 0, Σ σ(π) = 1
             b_π ∈ {0,1}

    Args:
        game_matrix: Square payoff matrix.
        discrete_factors: Number of breakpoints for entropy approximation.
        regret_tolerance: Maximum allowed regret for the solution.

    Returns:
        Nash equilibrium strategy.
    """
    shape = game_matrix.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Payoff matrix must be square")

    game_matrix_np = np.array(game_matrix, dtype=np.float64)

    # Handle NaN values by imputing column means
    if np.isnan(game_matrix_np).any():
        for j in range(game_matrix_np.shape[1]):
            col = game_matrix_np[:, j]
            if np.isnan(col).any():
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):
                    col_mean = 0
                game_matrix_np[np.isnan(col), j] = col_mean

    n = shape[0]
    big_m = np.max(game_matrix_np) - np.min(game_matrix_np)

    # Decision variables
    x = cp.Variable(n)  # Strategy
    u = cp.Variable(1)  # Max utility (u*)
    z = cp.Variable(n)  # Entropy approximation
    b = cp.Variable(n, boolean=True)  # Support indicators

    # Objective: minimize sum of z (approximates -entropy, so minimizing = max entropy)
    obj = cp.Minimize(cp.sum(z))

    # Expected utilities for each action
    u_actions = game_matrix_np @ x

    # Constraints
    constraints = [
        u_actions <= u + EPSILON,  # u* ≥ u_π
        cp.sum(x) == 1,  # Probability constraint
        x >= 0,  # Non-negativity
        u - u_actions <= big_m * b,  # Big-M: if b=0, u = u_π (best response)
        x <= 1 - b,  # If b=1 (not best response), x can be > 0
    ]

    # Piecewise linear approximation of x*log(x)
    k_vals = np.arange(discrete_factors)
    for k in k_vals:
        if k == 0:
            # At x=0, use tangent from 0 to 1/K
            slope = np.log(1 / discrete_factors)
            constraints.append(slope * x <= z)
        else:
            # Linear segment from k/K to (k+1)/K
            k_over_n = k / discrete_factors
            k1_over_n = (k + 1) / discrete_factors
            f_k = k_over_n * np.log(k_over_n)
            f_k1 = k1_over_n * np.log(k1_over_n)
            slope = (f_k1 - f_k) * discrete_factors
            intercept = f_k
            constraints.append(intercept + slope * (x - k_over_n) <= z)

    prob = cp.Problem(obj, constraints)

    # Try ECOS_BB first, fall back to GLPK_MI
    try:
        prob.solve(solver="ECOS_BB")
        if prob.status != "optimal":
            raise ValueError(f"ECOS_BB failed with status: {prob.status}")
    except Exception as e:
        warnings.warn(f"ECOS_BB failed: {e}, trying GLPK_MI")
        try:
            prob.solve(solver="GLPK_MI")
            if prob.status != "optimal":
                raise ValueError(f"GLPK_MI failed with status: {prob.status}")
        except Exception as e2:
            raise RuntimeError(f"Both solvers failed. GLPK_MI error: {e2}") from e2

    # Project to simplex and validate
    ne_strategy = simplex_projection(x.value.reshape(-1))

    # Verify it's actually a Nash equilibrium
    expected_utils = game_matrix_np @ ne_strategy
    nash_value = ne_strategy @ game_matrix_np @ ne_strategy
    max_regret = np.max(expected_utils - nash_value)

    if max_regret > regret_tolerance:
        raise RuntimeError(f"Solution has regret {max_regret:.6f} > {regret_tolerance}")

    return ne_strategy


def _replicator_dynamics_mene(
    game_matrix: NDArray[np.floating],
    regret_tolerance: float = 1e-4,
    iterations: int = 10000,
    lr: float = 0.3,
    entropy_weight: float = 0.01,
    n_random: int = 30,
    pure_concentration: float = 0.9,
) -> NDArray[np.floating]:
    """Find approximate max-entropy NE using entropy-regularized replicator dynamics.

    Runs from multiple starting points and returns the valid solution with
    the highest entropy (to approximate MENE). Starting points include random
    simplex points and points concentrated near each pure strategy.

    Args:
        game_matrix: Square payoff matrix for a symmetric game.
        regret_tolerance: Maximum allowed regret for a valid solution.
        iterations: Maximum number of iterations.
        lr: Learning rate for multiplicative weights update.
        entropy_weight: Entropy regularization strength.
        n_random: Number of random starting points.
        pure_concentration: Mass placed on the dominant action for
            pure-strategy-adjacent starting points.

    Returns:
        Approximate Nash equilibrium strategy with highest entropy.

    Raises:
        RuntimeError: If no starting point produces a valid NE.
    """
    game_matrix = np.asarray(game_matrix, dtype=np.float64)
    n = game_matrix.shape[0]

    # Build starting points
    starts = []

    # 30 random points on the simplex (Dirichlet)
    rng = np.random.default_rng()
    for _ in range(n_random):
        starts.append(rng.dirichlet(np.ones(n)))

    # One point near each pure strategy: 90% on that action, rest uniform
    remaining = (1 - pure_concentration) / max(n - 1, 1)
    for i in range(n):
        s = np.full(n, remaining)
        s[i] = pure_concentration
        starts.append(s / s.sum())

    # Run replicator dynamics from each start, collect valid solutions
    candidates = []
    for s0 in starts:
        sigma = _run_replicator(game_matrix, s0, iterations, lr, entropy_weight)
        sigma = simplex_projection(sigma)

        expected_utils = game_matrix @ sigma
        nash_value = sigma @ game_matrix @ sigma
        max_regret = np.max(expected_utils - nash_value)

        if max_regret <= regret_tolerance:
            entropy = -np.sum(sigma * np.log(sigma + 1e-30))
            candidates.append((entropy, max_regret, sigma))

    if not candidates:
        raise RuntimeError(
            f"Replicator dynamics failed from all {len(starts)} starting points "
            f"(none achieved regret <= {regret_tolerance})"
        )

    # Pick the highest-entropy valid solution
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_entropy, best_regret, best_sigma = candidates[0]
    warnings.warn(
        f"Replicator dynamics found solution with entropy={best_entropy:.4f}, "
        f"regret={best_regret:.6f} (from {len(candidates)}/{len(starts)} valid starts)."
    )
    return best_sigma


def _run_replicator(
    game_matrix: NDArray[np.floating],
    sigma: NDArray[np.floating],
    iterations: int,
    lr: float,
    entropy_weight: float,
) -> NDArray[np.floating]:
    """Run entropy-regularized replicator dynamics from a single starting point."""
    for _ in range(iterations):
        u = game_matrix @ sigma
        u = u + entropy_weight * (1 - np.log(sigma + 1e-10))

        sigma_new = sigma * np.exp(lr * (u - u.max()))
        sigma_new = sigma_new / sigma_new.sum()

        if np.allclose(sigma, sigma_new, atol=1e-10):
            break
        sigma = sigma_new

    return sigma
