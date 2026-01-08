"""Maximum Entropy Nash Equilibrium (MENE) solver.

Implementation based on Zun Li et al. (2024) using MILP formulation
with piecewise linear entropy approximation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

from causal_game_analysis.solvers.base import register_solver
from causal_game_analysis.utils import simplex_projection

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

    def __init__(self, discrete_factors: int = 100):
        self.discrete_factors = discrete_factors

    def solve(self, payoff_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute maximum entropy Nash equilibrium.

        Args:
            payoff_matrix: Square payoff matrix for symmetric 2-player game.

        Returns:
            Equilibrium mixed strategy.

        Raises:
            RuntimeError: If optimization fails or result is not a valid NE.
        """
        return milp_max_entropy_ne(payoff_matrix, self.discrete_factors)


def milp_max_entropy_ne(
    game_matrix: NDArray[np.floating], discrete_factors: int = 100
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

    if max_regret > EPSILON:
        raise RuntimeError(f"Solution has regret {max_regret:.6f} > {EPSILON}")

    return ne_strategy
