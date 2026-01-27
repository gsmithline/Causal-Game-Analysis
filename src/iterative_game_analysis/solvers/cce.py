"""Coarse Correlated Equilibrium (CCE) solver with Min KL-Divergence to Max-Aff-Ent.

Finds the CCE that is closest (in KL-divergence) to the maximum affinity
entropy joint distribution.

References:
    - Aumann (1987) - Correlated Equilibrium
    - Liu, Gemp, Marris, Piliouras, Lanctot (2025) - "Re-Evaluating Open-Ended
      Evaluation Of Large Language Models", ICLR 2025.
      https://arxiv.org/abs/2502.20170
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cvxpy as cp
import numpy as np

from iterative_game_analysis.solvers.base import register_solver
from iterative_game_analysis.solvers.lle import max_affinity_entropy, _payoff_tensor_from_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

EPSILON = 1e-10


def _build_cce_constraints(
    pt: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Build CCE incentive constraint matrix.

    For CCE, a player commits to following recommendations before seeing them.
    The constraint is: for each player p and deviation action a',
    E[u_p(recommended) - u_p(a')] >= 0

    Args:
        pt: Payoff tensor of shape (num_players, *action_dims)

    Returns:
        Constraint matrix where each row is a constraint, and
        constraint @ x <= 0 enforces CCE (note the sign flip).
    """
    npl = pt.shape[0]
    na = pt.shape[1:]
    total_actions = np.prod(na)

    constraints = []

    for player in range(npl):
        n_actions_p = na[player]

        for deviation_action in range(n_actions_p):
            # Constraint: E[u_p(s_p, s_{-p})] >= E[u_p(deviation, s_{-p})]
            # Rewritten: E[u_p(deviation, s_{-p}) - u_p(s_p, s_{-p})] <= 0

            constraint = np.zeros(total_actions)

            # Iterate over all joint action profiles
            for flat_idx in range(total_actions):
                joint_action = np.unravel_index(flat_idx, na)

                # Get player's recommended action
                recommended_action = joint_action[player]

                # Get payoff for recommended action
                u_recommended = pt[(player,) + joint_action]

                # Get payoff for deviation action
                deviation_joint = list(joint_action)
                deviation_joint[player] = deviation_action
                u_deviation = pt[(player,) + tuple(deviation_joint)]

                # Coefficient: u_deviation - u_recommended
                # We want: sum of (u_deviation - u_recommended) * x <= 0
                constraint[flat_idx] = u_deviation - u_recommended

            constraints.append(constraint)

    return np.array(constraints)


def solve_cce_min_kl(
    pt: NDArray[np.floating],
    target_joint: Optional[NDArray[np.floating]] = None,
    verbose: bool = False
) -> tuple[NDArray[np.floating], list[NDArray[np.floating]], float]:
    """Solve for CCE with minimum KL-divergence to target distribution.

    Args:
        pt: Payoff tensor of shape (num_players, *action_dims)
        target_joint: Target joint distribution (default: max-aff-ent product)
        verbose: Whether to print solver output

    Returns:
        Tuple of:
            - joint: CCE joint distribution over action profiles
            - marginals: List of marginal distributions per player
            - kl_div: KL-divergence from CCE to target
    """
    npl = pt.shape[0]
    na = pt.shape[1:]
    total_actions = int(np.prod(na))

    # Compute max affinity entropy if no target provided
    if target_joint is None:
        max_aff_ent_xs = max_affinity_entropy(pt)
        # Compute product distribution
        target_joint = max_aff_ent_xs[0]
        for i in range(1, npl):
            target_joint = np.outer(target_joint, max_aff_ent_xs[i]).reshape(
                target_joint.shape + (na[i],)
            )

    target_flat = np.ravel(target_joint)
    target_flat = np.clip(target_flat, EPSILON, 1.0)
    target_flat = target_flat / target_flat.sum()

    # Build CCE constraints
    constraints_matrix = _build_cce_constraints(pt)

    # CVX optimization
    x = cp.Variable(total_actions, nonneg=True)

    # Objective: minimize KL-divergence from x to target
    # KL(x || target) = sum(x * log(x / target))
    objective = cp.Minimize(cp.sum(cp.kl_div(x, target_flat)))

    # Constraints
    cons = [
        cp.sum(x) == 1,  # Valid distribution
        constraints_matrix @ x <= EPSILON,  # CCE incentive constraints
    ]

    prob = cp.Problem(objective, cons)

    try:
        result = prob.solve(solver=cp.SCS, verbose=verbose)
    except Exception:
        # Fallback to ECOS
        try:
            result = prob.solve(solver=cp.ECOS, verbose=verbose)
        except Exception as e:
            raise RuntimeError(f"CCE solver failed: {e}")

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"CCE solver failed with status: {prob.status}")

    # Extract solution
    joint = x.value.reshape(na)
    joint = np.clip(joint, 0, 1)
    joint = joint / joint.sum()

    # Compute marginals
    marginals = _marginals_from_joint(joint)

    return joint, marginals, result


def _marginals_from_joint(joint: NDArray[np.floating]) -> list[NDArray[np.floating]]:
    """Compute marginal distributions from joint distribution.

    Args:
        joint: Joint distribution over action profiles

    Returns:
        List of marginal distributions, one per player
    """
    marginals = []
    for p in range(joint.ndim):
        all_but_p = list(range(joint.ndim))
        all_but_p.pop(p)
        marginal = np.sum(joint, axis=tuple(all_but_p))
        marginals.append(marginal)
    return marginals


def compute_cce_regrets(
    pt: NDArray[np.floating],
    joint: NDArray[np.floating]
) -> dict[str, NDArray[np.floating]]:
    """Compute CCE regrets for a joint distribution.

    Args:
        pt: Payoff tensor
        joint: Joint distribution over action profiles

    Returns:
        Dictionary with:
            - 'per_player': array of max regret per player
            - 'total': sum of regrets
            - 'max': maximum regret across players
    """
    npl = pt.shape[0]
    na = pt.shape[1:]

    constraints = _build_cce_constraints(pt)
    joint_flat = np.ravel(joint)

    # Compute regret for each constraint
    regrets_all = constraints @ joint_flat

    # Group by player
    constraints_per_player = sum(na)  # Approximation for symmetric
    regrets_per_player = []

    idx = 0
    for p in range(npl):
        n_constraints = na[p]
        player_regrets = regrets_all[idx:idx + n_constraints]
        regrets_per_player.append(np.max(np.maximum(0, player_regrets)))
        idx += n_constraints

    return {
        'per_player': np.array(regrets_per_player),
        'total': sum(regrets_per_player),
        'max': max(regrets_per_player),
    }


@register_solver("cce")
class CCESolver:
    """CCE solver with minimum KL-divergence to maximum affinity entropy.

    Finds the Coarse Correlated Equilibrium that is closest (in KL-divergence)
    to the maximum affinity entropy product distribution.

    CCE is a relaxation of Nash equilibrium where players commit to following
    a mediator's recommendations before learning what they are.

    Args:
        verbose: Whether to print solver output (default False)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._last_joint = None
        self._last_kl = None

    def solve(self, payoff_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute CCE marginal strategy for symmetric game.

        Args:
            payoff_matrix: Square payoff matrix for symmetric 2-player game.

        Returns:
            Marginal equilibrium strategy for player 0.
        """
        n = payoff_matrix.shape[0]

        # Handle NaN values
        payoff_matrix = np.array(payoff_matrix, dtype=np.float64)
        if np.isnan(payoff_matrix).any():
            for j in range(payoff_matrix.shape[1]):
                col = payoff_matrix[:, j]
                if np.isnan(col).any():
                    col_mean = np.nanmean(col)
                    if np.isnan(col_mean):
                        col_mean = 0
                    payoff_matrix[np.isnan(col), j] = col_mean

        # Convert to payoff tensor
        pt = _payoff_tensor_from_matrix(payoff_matrix)

        # Solve CCE
        joint, marginals, kl = solve_cce_min_kl(pt, verbose=self.verbose)

        # Store for inspection
        self._last_joint = joint
        self._last_kl = kl

        # Return player 0's marginal (symmetric game)
        strategy = marginals[0]
        strategy = np.clip(strategy, 0, 1)
        strategy = strategy / strategy.sum()

        return strategy

    def solve_full(
        self,
        payoff_matrix: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], list[NDArray[np.floating]], float]:
        """Solve CCE and return full results.

        Args:
            payoff_matrix: Square payoff matrix.

        Returns:
            Tuple of (joint distribution, marginals, KL-divergence)
        """
        pt = _payoff_tensor_from_matrix(payoff_matrix)
        return solve_cce_min_kl(pt, verbose=self.verbose)

    @property
    def last_joint(self) -> Optional[NDArray[np.floating]]:
        """Get the joint distribution from the last solve call."""
        return self._last_joint

    @property
    def last_kl_divergence(self) -> Optional[float]:
        """Get the KL-divergence from the last solve call."""
        return self._last_kl


def is_cce(
    pt: NDArray[np.floating],
    joint: NDArray[np.floating],
    tolerance: float = 1e-6
) -> bool:
    """Check if a joint distribution is a CCE.

    Args:
        pt: Payoff tensor
        joint: Joint distribution to check
        tolerance: Maximum allowed regret

    Returns:
        True if joint is a valid CCE
    """
    regrets = compute_cce_regrets(pt, joint)
    return regrets['max'] <= tolerance
