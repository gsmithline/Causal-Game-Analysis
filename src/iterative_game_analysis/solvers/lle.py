"""Limiting Logit Equilibrium (LLE) solver with Maximum Affinity Entropy.

Implementation based on the QRE (Quantal Response Equilibrium) annealing approach.
The solver follows the QRE correspondence as temperature approaches zero,
converging to a Nash equilibrium refinement.

References:
    - McKelvey & Palfrey (1995) - Quantal Response Equilibria
    - Liu, Gemp, Marris, Piliouras, Lanctot (2025) - "Re-Evaluating Open-Ended
      Evaluation Of Large Language Models", ICLR 2025.
      https://arxiv.org/abs/2502.20170
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import special
from scipy.optimize import minimize

from iterative_game_analysis.solvers.base import register_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray

EPSILON = 1e-10


def _payoff_tensor_from_matrix(payoff_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert symmetric game payoff matrix to payoff tensor.

    Args:
        payoff_matrix: Square payoff matrix M where M[i,j] is row player's
            payoff when playing action i against column player's action j.

    Returns:
        Payoff tensor of shape (2, n, n) where tensor[p] gives player p's payoffs.
        For symmetric games: tensor[0] = M, tensor[1] = M.T
    """
    n = payoff_matrix.shape[0]
    pt = np.zeros((2, n, n), dtype=np.float64)
    pt[0] = payoff_matrix
    pt[1] = payoff_matrix.T
    return pt


def _pt_grad(pt: NDArray[np.floating], x: list[NDArray[np.floating]], player: int) -> NDArray[np.floating]:
    """Compute expected payoff gradient for a player.

    Given strategies x for all players, computes the gradient of player's
    expected payoff with respect to their own strategy.

    Args:
        pt: Payoff tensor of shape (num_players, *action_dims)
        x: List of strategy vectors, one per player
        player: Player index to compute gradient for

    Returns:
        Gradient vector of expected payoffs for each action
    """
    grad_p = np.moveaxis(pt[player], player, 0)
    for p, x_p in enumerate(x[::-1]):
        if p == len(x) - 1 - player:
            continue
        grad_p = np.dot(grad_p, x_p)
    return grad_p


def qre_exploitability(
    logits: NDArray[np.floating],
    target_logits: NDArray[np.floating],
    pt: NDArray[np.floating],
    temperature: float
) -> float:
    """Compute QRE exploitability (convergence measure).

    Measures the gap between the current strategy and the quantal best response
    in the annealed game with KL-divergence regularization toward target distribution.

    Args:
        logits: Concatenated log-odds for all players' strategies
        target_logits: Target distribution log-odds (e.g., max-aff-ent)
        pt: Payoff tensor
        temperature: Rationality parameter (lower = more rational)

    Returns:
        Sum of exploitability gaps across all players
    """
    npl = pt.shape[0]
    na = pt.shape[1:]

    # Split logits by player
    logits_split = np.split(logits, np.cumsum(na)[:-1])
    target_logits_split = np.split(target_logits, np.cumsum(na)[:-1])

    # Convert to probabilities
    x = [special.softmax(l) for l in logits_split]

    qre_exp_sum = 0.0
    for p, (x_p, target_p) in enumerate(zip(x, target_logits_split)):
        # Clip for numerical stability
        x_p = np.clip(x_p, EPSILON, 1.0)
        log_x_p = np.log(x_p)

        # KL divergence from target
        x_p_kl = np.sum(x_p * (log_x_p - target_p))

        # Expected payoff gradient
        grad_p = _pt_grad(pt, x, p)

        # Quantal best response
        br_logits = grad_p / temperature + target_p
        br_p = special.softmax(br_logits)
        br_p = np.clip(br_p, EPSILON, 1.0)
        log_br_p = np.log(br_p)
        br_p_kl = np.sum(br_p * (log_br_p - target_p))

        # Value gap in the annealed game
        u_p_x = np.dot(grad_p, x_p)
        u_p_dist = u_p_x - temperature * x_p_kl
        u_p_br = np.dot(grad_p, br_p) - temperature * br_p_kl

        qre_exp_sum += u_p_br - u_p_dist

    return qre_exp_sum


def _minimize_qre_exploitability(
    x0: NDArray[np.floating],
    target_logits: NDArray[np.floating],
    pt: NDArray[np.floating],
    temperature: float
) -> tuple[list[NDArray[np.floating]], float]:
    """Find QRE by minimizing exploitability at given temperature.

    Args:
        x0: Initial logits (concatenated for all players)
        target_logits: Target distribution logits
        pt: Payoff tensor
        temperature: Rationality parameter

    Returns:
        Tuple of (strategies per player, final exploitability)
    """
    na = pt.shape[1:]

    def objective(logits):
        return qre_exploitability(logits, target_logits, pt, temperature)

    res = minimize(objective, x0=x0, method='L-BFGS-B')

    # Convert result to strategies
    xf = [special.softmax(l) for l in np.split(res.x, np.cumsum(na)[:-1])]

    return xf, res.fun


def affinity_lle(
    pt: NDArray[np.floating],
    target_dists: list[NDArray[np.floating]],
    temperature_init: float = 1.0,
    gamma: float = 0.99,
    num_anneals: int = 500,
    verbose: bool = False
) -> NDArray[np.floating]:
    """Compute Limiting Logit Equilibrium via temperature annealing.

    Starting from a target distribution (e.g., max-affinity-entropy), follows
    the QRE correspondence as temperature decreases toward zero.

    Args:
        pt: Payoff tensor of shape (num_players, *action_dims)
        target_dists: Target distributions per player to anneal from
        temperature_init: Starting temperature (default 1.0)
        gamma: Temperature decay rate per iteration (default 0.99)
        num_anneals: Number of annealing steps (default 500)
        verbose: Whether to print progress

    Returns:
        Final strategy profile (concatenated across players)
    """
    x0 = np.concatenate(target_dists)
    x0 = np.clip(x0, EPSILON, 1.0)
    x0 = x0 / x0.sum()  # Ensure valid distribution
    target_logits = np.log(x0)

    temperature = temperature_init
    xs = np.zeros((num_anneals + 1, len(x0)))
    xs[0] = x0

    for t_idx in range(num_anneals):
        if verbose and t_idx % 100 == 0:
            print(f"LLE annealing: step {t_idx}, temperature={temperature:.6f}")

        current_logits = np.log(np.clip(xs[t_idx], EPSILON, 1.0))
        xf, exp = _minimize_qre_exploitability(current_logits, target_logits, pt, temperature)
        xs[t_idx + 1] = np.concatenate(xf)
        temperature *= gamma

    return xs[-1]


def max_affinity_entropy(
    pt: NDArray[np.floating],
    iterations: int = 100
) -> list[NDArray[np.floating]]:
    """Compute maximum affinity entropy distribution.

    This provides the target distribution for LLE annealing.
    Uses iterated best response with entropy regularization.

    Args:
        pt: Payoff tensor
        iterations: Number of iterations for convergence

    Returns:
        List of strategy distributions per player
    """
    npl = pt.shape[0]
    na = pt.shape[1:]

    # Initialize with uniform distributions
    x = [np.ones(n) / n for n in na]

    # Iterate to find max entropy fixed point
    for _ in range(iterations):
        new_x = []
        for p in range(npl):
            grad_p = _pt_grad(pt, x, p)
            # Softmax with temperature 1 gives max entropy best response
            new_x.append(special.softmax(grad_p))
        x = new_x

    return x


@register_solver("lle")
class LLESolver:
    """Limiting Logit Equilibrium solver.

    Computes Nash equilibrium by following the QRE correspondence as
    the rationality parameter increases (temperature decreases).

    The solver first computes the maximum affinity entropy distribution,
    then anneals from high to low temperature to find the limiting
    logit equilibrium.

    Args:
        temperature_init: Starting temperature for annealing (default 1.0)
        gamma: Temperature decay rate per step (default 0.99)
        num_anneals: Number of annealing steps (default 500)
        verbose: Whether to print progress (default False)
    """

    def __init__(
        self,
        temperature_init: float = 1.0,
        gamma: float = 0.99,
        num_anneals: int = 500,
        verbose: bool = False
    ):
        self.temperature_init = temperature_init
        self.gamma = gamma
        self.num_anneals = num_anneals
        self.verbose = verbose

    def solve(self, payoff_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute Limiting Logit Equilibrium for symmetric game.

        Args:
            payoff_matrix: Square payoff matrix for symmetric 2-player game.

        Returns:
            Equilibrium mixed strategy.
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

        # Compute max affinity entropy as starting point
        max_aff_ent = max_affinity_entropy(pt)

        # Run LLE annealing
        final_strategy = affinity_lle(
            pt,
            max_aff_ent,
            temperature_init=self.temperature_init,
            gamma=self.gamma,
            num_anneals=self.num_anneals,
            verbose=self.verbose
        )

        # Extract player 0's strategy (symmetric game)
        strategy = final_strategy[:n]

        # Ensure valid probability distribution
        strategy = np.clip(strategy, 0, 1)
        strategy = strategy / strategy.sum()

        return strategy

    def solve_with_target(
        self,
        payoff_matrix: NDArray[np.floating],
        target_dist: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute LLE starting from a custom target distribution.

        Args:
            payoff_matrix: Square payoff matrix.
            target_dist: Custom starting distribution for annealing.

        Returns:
            Equilibrium mixed strategy.
        """
        n = payoff_matrix.shape[0]
        pt = _payoff_tensor_from_matrix(payoff_matrix)

        # Use provided target for both players (symmetric)
        target_dists = [target_dist, target_dist]

        final_strategy = affinity_lle(
            pt,
            target_dists,
            temperature_init=self.temperature_init,
            gamma=self.gamma,
            num_anneals=self.num_anneals,
            verbose=self.verbose
        )

        strategy = final_strategy[:n]
        strategy = np.clip(strategy, 0, 1)
        strategy = strategy / strategy.sum()

        return strategy


def compute_qre_at_temperature(
    payoff_matrix: NDArray[np.floating],
    temperature: float,
    target_dist: Optional[NDArray[np.floating]] = None
) -> NDArray[np.floating]:
    """Compute QRE at a specific temperature.

    Utility function for analyzing the QRE correspondence.

    Args:
        payoff_matrix: Square payoff matrix.
        temperature: Rationality parameter.
        target_dist: Optional target distribution (default: uniform).

    Returns:
        QRE strategy at the given temperature.
    """
    n = payoff_matrix.shape[0]
    pt = _payoff_tensor_from_matrix(payoff_matrix)

    if target_dist is None:
        target_dist = np.ones(n) / n

    target_dists = [target_dist, target_dist]
    x0 = np.concatenate(target_dists)
    target_logits = np.log(np.clip(x0, EPSILON, 1.0))

    xf, _ = _minimize_qre_exploitability(
        np.log(np.clip(x0, EPSILON, 1.0)),
        target_logits,
        pt,
        temperature
    )

    return xf[0]
