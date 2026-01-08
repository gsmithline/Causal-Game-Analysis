"""Utility functions for causal game analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def simplex_projection(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Project a vector onto the probability simplex.

    Uses the algorithm from Duchi et al. (2008) for efficient projection.

    Args:
        x: Input vector of any real values.

    Returns:
        Projected vector on the probability simplex (non-negative, sums to 1).
    """
    x = np.asarray(x, dtype=np.float64).ravel()

    # Already on simplex
    if (x >= 0).all() and abs(np.sum(x) - 1) < 1e-10:
        return x

    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u * ind > cssv
    rho = ind[cond][-1]
    theta = cssv[rho - 1] / rho
    return np.maximum(x - theta, 0)


def compute_regret(
    strategy: NDArray[np.floating], payoff_matrix: NDArray[np.floating]
) -> tuple[NDArray[np.floating], float, NDArray[np.floating]]:
    """Compute regret for a strategy against a payoff matrix.

    Args:
        strategy: Mixed strategy (probability distribution over actions).
        payoff_matrix: Payoff matrix where M[i,j] is payoff for action i vs opponent j.

    Returns:
        Tuple of (regret per action, Nash value, expected utility per action).
    """
    expected_utils = payoff_matrix @ strategy
    nash_value = float(strategy @ payoff_matrix @ strategy)
    regret = expected_utils - nash_value
    return regret, nash_value, expected_utils


def l1_norm(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Compute L1 norm (sum of absolute differences) between two vectors."""
    return float(np.sum(np.abs(x - y)))
