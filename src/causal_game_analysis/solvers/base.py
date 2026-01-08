"""Base solver interface for equilibrium computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class Solver(Protocol):
    """Protocol for equilibrium solvers.

    All solvers must implement the solve method that takes a payoff matrix
    and returns an equilibrium strategy.
    """

    def solve(self, payoff_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute equilibrium strategy for a symmetric game.

        Args:
            payoff_matrix: Square payoff matrix where M[i,j] is the payoff
                for row player using action i when column player uses action j.

        Returns:
            Equilibrium mixed strategy (probability distribution over actions).
        """
        ...


_SOLVERS: dict[str, type[Solver]] = {}


def register_solver(name: str):
    """Decorator to register a solver class."""

    def decorator(cls: type[Solver]) -> type[Solver]:
        _SOLVERS[name] = cls
        return cls

    return decorator


def get_solver(name: str) -> Solver:
    """Get a solver instance by name.

    Args:
        name: Solver name (e.g., "mene", "uniform").

    Returns:
        Solver instance.

    Raises:
        ValueError: If solver name is not recognized.
    """
    if name not in _SOLVERS:
        available = ", ".join(_SOLVERS.keys())
        raise ValueError(f"Unknown solver '{name}'. Available: {available}")
    return _SOLVERS[name]()


@register_solver("uniform")
class UniformSolver:
    """Solver that returns uniform distribution (baseline)."""

    def solve(self, payoff_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return uniform distribution over actions."""
        n = payoff_matrix.shape[0]
        return np.ones(n) / n
