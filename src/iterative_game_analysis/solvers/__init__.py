"""Equilibrium solvers for meta-game analysis."""

from .base import Solver, get_solver
from .mene import MENESolver
from .lle import (
    LLESolver,
    qre_exploitability,
    compute_qre_at_temperature,
    max_affinity_entropy,
    construct_kernels,
    affinity_entropy,
)
from .cce import (
    CCESolver,
    solve_cce_min_kl,
    compute_cce_regrets,
    is_cce,
)

__all__ = [
    "Solver",
    "get_solver",
    # Nash equilibrium solvers
    "MENESolver",
    "LLESolver",
    # CCE solver
    "CCESolver",
    # Utility functions
    "qre_exploitability",
    "compute_qre_at_temperature",
    "max_affinity_entropy",
    "construct_kernels",
    "affinity_entropy",
    "solve_cce_min_kl",
    "compute_cce_regrets",
    "is_cce",
]
