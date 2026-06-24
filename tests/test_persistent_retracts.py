"""Tests for evaluation/persistent_retracts.py (σ-CURB via BHK 2013 route)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Project root on sys.path so we can import evaluation/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.persistent_retracts import (
    find_minimal_persistent_retracts,
    find_weakly_inferior,
    is_own_payoff_equivalent_to_mixture,
    is_weakly_dominated,
    perturb_for_sigma_curb,
)
from evaluation.curb_analysis import find_minimal_curb_sets_klimm_weibull


# Rock-paper-scissors row-player payoffs (role-symmetric, no inferior strategies).
RPS = np.array(
    [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ]
)


def test_no_weakly_inferior_in_rps():
    """Pure RPS has no weakly inferior strategies."""
    assert find_weakly_inferior(RPS) == frozenset()


def test_strict_dominance_flagged():
    """A strategy strictly dominated by another pure strategy is weakly inferior."""
    # 3-strategy game: strategy 2 is strictly worse than 0 against every k.
    M = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0],
        ]
    )
    assert is_weakly_dominated(M, 2) is True
    assert is_weakly_dominated(M, 0) is False
    assert is_weakly_dominated(M, 1) is False
    assert find_weakly_inferior(M) == frozenset({2})


def test_own_payoff_equivalent_to_mixture():
    """A strategy own-payoff-equivalent to a proper mixture is weakly inferior.

    Construction: RPS + a 4th strategy D whose row equals 0.5·R + 0.5·P
    coordinate-wise. D is not strictly dominated by any mixture (equality
    everywhere), but it *is* own-payoff-equivalent to (0.5, 0.5, 0) over
    {R, P, S}.
    """
    # M[D, :] = 0.5 * M[R, :] + 0.5 * M[P, :], padded with M[*, D] = 0.
    M = np.array(
        [
            [0.0, -1.0, 1.0, 0.0],   # R
            [1.0, 0.0, -1.0, 0.0],   # P
            [-1.0, 1.0, 0.0, 0.0],   # S
            [0.5, -0.5, 0.0, 0.0],   # D = 0.5R + 0.5P (against pure cols R, P, S, D)
        ]
    )
    # D is NOT strictly dominated (mixture only equals D, never strictly better).
    assert is_weakly_dominated(M, 3) is False
    # But D is own-payoff-equivalent to (0.5, 0.5, 0).
    assert is_own_payoff_equivalent_to_mixture(M, 3) is True
    assert find_weakly_inferior(M) == frozenset({3})


def test_sigma_curb_strictly_finer_than_beta_curb():
    """End-to-end: σ-CURB excludes a strategy that β-CURB cannot remove.

    Same RPS+D game as above. Because D is own-payoff-equivalent to a
    mixture of R and P, D is a *co-best response* at certain mixtures over
    {R, P, S}, so {R, P, S} is not β-CURB and the unique minimal β-CURB is
    the full strategy set. After the BHK 2013 perturbation, D is no longer
    a best response anywhere, and {R, P, S} becomes the minimal σ-CURB.
    """
    M = np.array(
        [
            [0.0, -1.0, 1.0, 0.0],
            [1.0, 0.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [0.5, -0.5, 0.0, 0.0],
        ]
    )

    beta_minimal = find_minimal_curb_sets_klimm_weibull(M, n_strategies=4)
    # The full set is the only minimal β-CURB (ρ_β = 1).
    assert frozenset({0, 1, 2, 3}) in beta_minimal
    assert all(len(C) == 4 for C in beta_minimal)

    sigma_minimal = find_minimal_persistent_retracts(M, n_strategies=4, epsilon=1e-6)
    # The persistent retract excludes D (ρ_σ = 3/4).
    assert frozenset({0, 1, 2}) in sigma_minimal
    assert all(3 not in C for C in sigma_minimal)


def test_perturb_only_touches_inferior_rows():
    """Perturbation should subtract ε from each inferior row and nothing else."""
    M = np.array(
        [
            [0.0, -1.0, 1.0, 0.0],
            [1.0, 0.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [0.5, -0.5, 0.0, 0.0],
        ]
    )
    weakly_inferior = frozenset({3})
    epsilon = 1e-3
    M_prime = perturb_for_sigma_curb(M, weakly_inferior, epsilon=epsilon)

    # Rows 0..2 unchanged.
    np.testing.assert_array_equal(M_prime[:3, :], M[:3, :])
    # Row 3 uniformly reduced by ε.
    np.testing.assert_allclose(M_prime[3, :], M[3, :] - epsilon)
    # Original not mutated.
    np.testing.assert_array_equal(
        M,
        np.array(
            [
                [0.0, -1.0, 1.0, 0.0],
                [1.0, 0.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [0.5, -0.5, 0.0, 0.0],
            ]
        ),
    )
