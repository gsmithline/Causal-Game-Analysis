"""Tests for the Limiting Logit Equilibrium (LLE) solver."""

import numpy as np
import pytest

from iterative_game_analysis.solvers.lle import (
    LLESolver,
    qre_exploitability,
    compute_qre_at_temperature,
    max_affinity_entropy,
    _payoff_tensor_from_matrix,
)
from iterative_game_analysis.solvers import get_solver


class TestLLESolver:
    """Tests for LLESolver class."""

    def test_solver_registration(self):
        """Test that LLE solver is registered."""
        solver = get_solver("lle")
        assert isinstance(solver, LLESolver)

    def test_rock_paper_scissors(self):
        """Test LLE on rock-paper-scissors (should be uniform)."""
        # RPS payoff matrix
        rps = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ], dtype=np.float64)

        solver = LLESolver(num_anneals=200, gamma=0.95)
        strategy = solver.solve(rps)

        # Should be close to uniform (1/3, 1/3, 1/3)
        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(strategy, expected, decimal=1)

    def test_prisoners_dilemma(self):
        """Test LLE on prisoner's dilemma."""
        # Payoff matrix (row player)
        # C = cooperate, D = defect
        pd = np.array([
            [-1, -3],  # C: vs C, vs D
            [0, -2],   # D: vs C, vs D
        ], dtype=np.float64)

        solver = LLESolver(num_anneals=300)
        strategy = solver.solve(pd)

        # Dominant strategy is D, so should converge to mostly D
        assert strategy[1] > 0.5  # D should have higher probability

    def test_output_is_valid_distribution(self):
        """Test that output is a valid probability distribution."""
        payoff = np.random.randn(5, 5)

        solver = LLESolver(num_anneals=100)
        strategy = solver.solve(payoff)

        assert len(strategy) == 5
        assert np.isclose(strategy.sum(), 1.0)
        assert np.all(strategy >= 0)
        assert np.all(strategy <= 1)

    def test_handles_nan_values(self):
        """Test that solver handles NaN values in payoff matrix."""
        payoff = np.array([
            [1.0, np.nan, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ])

        solver = LLESolver(num_anneals=100)
        strategy = solver.solve(payoff)

        assert len(strategy) == 3
        assert np.isclose(strategy.sum(), 1.0)
        assert not np.any(np.isnan(strategy))


class TestMaxAffinityEntropy:
    """Tests for max_affinity_entropy function."""

    def test_returns_valid_distributions(self):
        """Test that max_affinity_entropy returns valid distributions."""
        payoff = np.random.randn(4, 4)
        pt = _payoff_tensor_from_matrix(payoff)

        dists = max_affinity_entropy(pt)

        assert len(dists) == 2
        for dist in dists:
            assert len(dist) == 4
            assert np.isclose(dist.sum(), 1.0)
            assert np.all(dist >= 0)


class TestQREExploitability:
    """Tests for QRE exploitability computation."""

    def test_exploitability_non_negative(self):
        """Test that exploitability is non-negative."""
        payoff = np.random.randn(3, 3)
        pt = _payoff_tensor_from_matrix(payoff)

        # Uniform distribution
        logits = np.zeros(6)
        target_logits = np.zeros(6)

        exp = qre_exploitability(logits, target_logits, pt, temperature=1.0)

        # Exploitability should be >= 0 (it's a gap measure)
        assert exp >= -1e-6  # Allow small numerical error


class TestComputeQREAtTemperature:
    """Tests for compute_qre_at_temperature function."""

    def test_high_temperature_near_uniform(self):
        """Test that high temperature QRE is near uniform."""
        payoff = np.array([
            [1, 0],
            [0, 1],
        ], dtype=np.float64)

        qre = compute_qre_at_temperature(payoff, temperature=10.0)

        # High temperature should be close to uniform
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(qre, expected, decimal=1)

    def test_low_temperature_more_concentrated(self):
        """Test that low temperature QRE is more concentrated."""
        # Game with dominant strategy
        payoff = np.array([
            [2, 2],
            [0, 0],
        ], dtype=np.float64)

        qre_high = compute_qre_at_temperature(payoff, temperature=10.0)
        qre_low = compute_qre_at_temperature(payoff, temperature=0.1)

        # Low temperature should be more concentrated on action 0
        assert qre_low[0] > qre_high[0]


class TestPayoffTensorConversion:
    """Tests for payoff tensor conversion."""

    def test_symmetric_game_conversion(self):
        """Test conversion of symmetric game to tensor."""
        payoff = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float64)

        pt = _payoff_tensor_from_matrix(payoff)

        assert pt.shape == (2, 2, 2)
        np.testing.assert_array_equal(pt[0], payoff)
        np.testing.assert_array_equal(pt[1], payoff.T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
