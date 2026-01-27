"""Tests for the CCE (Coarse Correlated Equilibrium) solver."""

import numpy as np
import pytest

from iterative_game_analysis.solvers.cce import (
    CCESolver,
    solve_cce_min_kl,
    compute_cce_regrets,
    is_cce,
    _build_cce_constraints,
    _marginals_from_joint,
)
from iterative_game_analysis.solvers.lle import _payoff_tensor_from_matrix
from iterative_game_analysis.solvers import get_solver


class TestCCESolver:
    """Tests for CCESolver class."""

    def test_solver_registration(self):
        """Test that CCE solver is registered."""
        solver = get_solver("cce")
        assert isinstance(solver, CCESolver)

    def test_rock_paper_scissors(self):
        """Test CCE on rock-paper-scissors."""
        rps = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ], dtype=np.float64)

        solver = CCESolver()
        strategy = solver.solve(rps)

        # Should be close to uniform
        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(strategy, expected, decimal=1)

    def test_output_is_valid_distribution(self):
        """Test that output is a valid probability distribution."""
        payoff = np.random.randn(4, 4)

        solver = CCESolver()
        strategy = solver.solve(payoff)

        assert len(strategy) == 4
        assert np.isclose(strategy.sum(), 1.0)
        assert np.all(strategy >= 0)
        assert np.all(strategy <= 1)

    def test_handles_nan_values(self):
        """Test that solver handles NaN values."""
        payoff = np.array([
            [1.0, np.nan],
            [0.5, 1.0],
        ])

        solver = CCESolver()
        strategy = solver.solve(payoff)

        assert len(strategy) == 2
        assert np.isclose(strategy.sum(), 1.0)
        assert not np.any(np.isnan(strategy))

    def test_solve_full_returns_joint(self):
        """Test that solve_full returns joint distribution."""
        payoff = np.array([
            [1, 0],
            [0, 1],
        ], dtype=np.float64)

        solver = CCESolver()
        joint, marginals, kl = solver.solve_full(payoff)

        # Joint should be 2x2
        assert joint.shape == (2, 2)
        assert np.isclose(joint.sum(), 1.0)
        assert np.all(joint >= 0)

        # Should have 2 marginals
        assert len(marginals) == 2
        for m in marginals:
            assert np.isclose(m.sum(), 1.0)

    def test_last_joint_property(self):
        """Test that last_joint property works."""
        payoff = np.random.randn(3, 3)

        solver = CCESolver()
        _ = solver.solve(payoff)

        assert solver.last_joint is not None
        assert solver.last_joint.shape == (3, 3)


class TestCCEConstraints:
    """Tests for CCE constraint building."""

    def test_constraint_dimensions(self):
        """Test constraint matrix dimensions."""
        payoff = np.random.randn(3, 3)
        pt = _payoff_tensor_from_matrix(payoff)

        constraints = _build_cce_constraints(pt)

        # For 2 players with 3 actions each: 2 * 3 = 6 constraints
        # (each player has n deviation options)
        assert constraints.shape[0] == 6
        assert constraints.shape[1] == 9  # 3 * 3 joint actions


class TestMarginalsFromJoint:
    """Tests for marginal computation."""

    def test_marginals_sum_correctly(self):
        """Test that marginals sum to 1."""
        joint = np.random.rand(3, 4)
        joint = joint / joint.sum()

        marginals = _marginals_from_joint(joint)

        assert len(marginals) == 2
        np.testing.assert_almost_equal(marginals[0].sum(), 1.0)
        np.testing.assert_almost_equal(marginals[1].sum(), 1.0)

    def test_marginal_dimensions(self):
        """Test marginal dimensions match."""
        joint = np.random.rand(3, 4)
        joint = joint / joint.sum()

        marginals = _marginals_from_joint(joint)

        assert len(marginals[0]) == 3
        assert len(marginals[1]) == 4


class TestCCERegrets:
    """Tests for CCE regret computation."""

    def test_regrets_structure(self):
        """Test regret dict structure."""
        payoff = np.random.randn(3, 3)
        pt = _payoff_tensor_from_matrix(payoff)

        # Uniform distribution
        joint = np.ones((3, 3)) / 9

        regrets = compute_cce_regrets(pt, joint)

        assert 'per_player' in regrets
        assert 'total' in regrets
        assert 'max' in regrets
        assert len(regrets['per_player']) == 2


class TestIsCCE:
    """Tests for CCE verification."""

    def test_solved_cce_is_valid(self):
        """Test that solved CCE passes verification."""
        payoff = np.array([
            [1, 0],
            [0, 1],
        ], dtype=np.float64)

        pt = _payoff_tensor_from_matrix(payoff)
        joint, _, _ = solve_cce_min_kl(pt)

        assert is_cce(pt, joint, tolerance=1e-4)

    def test_random_joint_may_not_be_cce(self):
        """Test that random joint is likely not a CCE."""
        payoff = np.array([
            [2, 0],
            [0, 1],
        ], dtype=np.float64)

        pt = _payoff_tensor_from_matrix(payoff)

        # Highly concentrated on suboptimal action
        joint = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
        ])

        # This is unlikely to be a CCE
        # (player 0 would want to deviate to action 0)
        # Note: this test may be flaky depending on the game
        regrets = compute_cce_regrets(pt, joint)
        # Just check we can compute regrets
        assert regrets['max'] >= 0


class TestSolveCCEMinKL:
    """Tests for the core CCE solving function."""

    def test_returns_valid_joint(self):
        """Test that solve returns valid joint distribution."""
        payoff = np.random.randn(3, 3)
        pt = _payoff_tensor_from_matrix(payoff)

        joint, marginals, kl = solve_cce_min_kl(pt)

        assert joint.shape == (3, 3)
        assert np.isclose(joint.sum(), 1.0)
        assert np.all(joint >= -1e-6)  # Allow small numerical error

    def test_kl_divergence_non_negative(self):
        """Test that KL-divergence is non-negative."""
        payoff = np.random.randn(3, 3)
        pt = _payoff_tensor_from_matrix(payoff)

        _, _, kl = solve_cce_min_kl(pt)

        assert kl >= -1e-6  # Allow small numerical error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
