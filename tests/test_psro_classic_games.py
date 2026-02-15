#!/usr/bin/env python3
"""
Test PSRO implementation on classic games.

Games tested:
- Rock-Paper-Scissors: Nash = (1/3, 1/3, 1/3) for both players
- Prisoner's Dilemma: Nash = (Defect, Defect) = pure strategy on action 1
- Matching Pennies: Nash = (1/2, 1/2) for both players

Run with: python tests/test_psro_classic_games.py
"""

import numpy as np
from typing import Tuple, List


# =============================================================================
# Classic Game Payoff Matrices
# =============================================================================

def get_rps_payoff() -> np.ndarray:
    """
    Rock-Paper-Scissors payoff matrix.
    Nash equilibrium: (1/3, 1/3, 1/3) for both players.

    Returns:
        payoff[i, j, p] = player p's payoff when P1 plays i, P2 plays j
    """
    return np.array([
        [[0, 0], [-1, 1], [1, -1]],   # Rock vs Rock, Paper, Scissors
        [[1, -1], [0, 0], [-1, 1]],   # Paper
        [[-1, 1], [1, -1], [0, 0]],   # Scissors
    ], dtype=np.float64)


def get_prisoners_dilemma_payoff() -> np.ndarray:
    """
    Prisoner's Dilemma payoff matrix.
    Nash equilibrium: Both players Defect (action 1).

    Actions: 0 = Cooperate, 1 = Defect
    """
    return np.array([
        [[-1, -1], [-3, 0]],   # Cooperate vs Cooperate, Defect
        [[0, -3], [-2, -2]],   # Defect vs Cooperate, Defect
    ], dtype=np.float64)


def get_matching_pennies_payoff() -> np.ndarray:
    """
    Matching Pennies payoff matrix.
    Nash equilibrium: (1/2, 1/2) for both players.

    Actions: 0 = Heads, 1 = Tails
    P1 wins if same, P2 wins if different.
    """
    return np.array([
        [[1, -1], [-1, 1]],   # Heads vs Heads, Tails
        [[-1, 1], [1, -1]],   # Tails
    ], dtype=np.float64)


# =============================================================================
# Nash Equilibrium Solvers
# =============================================================================

def solve_nash_replicator(
    payoff: np.ndarray,
    iterations: int = 10000,
    dt: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for Nash equilibrium using replicator dynamics.

    Args:
        payoff: Shape [n_actions_p1, n_actions_p2, 2]
        iterations: Number of iterations
        dt: Step size

    Returns:
        (sigma_p1, sigma_p2): Nash equilibrium strategies
    """
    n1 = payoff.shape[0]
    n2 = payoff.shape[1]

    # Start near uniform with small perturbation
    sigma_p1 = np.ones(n1) / n1 + np.random.randn(n1) * 0.01
    sigma_p2 = np.ones(n2) / n2 + np.random.randn(n2) * 0.01
    sigma_p1 = np.clip(sigma_p1, 0.01, 1)
    sigma_p2 = np.clip(sigma_p2, 0.01, 1)
    sigma_p1 /= sigma_p1.sum()
    sigma_p2 /= sigma_p2.sum()

    for _ in range(iterations):
        # Expected payoffs
        u1 = payoff[:, :, 0] @ sigma_p2  # P1's payoff for each action
        u2 = payoff[:, :, 1].T @ sigma_p1  # P2's payoff for each action

        # Replicator dynamics
        avg_u1 = sigma_p1 @ u1
        avg_u2 = sigma_p2 @ u2

        sigma_p1 = sigma_p1 * (1 + dt * (u1 - avg_u1))
        sigma_p2 = sigma_p2 * (1 + dt * (u2 - avg_u2))

        # Normalize
        sigma_p1 = np.clip(sigma_p1, 1e-10, 1)
        sigma_p2 = np.clip(sigma_p2, 1e-10, 1)
        sigma_p1 /= sigma_p1.sum()
        sigma_p2 /= sigma_p2.sum()

    return sigma_p1, sigma_p2


def solve_nash_support_enumeration(
    payoff: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for Nash equilibrium using support enumeration (exact).
    Only works for small games.
    """
    try:
        import nashpy as nash
        payoff_p1 = payoff[:, :, 0]
        payoff_p2 = payoff[:, :, 1]
        game = nash.Game(payoff_p1, payoff_p2)
        equilibria = list(game.support_enumeration())
        if equilibria:
            return equilibria[0]
    except ImportError:
        pass

    # Fallback to replicator
    return solve_nash_replicator(payoff)


# =============================================================================
# PSRO Implementation for Matrix Games
# =============================================================================

class PSRO:
    """
    Policy Space Response Oracles for matrix games.

    In matrix games, policies are simply mixed strategies (probability distributions
    over actions). Best responses are pure strategies.
    """

    def __init__(self, payoff: np.ndarray, seed: int = 42):
        """
        Args:
            payoff: Shape [n_actions_p1, n_actions_p2, 2]
        """
        self.payoff = payoff
        self.n_actions_p1 = payoff.shape[0]
        self.n_actions_p2 = payoff.shape[1]
        np.random.seed(seed)

        # Population of policies (as one-hot pure strategies initially)
        self.population_p1: List[np.ndarray] = [np.eye(self.n_actions_p1)[0]]
        self.population_p2: List[np.ndarray] = [np.eye(self.n_actions_p2)[0]]

    def compute_empirical_payoff_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute payoff matrix over populations.

        Returns:
            (emp_payoff_p1, emp_payoff_p2): Payoffs for meta-game
        """
        n1 = len(self.population_p1)
        n2 = len(self.population_p2)

        emp_payoff_p1 = np.zeros((n1, n2))
        emp_payoff_p2 = np.zeros((n1, n2))

        for i, pi in enumerate(self.population_p1):
            for j, pj in enumerate(self.population_p2):
                # Expected payoff when P1 uses policy i, P2 uses policy j
                for a1 in range(self.n_actions_p1):
                    for a2 in range(self.n_actions_p2):
                        emp_payoff_p1[i, j] += pi[a1] * pj[a2] * self.payoff[a1, a2, 0]
                        emp_payoff_p2[i, j] += pi[a1] * pj[a2] * self.payoff[a1, a2, 1]

        return emp_payoff_p1, emp_payoff_p2

    def solve_meta_game(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Nash equilibrium of the meta-game.

        Returns:
            (sigma_p1, sigma_p2): Meta-strategies over populations
        """
        emp_p1, emp_p2 = self.compute_empirical_payoff_matrix()

        # Stack into standard format
        n1, n2 = emp_p1.shape
        emp_payoff = np.zeros((n1, n2, 2))
        emp_payoff[:, :, 0] = emp_p1
        emp_payoff[:, :, 1] = emp_p2

        return solve_nash_replicator(emp_payoff, iterations=5000)

    def compute_best_response(
        self,
        opponent_mixture: np.ndarray,
        player_id: int
    ) -> np.ndarray:
        """
        Compute best response to opponent's meta-strategy.

        Args:
            opponent_mixture: Distribution over opponent's population
            player_id: 0 for P1, 1 for P2

        Returns:
            Best response pure strategy (one-hot)
        """
        if player_id == 0:
            # P1 computing BR against P2's meta-strategy
            opponent_policy = sum(
                opponent_mixture[j] * self.population_p2[j]
                for j in range(len(self.population_p2))
            )
            # Expected payoff for each P1 action
            ev = self.payoff[:, :, 0] @ opponent_policy
            br_action = np.argmax(ev)
            return np.eye(self.n_actions_p1)[br_action]
        else:
            # P2 computing BR against P1's meta-strategy
            opponent_policy = sum(
                opponent_mixture[i] * self.population_p1[i]
                for i in range(len(self.population_p1))
            )
            # Expected payoff for each P2 action
            ev = self.payoff[:, :, 1].T @ opponent_policy
            br_action = np.argmax(ev)
            return np.eye(self.n_actions_p2)[br_action]

    def compute_nashconv(self, sigma_p1: np.ndarray, sigma_p2: np.ndarray) -> float:
        """
        Compute NashConv (exploitability) of current meta-strategies.
        """
        # Convert meta-strategies to full strategy space
        full_p1 = sum(sigma_p1[i] * self.population_p1[i] for i in range(len(sigma_p1)))
        full_p2 = sum(sigma_p2[j] * self.population_p2[j] for j in range(len(sigma_p2)))

        # Current values
        v1_current = full_p1 @ self.payoff[:, :, 0] @ full_p2
        v2_current = full_p1 @ self.payoff[:, :, 1] @ full_p2

        # Best response values
        br_ev_p1 = self.payoff[:, :, 0] @ full_p2
        v1_br = br_ev_p1.max()

        br_ev_p2 = self.payoff[:, :, 1].T @ full_p1
        v2_br = br_ev_p2.max()

        return (v1_br - v1_current) + (v2_br - v2_current)

    def run(self, n_iterations: int = 10, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run PSRO for n_iterations.

        Returns:
            (final_strategy_p1, final_strategy_p2): Converged strategies
        """
        if verbose:
            print(f"Running PSRO for {n_iterations} iterations...")

        for iteration in range(n_iterations):
            # Solve meta-game
            sigma_p1, sigma_p2 = self.solve_meta_game()

            # Compute NashConv
            nashconv = self.compute_nashconv(sigma_p1, sigma_p2)

            if verbose and (iteration + 1) % max(1, n_iterations // 5) == 0:
                print(f"  Iter {iteration+1}: NashConv={nashconv:.6f}, Pop=({len(self.population_p1)}, {len(self.population_p2)})")

            # Compute best responses
            br_p1 = self.compute_best_response(sigma_p2, player_id=0)
            br_p2 = self.compute_best_response(sigma_p1, player_id=1)

            # Add to population if not already present
            if not any(np.allclose(br_p1, p) for p in self.population_p1):
                self.population_p1.append(br_p1)
            if not any(np.allclose(br_p2, p) for p in self.population_p2):
                self.population_p2.append(br_p2)

        # Final solve
        sigma_p1, sigma_p2 = self.solve_meta_game()

        # Convert to full strategy space
        final_p1 = sum(sigma_p1[i] * self.population_p1[i] for i in range(len(sigma_p1)))
        final_p2 = sum(sigma_p2[j] * self.population_p2[j] for j in range(len(sigma_p2)))

        return final_p1, final_p2


# =============================================================================
# Tests
# =============================================================================

def test_psro_rps():
    """Test PSRO converges to Nash in Rock-Paper-Scissors."""
    print("\n" + "=" * 60)
    print("TEST: PSRO on Rock-Paper-Scissors")
    print("=" * 60)

    payoff = get_rps_payoff()
    psro = PSRO(payoff, seed=42)

    final_p1, final_p2 = psro.run(n_iterations=10, verbose=True)

    expected = np.array([1/3, 1/3, 1/3])
    error_p1 = np.abs(final_p1 - expected).max()
    error_p2 = np.abs(final_p2 - expected).max()

    print(f"\nResults:")
    print(f"  P1 strategy: {np.round(final_p1, 4)}")
    print(f"  P2 strategy: {np.round(final_p2, 4)}")
    print(f"  Expected:    {expected}")
    print(f"  Max error P1: {error_p1:.6f}")
    print(f"  Max error P2: {error_p2:.6f}")

    passed = error_p1 < 0.05 and error_p2 < 0.05
    print(f"\n{'PASSED' if passed else 'FAILED'}")
    return passed


def test_psro_prisoners_dilemma():
    """Test PSRO converges to Nash in Prisoner's Dilemma."""
    print("\n" + "=" * 60)
    print("TEST: PSRO on Prisoner's Dilemma")
    print("=" * 60)

    payoff = get_prisoners_dilemma_payoff()
    psro = PSRO(payoff, seed=42)

    final_p1, final_p2 = psro.run(n_iterations=10, verbose=True)

    expected = np.array([0, 1])  # Defect
    error_p1 = 1 - final_p1[1]  # Should be 1 for defect
    error_p2 = 1 - final_p2[1]

    print(f"\nResults:")
    print(f"  P1 strategy: {np.round(final_p1, 4)} (expected [0, 1] = Defect)")
    print(f"  P2 strategy: {np.round(final_p2, 4)} (expected [0, 1] = Defect)")
    print(f"  P1 defect prob: {final_p1[1]:.4f}")
    print(f"  P2 defect prob: {final_p2[1]:.4f}")

    passed = final_p1[1] > 0.95 and final_p2[1] > 0.95
    print(f"\n{'PASSED' if passed else 'FAILED'}")
    return passed


def test_psro_matching_pennies():
    """Test PSRO converges to Nash in Matching Pennies."""
    print("\n" + "=" * 60)
    print("TEST: PSRO on Matching Pennies")
    print("=" * 60)

    payoff = get_matching_pennies_payoff()
    psro = PSRO(payoff, seed=42)

    final_p1, final_p2 = psro.run(n_iterations=10, verbose=True)

    expected = np.array([0.5, 0.5])
    error_p1 = np.abs(final_p1 - expected).max()
    error_p2 = np.abs(final_p2 - expected).max()

    print(f"\nResults:")
    print(f"  P1 strategy: {np.round(final_p1, 4)}")
    print(f"  P2 strategy: {np.round(final_p2, 4)}")
    print(f"  Expected:    {expected}")
    print(f"  Max error P1: {error_p1:.6f}")
    print(f"  Max error P2: {error_p2:.6f}")

    passed = error_p1 < 0.05 and error_p2 < 0.05
    print(f"\n{'PASSED' if passed else 'FAILED'}")
    return passed


def test_replicator_dynamics():
    """Test replicator dynamics solver directly."""
    print("\n" + "=" * 60)
    print("TEST: Replicator Dynamics Solver")
    print("=" * 60)

    results = {}

    # RPS
    payoff = get_rps_payoff()
    sigma_p1, sigma_p2 = solve_nash_replicator(payoff, iterations=10000)
    expected = np.array([1/3, 1/3, 1/3])
    error = max(np.abs(sigma_p1 - expected).max(), np.abs(sigma_p2 - expected).max())
    results['RPS'] = {'error': error, 'p1': sigma_p1, 'p2': sigma_p2, 'expected': expected}

    # PD
    payoff = get_prisoners_dilemma_payoff()
    sigma_p1, sigma_p2 = solve_nash_replicator(payoff, iterations=10000)
    expected = np.array([0, 1])
    error = max(1 - sigma_p1[1], 1 - sigma_p2[1])
    results['PD'] = {'error': error, 'p1': sigma_p1, 'p2': sigma_p2, 'expected': expected}

    # Matching Pennies
    payoff = get_matching_pennies_payoff()
    sigma_p1, sigma_p2 = solve_nash_replicator(payoff, iterations=10000)
    expected = np.array([0.5, 0.5])
    error = max(np.abs(sigma_p1 - expected).max(), np.abs(sigma_p2 - expected).max())
    results['MP'] = {'error': error, 'p1': sigma_p1, 'p2': sigma_p2, 'expected': expected}

    print("\nResults:")
    all_passed = True
    for name, data in results.items():
        status = "PASS" if data['error'] < 0.05 else "FAIL"
        if data['error'] >= 0.05:
            all_passed = False
        print(f"  {name}: P1={np.round(data['p1'], 3)}, P2={np.round(data['p2'], 3)}, error={data['error']:.4f} [{status}]")

    print(f"\n{'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


def run_all_tests():
    """Run all PSRO tests."""
    print("\n" + "=" * 70)
    print("PSRO CLASSIC GAMES TEST SUITE")
    print("=" * 70)

    results = {}

    results['Replicator Dynamics'] = test_replicator_dynamics()
    results['PSRO RPS'] = test_psro_rps()
    results['PSRO Prisoner\'s Dilemma'] = test_psro_prisoners_dilemma()
    results['PSRO Matching Pennies'] = test_psro_matching_pennies()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"  {name}: {status}")

    print("-" * 70)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
