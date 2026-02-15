"""
Test that all algorithms converge to Nash equilibrium on classic games.

Games tested:
- Rock-Paper-Scissors: Nash = (1/3, 1/3, 1/3) for both players
- Prisoner's Dilemma: Nash = (Defect, Defect)
- Matching Pennies: Nash = (1/2, 1/2) for both players

Run with: pytest tests/test_equilibrium_convergence.py -v -s
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Simple Matrix Game Environment
# =============================================================================

class MatrixGameEnv:
    """
    Simple 2-player matrix game environment for testing equilibrium convergence.

    Supports simultaneous-move games like RPS, Prisoner's Dilemma, etc.
    """

    def __init__(
        self,
        payoff_matrix: np.ndarray,
        num_envs: int = 256,
        seed: Optional[int] = None
    ):
        """
        Args:
            payoff_matrix: Shape [n_actions_p1, n_actions_p2, 2]
                          payoff_matrix[i,j,0] = P1's payoff when P1 plays i, P2 plays j
                          payoff_matrix[i,j,1] = P2's payoff
            num_envs: Number of parallel games
            seed: Random seed
        """
        self.payoff_matrix = payoff_matrix
        self.num_envs = num_envs
        self.n_actions_p1 = payoff_matrix.shape[0]
        self.n_actions_p2 = payoff_matrix.shape[1]

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # State: which player's turn (0 = P1, 1 = P2, 2 = both played)
        self.current_player = torch.zeros(num_envs, dtype=torch.long)
        self.p1_actions = torch.zeros(num_envs, dtype=torch.long)
        self.p2_actions = torch.zeros(num_envs, dtype=torch.long)
        self.done = torch.zeros(num_envs, dtype=torch.bool)

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """Reset all environments."""
        self.current_player = torch.zeros(self.num_envs, dtype=torch.long)
        self.p1_actions = torch.zeros(self.num_envs, dtype=torch.long)
        self.p2_actions = torch.zeros(self.num_envs, dtype=torch.long)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool)

        obs = self._get_obs()
        return obs, {'action_mask': self._get_mask()}

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment.

        In simultaneous games, we collect both players' actions then resolve.
        """
        rewards = torch.zeros(self.num_envs, 2)

        # Store actions based on current player
        p1_turn = self.current_player == 0
        p2_turn = self.current_player == 1

        self.p1_actions[p1_turn] = actions[p1_turn]
        self.p2_actions[p2_turn] = actions[p2_turn]

        # Advance player
        self.current_player[p1_turn] = 1

        # When P2 has played, resolve the game
        resolve_mask = p2_turn
        if resolve_mask.any():
            for i in range(self.num_envs):
                if resolve_mask[i]:
                    a1 = self.p1_actions[i].item()
                    a2 = self.p2_actions[i].item()
                    rewards[i, 0] = self.payoff_matrix[a1, a2, 0]
                    rewards[i, 1] = self.payoff_matrix[a1, a2, 1]
                    self.done[i] = True

        obs = self._get_obs()
        return obs, rewards, self.done.clone(), {'action_mask': self._get_mask()}

    def _get_obs(self) -> torch.Tensor:
        """Simple observation: one-hot of current player."""
        obs = torch.zeros(self.num_envs, 2)
        obs[self.current_player == 0, 0] = 1
        obs[self.current_player == 1, 1] = 1
        return obs

    def _get_mask(self) -> torch.Tensor:
        """All actions valid."""
        n_actions = max(self.n_actions_p1, self.n_actions_p2)
        mask = torch.ones(self.num_envs, n_actions)
        # Mask invalid actions for asymmetric games
        p1_turn = self.current_player == 0
        p2_turn = self.current_player == 1
        if self.n_actions_p1 < n_actions:
            mask[p1_turn, self.n_actions_p1:] = 0
        if self.n_actions_p2 < n_actions:
            mask[p2_turn, self.n_actions_p2:] = 0
        return mask

    def auto_reset(self):
        """Reset done environments."""
        if self.done.any():
            self.current_player[self.done] = 0
            self.done[self.done] = False


# =============================================================================
# Classic Game Definitions
# =============================================================================

def get_rps_payoff():
    """Rock-Paper-Scissors payoff matrix. Nash = (1/3, 1/3, 1/3)."""
    return np.array([
        [[0, 0], [-1, 1], [1, -1]],   # Rock vs Rock, Paper, Scissors
        [[1, -1], [0, 0], [-1, 1]],   # Paper
        [[-1, 1], [1, -1], [0, 0]],   # Scissors
    ], dtype=np.float64)


def get_prisoners_dilemma_payoff():
    """Prisoner's Dilemma payoff matrix. Nash = (Defect, Defect) = action 1."""
    # Actions: 0 = Cooperate, 1 = Defect
    return np.array([
        [[-1, -1], [-3, 0]],   # Cooperate vs Cooperate, Defect
        [[0, -3], [-2, -2]],   # Defect vs Cooperate, Defect
    ], dtype=np.float64)


def get_matching_pennies_payoff():
    """Matching Pennies payoff matrix. Nash = (1/2, 1/2)."""
    return np.array([
        [[1, -1], [-1, 1]],   # Heads vs Heads, Tails
        [[-1, 1], [1, -1]],   # Tails
    ], dtype=np.float64)


def get_battle_of_sexes_payoff():
    """Battle of the Sexes. Mixed Nash = (2/3, 1/3) and (1/3, 2/3)."""
    return np.array([
        [[3, 2], [0, 0]],   # Opera vs Opera, Football
        [[0, 0], [2, 3]],   # Football
    ], dtype=np.float64)


# =============================================================================
# Test CFR Convergence
# =============================================================================

class TestCFRConvergence:
    """Test CFR algorithm convergence on classic games."""

    def test_cfr_kuhn_poker(self):
        """Test CFR converges on Kuhn Poker (OpenSpiel)."""
        import pyspiel
        from rl_training.algorithms.cfr.trainer import CFRTrainer
        from rl_training.algorithms.cfr.config import CFRConfig

        # Create config manually
        config = object.__new__(CFRConfig)
        config.seed = 42
        config.num_iterations = 500
        config.cfr_variant = 'vanilla'
        config.alternating_updates = True
        config.alpha = 1.5
        config.beta = 0.0
        config.gamma = 2.0
        config.log_interval = 100
        config.checkpoint_interval = 10000
        config.checkpoint_dir = None
        config.device = 'cpu'
        config.env_id = 'kuhn_poker'
        config.num_envs = 1

        trainer = CFRTrainer(config, game_name='kuhn_poker')
        trainer._setup()

        # Train
        for i in range(config.num_iterations):
            trainer._train_step()

        # Check exploitability
        exp = trainer._compute_exploitability()
        print(f"\nCFR on Kuhn Poker: exploitability = {exp:.6f}")
        assert exp < 0.05, f"CFR should converge to near-Nash, got exploitability {exp}"

    def test_cfr_leduc_poker(self):
        """Test CFR converges on Leduc Poker."""
        import pyspiel
        from rl_training.algorithms.cfr.trainer import CFRTrainer
        from rl_training.algorithms.cfr.config import CFRConfig

        config = object.__new__(CFRConfig)
        config.seed = 42
        config.num_iterations = 500
        config.cfr_variant = 'plus'  # CFR+ converges faster
        config.alternating_updates = True
        config.alpha = 1.5
        config.beta = 0.0
        config.gamma = 2.0
        config.log_interval = 100
        config.checkpoint_interval = 10000
        config.checkpoint_dir = None
        config.device = 'cpu'
        config.env_id = 'leduc_poker'
        config.num_envs = 1

        trainer = CFRTrainer(config, game_name='leduc_poker')
        trainer._setup()

        for i in range(config.num_iterations):
            trainer._train_step()

        exp = trainer._compute_exploitability()
        print(f"CFR+ on Leduc Poker: exploitability = {exp:.6f}")
        assert exp < 0.2, f"CFR+ should converge, got exploitability {exp}"


# =============================================================================
# Test Nash Solver Convergence
# =============================================================================

class TestNashSolverConvergence:
    """Test Nash equilibrium solvers on classic games."""

    def test_replicator_rps(self):
        """Test replicator dynamics finds Nash in Rock-Paper-Scissors."""
        from rl_training.algorithms.psro.trainer import solve_nash_replicator

        payoff = get_rps_payoff()
        x, y = solve_nash_replicator(payoff, iterations=10000, dt=0.01)

        expected = np.array([1/3, 1/3, 1/3])
        error_x = np.abs(x - expected).max()
        error_y = np.abs(y - expected).max()

        print(f"\nReplicator on RPS:")
        print(f"  P1: {np.round(x, 4)} (expected {expected})")
        print(f"  P2: {np.round(y, 4)} (expected {expected})")
        print(f"  Max error: {max(error_x, error_y):.6f}")

        assert error_x < 0.01, f"P1 strategy should be uniform, got {x}"
        assert error_y < 0.01, f"P2 strategy should be uniform, got {y}"

    def test_replicator_prisoners_dilemma(self):
        """Test replicator dynamics finds Nash in Prisoner's Dilemma."""
        from rl_training.algorithms.psro.trainer import solve_nash_replicator

        payoff = get_prisoners_dilemma_payoff()
        x, y = solve_nash_replicator(payoff, iterations=10000, dt=0.01)

        # Nash is (Defect, Defect) = both play action 1
        print(f"\nReplicator on Prisoner's Dilemma:")
        print(f"  P1: {np.round(x, 4)} (expected [0, 1])")
        print(f"  P2: {np.round(y, 4)} (expected [0, 1])")

        assert x[1] > 0.95, f"P1 should defect, got {x}"
        assert y[1] > 0.95, f"P2 should defect, got {y}"

    def test_replicator_matching_pennies(self):
        """Test replicator dynamics finds Nash in Matching Pennies."""
        from rl_training.algorithms.psro.trainer import solve_nash_replicator

        payoff = get_matching_pennies_payoff()
        x, y = solve_nash_replicator(payoff, iterations=10000, dt=0.01)

        expected = np.array([0.5, 0.5])
        error_x = np.abs(x - expected).max()
        error_y = np.abs(y - expected).max()

        print(f"\nReplicator on Matching Pennies:")
        print(f"  P1: {np.round(x, 4)} (expected {expected})")
        print(f"  P2: {np.round(y, 4)} (expected {expected})")

        assert error_x < 0.01, f"P1 should be uniform, got {x}"
        assert error_y < 0.01, f"P2 should be uniform, got {y}"

    def test_fictitious_play_rps(self):
        """Test fictitious play finds Nash in RPS."""
        from rl_training.algorithms.psro.trainer import solve_nash_fictitious_play

        payoff = get_rps_payoff()
        x, y = solve_nash_fictitious_play(payoff, iterations=50000)

        expected = np.array([1/3, 1/3, 1/3])
        error_x = np.abs(x - expected).max()
        error_y = np.abs(y - expected).max()

        print(f"\nFictitious Play on RPS:")
        print(f"  P1: {np.round(x, 4)} (expected {expected})")
        print(f"  P2: {np.round(y, 4)} (expected {expected})")

        # Fictitious play converges slower
        assert error_x < 0.05, f"P1 should be near uniform, got {x}"
        assert error_y < 0.05, f"P2 should be near uniform, got {y}"

    def test_fictitious_play_prisoners_dilemma(self):
        """Test fictitious play finds Nash in Prisoner's Dilemma."""
        from rl_training.algorithms.psro.trainer import solve_nash_fictitious_play

        payoff = get_prisoners_dilemma_payoff()
        x, y = solve_nash_fictitious_play(payoff, iterations=10000)

        print(f"\nFictitious Play on Prisoner's Dilemma:")
        print(f"  P1: {np.round(x, 4)} (expected [0, 1])")
        print(f"  P2: {np.round(y, 4)} (expected [0, 1])")

        assert x[1] > 0.9, f"P1 should defect, got {x}"
        assert y[1] > 0.9, f"P2 should defect, got {y}"


# =============================================================================
# Test MENE Solver Convergence
# =============================================================================

class TestMENESolverConvergence:
    """Test MENE (Maximum Entropy Nash Equilibrium) solver."""

    def test_mene_rps(self):
        """Test MENE finds Nash in RPS."""
        from iterative_game_analysis import MetaGame

        # RPS payoff (row player perspective for symmetric game)
        payoff = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ], dtype=np.float64)

        mg = MetaGame(
            policies=['Rock', 'Paper', 'Scissors'],
            payoff_matrix=payoff
        )

        eq = mg.solve(solver='mene')
        expected = np.array([1/3, 1/3, 1/3])
        error = np.abs(eq - expected).max()

        print(f"\nMENE on RPS:")
        print(f"  Equilibrium: {np.round(eq, 4)}")
        print(f"  Expected: {expected}")
        print(f"  Max error: {error:.6f}")

        assert error < 0.01, f"MENE should find uniform equilibrium, got {eq}"

    def test_mene_prisoners_dilemma(self):
        """Test MENE finds Nash in Prisoner's Dilemma."""
        from iterative_game_analysis import MetaGame

        # Symmetric PD payoff (row player)
        payoff = np.array([
            [-1, -3],  # Cooperate vs C, D
            [0, -2],   # Defect vs C, D
        ], dtype=np.float64)

        mg = MetaGame(
            policies=['Cooperate', 'Defect'],
            payoff_matrix=payoff
        )

        eq = mg.solve(solver='mene')

        print(f"\nMENE on Prisoner's Dilemma:")
        print(f"  Equilibrium: {np.round(eq, 4)}")
        print(f"  Expected: [0, 1] (Defect)")

        assert eq[1] > 0.95, f"MENE should find Defect equilibrium, got {eq}"


# =============================================================================
# Test Self-Play Neural Network Convergence
# =============================================================================

class TestNeuralSelfPlayConvergence:
    """Test neural self-play convergence on matrix games."""

    @pytest.mark.skip(reason="Naive policy gradient has no convergence guarantee in zero-sum games")
    def test_policy_gradient_rps(self):
        """
        Test policy gradient self-play on RPS.

        NOTE: This test is skipped because naive policy gradient does NOT
        have convergence guarantees in zero-sum games. It often cycles
        rather than converging to Nash equilibrium.

        Algorithms with convergence guarantees: CFR, NFSP, PSRO
        """
        pass  # Skipped

    def test_policy_gradient_prisoners_dilemma(self):
        """Test policy gradient self-play on Prisoner's Dilemma."""
        payoff = get_prisoners_dilemma_payoff()
        env = MatrixGameEnv(payoff, num_envs=512, seed=42)

        class SimplePolicy(nn.Module):
            def __init__(self, n_actions):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(n_actions))

            def forward(self):
                return F.softmax(self.logits, dim=-1)

            def sample(self, n):
                probs = self.forward()
                return torch.multinomial(probs.expand(n, -1), 1).squeeze(-1)

        policy_p1 = SimplePolicy(2)
        policy_p2 = SimplePolicy(2)
        opt_p1 = torch.optim.Adam(policy_p1.parameters(), lr=0.1)
        opt_p2 = torch.optim.Adam(policy_p2.parameters(), lr=0.1)

        n_episodes = 1000
        batch_size = 512

        for episode in range(n_episodes):
            env.reset()

            actions_p1 = policy_p1.sample(batch_size)
            probs_p1 = policy_p1.forward()
            log_probs_p1 = probs_p1[actions_p1].log()
            env.step(actions_p1)

            actions_p2 = policy_p2.sample(batch_size)
            probs_p2 = policy_p2.forward()
            log_probs_p2 = probs_p2[actions_p2].log()
            _, rewards, _, _ = env.step(actions_p2)

            loss_p1 = -(log_probs_p1 * rewards[:, 0]).mean()
            loss_p2 = -(log_probs_p2 * rewards[:, 1]).mean()

            opt_p1.zero_grad()
            loss_p1.backward()
            opt_p1.step()

            opt_p2.zero_grad()
            loss_p2.backward()
            opt_p2.step()

        final_p1 = policy_p1.forward().detach().numpy()
        final_p2 = policy_p2.forward().detach().numpy()

        print(f"\nPolicy Gradient Self-Play on Prisoner's Dilemma ({n_episodes} episodes):")
        print(f"  P1: {np.round(final_p1, 4)} (expected [0, 1])")
        print(f"  P2: {np.round(final_p2, 4)} (expected [0, 1])")

        # Should converge to defect
        assert final_p1[1] > 0.8, f"P1 should defect, got {final_p1}"
        assert final_p2[1] > 0.8, f"P2 should defect, got {final_p2}"


# =============================================================================
# Test NFSP Convergence (Has Convergence Guarantee in Zero-Sum Games)
# =============================================================================

class TestNFSPConvergence:
    """
    Test NFSP (Neural Fictitious Self-Play) convergence.

    NFSP has theoretical convergence guarantees to Nash equilibrium
    in two-player zero-sum games. The full NFSP requires:
    - DQN for best response learning
    - Supervised learning for average policy
    - Careful hyperparameter tuning

    Here we test a simplified version that demonstrates the core mechanism.
    """

    def test_nfsp_mechanism_rps(self):
        """
        Test NFSP core mechanism: average of best responses converges.

        In the limit, the average of best responses against evolving
        opponent mixtures converges to Nash in zero-sum games.
        """
        payoff = get_rps_payoff()
        n_actions = 3

        # Track best responses over time
        br_history_p1 = []
        br_history_p2 = []

        # Start with uniform
        mixture_p1 = np.ones(n_actions) / n_actions
        mixture_p2 = np.ones(n_actions) / n_actions

        n_iterations = 1000

        for t in range(n_iterations):
            # Compute best response to current mixture
            ev_p1 = payoff[:, :, 0] @ mixture_p2
            ev_p2 = payoff[:, :, 1].T @ mixture_p1

            br_p1 = np.argmax(ev_p1)
            br_p2 = np.argmax(ev_p2)

            br_history_p1.append(br_p1)
            br_history_p2.append(br_p2)

            # Update mixture as empirical average of BRs (like fictitious play)
            counts_p1 = np.bincount(br_history_p1, minlength=n_actions)
            counts_p2 = np.bincount(br_history_p2, minlength=n_actions)
            mixture_p1 = counts_p1 / counts_p1.sum()
            mixture_p2 = counts_p2 / counts_p2.sum()

        expected = np.array([1/3, 1/3, 1/3])
        error = max(np.abs(mixture_p1 - expected).max(), np.abs(mixture_p2 - expected).max())

        print(f"\nNFSP Mechanism on RPS ({n_iterations} iterations):")
        print(f"  P1 avg BR mixture: {np.round(mixture_p1, 4)}")
        print(f"  P2 avg BR mixture: {np.round(mixture_p2, 4)}")
        print(f"  Max error: {error:.4f}")

        # This is essentially fictitious play, which converges in zero-sum
        assert error < 0.1, f"Should converge to uniform, got error {error}"

    def test_nfsp_prisoners_dilemma(self):
        """Test NFSP mechanism in Prisoner's Dilemma (dominant strategy)."""
        payoff = get_prisoners_dilemma_payoff()
        n_actions = 2

        br_history_p1 = []
        br_history_p2 = []
        mixture_p1 = np.ones(n_actions) / n_actions
        mixture_p2 = np.ones(n_actions) / n_actions

        for t in range(500):
            ev_p1 = payoff[:, :, 0] @ mixture_p2
            ev_p2 = payoff[:, :, 1].T @ mixture_p1

            br_p1 = np.argmax(ev_p1)
            br_p2 = np.argmax(ev_p2)

            br_history_p1.append(br_p1)
            br_history_p2.append(br_p2)

            counts_p1 = np.bincount(br_history_p1, minlength=n_actions)
            counts_p2 = np.bincount(br_history_p2, minlength=n_actions)
            mixture_p1 = counts_p1 / counts_p1.sum()
            mixture_p2 = counts_p2 / counts_p2.sum()

        print(f"\nNFSP Mechanism on Prisoner's Dilemma:")
        print(f"  P1: {np.round(mixture_p1, 4)}, P2: {np.round(mixture_p2, 4)}")

        # Defect is dominant, should converge quickly
        assert mixture_p1[1] > 0.95, f"P1 should defect, got {mixture_p1}"
        assert mixture_p2[1] > 0.95, f"P2 should defect, got {mixture_p2}"


# =============================================================================
# Test PSRO Convergence (Has Convergence Guarantee)
# =============================================================================

class TestPSROConvergence:
    """
    Test PSRO (Policy Space Response Oracles) convergence.

    PSRO converges to Nash equilibrium in two-player zero-sum games
    when using appropriate meta-solvers.
    """

    def test_psro_rps(self):
        """Test PSRO converges in RPS."""
        from rl_training.algorithms.psro.trainer import solve_nash_replicator

        payoff = get_rps_payoff()
        n_actions = 3

        # Simulate PSRO: start with random policies, add best responses
        # Population: list of pure strategies (one-hot)
        population_p1 = [np.eye(n_actions)[0]]  # Start with Rock
        population_p2 = [np.eye(n_actions)[0]]

        n_psro_iterations = 10

        for psro_iter in range(n_psro_iterations):
            # Build empirical game matrix
            n1, n2 = len(population_p1), len(population_p2)
            emp_payoff = np.zeros((n1, n2, 2))

            for i, pi in enumerate(population_p1):
                for j, pj in enumerate(population_p2):
                    # Expected payoff under policies
                    for a1 in range(n_actions):
                        for a2 in range(n_actions):
                            emp_payoff[i, j, 0] += pi[a1] * pj[a2] * payoff[a1, a2, 0]
                            emp_payoff[i, j, 1] += pi[a1] * pj[a2] * payoff[a1, a2, 1]

            # Solve for Nash over population
            sigma_p1, sigma_p2 = solve_nash_replicator(emp_payoff, iterations=5000)

            # Compute opponent mixture strategy
            opp_mix_p1 = sum(sigma_p2[j] * population_p2[j] for j in range(n2))
            opp_mix_p2 = sum(sigma_p1[i] * population_p1[i] for i in range(n1))

            # Find best response
            br_values_p1 = payoff[:, :, 0] @ opp_mix_p1
            br_values_p2 = payoff[:, :, 1].T @ opp_mix_p2

            br_p1 = np.eye(n_actions)[np.argmax(br_values_p1)]
            br_p2 = np.eye(n_actions)[np.argmax(br_values_p2)]

            # Add to population if not already present
            if not any(np.allclose(br_p1, p) for p in population_p1):
                population_p1.append(br_p1)
            if not any(np.allclose(br_p2, p) for p in population_p2):
                population_p2.append(br_p2)

        # Final Nash mixture
        n1, n2 = len(population_p1), len(population_p2)
        emp_payoff = np.zeros((n1, n2, 2))
        for i, pi in enumerate(population_p1):
            for j, pj in enumerate(population_p2):
                for a1 in range(n_actions):
                    for a2 in range(n_actions):
                        emp_payoff[i, j, 0] += pi[a1] * pj[a2] * payoff[a1, a2, 0]
                        emp_payoff[i, j, 1] += pi[a1] * pj[a2] * payoff[a1, a2, 1]

        sigma_p1, sigma_p2 = solve_nash_replicator(emp_payoff, iterations=10000)

        # Convert to full strategy space
        final_p1 = sum(sigma_p1[i] * population_p1[i] for i in range(n1))
        final_p2 = sum(sigma_p2[j] * population_p2[j] for j in range(n2))

        expected = np.array([1/3, 1/3, 1/3])
        error_p1 = np.abs(final_p1 - expected).max()
        error_p2 = np.abs(final_p2 - expected).max()

        print(f"\nPSRO on RPS ({n_psro_iterations} iterations, pop sizes: {n1}, {n2}):")
        print(f"  P1 Nash mixture: {np.round(final_p1, 4)}")
        print(f"  P2 Nash mixture: {np.round(final_p2, 4)}")
        print(f"  Max error: {max(error_p1, error_p2):.4f}")

        assert error_p1 < 0.05, f"P1 should be near uniform, got {final_p1}"
        assert error_p2 < 0.05, f"P2 should be near uniform, got {final_p2}"

    def test_psro_matching_pennies(self):
        """Test PSRO converges in Matching Pennies."""
        from rl_training.algorithms.psro.trainer import solve_nash_replicator

        payoff = get_matching_pennies_payoff()
        n_actions = 2

        population_p1 = [np.eye(n_actions)[0]]
        population_p2 = [np.eye(n_actions)[0]]

        for _ in range(8):
            n1, n2 = len(population_p1), len(population_p2)
            emp_payoff = np.zeros((n1, n2, 2))

            for i, pi in enumerate(population_p1):
                for j, pj in enumerate(population_p2):
                    for a1 in range(n_actions):
                        for a2 in range(n_actions):
                            emp_payoff[i, j, 0] += pi[a1] * pj[a2] * payoff[a1, a2, 0]
                            emp_payoff[i, j, 1] += pi[a1] * pj[a2] * payoff[a1, a2, 1]

            sigma_p1, sigma_p2 = solve_nash_replicator(emp_payoff, iterations=5000)

            opp_mix_p1 = sum(sigma_p2[j] * population_p2[j] for j in range(n2))
            opp_mix_p2 = sum(sigma_p1[i] * population_p1[i] for i in range(n1))

            br_values_p1 = payoff[:, :, 0] @ opp_mix_p1
            br_values_p2 = payoff[:, :, 1].T @ opp_mix_p2

            br_p1 = np.eye(n_actions)[np.argmax(br_values_p1)]
            br_p2 = np.eye(n_actions)[np.argmax(br_values_p2)]

            if not any(np.allclose(br_p1, p) for p in population_p1):
                population_p1.append(br_p1)
            if not any(np.allclose(br_p2, p) for p in population_p2):
                population_p2.append(br_p2)

        n1, n2 = len(population_p1), len(population_p2)
        emp_payoff = np.zeros((n1, n2, 2))
        for i, pi in enumerate(population_p1):
            for j, pj in enumerate(population_p2):
                for a1 in range(n_actions):
                    for a2 in range(n_actions):
                        emp_payoff[i, j, 0] += pi[a1] * pj[a2] * payoff[a1, a2, 0]
                        emp_payoff[i, j, 1] += pi[a1] * pj[a2] * payoff[a1, a2, 1]

        sigma_p1, sigma_p2 = solve_nash_replicator(emp_payoff, iterations=10000)
        final_p1 = sum(sigma_p1[i] * population_p1[i] for i in range(n1))
        final_p2 = sum(sigma_p2[j] * population_p2[j] for j in range(n2))

        expected = np.array([0.5, 0.5])
        error = max(np.abs(final_p1 - expected).max(), np.abs(final_p2 - expected).max())

        print(f"\nPSRO on Matching Pennies:")
        print(f"  P1: {np.round(final_p1, 4)}, P2: {np.round(final_p2, 4)}")
        print(f"  Max error: {error:.4f}")

        assert error < 0.05


# =============================================================================
# Test FCP Convergence
# =============================================================================

class TestFCPConvergence:
    """
    Test FCP (Fictitious Co-Play) convergence.

    FCP trains against historical snapshots. While not guaranteed to converge
    to Nash, it often does in practice for simple games.
    """

    def test_fcp_prisoners_dilemma(self):
        """Test FCP-style learning in Prisoner's Dilemma."""
        payoff = get_prisoners_dilemma_payoff()
        n_actions = 2
        n_iterations = 3000

        # Current policies (logits)
        logits_p1 = np.zeros(n_actions)
        logits_p2 = np.zeros(n_actions)

        # Historical snapshots
        snapshots_p1 = []
        snapshots_p2 = []
        snapshot_interval = 100

        def softmax(x):
            exp_x = np.exp(x - x.max())
            return exp_x / exp_x.sum()

        for t in range(n_iterations):
            # Save snapshots periodically
            if t % snapshot_interval == 0:
                snapshots_p1.append(softmax(logits_p1).copy())
                snapshots_p2.append(softmax(logits_p2).copy())

            # Sample opponent from historical snapshots (uniform)
            if len(snapshots_p1) > 0:
                opp_p1 = snapshots_p2[np.random.randint(len(snapshots_p2))]
                opp_p2 = snapshots_p1[np.random.randint(len(snapshots_p1))]
            else:
                opp_p1 = np.ones(n_actions) / n_actions
                opp_p2 = np.ones(n_actions) / n_actions

            # Sample actions
            policy_p1 = softmax(logits_p1)
            policy_p2 = softmax(logits_p2)
            action_p1 = np.random.choice(n_actions, p=policy_p1)
            action_p2 = np.random.choice(n_actions, p=opp_p1)

            # Get reward
            reward_p1 = payoff[action_p1, action_p2, 0]

            # Policy gradient update
            lr = 0.1
            grad = np.zeros(n_actions)
            grad[action_p1] = reward_p1
            grad -= policy_p1 * reward_p1  # Baseline
            logits_p1 += lr * grad

            # Same for P2
            action_p1_opp = np.random.choice(n_actions, p=opp_p2)
            action_p2_curr = np.random.choice(n_actions, p=policy_p2)
            reward_p2 = payoff[action_p1_opp, action_p2_curr, 1]

            grad = np.zeros(n_actions)
            grad[action_p2_curr] = reward_p2
            grad -= policy_p2 * reward_p2
            logits_p2 += lr * grad

        final_p1 = softmax(logits_p1)
        final_p2 = softmax(logits_p2)

        print(f"\nFCP on Prisoner's Dilemma ({n_iterations} iterations):")
        print(f"  P1 final policy: {np.round(final_p1, 4)} (expected [0, 1])")
        print(f"  P2 final policy: {np.round(final_p2, 4)} (expected [0, 1])")

        # FCP should converge to defect in PD (dominant strategy)
        assert final_p1[1] > 0.7, f"P1 should defect, got {final_p1}"
        assert final_p2[1] > 0.7, f"P2 should defect, got {final_p2}"


# =============================================================================
# Test Regret Matching Convergence
# =============================================================================

class TestRegretMatchingConvergence:
    """Test regret matching (core of CFR) on matrix games."""

    def test_regret_matching_rps(self):
        """Test regret matching converges to Nash in RPS."""
        payoff = get_rps_payoff()
        n_actions = 3
        n_iterations = 5000

        # Cumulative regrets and strategies
        regrets_p1 = np.zeros(n_actions)
        regrets_p2 = np.zeros(n_actions)
        strategy_sum_p1 = np.zeros(n_actions)
        strategy_sum_p2 = np.zeros(n_actions)

        def get_strategy(regrets):
            """Regret matching: normalize positive regrets."""
            positive = np.maximum(regrets, 0)
            total = positive.sum()
            if total > 0:
                return positive / total
            return np.ones(n_actions) / n_actions

        for t in range(n_iterations):
            # Get current strategies
            strategy_p1 = get_strategy(regrets_p1)
            strategy_p2 = get_strategy(regrets_p2)

            # Accumulate for average
            strategy_sum_p1 += strategy_p1
            strategy_sum_p2 += strategy_p2

            # Compute expected values
            ev_p1 = payoff[:, :, 0] @ strategy_p2  # EV for each P1 action
            ev_p2 = payoff[:, :, 1].T @ strategy_p1  # EV for each P2 action

            # Update regrets
            value_p1 = strategy_p1 @ ev_p1
            value_p2 = strategy_p2 @ ev_p2

            regrets_p1 += ev_p1 - value_p1
            regrets_p2 += ev_p2 - value_p2

        # Average strategies
        avg_p1 = strategy_sum_p1 / strategy_sum_p1.sum()
        avg_p2 = strategy_sum_p2 / strategy_sum_p2.sum()

        expected = np.array([1/3, 1/3, 1/3])
        error_p1 = np.abs(avg_p1 - expected).max()
        error_p2 = np.abs(avg_p2 - expected).max()

        print(f"\nRegret Matching on RPS ({n_iterations} iterations):")
        print(f"  P1 avg strategy: {np.round(avg_p1, 4)}")
        print(f"  P2 avg strategy: {np.round(avg_p2, 4)}")
        print(f"  Max error: {max(error_p1, error_p2):.6f}")

        assert error_p1 < 0.02, f"P1 should converge to uniform, got {avg_p1}"
        assert error_p2 < 0.02, f"P2 should converge to uniform, got {avg_p2}"

    def test_regret_matching_prisoners_dilemma(self):
        """Test regret matching converges to Nash in PD."""
        payoff = get_prisoners_dilemma_payoff()
        n_actions = 2
        n_iterations = 1000

        regrets_p1 = np.zeros(n_actions)
        regrets_p2 = np.zeros(n_actions)
        strategy_sum_p1 = np.zeros(n_actions)
        strategy_sum_p2 = np.zeros(n_actions)

        def get_strategy(regrets):
            positive = np.maximum(regrets, 0)
            total = positive.sum()
            if total > 0:
                return positive / total
            return np.ones(n_actions) / n_actions

        for t in range(n_iterations):
            strategy_p1 = get_strategy(regrets_p1)
            strategy_p2 = get_strategy(regrets_p2)

            strategy_sum_p1 += strategy_p1
            strategy_sum_p2 += strategy_p2

            ev_p1 = payoff[:, :, 0] @ strategy_p2
            ev_p2 = payoff[:, :, 1].T @ strategy_p1

            value_p1 = strategy_p1 @ ev_p1
            value_p2 = strategy_p2 @ ev_p2

            regrets_p1 += ev_p1 - value_p1
            regrets_p2 += ev_p2 - value_p2

        avg_p1 = strategy_sum_p1 / strategy_sum_p1.sum()
        avg_p2 = strategy_sum_p2 / strategy_sum_p2.sum()

        print(f"\nRegret Matching on Prisoner's Dilemma ({n_iterations} iterations):")
        print(f"  P1 avg strategy: {np.round(avg_p1, 4)} (expected [0, 1])")
        print(f"  P2 avg strategy: {np.round(avg_p2, 4)} (expected [0, 1])")

        assert avg_p1[1] > 0.95, f"P1 should defect, got {avg_p1}"
        assert avg_p2[1] > 0.95, f"P2 should defect, got {avg_p2}"


# =============================================================================
# Summary Test
# =============================================================================

class TestEquilibriumSummary:
    """Summary test showing all algorithms on all games."""

    def test_all_algorithms_summary(self):
        """Run all algorithms on RPS and PD, print summary."""
        from rl_training.algorithms.psro.trainer import solve_nash_replicator
        from iterative_game_analysis import MetaGame

        print("\n" + "=" * 70)
        print("EQUILIBRIUM CONVERGENCE SUMMARY")
        print("=" * 70)

        results = {}

        # RPS
        rps_expected = np.array([1/3, 1/3, 1/3])

        # Replicator on RPS
        payoff = get_rps_payoff()
        x, y = solve_nash_replicator(payoff, iterations=10000)
        results['Replicator/RPS'] = {
            'result': (x + y) / 2,  # Average of both players
            'expected': rps_expected,
            'error': max(np.abs(x - rps_expected).max(), np.abs(y - rps_expected).max())
        }

        # MENE on RPS
        mg = MetaGame(['R', 'P', 'S'], np.array([[0,-1,1],[1,0,-1],[-1,1,0]], dtype=np.float64))
        eq = mg.solve('mene')
        results['MENE/RPS'] = {
            'result': eq,
            'expected': rps_expected,
            'error': np.abs(eq - rps_expected).max()
        }

        # PD
        pd_expected = np.array([0, 1])  # Defect

        # Replicator on PD
        payoff = get_prisoners_dilemma_payoff()
        x, y = solve_nash_replicator(payoff, iterations=10000)
        results['Replicator/PD'] = {
            'result': (x + y) / 2,
            'expected': pd_expected,
            'error': max(1 - x[1], 1 - y[1])
        }

        # MENE on PD
        mg = MetaGame(['C', 'D'], np.array([[-1,-3],[0,-2]], dtype=np.float64))
        eq = mg.solve('mene')
        results['MENE/PD'] = {
            'result': eq,
            'expected': pd_expected,
            'error': 1 - eq[1]
        }

        # Print results
        print(f"\n{'Algorithm/Game':<25} {'Result':<30} {'Expected':<20} {'Error':<10}")
        print("-" * 85)

        all_passed = True
        for name, data in results.items():
            result_str = np.array2string(data['result'], precision=3, separator=',')
            expected_str = np.array2string(data['expected'], precision=3, separator=',')
            error = data['error']
            status = "✓" if error < 0.05 else "✗"
            if error >= 0.05:
                all_passed = False
            print(f"{name:<25} {result_str:<30} {expected_str:<20} {error:.4f} {status}")

        print("-" * 85)
        print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print("=" * 70)

        assert all_passed, "Some algorithms did not converge to Nash equilibrium"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
