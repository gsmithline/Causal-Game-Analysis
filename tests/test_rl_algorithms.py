"""
Comprehensive tests for all RL algorithms using mock CPU environment.

This test suite validates that all RL algorithms work correctly by using
a CPU-based mock bargaining environment instead of the CUDA version.

Run with: pytest tests/test_rl_algorithms.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_training.envs.mock_bargain_env import MockBargainEnv, MockBargainEnvWrapper


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_env():
    """Small environment for quick tests."""
    return MockBargainEnv(num_envs=32, seed=42)


@pytest.fixture
def medium_env():
    """Medium environment for algorithm tests."""
    return MockBargainEnv(num_envs=256, seed=42)


# =============================================================================
# Mock Environment Tests
# =============================================================================

class TestMockEnvironment:
    """Test the mock environment itself."""

    def test_reset(self, small_env):
        """Test environment reset."""
        obs, info = small_env.reset()
        assert obs.shape == (32, 92)
        assert info['action_mask'].shape == (32, 82)
        assert info['current_player'].shape == (32,)

    def test_step_counteroffer(self, small_env):
        """Test counteroffer action."""
        obs, info = small_env.reset()
        actions = torch.zeros(32, dtype=torch.long)  # Action 0 = offer (0,0,0)
        result = small_env.step(actions)

        assert result.observations.shape == (32, 92)
        assert result.rewards.shape == (32, 2)
        assert not result.terminated.all()  # Counteroffers don't end game

    def test_step_accept(self, small_env):
        """Test accept action after an offer."""
        obs, info = small_env.reset()

        # Player 0 makes offer
        actions = torch.zeros(32, dtype=torch.long)
        small_env.step(actions)

        # Player 1 accepts
        actions = torch.full((32,), MockBargainEnv.ACTION_ACCEPT)
        result = small_env.step(actions)

        assert result.terminated.all()  # Accept ends game
        assert (result.rewards.sum(dim=1) > 0).any()  # Someone gets rewards

    def test_step_walk(self, small_env):
        """Test walk action."""
        obs, info = small_env.reset()
        actions = torch.full((32,), MockBargainEnv.ACTION_WALK)
        result = small_env.step(actions)

        assert result.terminated.all()  # Walk ends game

    def test_auto_reset(self, small_env):
        """Test automatic reset of done environments."""
        obs, info = small_env.reset()

        # Walk to end games
        actions = torch.full((32,), MockBargainEnv.ACTION_WALK)
        result = small_env.step(actions)
        assert result.terminated.all()

        # Auto reset
        small_env.auto_reset()
        assert not small_env.dones.any()

    def test_action_masking(self, small_env):
        """Test action masking logic."""
        obs, info = small_env.reset()

        # Initially, accept should be masked (no offer yet)
        mask = info['action_mask']
        assert (mask[:, MockBargainEnv.ACTION_ACCEPT] == 0).all()

        # After an offer, accept should be valid
        actions = torch.zeros(32, dtype=torch.long)
        result = small_env.step(actions)
        mask = result.info['action_mask']
        assert (mask[:, MockBargainEnv.ACTION_ACCEPT] == 1).all()


# =============================================================================
# PPO Tests
# =============================================================================

class TestPPO:
    """Test PPO algorithm."""

    def test_ppo_import(self):
        """Test PPO can be imported."""
        from rl_training.algorithms.ppo_bargain.trainer import PPOBargainTrainer
        from rl_training.algorithms.ppo_bargain.config import PPOBargainConfig
        assert PPOBargainTrainer is not None
        assert PPOBargainConfig is not None

    def test_ppo_config(self):
        """Test PPO config creation."""
        from rl_training.algorithms.ppo_bargain.config import PPOBargainConfig
        config = PPOBargainConfig(
            num_envs=32,
            total_timesteps=100,
            rollout_steps=16,
        )
        config.validate()
        assert config.num_envs == 32

    def test_ppo_network(self, small_env):
        """Test PPO with BargainMLP network."""
        from rl_training.networks.bargain_mlp import BargainMLP

        net = BargainMLP(obs_dim=92, num_actions=82)
        obs, info = small_env.reset()
        mask = info['action_mask']

        # Forward pass
        logits, values = net(obs, mask)
        assert logits.shape == (32, 82)
        assert values.shape == (32,)

        # Get action
        actions, log_probs, values = net.get_action(obs, mask)
        assert actions.shape == (32,)
        assert (actions < 82).all()

    def test_ppo_training_loop(self, small_env):
        """Test a mini PPO training loop."""
        from rl_training.networks.bargain_mlp import BargainMLP
        import torch.nn.functional as F

        net = BargainMLP(obs_dim=92, num_actions=82)
        optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

        # Collect rollout
        obs, info = small_env.reset()
        mask = info['action_mask']

        obs_list, action_list, reward_list, logprob_list, value_list = [], [], [], [], []

        for _ in range(8):  # Short rollout
            with torch.no_grad():
                actions, log_probs, values = net.get_action(obs, mask)

            obs_list.append(obs)
            action_list.append(actions)
            logprob_list.append(log_probs)
            value_list.append(values)

            # Step with current player's action
            result = small_env.step(actions)
            current_player = small_env.get_current_player()

            # Get rewards for learning player (player 0)
            rewards = result.rewards[:, 0]
            reward_list.append(rewards)

            if result.terminated.any():
                small_env.auto_reset()

            obs = result.observations
            mask = result.info['action_mask']

        # PPO update
        obs_batch = torch.stack(obs_list)
        action_batch = torch.stack(action_list)
        old_logprobs = torch.stack(logprob_list)
        returns = torch.stack(reward_list)  # Simplified returns

        # Flatten
        obs_batch = obs_batch.view(-1, 92)
        action_batch = action_batch.view(-1)
        old_logprobs = old_logprobs.view(-1)
        returns = returns.view(-1)

        # Get fresh predictions
        logits, values = net(obs_batch)
        dist = torch.distributions.Categorical(logits=logits)
        new_logprobs = dist.log_prob(action_batch)
        entropy = dist.entropy()

        # PPO loss
        ratio = torch.exp(new_logprobs - old_logprobs)
        advantages = returns - values.detach()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)
        print(f"PPO loss: {loss.item():.4f}")


# =============================================================================
# NFSP Tests
# =============================================================================

class TestNFSP:
    """Test NFSP algorithm."""

    def test_nfsp_import(self):
        """Test NFSP can be imported."""
        from rl_training.algorithms.nfsp_bargain.trainer import NFSPBargainTrainer
        from rl_training.algorithms.nfsp_bargain.config import NFSPBargainConfig
        assert NFSPBargainTrainer is not None

    def test_nfsp_config(self):
        """Test NFSP config."""
        from rl_training.algorithms.nfsp_bargain.config import NFSPBargainConfig
        config = NFSPBargainConfig(
            num_envs=32,
            total_timesteps=100,
            eta=0.1,
        )
        config.validate()
        assert config.eta == 0.1

    def test_nfsp_networks(self, small_env):
        """Test NFSP dual network architecture."""
        from rl_training.networks.bargain_mlp import BargainMLP

        # Best Response network (Q-network style)
        br_net = BargainMLP(obs_dim=92, num_actions=82)

        # Average policy network
        avg_net = BargainMLP(obs_dim=92, num_actions=82)

        obs, info = small_env.reset()
        mask = info['action_mask']

        # BR network forward
        br_logits, _ = br_net(obs, mask)
        assert br_logits.shape == (32, 82)

        # Average policy forward
        avg_logits, _ = avg_net(obs, mask)
        assert avg_logits.shape == (32, 82)

    def test_nfsp_mode_selection(self, small_env):
        """Test NFSP eta-greedy mode selection."""
        eta = 0.1  # Probability of using BR

        n_samples = 1000
        br_count = sum(1 for _ in range(n_samples) if np.random.random() < eta)

        # Should be roughly 10% BR
        assert 50 < br_count < 150  # Allow variance


# =============================================================================
# MAPPO Tests
# =============================================================================

class TestMAPPO:
    """Test MAPPO algorithm."""

    def test_mappo_import(self):
        """Test MAPPO can be imported."""
        from rl_training.algorithms.mappo.trainer import MAPPOTrainer
        from rl_training.algorithms.mappo.config import MAPPOConfig
        assert MAPPOTrainer is not None

    def test_mappo_config(self):
        """Test MAPPO config."""
        from rl_training.algorithms.mappo.config import MAPPOConfig
        config = MAPPOConfig(
            num_envs=32,
            total_timesteps=100,
        )
        config.validate()

    def test_mappo_separate_actors(self, small_env):
        """Test MAPPO with separate actor networks per player."""
        from rl_training.networks.bargain_mlp import BargainMLP

        # Separate actors for each player
        actor_p0 = BargainMLP(obs_dim=92, num_actions=82)
        actor_p1 = BargainMLP(obs_dim=92, num_actions=82)

        # Shared critic
        class SharedCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(92, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )

            def forward(self, obs):
                return self.net(obs).squeeze(-1)

        critic = SharedCritic()

        obs, info = small_env.reset()
        mask = info['action_mask']
        current_player = small_env.get_current_player()

        # Player 0's turn
        p0_mask = current_player == 0
        if p0_mask.any():
            p0_logits, _ = actor_p0(obs[p0_mask], mask[p0_mask])
            p0_values = critic(obs[p0_mask])
            assert p0_logits.shape[1] == 82

        print("MAPPO separate actors test passed")


# =============================================================================
# PSRO Tests
# =============================================================================

class TestPSRO:
    """Test PSRO algorithm."""

    def test_psro_import(self):
        """Test PSRO can be imported."""
        from rl_training.algorithms.psro.trainer import PSROTrainer
        from rl_training.algorithms.psro.config import PSROConfig
        assert PSROTrainer is not None

    def test_psro_config(self):
        """Test PSRO config."""
        from rl_training.algorithms.psro.config import PSROConfig
        config = PSROConfig(
            num_envs=32,
            max_policies=5,
            br_training_steps=100,
        )
        config.validate()

    def test_psro_nash_solvers(self):
        """Test PSRO Nash equilibrium solvers."""
        from rl_training.algorithms.psro.trainer import (
            solve_nash_replicator,
            solve_nash_fictitious_play
        )

        # Simple 2x2 game
        payoff = np.array([
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ], dtype=np.float64)

        # Replicator dynamics
        x, y = solve_nash_replicator(payoff, iterations=5000)
        assert abs(x[0] - 0.5) < 0.1
        assert abs(y[0] - 0.5) < 0.1

        # Fictitious play
        x, y = solve_nash_fictitious_play(payoff, iterations=5000)
        assert abs(x[0] - 0.5) < 0.2
        assert abs(y[0] - 0.5) < 0.2

    def test_psro_population(self, small_env):
        """Test PSRO policy population management."""
        from rl_training.networks.bargain_mlp import BargainMLP
        from copy import deepcopy

        # Simulate policy population
        policies = []
        for _ in range(3):
            policy = BargainMLP(obs_dim=92, num_actions=82)
            policies.append(policy)

        # Sample from mixture
        mixture = np.array([0.5, 0.3, 0.2])

        obs, info = small_env.reset()
        mask = info['action_mask']
        batch_size = obs.shape[0]

        # Sample which policy for each env
        policy_indices = np.random.choice(len(policies), size=batch_size, p=mixture)

        actions = torch.zeros(batch_size, dtype=torch.long)
        for idx in range(len(policies)):
            policy_mask = torch.tensor(policy_indices == idx)
            if policy_mask.any():
                with torch.no_grad():
                    policy_actions, _, _ = policies[idx].get_action(
                        obs[policy_mask], mask[policy_mask]
                    )
                actions[policy_mask] = policy_actions

        assert actions.shape == (batch_size,)
        print("PSRO population sampling test passed")


# =============================================================================
# Ex2PSRO Tests
# =============================================================================

class TestEx2PSRO:
    """Test Ex2PSRO algorithm."""

    def test_ex2psro_import(self):
        """Test Ex2PSRO can be imported."""
        from rl_training.algorithms.ex2psro.trainer import Ex2PSROTrainer
        from rl_training.algorithms.ex2psro.config import Ex2PSROConfig
        assert Ex2PSROTrainer is not None

    def test_ex2psro_config(self):
        """Test Ex2PSRO config."""
        from rl_training.algorithms.ex2psro.config import Ex2PSROConfig
        config = Ex2PSROConfig(
            num_envs=32,
            welfare_fn="utilitarian",
            kl_coef=0.1,
        )
        config.validate()
        assert config.welfare_fn == "utilitarian"

    def test_ex2psro_welfare_functions(self):
        """Test Ex2PSRO welfare function calculations."""
        # Simulate payoffs
        payoffs = np.array([0.5, 0.3, 0.8, 0.2])

        # Utilitarian welfare
        utilitarian = np.mean(payoffs)
        assert abs(utilitarian - 0.45) < 0.01

        # Egalitarian welfare (minimum)
        egalitarian = np.min(payoffs)
        assert egalitarian == 0.2

        # Nash welfare (product, use log for stability)
        nash = np.exp(np.sum(np.log(payoffs)))
        assert nash > 0

    def test_ex2psro_kl_regularization(self, small_env):
        """Test KL divergence regularization."""
        import torch.nn.functional as F

        # Two policies
        from rl_training.networks.bargain_mlp import BargainMLP
        policy = BargainMLP(obs_dim=92, num_actions=82)
        exploration = BargainMLP(obs_dim=92, num_actions=82)

        obs, info = small_env.reset()
        mask = info['action_mask']

        # Get action distributions (without masking for clean KL computation)
        logits_policy, _ = policy(obs)
        logits_explore, _ = exploration(obs)

        probs_policy = F.softmax(logits_policy, dim=-1)
        probs_explore = F.softmax(logits_explore, dim=-1)

        # Add small epsilon to avoid log(0)
        eps = 1e-8
        probs_policy = probs_policy + eps
        probs_explore = probs_explore + eps

        # Renormalize
        probs_policy = probs_policy / probs_policy.sum(dim=-1, keepdim=True)
        probs_explore = probs_explore / probs_explore.sum(dim=-1, keepdim=True)

        # KL divergence: KL(explore || policy) = sum(explore * log(explore/policy))
        kl = (probs_explore * (probs_explore.log() - probs_policy.log())).sum(dim=-1).mean()
        assert kl >= -eps  # KL is always non-negative (allow small numerical error)
        assert not torch.isnan(kl)
        print(f"KL divergence: {kl.item():.4f}")


# =============================================================================
# FCP Tests
# =============================================================================

class TestFCP:
    """Test Fictitious Co-Play algorithm."""

    def test_fcp_import(self):
        """Test FCP can be imported."""
        from rl_training.algorithms.fcp.trainer import FCPTrainer
        from rl_training.algorithms.fcp.config import FCPConfig
        assert FCPTrainer is not None

    def test_fcp_config(self):
        """Test FCP config."""
        from rl_training.algorithms.fcp.config import FCPConfig
        config = FCPConfig(
            num_envs=32,
            snapshot_interval=100,
        )
        config.validate()

    def test_fcp_snapshot_management(self, small_env):
        """Test FCP policy snapshot mechanism."""
        from rl_training.networks.bargain_mlp import BargainMLP
        from copy import deepcopy

        # Simulate snapshot collection
        snapshots = []
        policy = BargainMLP(obs_dim=92, num_actions=82)

        for i in range(5):
            # Simulate training changing weights
            with torch.no_grad():
                for param in policy.parameters():
                    param.add_(torch.randn_like(param) * 0.01)

            # Take snapshot
            snapshot = deepcopy(policy)
            snapshot.eval()
            snapshots.append(snapshot)

        # Verify snapshots are different
        params_0 = list(snapshots[0].parameters())[0]
        params_4 = list(snapshots[4].parameters())[0]
        assert not torch.allclose(params_0, params_4)

        # Sample opponent from snapshots
        obs, info = small_env.reset()
        mask = info['action_mask']

        opp_idx = np.random.randint(len(snapshots))
        with torch.no_grad():
            opp_actions, _, _ = snapshots[opp_idx].get_action(obs, mask)

        assert opp_actions.shape == (32,)
        print("FCP snapshot management test passed")


# =============================================================================
# Sampled CFR Tests
# =============================================================================

class TestSampledCFR:
    """Test Sampled/Deep CFR algorithm."""

    def test_sampled_cfr_import(self):
        """Test Sampled CFR can be imported."""
        from rl_training.algorithms.sampled_cfr.trainer import SampledCFRTrainer
        from rl_training.algorithms.sampled_cfr.config import SampledCFRConfig
        assert SampledCFRTrainer is not None

    def test_sampled_cfr_config(self):
        """Test Sampled CFR config."""
        from rl_training.algorithms.sampled_cfr.config import SampledCFRConfig
        config = SampledCFRConfig(
            num_envs=32,
            total_timesteps=100,
        )
        config.validate()

    def test_sampled_cfr_networks(self, small_env):
        """Test Sampled CFR advantage and strategy networks."""
        from rl_training.algorithms.sampled_cfr.trainer import (
            AdvantageNetwork,
            StrategyNetwork
        )

        obs, info = small_env.reset()
        mask = info['action_mask']

        # Advantage network
        adv_net = AdvantageNetwork(obs_dim=92, num_actions=82)
        advantages = adv_net(obs)
        assert advantages.shape == (32, 82)

        # Strategy network
        strat_net = StrategyNetwork(obs_dim=92, num_actions=82)
        strategy = strat_net(obs, mask)
        assert strategy.shape == (32, 82)
        assert torch.allclose(strategy.sum(dim=-1), torch.ones(32), atol=1e-5)

    def test_sampled_cfr_regret_matching(self, small_env):
        """Test regret matching in Sampled CFR."""
        import torch.nn.functional as F

        # Simulate advantages
        advantages = torch.randn(32, 82)

        # Regret matching: positive regrets normalized
        positive_adv = F.relu(advantages)
        strategy = positive_adv / (positive_adv.sum(dim=-1, keepdim=True) + 1e-8)

        # Strategy should be valid probability distribution
        assert (strategy >= 0).all()
        assert torch.allclose(strategy.sum(dim=-1), torch.ones(32), atol=1e-5)


# =============================================================================
# Integration Test: Full Training Loop
# =============================================================================

class TestIntegration:
    """Integration tests running mini training loops."""

    def test_full_game_episode(self, small_env):
        """Test playing full game episodes."""
        from rl_training.networks.bargain_mlp import BargainMLP

        policy = BargainMLP(obs_dim=92, num_actions=82)

        obs, info = small_env.reset()
        mask = info['action_mask']

        games_completed = 0
        max_steps = 100

        for step in range(max_steps):
            with torch.no_grad():
                actions, _, _ = policy.get_action(obs, mask)

            result = small_env.step(actions)

            if result.terminated.any():
                games_completed += result.terminated.sum().item()
                small_env.auto_reset()

            obs = result.observations
            mask = result.info['action_mask']

        assert games_completed > 0
        print(f"Completed {games_completed} games in {max_steps} steps")

    def test_self_play_rewards(self, medium_env):
        """Test self-play produces reasonable rewards."""
        from rl_training.networks.bargain_mlp import BargainMLP

        policy = BargainMLP(obs_dim=92, num_actions=82)

        obs, info = medium_env.reset()
        mask = info['action_mask']

        total_rewards = torch.zeros(2)
        games = 0

        for _ in range(200):
            with torch.no_grad():
                actions, _, _ = policy.get_action(obs, mask)

            result = medium_env.step(actions)

            if result.terminated.any():
                total_rewards += result.rewards[result.terminated].sum(dim=0)
                games += result.terminated.sum().item()
                medium_env.auto_reset()

            obs = result.observations
            mask = result.info['action_mask']

        if games > 0:
            avg_rewards = total_rewards / games
            print(f"Average rewards: P0={avg_rewards[0]:.3f}, P1={avg_rewards[1]:.3f}")
            # Rewards should be in reasonable range
            assert avg_rewards.abs().max() < 10


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
