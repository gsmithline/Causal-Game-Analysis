#!/usr/bin/env python3
"""Tests for the CPU bargaining environment implementation."""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from bargain_env_cpu import (
    BargainEnvCPU,
    run_games_with_policy,
    NUM_ACTIONS,
    OBS_DIM,
    ACTION_ACCEPT,
    ACTION_WALK,
    NUM_ITEM_TYPES,
    ITEM_QUANTITIES,
)


class TestBargainEnvCPU:
    """Tests for BargainEnvCPU class."""

    def test_init(self):
        """Test environment initialization."""
        env = BargainEnvCPU(num_envs=10, seed=42)
        assert env.num_envs == 10
        assert env.self_play is True

    def test_reset_returns_correct_shapes(self):
        """Test that reset returns arrays with correct shapes."""
        env = BargainEnvCPU(num_envs=16, seed=42)
        obs, info = env.reset()

        assert obs.shape == (16, OBS_DIM)
        assert info['action_mask'].shape == (16, NUM_ACTIONS)
        assert info['current_player'].shape == (16,)

    def test_reset_initializes_player_0(self):
        """Test that player 0 starts."""
        env = BargainEnvCPU(num_envs=10, seed=42)
        obs, info = env.reset()

        # All games should start with player 0
        assert np.all(info['current_player'] == 0)

    def test_action_mask_valid_at_start(self):
        """Test that action mask is valid at game start."""
        env = BargainEnvCPU(num_envs=10, seed=42)
        obs, info = env.reset()

        mask = info['action_mask']

        # At start: all counteroffers valid, no ACCEPT (no offer yet), WALK valid
        assert np.all(mask[:, 0:80] == 1.0)  # Counteroffers
        assert np.all(mask[:, ACTION_ACCEPT] == 0.0)  # No offer to accept
        assert np.all(mask[:, ACTION_WALK] == 1.0)  # Can walk

    def test_step_changes_player(self):
        """Test that step changes current player."""
        env = BargainEnvCPU(num_envs=10, seed=42)
        obs, info = env.reset()

        # Player 0 makes an offer
        actions = np.zeros(10, dtype=np.int32)  # Action 0 = offer (0,0,0)
        obs, rewards, dones, truncs, info = env.step(actions)

        # Now should be player 1's turn
        assert np.all(info['current_player'] == 1)

    def test_accept_with_offer(self):
        """Test accepting an offer ends the game."""
        env = BargainEnvCPU(num_envs=1, seed=42)
        obs, info = env.reset()

        # P1 makes offer
        actions = np.array([0], dtype=np.int32)
        obs, rewards, dones, truncs, info = env.step(actions)

        assert not dones[0]  # Game not done yet

        # P2 accepts
        actions = np.array([ACTION_ACCEPT], dtype=np.int32)
        obs, rewards, dones, truncs, info = env.step(actions)

        assert dones[0]  # Game should be done
        assert rewards[0, 0] >= 0  # Player 0 should have reward
        assert rewards[0, 1] >= 0  # Player 1 should have reward

    def test_walk_ends_game(self):
        """Test that walking ends the game."""
        env = BargainEnvCPU(num_envs=1, seed=42)
        obs, info = env.reset()

        # P1 walks immediately
        actions = np.array([ACTION_WALK], dtype=np.int32)
        obs, rewards, dones, truncs, info = env.step(actions)

        assert dones[0]
        # Rewards should be normalized outside options
        assert 0 <= rewards[0, 0] <= 1
        assert 0 <= rewards[0, 1] <= 1

    def test_auto_reset(self):
        """Test that auto_reset works."""
        env = BargainEnvCPU(num_envs=10, seed=42)
        obs, info = env.reset()

        # Walk all games to end them
        actions = np.full(10, ACTION_WALK, dtype=np.int32)
        obs, rewards, dones, truncs, info = env.step(actions)

        assert np.all(dones)

        # Auto reset
        env.auto_reset()

        # Games should be reset
        assert not np.any(env._done)
        assert np.all(env._current_player == 0)

    def test_sample_valid_actions(self):
        """Test that sampled actions are valid."""
        env = BargainEnvCPU(num_envs=100, seed=42)
        obs, info = env.reset()

        for _ in range(10):
            actions = env.sample_valid_actions()

            # Check all actions are valid according to current internal mask
            for i in range(env.num_envs):
                if not env._done[i]:
                    assert env._action_masks[i, actions[i]] == 1.0

            obs, rewards, dones, truncs, info = env.step(actions)
            env.auto_reset()

    def test_decode_action(self):
        """Test action decoding."""
        # Test ACCEPT
        action_type, offer = BargainEnvCPU.decode_action(ACTION_ACCEPT)
        assert action_type == "ACCEPT"
        assert offer is None

        # Test WALK
        action_type, offer = BargainEnvCPU.decode_action(ACTION_WALK)
        assert action_type == "WALK"
        assert offer is None

        # Test counteroffer 0 = (0, 0, 0)
        action_type, offer = BargainEnvCPU.decode_action(0)
        assert action_type == "COUNTEROFFER"
        assert offer == (0, 0, 0)

        # Test counteroffer encoding/decoding roundtrip
        for n0 in range(8):
            for n1 in range(5):
                for n2 in range(2):
                    action = BargainEnvCPU.encode_offer((n0, n1, n2))
                    action_type, decoded = BargainEnvCPU.decode_action(action)
                    assert decoded == (n0, n1, n2)

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        env1 = BargainEnvCPU(num_envs=10, seed=12345)
        env2 = BargainEnvCPU(num_envs=10, seed=12345)

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        np.testing.assert_array_equal(obs1, obs2)

    def test_final_turn_restrictions(self):
        """Test that final turn only allows ACCEPT or WALK."""
        env = BargainEnvCPU(num_envs=1, seed=42)
        obs, info = env.reset()

        # Play 5 actions (alternating offers)
        for i in range(5):
            actions = np.array([i % 80], dtype=np.int32)  # Various counteroffers
            obs, rewards, dones, truncs, info = env.step(actions)

        # On 6th action (final turn), should only allow ACCEPT or WALK
        mask = info['action_mask'][0]
        assert mask[ACTION_ACCEPT] == 1.0 or mask[ACTION_WALK] == 1.0
        # Counteroffers should be disabled
        assert np.all(mask[0:80] == 0.0)

    def test_game_data_collection(self):
        """Test get_game_data returns correct data."""
        env = BargainEnvCPU(num_envs=5, seed=42)
        env.reset()

        data = env.get_game_data()

        assert data['player_values'].shape == (5, 2, 3)
        assert data['outside_options'].shape == (5, 2)
        assert data['rewards'].shape == (5, 2)
        assert data['outcome'].shape == (5,)
        assert data['done'].shape == (5,)


class TestRunGamesWithPolicy:
    """Tests for run_games_with_policy helper."""

    def test_random_policy(self):
        """Test running games with random policy."""
        env = BargainEnvCPU(num_envs=32, seed=42)

        def random_policy(obs, mask):
            actions = np.zeros(obs.shape[0], dtype=np.int32)
            for i in range(obs.shape[0]):
                valid = np.where(mask[i] > 0)[0]
                if len(valid) > 0:
                    actions[i] = np.random.choice(valid)
            return actions

        results = run_games_with_policy(
            env, random_policy, num_games=100, collect_data=True
        )

        assert len(results['rewards_p0']) == 100
        assert len(results['rewards_p1']) == 100
        assert len(results['outcomes']) == 100
        assert len(results['game_data']) == 100

    def test_always_walk_policy(self):
        """Test policy that always walks."""
        env = BargainEnvCPU(num_envs=16, seed=42)

        def always_walk(obs, mask):
            return np.full(obs.shape[0], ACTION_WALK, dtype=np.int32)

        results = run_games_with_policy(
            env, always_walk, num_games=50, collect_data=False
        )

        assert len(results['rewards_p0']) == 50
        # All outcomes should be WALK (code 2)
        assert all(o == 2 for o in results['outcomes'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
