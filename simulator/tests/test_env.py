"""
Unit tests for CUDA Bargaining Game Environment.

Run with: pytest tests/test_env.py -v
"""

import pytest
import torch
import numpy as np

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


@pytest.fixture
def env():
    """Create a small test environment."""
    from cuda_bargain import BargainEnv
    return BargainEnv(num_envs=128, device=0)


@pytest.fixture
def large_env():
    """Create a larger environment for stress testing."""
    from cuda_bargain import BargainEnv
    return BargainEnv(num_envs=8192, device=0)


class TestInitialization:
    """Test environment initialization."""

    def test_create_env(self):
        from cuda_bargain import BargainEnv
        env = BargainEnv(num_envs=64)
        assert env.num_envs == 64

    def test_create_large_env(self):
        from cuda_bargain import BargainEnv
        env = BargainEnv(num_envs=32768)
        assert env.num_envs == 32768

    def test_constants(self):
        from cuda_bargain import (
            NUM_ACTIONS, OBS_DIM, ACTION_ACCEPT, ACTION_WALK,
            NUM_ITEM_TYPES, MAX_ROUNDS, ITEM_QUANTITIES
        )
        assert NUM_ACTIONS == 82
        assert OBS_DIM == 92
        assert ACTION_ACCEPT == 80
        assert ACTION_WALK == 81
        assert NUM_ITEM_TYPES == 3
        assert MAX_ROUNDS == 3
        assert ITEM_QUANTITIES == (7, 4, 1)


class TestReset:
    """Test reset functionality."""

    def test_reset_returns_correct_shapes(self, env):
        obs, info = env.reset(seed=42)

        assert obs.shape == (128, 92)
        assert obs.dtype == torch.float32
        assert obs.is_cuda

        assert 'action_mask' in info
        assert info['action_mask'].shape == (128, 82)
        assert info['action_mask'].dtype == torch.float32

        assert 'current_player' in info
        assert info['current_player'].shape == (128,)

    def test_reset_deterministic_with_seed(self, env):
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        assert torch.allclose(obs1, obs2)

    def test_reset_different_with_different_seeds(self, env):
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=456)
        assert not torch.allclose(obs1, obs2)

    def test_initial_player_is_zero(self, env):
        _, info = env.reset()
        current_players = info['current_player']
        assert (current_players == 0).all()


class TestActionMasking:
    """Test action mask validity."""

    def test_round1_player1_no_accept(self, env):
        """Round 1, Player 1 cannot ACCEPT (no offer yet)."""
        from cuda_bargain import ACTION_ACCEPT, ACTION_WALK

        _, info = env.reset()
        mask = info['action_mask']

        # Cannot accept (no offer)
        assert (mask[:, ACTION_ACCEPT] == 0).all()
        # Can always walk
        assert (mask[:, ACTION_WALK] == 1).all()
        # Can make counteroffers (first 80 actions)
        assert (mask[:, :80] == 1).all()

    def test_counteroffer_enables_accept(self, env):
        """After a counteroffer, opponent can ACCEPT."""
        from cuda_bargain import ACTION_ACCEPT, ACTION_WALK

        _, info = env.reset()

        # P1 makes a counteroffer (action 0 = offer [0,0,0])
        actions = torch.zeros(128, dtype=torch.int32, device='cuda')
        _, _, _, _, info = env.step(actions)

        mask = info['action_mask']
        current_player = info['current_player']

        # Now it's P2's turn
        assert (current_player == 1).all()
        # P2 can accept
        assert (mask[:, ACTION_ACCEPT] == 1).all()
        # P2 can walk
        assert (mask[:, ACTION_WALK] == 1).all()


class TestStep:
    """Test step functionality."""

    def test_step_returns_correct_shapes(self, env):
        obs, info = env.reset()

        actions = torch.zeros(128, dtype=torch.int32, device='cuda')
        obs, rewards, dones, truncateds, info = env.step(actions)

        assert obs.shape == (128, 92)
        assert rewards.shape == (128, 2)
        assert dones.shape == (128,)
        assert truncateds.shape == (128,)
        assert info['action_mask'].shape == (128, 82)

    def test_walk_terminates_game(self, env):
        """WALK action should terminate the game."""
        from cuda_bargain import ACTION_WALK

        _, info = env.reset()

        actions = torch.full((128,), ACTION_WALK, dtype=torch.int32, device='cuda')
        _, rewards, dones, _, _ = env.step(actions)

        assert dones.all()
        # Rewards should be non-negative (outside offers)
        assert (rewards >= 0).all()

    def test_accept_terminates_game(self, env):
        """ACCEPT action should terminate the game."""
        from cuda_bargain import ACTION_ACCEPT

        _, info = env.reset()

        # P1 makes counteroffer
        actions = torch.zeros(128, dtype=torch.int32, device='cuda')
        env.step(actions)

        # P2 accepts
        actions = torch.full((128,), ACTION_ACCEPT, dtype=torch.int32, device='cuda')
        _, rewards, dones, _, _ = env.step(actions)

        assert dones.all()
        # Both players should have rewards
        assert (rewards[:, 0] >= 0).all()
        assert (rewards[:, 1] >= 0).all()

    def test_counteroffer_advances_turn(self, env):
        """Counteroffer should switch to other player."""
        _, info = env.reset()

        # Initially P1's turn
        assert (info['current_player'] == 0).all()

        # P1 counteroffers
        actions = torch.zeros(128, dtype=torch.int32, device='cuda')
        _, _, dones, _, info = env.step(actions)

        # Now P2's turn
        assert not dones.any()
        assert (info['current_player'] == 1).all()


class TestGameLogic:
    """Test game logic correctness."""

    def test_full_game_to_accept(self, env):
        """Play a full game ending in accept."""
        from cuda_bargain import ACTION_ACCEPT

        obs, info = env.reset()
        total_steps = 0
        max_steps = 10  # Should finish well before this

        while not info.get('dones', torch.zeros(1)).all() and total_steps < max_steps:
            # Get current player
            current_player = info['current_player']
            mask = info['action_mask']

            # Choose first valid action (counteroffer or accept if available)
            actions = torch.zeros(128, dtype=torch.int32, device='cuda')
            for i in range(128):
                for a in range(82):
                    if mask[i, a] > 0.5:
                        actions[i] = a
                        break

            obs, rewards, dones, _, info = env.step(actions)
            total_steps += 1

            if dones.all():
                break

        # Game should have ended
        assert dones.all() or total_steps == max_steps

    def test_final_round_p2_no_counteroffer(self, env):
        """In final round, P2 can only ACCEPT or WALK."""
        from cuda_bargain import ACTION_ACCEPT, ACTION_WALK

        _, info = env.reset()

        # Simulate to final round
        # Round 1: P1 offer, P2 offer
        # Round 2: P1 offer, P2 offer
        # Round 3: P1 offer, P2 (final - no counteroffer)

        for step in range(5):  # 5 counteroffers to get to round 3 P2
            mask = info['action_mask']
            # Use first valid counteroffer
            actions = torch.zeros(128, dtype=torch.int32, device='cuda')
            for i in range(128):
                for a in range(80):  # Only counteroffers
                    if mask[i, a] > 0.5:
                        actions[i] = a
                        break

            _, _, dones, _, info = env.step(actions)

        # Now at round 3, P2's turn
        mask = info['action_mask']
        current_player = info['current_player']

        # Should be P2's turn
        assert (current_player == 1).all()
        # P2 cannot counteroffer (actions 0-79 should be masked)
        assert (mask[:, :80] == 0).all()
        # Can only ACCEPT or WALK
        assert (mask[:, ACTION_ACCEPT] == 1).all()
        assert (mask[:, ACTION_WALK] == 1).all()


class TestRewards:
    """Test reward calculation."""

    def test_rewards_normalized(self, env):
        """Rewards should be normalized between 0 and 1."""
        from cuda_bargain import ACTION_WALK

        _, _ = env.reset()

        # Walk immediately
        actions = torch.full((128,), ACTION_WALK, dtype=torch.int32, device='cuda')
        _, rewards, _, _, _ = env.step(actions)

        assert (rewards >= 0).all()
        assert (rewards <= 1).all()

    def test_accept_rewards_consistent(self, env):
        """Accept rewards should reflect item division."""
        from cuda_bargain import ACTION_ACCEPT

        _, _ = env.reset()

        # P1 offers all items (action 79 = [7,4,1])
        actions = torch.full((128,), 79, dtype=torch.int32, device='cuda')
        env.step(actions)

        # P2 accepts
        actions = torch.full((128,), ACTION_ACCEPT, dtype=torch.int32, device='cuda')
        _, rewards, _, _, _ = env.step(actions)

        # P2 should have positive reward (got all items)
        # P1 should have zero reward (kept nothing)
        assert (rewards[:, 1] > 0).all()
        assert (rewards[:, 0] == 0).all()


class TestAutoReset:
    """Test auto-reset functionality."""

    def test_auto_reset_resets_done_games(self, env):
        from cuda_bargain import ACTION_WALK

        _, _ = env.reset()

        # Walk to end all games
        actions = torch.full((128,), ACTION_WALK, dtype=torch.int32, device='cuda')
        _, _, dones, _, _ = env.step(actions)

        assert dones.all()

        # Auto reset
        env.auto_reset()

        # Games should be reset
        dones = env.get_dones()
        assert not dones.any()


class TestRandomActions:
    """Test random action generation."""

    def test_get_opponent_actions(self, env):
        _, _ = env.reset()

        # P1 makes offer first
        actions = torch.zeros(128, dtype=torch.int32, device='cuda')
        env.step(actions)

        # Get random actions for P2 (opponent)
        opponent_actions = env.get_opponent_actions(opponent_player=1)

        assert opponent_actions.shape == (128,)
        assert opponent_actions.dtype == torch.int32
        # All should be valid actions (0-81)
        assert (opponent_actions >= 0).all()
        assert (opponent_actions < 82).all()

    def test_sample_valid_actions(self, env):
        _, info = env.reset()

        actions = env.sample_valid_actions()
        mask = info['action_mask']

        assert actions.shape == (128,)
        # All sampled actions should be valid according to mask
        for i in range(128):
            assert mask[i, actions[i]] > 0.5


class TestActionEncoding:
    """Test action encoding/decoding."""

    def test_decode_accept(self):
        from cuda_bargain import BargainEnv
        action_type, offer = BargainEnv.decode_action(80)
        assert action_type == "ACCEPT"
        assert offer is None

    def test_decode_walk(self):
        from cuda_bargain import BargainEnv
        action_type, offer = BargainEnv.decode_action(81)
        assert action_type == "WALK"
        assert offer is None

    def test_decode_counteroffer(self):
        from cuda_bargain import BargainEnv

        # Test a few known encodings
        # action = n0 * 10 + n1 * 2 + n2
        action_type, offer = BargainEnv.decode_action(0)
        assert action_type == "COUNTEROFFER"
        assert offer == (0, 0, 0)

        action_type, offer = BargainEnv.decode_action(79)
        assert action_type == "COUNTEROFFER"
        assert offer == (7, 4, 1)

        action_type, offer = BargainEnv.decode_action(35)
        assert action_type == "COUNTEROFFER"
        # 35 = 3 * 10 + 2 * 2 + 1 = 30 + 4 + 1 = 35
        assert offer == (3, 2, 1)

    def test_encode_offer(self):
        from cuda_bargain import BargainEnv

        assert BargainEnv.encode_offer((0, 0, 0)) == 0
        assert BargainEnv.encode_offer((7, 4, 1)) == 79
        assert BargainEnv.encode_offer((3, 2, 1)) == 35

    def test_encode_decode_roundtrip(self):
        from cuda_bargain import BargainEnv

        for action_idx in range(80):
            action_type, offer = BargainEnv.decode_action(action_idx)
            assert action_type == "COUNTEROFFER"
            encoded = BargainEnv.encode_offer(offer)
            assert encoded == action_idx


class TestLargeScale:
    """Test with larger environments."""

    def test_large_batch_performance(self, large_env):
        """Test that large batches work correctly."""
        obs, info = large_env.reset()

        assert obs.shape == (8192, 92)
        assert info['action_mask'].shape == (8192, 82)

        # Run several steps
        for _ in range(10):
            actions = large_env.sample_valid_actions()
            obs, rewards, dones, _, info = large_env.step(actions)

            if dones.any():
                large_env.auto_reset()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
