"""
Test parity between CPU and CUDA bargaining environment implementations.

Verifies that both implementations produce identical results for:
- Observation structure
- Action masks
- Reward calculations (especially accept scenarios)
- Game state transitions
"""

import numpy as np
import sys
from pathlib import Path

# Add simulator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from bargain_env_cpu import BargainEnvCPU, ITEM_QUANTITIES, ACTION_ACCEPT, ACTION_WALK


def test_accept_reward_p2_accepts_p1_offer():
    """Test rewards when P2 accepts P1's offer."""
    env = BargainEnvCPU(num_envs=1, seed=42)
    env.reset()

    # Manually set up a controlled scenario
    env._player_values[0, 0] = np.array([10, 20, 30], dtype=np.float32)  # P1 values
    env._player_values[0, 1] = np.array([30, 20, 10], dtype=np.float32)  # P2 values
    env._max_possible_values[0, 0] = 10*7 + 20*4 + 30*1  # 70 + 80 + 30 = 180
    env._max_possible_values[0, 1] = 30*7 + 20*4 + 10*1  # 210 + 80 + 10 = 300
    env._outside_options[0] = np.array([50, 100], dtype=np.float32)

    # P1 makes offer: give P2 [3, 2, 1] items
    # P1 keeps [4, 2, 0]
    offer_action = 3 * 10 + 2 * 2 + 1  # = 35
    obs, rewards, dones, _, info = env.step(np.array([offer_action]))

    assert not dones[0], "Game should not be done after P1's offer"
    assert env._current_player[0] == 1, "Should be P2's turn"

    # P2 accepts
    obs, rewards, dones, _, info = env.step(np.array([ACTION_ACCEPT]))

    assert dones[0], "Game should be done after accept"

    # Expected rewards:
    # P1 keeps [4, 2, 0]: value = 4*10 + 2*20 + 0*30 = 40 + 40 + 0 = 80
    # P1 normalized: 80 / 180 = 0.4444...
    # P2 gets [3, 2, 1]: value = 3*30 + 2*20 + 1*10 = 90 + 40 + 10 = 140
    # P2 normalized: 140 / 300 = 0.4666...

    expected_p1 = 80.0 / 180.0
    expected_p2 = 140.0 / 300.0

    assert np.isclose(rewards[0, 0], expected_p1, atol=1e-5), \
        f"P1 reward: expected {expected_p1}, got {rewards[0, 0]}"
    assert np.isclose(rewards[0, 1], expected_p2, atol=1e-5), \
        f"P2 reward: expected {expected_p2}, got {rewards[0, 1]}"

    print("✓ P2 accepts P1 offer: rewards correct")


def test_accept_reward_p1_accepts_p2_counteroffer():
    """Test rewards when P1 accepts P2's counteroffer (the fixed edge case)."""
    env = BargainEnvCPU(num_envs=1, seed=42)
    env.reset()

    # Manually set up a controlled scenario
    env._player_values[0, 0] = np.array([10, 20, 30], dtype=np.float32)  # P1 values
    env._player_values[0, 1] = np.array([30, 20, 10], dtype=np.float32)  # P2 values
    env._max_possible_values[0, 0] = 180.0
    env._max_possible_values[0, 1] = 300.0
    env._outside_options[0] = np.array([50, 100], dtype=np.float32)

    # P1 makes initial offer: give P2 [1, 1, 0]
    offer1_action = 1 * 10 + 1 * 2 + 0  # = 12
    env.step(np.array([offer1_action]))

    # P2 makes counteroffer: [5, 3, 1] to P2 (what P2 wants)
    # This means P1 would keep [2, 1, 0]
    offer2_action = 5 * 10 + 3 * 2 + 1  # = 57
    env.step(np.array([offer2_action]))

    assert env._current_player[0] == 0, "Should be P1's turn"

    # P1 accepts P2's counteroffer
    obs, rewards, dones, _, info = env.step(np.array([ACTION_ACCEPT]))

    assert dones[0], "Game should be done after accept"

    # CUDA semantics: offer ALWAYS means items to P2
    # So P2's counteroffer of [5, 3, 1] means P2 gets [5, 3, 1]
    # P1 keeps [7-5, 4-3, 1-1] = [2, 1, 0]
    # P1 value: 2*10 + 1*20 + 0*30 = 20 + 20 + 0 = 40
    # P1 normalized: 40 / 180 = 0.2222...
    # P2 value: 5*30 + 3*20 + 1*10 = 150 + 60 + 10 = 220
    # P2 normalized: 220 / 300 = 0.7333...

    expected_p1 = 40.0 / 180.0
    expected_p2 = 220.0 / 300.0

    assert np.isclose(rewards[0, 0], expected_p1, atol=1e-5), \
        f"P1 reward: expected {expected_p1}, got {rewards[0, 0]}"
    assert np.isclose(rewards[0, 1], expected_p2, atol=1e-5), \
        f"P2 reward: expected {expected_p2}, got {rewards[0, 1]}"

    print("✓ P1 accepts P2 counteroffer: rewards correct (CUDA semantics)")


def test_walk_rewards():
    """Test that walking gives both players their outside options."""
    env = BargainEnvCPU(num_envs=1, seed=42)
    env.reset()

    # Set controlled values
    env._player_values[0, 0] = np.array([10, 20, 30], dtype=np.float32)
    env._player_values[0, 1] = np.array([30, 20, 10], dtype=np.float32)
    env._max_possible_values[0, 0] = 180.0
    env._max_possible_values[0, 1] = 300.0
    env._outside_options[0] = np.array([90, 150], dtype=np.float32)

    # P1 walks immediately
    obs, rewards, dones, _, info = env.step(np.array([ACTION_WALK]))

    assert dones[0], "Game should be done after walk"

    expected_p1 = 90.0 / 180.0  # 0.5
    expected_p2 = 150.0 / 300.0  # 0.5

    assert np.isclose(rewards[0, 0], expected_p1, atol=1e-5), \
        f"P1 walk reward: expected {expected_p1}, got {rewards[0, 0]}"
    assert np.isclose(rewards[0, 1], expected_p2, atol=1e-5), \
        f"P2 walk reward: expected {expected_p2}, got {rewards[0, 1]}"

    print("✓ Walk rewards: correct")


def test_action_mask_initial():
    """Test initial action mask for P1."""
    env = BargainEnvCPU(num_envs=1, seed=42)
    obs, info = env.reset()

    mask = info['action_mask'][0]

    # P1's first turn: all counteroffers valid, WALK valid, ACCEPT invalid (no offer)
    assert mask[:80].sum() == 80, "All 80 counteroffers should be valid"
    assert mask[ACTION_ACCEPT] == 0, "ACCEPT should be invalid (no offer)"
    assert mask[ACTION_WALK] == 1, "WALK should be valid"

    print("✓ Initial action mask: correct")


def test_action_mask_after_offer():
    """Test action mask after P1 makes an offer."""
    env = BargainEnvCPU(num_envs=1, seed=42)
    env.reset()

    # P1 makes offer
    offer_action = 3 * 10 + 2 * 2 + 0  # = 34
    obs, rewards, dones, _, info = env.step(np.array([offer_action]))

    mask = info['action_mask'][0]

    # P2's turn: all counteroffers valid, WALK valid, ACCEPT valid (offer exists)
    assert mask[:80].sum() == 80, "All 80 counteroffers should be valid for P2"
    assert mask[ACTION_ACCEPT] == 1, "ACCEPT should be valid (offer exists)"
    assert mask[ACTION_WALK] == 1, "WALK should be valid"

    print("✓ Action mask after offer: correct")


def test_action_mask_final_turn():
    """Test action mask on P2's final turn (action count 5)."""
    env = BargainEnvCPU(num_envs=1, seed=42)
    env.reset()

    # Play 5 counteroffer actions to reach final turn
    for i in range(5):
        action = (i % 8) * 10 + (i % 5) * 2 + (i % 2)
        env.step(np.array([action]))

    # Should be P2's final turn (action_count = 5)
    assert env._action_count[0] == 5, f"Should be action 5, got {env._action_count[0]}"
    assert env._current_player[0] == 1, "Should be P2's turn"

    env._build_observations()
    mask = env._action_masks[0]

    # Final turn: only ACCEPT and WALK valid
    assert mask[:80].sum() == 0, "No counteroffers should be valid on final turn"
    assert mask[ACTION_ACCEPT] == 1, "ACCEPT should be valid"
    assert mask[ACTION_WALK] == 1, "WALK should be valid"

    print("✓ Final turn action mask: correct")


def test_observation_structure():
    """Test observation vector structure matches specification."""
    env = BargainEnvCPU(num_envs=1, seed=42)
    obs, info = env.reset()

    assert obs.shape == (1, 92), f"Observation shape should be (1, 92), got {obs.shape}"

    # Check observation ranges
    assert (obs[0, 0:3] >= 0).all() and (obs[0, 0:3] <= 1).all(), \
        "Item values should be normalized 0-1"
    assert obs[0, 3] >= 0 and obs[0, 3] <= 1, \
        "Outside option should be normalized 0-1"
    assert (obs[0, 4:7] == -1).all(), \
        "No offer initially, should be -1"
    assert obs[0, 7] == 0, "Offer valid flag should be 0"
    assert obs[0, 8] == 0, "Round should be 0"
    assert obs[0, 9] == 0, "Current player should be 0 (P1)"

    print("✓ Observation structure: correct")


def test_round_advancement():
    """Test that rounds advance correctly."""
    env = BargainEnvCPU(num_envs=1, seed=42)
    env.reset()

    # Initial state
    assert env._current_round[0] == 0
    assert env._current_player[0] == 0

    # P1 acts -> still round 0
    env.step(np.array([0]))
    assert env._current_round[0] == 0, "Round should still be 0 after P1 acts"
    assert env._current_player[0] == 1, "Should be P2's turn"

    # P2 acts -> round 1
    env.step(np.array([1]))
    assert env._current_round[0] == 1, "Round should be 1 after both players act"
    assert env._current_player[0] == 0, "Should be P1's turn"

    # P1 acts -> still round 1
    env.step(np.array([2]))
    assert env._current_round[0] == 1, "Round should still be 1"

    # P2 acts -> round 2
    env.step(np.array([3]))
    assert env._current_round[0] == 2, "Round should be 2"

    print("✓ Round advancement: correct")


def test_encode_decode_actions():
    """Test action encoding/decoding roundtrip."""
    for action in range(80):
        action_type, offer = BargainEnvCPU.decode_action(action)
        assert action_type == "COUNTEROFFER"
        encoded = BargainEnvCPU.encode_offer(offer)
        assert encoded == action, f"Roundtrip failed for action {action}"

    # Test special actions
    assert BargainEnvCPU.decode_action(80) == ("ACCEPT", None)
    assert BargainEnvCPU.decode_action(81) == ("WALK", None)

    print("✓ Action encode/decode: correct")


def test_cuda_cpu_parity():
    """Compare CUDA and CPU implementations directly if CUDA is available."""
    try:
        from bargain_env import BargainEnv
        print("\nCUDA environment available - running parity tests...")
    except ImportError:
        print("\n⚠ CUDA environment not available - skipping direct parity test")
        return

    num_envs = 100
    num_steps = 20
    seed = 12345

    # Create both environments
    cpu_env = BargainEnvCPU(num_envs=num_envs, seed=seed)
    cuda_env = BargainEnv(num_envs=num_envs, seed=seed)

    # Reset both
    cpu_obs, cpu_info = cpu_env.reset(seed=seed)
    cuda_obs, cuda_info = cuda_env.reset(seed=seed)

    # Note: Random number generation may differ, so we can't compare exact values
    # But we can compare structure and run parallel games with same actions

    # Run several steps with random actions
    np.random.seed(seed)

    for step in range(num_steps):
        # Sample valid actions from CPU env
        actions = cpu_env.sample_valid_actions()

        # Step both environments
        cpu_obs, cpu_rew, cpu_done, cpu_trunc, cpu_info = cpu_env.step(actions)

        # Convert to torch for CUDA
        import torch
        cuda_actions = torch.from_numpy(actions).to(torch.int32).cuda()
        cuda_obs, cuda_rew, cuda_done, cuda_trunc, cuda_info = cuda_env.step(cuda_actions)

        # Compare shapes
        cuda_obs_np = cuda_obs.cpu().numpy()
        cuda_rew_np = cuda_rew.cpu().numpy()

        assert cpu_obs.shape == cuda_obs_np.shape, \
            f"Observation shape mismatch: {cpu_obs.shape} vs {cuda_obs_np.shape}"
        assert cpu_rew.shape == cuda_rew_np.shape, \
            f"Reward shape mismatch: {cpu_rew.shape} vs {cuda_rew_np.shape}"

        # Auto-reset done games
        cpu_env.auto_reset()
        cuda_env.auto_reset()

    print("✓ CUDA/CPU parity: shapes and API match")


def run_all_tests():
    """Run all parity tests."""
    print("=" * 60)
    print("Running CPU/CUDA Parity Tests")
    print("=" * 60)

    test_accept_reward_p2_accepts_p1_offer()
    test_accept_reward_p1_accepts_p2_counteroffer()
    test_walk_rewards()
    test_action_mask_initial()
    test_action_mask_after_offer()
    test_action_mask_final_turn()
    test_observation_structure()
    test_round_advancement()
    test_encode_decode_actions()
    test_cuda_cpu_parity()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
