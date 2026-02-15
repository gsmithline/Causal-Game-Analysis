#!/usr/bin/env python3
"""
Test the best history transformer model with specific scenarios.
"""

import torch
import torch.nn.functional as F
from cuda_bargain import BargainEnv, NUM_ACTIONS, OBS_DIM
from policy import HistoryTransformerPolicy

# Action indices
ACTION_ACCEPT = 80
ACTION_WALK = 81
MAX_SEQ_LEN = 6


def decode_action(action: int) -> str:
    """Decode action to human-readable string."""
    if action == ACTION_ACCEPT:
        return "ACCEPT"
    elif action == ACTION_WALK:
        return "WALK"
    else:
        # Offer encoding: action = item0*10 + item1*2 + item2
        item0 = action // 10
        item1 = (action % 10) // 2
        item2 = action % 2
        return f"OFFER [{item0}, {item1}, {item2}]"


def create_test_observation(
    player_values: list,      # [v0, v1, v2] - values for 3 item types (0-100)
    outside_option: float,    # Walk-away value (0-100)
    current_offer: list = None,  # [a, b, c] or None if no offer
    current_round: int = 0,   # 0, 1, or 2
    current_player: int = 0,  # 0 for P1, 1 for P2
) -> tuple:
    """Create a test observation tensor."""

    # Item quantities: [7, 4, 1]
    item_counts = [7, 4, 1]
    max_value = sum(100 * c for c in item_counts)  # 700 + 400 + 100 = 1200

    obs = torch.zeros(OBS_DIM, device='cuda')

    # Normalize values to 0-1
    obs[0] = player_values[0] / 100.0
    obs[1] = player_values[1] / 100.0
    obs[2] = player_values[2] / 100.0

    # Outside option (normalized by max possible value)
    obs[3] = outside_option / max_value

    # Current offer
    if current_offer is not None:
        obs[4] = current_offer[0] / item_counts[0]
        obs[5] = current_offer[1] / item_counts[1]
        obs[6] = current_offer[2] / item_counts[2]
        obs[7] = 1.0  # Offer validity flag
    else:
        obs[4:7] = -1.0
        obs[7] = 0.0

    # Round (0, 0.5, or 1.0)
    obs[8] = current_round / 2.0

    # Current player
    obs[9] = current_player

    # Action mask (all valid for now, simplified)
    obs[10:] = 1.0
    if current_offer is None:
        obs[10 + ACTION_ACCEPT] = 0.0  # Can't accept if no offer

    return obs


def test_outside_option_scenario():
    """Test what the model does when outside option equals max item value."""

    print("=" * 70)
    print("TESTING: Outside option = Max item value")
    print("=" * 70)

    # Load best model
    policy = HistoryTransformerPolicy(token_dim=175, num_actions=NUM_ACTIONS).cuda()
    policy.load_state_dict(torch.load("solver/history_30M_best_p1.pt"))
    policy.eval()

    # Scenario: Player values items at [100, 100, 100]
    # Item counts are [7, 4, 1]
    # Max value from items = 7*100 + 4*100 + 1*100 = 1200
    # Outside option = 1200 (same as max)

    player_values = [100, 100, 100]
    max_item_value = 7*100 + 4*100 + 1*100  # 1200

    print(f"\nPlayer values: {player_values}")
    print(f"Max possible from items: {max_item_value}")

    test_cases = [
        (max_item_value, "Outside = Max (1200)"),
        (max_item_value * 0.9, "Outside = 90% Max (1080)"),
        (max_item_value * 0.5, "Outside = 50% Max (600)"),
        (max_item_value * 0.1, "Outside = 10% Max (120)"),
    ]

    for outside, desc in test_cases:
        print(f"\n--- {desc} ---")

        # Create observation
        obs = create_test_observation(
            player_values=player_values,
            outside_option=outside,
            current_offer=None,  # P1's first move, no offer yet
            current_round=0,
            current_player=0,
        )

        # Create history tensor (single token)
        history = torch.zeros(1, MAX_SEQ_LEN, 175, device='cuda')
        history[0, 0, :OBS_DIM] = obs
        history[0, 0, 174] = 1.0  # validity
        seq_len = torch.tensor([1], device='cuda')

        # Create action mask
        action_mask = obs[10:].unsqueeze(0)

        # Get action distribution
        with torch.no_grad():
            logits, value = policy(history, seq_len, action_mask)
            probs = F.softmax(logits, dim=-1)

            # Get top 5 actions
            top_probs, top_actions = probs[0].topk(5)

            print(f"Value estimate: {value.item():.4f}")
            print("Top 5 actions:")
            for p, a in zip(top_probs, top_actions):
                print(f"  {decode_action(a.item())}: {p.item()*100:.1f}%")

            # Specifically check walk probability
            walk_prob = probs[0, ACTION_WALK].item()
            print(f"Walk probability: {walk_prob*100:.1f}%")


def test_with_offer():
    """Test response to various offers when outside option is high."""

    print("\n" + "=" * 70)
    print("TESTING: Response to offers with high outside option")
    print("=" * 70)

    # Load best model
    policy = HistoryTransformerPolicy(token_dim=175, num_actions=NUM_ACTIONS).cuda()
    policy.load_state_dict(torch.load("solver/history_30M_best_p1.pt"))
    policy.eval()

    player_values = [100, 100, 100]
    max_item_value = 1200
    outside = max_item_value  # Outside = max

    print(f"\nPlayer values: {player_values}")
    print(f"Outside option: {outside}")

    # Test different offers (opponent is giving us items)
    test_offers = [
        ([7, 4, 1], "All items (max)"),
        ([6, 3, 1], "Most items"),
        ([3, 2, 0], "Half items"),
        ([1, 1, 0], "Few items"),
        ([0, 0, 0], "No items"),
    ]

    for offer, desc in test_offers:
        value_from_offer = sum(o * v for o, v in zip(offer, player_values))
        print(f"\n--- Offer: {offer} ({desc}) ---")
        print(f"Value from offer: {value_from_offer}")

        obs = create_test_observation(
            player_values=player_values,
            outside_option=outside,
            current_offer=offer,
            current_round=0,
            current_player=1,  # P2's turn (responding to offer)
        )

        history = torch.zeros(1, MAX_SEQ_LEN, 175, device='cuda')
        history[0, 0, :OBS_DIM] = obs
        history[0, 0, 174] = 1.0
        seq_len = torch.tensor([1], device='cuda')
        action_mask = obs[10:].unsqueeze(0)

        with torch.no_grad():
            logits, value = policy(history, seq_len, action_mask)
            probs = F.softmax(logits, dim=-1)

            accept_prob = probs[0, ACTION_ACCEPT].item()
            walk_prob = probs[0, ACTION_WALK].item()

            print(f"Accept: {accept_prob*100:.1f}%, Walk: {walk_prob*100:.1f}%")

            # Should walk if offer value < outside option
            expected = "WALK" if value_from_offer < outside else "ACCEPT"
            actual = "ACCEPT" if accept_prob > walk_prob else "WALK (or counter)"
            print(f"Expected: {expected}, Actual preference: {actual}")


if __name__ == "__main__":
    test_outside_option_scenario()
    test_with_offer()
