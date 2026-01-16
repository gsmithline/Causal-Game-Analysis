"""Greedy baseline policy."""

import torch
from typing import Optional


class GreedyPolicy:
    """
    Greedy myopic policy.

    Player 1: Makes offers that maximize own value (gives minimum to P2)
    Player 2: Accepts if value > outside offer, else walks

    This is a simple baseline that doesn't consider opponent's values.
    """

    # Constants from bargaining game
    ACTION_ACCEPT = 80
    ACTION_WALK = 81
    ITEM_QUANTITIES = (7, 4, 1)

    def __init__(self):
        self.name = "greedy"

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Get greedy actions.

        Args:
            obs: [batch, obs_dim] observations
            action_mask: [batch, num_actions]
            deterministic: Ignored (always deterministic)

        Returns:
            actions: [batch] selected actions
        """
        device = obs.device
        batch_size = obs.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            action = self._get_greedy_action(obs[i], action_mask[i])
            actions[i] = action

        return actions

    def _get_greedy_action(self, obs: torch.Tensor, mask: torch.Tensor) -> int:
        """Get greedy action for single observation."""

        # Parse observation
        # [0-2] values, [3] outside, [4-6] offer, [7] valid, [8] round, [9] player
        player_values = obs[0:3] * 100  # Denormalize
        outside_offer = obs[3]
        current_offer = obs[4:7]
        offer_valid = obs[7] > 0.5
        current_player = int(obs[9].item())

        # Check if ACCEPT is valid
        can_accept = mask[self.ACTION_ACCEPT] > 0.5

        if current_player == 1:  # Player 2
            if offer_valid and can_accept:
                # Calculate value of current offer
                offer_value = 0.0
                for j, qty in enumerate(self.ITEM_QUANTITIES):
                    offer_items = current_offer[j].item() * qty
                    offer_value += offer_items * player_values[j].item()

                # Accept if better than outside offer
                # (outside_offer is normalized, need to compare properly)
                max_possible = sum(v * q for v, q in zip(player_values.tolist(), self.ITEM_QUANTITIES))
                outside_value = outside_offer.item() * max_possible

                if offer_value >= outside_value:
                    return self.ACTION_ACCEPT

            # If can't accept or offer not good, walk
            return self.ACTION_WALK

        else:  # Player 1
            # Make an offer that maximizes own value (give minimum to P2)
            # Find counteroffer that keeps most for self

            best_action = self.ACTION_WALK
            best_value = -float('inf')

            # Check all counteroffer actions
            for action in range(80):
                if mask[action] < 0.5:
                    continue

                # Decode action to offer (items given to P2)
                offer = self._decode_action(action)

                # Calculate value kept by P1
                p1_value = 0.0
                for j, qty in enumerate(self.ITEM_QUANTITIES):
                    p1_items = qty - offer[j]
                    p1_value += p1_items * player_values[j].item()

                if p1_value > best_value:
                    best_value = p1_value
                    best_action = action

            return best_action

    def _decode_action(self, action: int):
        """Decode action to offer tuple."""
        item2 = action % 2
        action //= 2
        item1 = action % 5
        item0 = action // 5
        return (item0, item1, item2)

    def get_action_probs(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get deterministic action as one-hot."""
        batch_size = obs.shape[0]
        num_actions = action_mask.shape[1]
        probs = torch.zeros(batch_size, num_actions, device=obs.device)

        actions = self.get_action(obs, action_mask)
        probs[torch.arange(batch_size, device=obs.device), actions] = 1.0

        return probs
