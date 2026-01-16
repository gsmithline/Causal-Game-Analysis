"""Fair split baseline policy (approximates Nash bargaining)."""

import torch


class FairSplitPolicy:
    """
    Fair split policy that proposes roughly equal value splits.

    This approximates a Nash bargaining solution by trying to
    split items so both players get similar value (from their
    own perspective, since values are private).

    Since we don't know opponent's values, this uses the assumption
    that opponent's values are similar to our own.
    """

    ACTION_ACCEPT = 80
    ACTION_WALK = 81
    ITEM_QUANTITIES = (7, 4, 1)

    def __init__(self, accept_threshold: float = 0.4):
        """
        Args:
            accept_threshold: Accept if offer gives us at least this
                            fraction of our max possible value
        """
        self.name = "fair_split"
        self.accept_threshold = accept_threshold

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Get fair split actions.

        Args:
            obs: [batch, obs_dim] observations
            action_mask: [batch, num_actions]
            deterministic: Ignored

        Returns:
            actions: [batch] selected actions
        """
        device = obs.device
        batch_size = obs.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            action = self._get_fair_action(obs[i], action_mask[i])
            actions[i] = action

        return actions

    def _get_fair_action(self, obs: torch.Tensor, mask: torch.Tensor) -> int:
        """Get fair action for single observation."""

        # Parse observation
        player_values = obs[0:3] * 100  # Denormalize
        outside_offer = obs[3]
        current_offer = obs[4:7]
        offer_valid = obs[7] > 0.5
        current_player = int(obs[9].item())

        # Calculate max possible value for this player
        max_value = sum(
            v * q for v, q in zip(player_values.tolist(), self.ITEM_QUANTITIES)
        )

        can_accept = mask[self.ACTION_ACCEPT] > 0.5

        if current_player == 1:  # Player 2 responding
            if offer_valid and can_accept:
                # Calculate value of current offer
                offer_value = 0.0
                for j, qty in enumerate(self.ITEM_QUANTITIES):
                    offer_items = current_offer[j].item() * qty
                    offer_value += offer_items * player_values[j].item()

                # Accept if we get at least threshold of our max value
                if offer_value >= self.accept_threshold * max_value:
                    return self.ACTION_ACCEPT

            # Otherwise walk
            return self.ACTION_WALK

        else:  # Player 1 making offer
            # Find offer that gives roughly 50% of total value (from our perspective)
            # Assuming opponent values items similarly

            target_p2_fraction = 0.5  # Give opponent ~50%
            best_action = self.ACTION_WALK
            best_diff = float('inf')

            for action in range(80):
                if mask[action] < 0.5:
                    continue

                offer = self._decode_action(action)

                # Calculate P2's share (from our value perspective)
                p2_value = 0.0
                for j, qty in enumerate(self.ITEM_QUANTITIES):
                    p2_value += offer[j] * player_values[j].item()

                p2_fraction = p2_value / max_value if max_value > 0 else 0

                # How close to target?
                diff = abs(p2_fraction - target_p2_fraction)
                if diff < best_diff:
                    best_diff = diff
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
